// Model runner, Phase 1: one sequence, chunked prefill and decode through the same path.
#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "draft.h"
#include "kernels.h"
#include "memory.h"
#include "prefix_cache.h"
#include "safetensors.h"
#include "vision.h"
#include "weights.h"

namespace q27 {

struct Options {
  std::string model_dir;
  int max_tokens = 32768;     // context capacity of the sequence (KV pages)
  int prefill_chunk = 2048;   // tokens per forward chunk
  int logits_rows = 256;      // rows per lm_head slice when all-position logits are requested
  KvType kv_type = KvType::BF16;
  std::string draft_dir;  // DFlash2 checkpoint (speculative decoding); empty: none
  // Prefix cache (off when state_cache_gib is 0): pinned host memory for state checkpoints, the KV
  // pool (0: what the VRAM budget leaves; without the cache, max_tokens) and the interval of the
  // checkpoints inside long prompts.
  double state_cache_gib = 0;
  int kv_pool_tokens = 0;
  int checkpoint_interval = 4096;
  // Tiered cache (needs the prefix cache): GiB of host memory for checkpoints the KV pool gives up (0: off, they are
  // dropped as before), and the folder they are written to when that is full (empty: the oldest is dropped then).
  double spill_ram_gib = 0;
  std::string spill_dir;
  // Tiered cache speed-ups, both OFF by default (docs/tiered-cache.md, "加速選項"). spill_pinned_gib: pinned host memory
  // (in slabs of spill_pinned_slab_mib, pinned while idle by spill_grow_pinned) that spill entries are stored in when
  // it has room, else they use pageable memory as before (0: off). spill_async_io: copy checkpoint pages on a
  // dedicated I/O stream instead of the compute stream, and do not stall the compute stream or the CPU for them.
  double spill_pinned_gib = 0;
  int spill_pinned_slab_mib = 1024;
  bool spill_async_io = false;
  // SPILL_ASYNC_PROMOTE (OFF by default; needs spill_ram_gib > 0): the frontend may restore a spilled checkpoint in the
  // background (spill_match / spill_restore_begin / spill_restore_poll) while other requests keep running, instead of
  // prefix_begin promoting it synchronously. With it off those calls do nothing and every code path is unchanged.
  bool spill_async_promote = false;
  // Vision tower (P4-12): 0 off, 1 BF16 weights on the device, 2 BF16 weights in pinned host memory,
  // 3 FP8 weights on the device (VisionMode). Images of up to image_max_pixels; the image embeddings of a
  // request (up to image_tokens rows) live in pinned host memory.
  int vision = 0;
  int image_max_pixels = 4194304;
  int image_tokens = 16384;
  // Several sequences (Phase 6): device areas a running sequence's state can be parked in while
  // another one runs (MAX_SEQS - 1 in the server; 0: one sequence at a time).
  int park_slots = 0;
  // Lower max_tokens to what the VRAM budget leaves for the KV pool (rounded down to 1,024) instead of
  // failing (MAX_TOKENS_FIT; the server sets it). Not with kv_pool_tokens.
  bool fit_max_tokens = false;
  // Fail instead of starting with a KV pool of fewer tokens (MIN_KV_TOKENS; 0: any size).
  int min_kv_tokens = 0;
  // Plan only (q27_plan): size the pools for a device with plan_free of plan_total bytes free and stop there, with
  // no CUDA call (nothing allocated or loaded; memory_report, max_tokens, cache_stats and device_memory answer).
  bool plan_only = false;
  size_t plan_free = 0, plan_total = 0;
};

class Model {
 public:
  explicit Model(const Options& opt);
  ~Model();

  void reset();
  void configure();  // re-reads the Q27_* switches from the environment (see model.cc)
  int pos() const { return pos_; }
  int max_tokens() const { return opt_.max_tokens; }  // (after fit_max_tokens)

  // Runs ids[0..n) after the tokens already in the sequence. For each position p the callback gets
  // the top-k (ids, log-probs) of the next token. k <= 64.
  void forward_topk(const int32_t* ids, int n, int k,
                    const std::function<void(int pos, const int32_t* ids, const float* logp)>& cb);
  // Runs ids[0..n) and samples the token after the last one. presence / frequency: penalties (Penalty) by
  // the running sequence's output counts, which this and the speculative steps keep (see penalty_init).
  // logprobs >= 0: the token's logprobs with that many top entries (see logprobs()). masked: the logits are masked by
  // row 0 of mask_rows() first. biased: the running sequence's logit bias (bias_set) is added to the logits first.
  int32_t forward_sample(const int32_t* ids, int n, float temperature, int top_k, float top_p, float uniform,
                         float min_p = 0.f, float presence = 0.f, float frequency = 0.f, int logprobs = -1,
                         bool masked = false, bool biased = false);
  // The logprobs of the tokens the last forward_sample / spec_begin / spec_step returned (lane -1), or of lane z of
  // the last spec_step_lanes; n = 0 when that sequence's sampling has none (SpecSampling::logprobs).
  const Logprobs& logprobs(int lane = -1) const { return last_lp_[lane < 0 ? 0 : lane]; }
  // The running sequence's output so far, ids[0..n), as the penalty counts (before the first sampled
  // token of a request with penalties; parked sequences keep theirs).
  void penalty_init(const int32_t* ids, int n);
  // The running sequence's logit bias (BiasList; n <= kMaxBias entries, n = 0: none) and min_tokens (eos_before: the
  // end-of-sequence ids are banned for the tokens at positions below it; 0: none), used by the calls whose sampling
  // is biased (forward_sample's biased, SpecSampling::biased); parked sequences keep theirs.
  void bias_set(const int32_t* ids, const float* vals, int n, int eos_before = 0);

  // Per-op GPU time (Q27_PROFILE=1): every op records an event when it starts; prints the totals
  // per label since the last dump to stderr and resets them.
  void profile_dump();

  // Speculative decoding (DFlash2; needs Options::draft_dir). spec_begin runs the prompt like
  // forward_sample (also filling the draft's context) and returns the first token; each spec_step
  // drafts 7 tokens after the last one, verifies them and writes the new tokens (accepted drafts and
  // the bonus token, 1..8) to out. Other forward calls may follow (pending GDN commits are flushed).
  // max_new > 0: the steps output at most max_new tokens in all (with the first one), so the sequence
  // never holds tokens past the caller's limit (the prefix cache checkpoints it at the end).
  int32_t spec_begin(const int32_t* ids, int n, const SpecSampling& smp, uint64_t seed, float uniform,
                     int max_new = 0);
  // ahead (default): launch the next step before returning this one (Q27_SPEC_PIPELINE); false when
  // the caller will not continue right away (another sequence runs next), so no step is wasted.
  int spec_step(int32_t* out, bool ahead = true);
  // Grammar masks (structured output): kMaskRows bitmasks of kMaskWords words in pinned host memory that the
  // host writes (bit t % 32 of word t / 32: token t allowed); row 8 z + i masks verify row i of lane z (z = 0 alone).
  // A sequence begun with SpecSampling::masked steps in two calls: spec_draft drafts the step and returns its verify
  // tokens (chain[8]: the last token, then the 7 drafted ones; such a sequence verifies a chain), the host writes the
  // mask of each row (the grammar after the row's tokens), and spec_step verifies and accepts with them (no step is
  // launched ahead). spec_draft_lanes is the same for spec_step_lanes (chain [n][8]; lanes without masks need it too
  // when one lane has them).
  static constexpr int kMaskWords = (dims::kVocab + 31) / 32;
  static constexpr int kMaskRows = kMaxLanes * kSpecBlock;
  uint32_t* mask_rows() const { return mask_host_; }
  void spec_draft(int32_t* chain);
  void spec_draft_lanes(const int32_t* lanes, int n, int32_t* chain);
  // Masks a decoding sequence (area -1: the running one, else a parked one) from its next step on, as if begun with
  // SpecSampling::masked (the grammar starts after the reasoning; its end, stop[2], no longer ends a step); a step
  // launched ahead is dropped.
  void spec_set_masked(int area);
  bool has_draft() const { return draft_ != nullptr; }
  // Counters since the last call: verify steps, steps that used the lookup chain, tokens produced.
  void spec_stats(int64_t* out);
  // Debug: KV rows (K then V, raw bytes) of attention layer `layer` at positions pos..pos+n-1, and the
  // committed recurrent state of GDN layer `gdn_layer` (fp32 [48][128][128]); pending commits flushed.
  void debug_kv(int layer, int pos, int n, void* out);
  void debug_ssm(int gdn_layer, float* out);

  // Prefix cache (Options::state_cache_gib > 0; otherwise this is reset() and returns 0).
  // prefix_begin resets the sequence and restores the deepest checkpoint whose tokens are a prefix of
  // ids[0..n-1) (one token at least is left to run) and returns its length `cached`. The caller then
  // runs ids[cached..n) (forward_sample or spec_begin); that prefill saves a checkpoint at each of
  // at[0..nat) in (cached, n) and at the last few multiples of checkpoint_interval. spec: the caller
  // continues with spec_begin, so only checkpoints with the draft's context qualify.
  int prefix_begin(const int32_t* ids, int n, const int32_t* at, int nat, bool spec);
  // Saves a checkpoint of the sequence so far (e.g. at the end of a generation).
  void checkpoint();
  // out[12]: lookups, hits, prompt tokens, cached tokens, checkpoints saved, evicted, live checkpoints, free
  // pages, pages only checkpoints hold (free after evicting them), pool pages, KV pages of a parking area, state
  // slots allocated.
  void cache_stats(int64_t* out) const;
  // Allocates up to n more pinned state slots of the prefix cache (they are not allocated at startup); returns how
  // many it allocated (0: all are). A checkpoint that finds no free slot allocates one itself.
  int grow_state_cache(int n);
  // Tiered cache (Options::spill_ram_gib > 0). Only the frontend's policy calls these, between engine steps.
  // spill_list: one line per entry, "id\tpos\tram_bytes\ton_disk\tlast_use\tlive\tpath" (live: 1 = an idle backup whose
  // checkpoint is still in the pool). spill_export writes an entry
  // that is in memory to `path` and frees the memory (it stays found by the lookup). spill_drop forgets an entry and
  // deletes its file. spill_import indexes a file another run wrote (0 when it is not a file of this model and KV
  // type, or unreadable: the caller deletes it). spill_stats writes min(n, kSpillStats) values to out: entries, in memory, on disk, memory bytes,
  // spilled, restored, written, dropped, subsumed (a deeper checkpoint of the same tokens was kept), failed, backed_up;
  // out[11..14] (appended): pinned pool bytes, pinned bytes in use, entries given pinned memory, entries that fell back to
  // pageable memory (both only counted when SPILL_PINNED_GIB > 0); out[15..19] (appended, SPILL_ASYNC_PROMOTE): restores
  // in progress, async restores finished, async restores that failed, their total wall time in ms and bytes (begin ->
  // visible). kSpillStats = 20 values in all. spill_list's "live" column is 0 none, 1 an idle backup, 2 busy (being restored by
  // spill_restore_*: neither written out nor dropped meanwhile); the line format is unchanged.
  std::string spill_list() const;
  void spill_export(uint64_t id, const std::string& path);
  void spill_drop(uint64_t id);
  uint64_t spill_import(const std::string& path);
  static constexpr int kSpillStats = 20;
  void spill_stats(int64_t* out, int n) const;
  // SPILL_PINNED_GIB: pins up to `chunks` more slabs (Options::spill_pinned_slab_mib each, cudaHostAlloc) of the pinned
  // pool; returns how many it added, 0 when the pool is full (or pinning failed once: it is not retried). Pinning
  // takes ~0.45 s per GiB on WSL2 and other CUDA calls wait meanwhile, so only the idle frontend calls this.
  int spill_grow_pinned(int chunks);
  // Idle backup: copies the least recently used pool checkpoint that has no valid backup to host memory and keeps it
  // in the pool too. Its eviction then only marks the copy as the sole one (no copy stall). 1: one was copied, 0: none
  // needed it (or no tiered cache). Called by the frontend while the engine is idle, repeatedly until 0.
  int spill_backup_one(int victims = 0);
  // SPILL_ASYNC_PROMOTE (Options::spill_async_promote; delay-hit scheduling: the frontend admits other requests while
  // an entry loads). All three run on the engine thread between steps and never block on the GPU or the disk.
  // spill_match: the best spilled entry (not an idle backup) deeper than the pool's best visible match, by spill_lookup's
  // criteria (tokens ids[0..n) start with the entry's tokens, pos < n, ring when spec). A restoring entry is returned
  // too (so a second request waits for the same load). out4 = {id, pos, kv_bytes + state_bytes, 1 if it is only on disk}.
  // 1: found. Read-only.
  int spill_match(const int32_t* ids, int n, bool spec, int64_t* out4) const;
  // Starts restoring the entry into the pool. 1: its host -> device copy was queued (RAM entry), 2: a thread reads its
  // file (disk entry), 0: not started (flag off, unknown / live / busy entry, no room for its pages, no memory for its
  // buffers; a damaged entry is dropped). The finished id comes out of spill_restore_poll, from when prefix_begin finds it.
  int spill_restore_begin(uint64_t id);
  // Advances every restore (a finished file read gets its pages and copy queued; a finished copy makes its checkpoint
  // visible) and writes the ids that ended, whether they succeeded or not, to done_ids[0..cap). Returns how many.
  // done_ok[i]: 1 restored, 0 failed (the entry is gone or back on disk only); a null done_ok is allowed.
  int spill_restore_poll(int64_t* done_ids, int32_t* done_ok, int cap);
  // The request that waited for restore `id` no longer needs it (timeout, cancel): its checkpoint stops being held (or, if
  // it is still loading, will not be). 1: something was released.
  int spill_release(uint64_t id);
  // false: prefix_begin does not promote a spilled match synchronously (the frontend decided to compute it instead).
  void spill_set_sync_lookup(bool on) { spill_sync_lookup_ = on; }
  // Memory (stats UI): one line per pool ("pool\t<name>\t<host 0|1>\t<bytes>") followed by its reservations
  // ("item\t<name>\t<what>\t<bytes>"), and "slots\t<state cache slots>\t<bytes of a checkpoint>". Fixed after
  // the constructor. device_memory: out[4] = free now, total (cudaMemGetInfo), free at the start, budget.
  std::string memory_report() const;
  void device_memory(int64_t* out) const;

  // Prompt chunks before forward_sample / spec_begin (Phase 6: other sequences can run between them).
  // ids[0..n) is the rest of the prompt after the sequence's position; runs chunks while more than one
  // prefill chunk is left, stopping after at least `budget` tokens, and returns the tokens run. The rest
  // then goes to forward_sample / spec_begin with the same chunk cuts as one call over all of it.
  // spec: the caller continues with spec_begin (the draft's context is filled the same way).
  int prefill(const int32_t* ids, int n, int budget, bool spec);

  // Several sequences (Phase 6). The running sequence (its GDN / conv state, the draft's context ring,
  // device scalars, KV pages, position and tokens) can be moved to one of Options::park_slots device
  // areas and resumed later, bit for bit. seq_park moves it to a free area (returned; the engine then
  // has no sequence, as after reset). seq_unpark(a) drops the running sequence (reset) and resumes the
  // one in area a. seq_swap(a) exchanges the running sequence with the one in area a. seq_drop(a) gives
  // a parked sequence's pages back; seq_suspend(a) first saves it as a prefix cache checkpoint (a
  // preempted sequence resumes from the cache). A parked sequence keeps its pages (KV is never copied).
  // P7-13: an area's pages hold KV while it is free; seq_park first moves those pages elsewhere (KvFull,
  // with nothing changed but evicted checkpoints, when the pool cannot take them).
  int seq_park();
  void seq_unpark(int a);
  void seq_swap(int a);
  void seq_drop(int a);
  void seq_suspend(int a);
  int park_slots() const { return int(parked_.size()); }
  // Adds a checkpoint position to the plan of a sequence that is already prefilling: the running one (a < 0) or the
  // one in parking area a. pos is rounded down to a multiple of 64 and must lie more than 64 tokens past the
  // sequence's position and from every planned one (a shorter chunk takes the decode GEMMs and would not match an
  // uncut prefill bit for bit, prefix_begin); the caller also keeps it more than 64 before the end of that prompt
  // (the engine does not know it). True also when it is planned already; false, with nothing changed, when it cannot be.
  bool add_checkpoint_at(int a, int pos);

  // Batched speculative steps (P6-10). One step of each of n (2..4) sequences in decode (after spec_begin):
  // lanes[z] is -1 for the running sequence or a parking area. Their draft blocks and verify rows run as one
  // batch (the GEMMs read the weights once for all), each lane exactly as its own spec_step: the tokens are
  // the same, bit for bit. Lane z's new tokens go to out[8 z ..], their count to counts[z]. While the same
  // lanes keep stepping their state stays in the batch; any other call first ends it (the lanes then hold
  // their sequences as before, nothing pending). Chains only (Q27_SPEC_PATHS 1); no lookup drafting.
  void spec_step_lanes(const int32_t* lanes, int n, int32_t* out, int32_t* counts);
  int max_lanes() const { return draft_ ? draft_->lanes() : 0; }

  // Images (P4-12). The prompt holds each image as a run of gh * gw ids >= kVocab (any values: the
  // frontend derives them from the image, so the prefix cache tells images apart) at positions
  // start .. start + gh * gw - 1, gh x gw the merged grid. mm_set (after reset / prefix_begin, before the
  // forward call) gives the request's images: spans [n][3] = (start, gh, gw), ascending. mm_encode runs
  // the vision tower on image i (its pixels in host memory: VisionPixels, 2 gh x 2 gw patches) into its
  // embedding rows; images whose positions the restored prefix covers need no encoding. mm_embeds copies them
  // out (tests).
  void mm_set(const int32_t* spans, int n);
  void mm_encode(int i, const VisionPixels& pixels);
  void mm_embeds(int i, bf16* out);
  void mm_put(int i, const bf16* in);  // replaces image i's embeddings (tests: another vision tower)
  bool has_vision() const { return opt_.vision != 0; }
  VisionMode vision_mode() const {
    return opt_.vision == 2 ? VisionMode::kHost : opt_.vision == 3 ? VisionMode::kFp8 : VisionMode::kDevice;
  }

 private:
  struct Act;
  void upload_ids(const int32_t* ids, int T);  // ids_ of the chunk at pos_ (image rows, M-RoPE positions)
  void copy_hist(int32_t* dst, const int32_t* ids, int n);  // lookup history (image rows as the image pad)
  void chunk(const int32_t* ids_dev, int T, bool advance = true);  // advances the device position (not pos_)
  int prefill_cut(int T, bool last) const;  // chunk length from pos_: up to the next planned checkpoint, or a multiple of 64
  void ensure_pages(int tokens);     // maps the logical pages of positions [0, tokens)
  // ... of a sequence with these pages and device page table (the running one's, or a batch lane's)
  void ensure_pages_of(std::vector<int32_t>& pages, int32_t* table, int tokens);
  // Writes physical pages to a device page table (as the kernels read it: page * kKvSlices, P7-13).
  void upload_pages(int32_t* table, const int32_t* pages, size_t n);
  void release_seq();                // returns the running sequence's pages
  void save_checkpoint(bool ring);   // at pos_ (nothing pending)
  // A checkpoint of tokens ids (pages: their KV pages) whose state is in a parking area (null: the running sequence's).
  void save_checkpoint_from(const char* area, const std::vector<int32_t>& ids, const std::vector<int32_t>& pages,
                            bool ring);
  int prompt_chunks(const int32_t* ids, int n, int budget, bool keep_tail, int* last);
  bool evict_checkpoint(bool for_slot = false);           // the least recently used one (for_slot: a covered one first); false when there is none
  // Tiered cache (spill.cc).
  bool spill_on() const { return cache_on_ && opt_.spill_ram_gib > 0; }
  void spill_checkpoint(const Checkpoint& c);    // a copy of c in host memory (before evict_checkpoint unrefs it)
  bool spill_subsumed(const Checkpoint& c, uint64_t ignore_id) const;  // a deeper checkpoint covers c (ignore_id: skipped)
  bool ckpt_covered(const Checkpoint& c) const;  // a deeper, usable checkpoint of the same chain is in the pool and c is long
  uint64_t spill_copy(const Checkpoint& c, bool live);  // the copy of c's pages and state as a new entry; 0: no memory
  uint64_t spill_add(const Checkpoint& c, Spilled&& s, bool live);  // s (copied from c) becomes entry spill_next_; its id
  void spill_lookup(const int32_t* ids, int n, bool spec);  // brings back a spilled match deeper than the pool's
  // The best spilled entry deeper than the pool's visible match (0: none); *depth its pos. with_busy: restoring ones too.
  uint64_t spill_best(const int32_t* ids, int n, bool spec, bool with_busy, int* depth) const;
  bool spill_room(int npages) const;  // free pages + pages only visible checkpoints hold cover npages
  // A state slot and npages pages (evicting checkpoints as needed); false: nothing is held.
  bool spill_acquire(int npages, int* slot, std::vector<int32_t>* pages);
  int spill_restore_issue(uint64_t id);  // entry in memory + busy: slot, pages, copy. 0 no room, 1 loading checkpoint, 2 done
  void spill_restore_finish(uint64_t id);  // counters; the entry is erased
  void spill_loading_done(Checkpoint& c);  // a loading checkpoint's copy is complete: visible, held (unless orphaned), reported ok
  bool spill_promote(uint64_t id);               // a spilled entry -> a checkpoint in the pool
  void spill_enforce_ram();                      // over spill_ram_gib: write out (or drop) the least recently used
  // Host bytes the tiered cache holds: spill_ram_ counts the entries (a pinned entry's KV included), and adding the
  // pool's free part makes it pageable + state + the whole pinned pool.
  size_t spill_host_use() const { return spill_ram_ + (pin_pool_.total() - pin_pool_.used()); }
  void spill_write(Spilled& s, const std::string& path);
  bool spill_load(Spilled& s);                   // the file's pages and state into memory
  void spill_erase(std::map<uint64_t, Spilled>::iterator it, bool remove_file);
  // pages' K / V slices <-> host [kKvSlices][pages][kv_page_bytes_] (runs of adjacent pages in one copy)
  void copy_pages(const std::vector<int32_t>& pages, const HostKv& host, bool to_host);  // on s_, then waits for the copy
  void copy_pages_on(const std::vector<int32_t>& pages, const HostKv& host, bool to_host, cudaStream_t st);  // queues only
  // SPILL_ASYNC_IO (io_s_ exists): see the invariants at the top of spill.cc.
  bool spill_async() const { return io_s_ != nullptr; }
  void io_begin();                       // io_s_ waits for everything queued on s_ so far (before it reads or writes pool pages)
  void spill_wait(Spilled& s);           // CPU: the entry's in-flight copy has completed (no-op unless inflight)
  void spill_reap();                     // frees the host memory of promoted entries whose host -> device copy is done
  HostKv spill_alloc_kv(size_t bytes);   // pinned segments if the pool holds ALL of it (SPILL_PINNED_GIB > 0), else one new char[]
  cudaEvent_t spill_event();             // a recycled / new event (no timing)
  void spill_event_put(cudaEvent_t ev);
  uint64_t spill_fingerprint() const;
  void spec_body(const SpecSampling& smp, int part = 0);  // part 1: the draft only, 2: the rest (0: all)
  void spec_part(const SpecSampling& smp, int part);      // a masked step's part, captured by its settings
  cudaGraphExec_t capture(const std::function<void()>& body);
  void spec_flush();  // replays the last verify's commits into the GDN / conv states
  void set_sbuf(int b);  // host side of the GDN state buffer switch
  int spec_rows(const SpecSampling& smp) const { return 1 + (kSpecBlock - 1) * std::max(1, smp.paths); }
  GdnSpec gdn_spec(int gdn_layer) const;
  bool fused(int T) const;
  bf16* cap_for(int layer) const;
  int32_t decode_step(int32_t id, float temperature, int top_k, float top_p, float uniform, float min_p, Penalty pen,
                      LogprobSink lp, bool masked, bool biased);
  // Penalty counts [vocab]: the running sequence's (-1), or parking area a's.
  uint32_t* pen_counts(int a = -1) const { return pen_ + size_t(a + 1) * dims::kVocab; }
  // Logit bias list: the running sequence's (-1), or parking area a's.
  BiasList* bias_list(int a = -1) const { return bias_ + (a + 1); }
  BiasLanes running_bias() const {  // (a decode or prompt row)
    BiasLanes b;
    b.list[0] = bias_, b.pos[0] = pos_dev_;
    return b;
  }
  void logits(const float* x_rows, int rows);  // final norm + lm_head into logits_
  void gdn(int layer, int T);
  void attn(int layer, int T);
  void mlp(int layer, int T);
  // GEMM on the activation already quantized into aq_ / asf_ (out += when residual).
  void gemm_q(const char* what, const LinearFp8& l, int T, float* out, bool residual);
  void gemm_q(const char* what, const LinearFp4& l, int T, float* out, bool residual);
  void gemm_q(const char* what, const LinearQ& l, int T, float* out, bool residual) {
    l.fp4 ? gemm_q(what, l.f4, T, out, residual) : gemm_q(what, l.f8, T, out, residual);
  }
  QuantOut q8(float in_scale) const;  // quantize into aq_
  QuantOut q4(float in_scale) const;  // quantize into aq_ / asf_
  QuantOut q8row() const;             // quantize into aq_ per token, the row scales into ascale_
  QuantOut qin(const LinearQ& l) const { return l.fp4 ? q4(l.in_scale()) : l.dyn ? q8row() : q8(l.in_scale()); }
  // The input of a projection from its fp32 rows x [T][K] (FP8 projections: the producing kernel already
  // wrote aq_; NVFP4 ones quantize here).
  void quant_in(const LinearQ& l, const float* x, int T, int K);
  void mlp_fp8(const LayerWeights& L, int T);
  void mark(const char* label);  // start of an op (profiling only)
  void profile_collect(bool graph = false);  // after a step (graph: the replayed marks)

  struct Prof {
    struct Marks {
      std::vector<cudaEvent_t> events;
      std::vector<const char*> labels;  // labels[i] is timed from events[i] to events[i + 1]
      int used = 0;
    } eager, graph;
    bool on = false, capturing = false;
    std::vector<std::pair<std::string, std::pair<double, long>>> totals;  // label -> (ms, count)
    int steps = 0;
  } prof_;

  Options opt_;
  std::unique_ptr<SafeTensors> ckpt_;
  std::unique_ptr<SafeTensors> lm_ckpt_;  // Q27_LM_HEAD_CKPT: the NVFP4 lm_head of a BF16-lm_head model
  Budget vram_;                      // the VRAM budget (make_budget), for device_memory
  Pool dev_weights_{"weights"}, host_weights_{"embedding", true}, kv_{"kv"}, state_{"state"}, act_{"activations"},
      host_cache_{"state cache", true}, host_mm_{"image embeddings", true};
  // Load-time scratch placed in the KV pool (unused until the first request): the NVFP4 copy of an FP8 lm_head that
  // the draft model only gathers its vocabulary head from.
  Pool load_tmp_{"load-time scratch"};
  Weights w_;
  cudaStream_t s_ = nullptr;
  int pos_ = 0;
  int max_pages_ = 0;

  // Sequence state.
  std::vector<KvLayer> kv_layers_;   // per attention layer
  int32_t* page_table_ = nullptr;    // logical -> physical page; unmapped entries hold the scratch page 0
  PagePool pool_;                    // physical pages 1 .. pool size
  std::vector<int32_t> seq_pages_;   // the running sequence's pages (host mirror of page_table_)
  std::vector<int32_t> seq_ids_;     // tokens fed so far (== pos_ once nothing is pending)
  int32_t spec_last_ = -1;           // last token returned by spec_begin / spec_step (fed by the next step)
  bool ring_valid_ = false;          // the draft's context ring holds the last 2,048 positions
  int ring_from_ = 0;                // prefill with a draft: the context rows from this position on were all appended (earlier chunks skip those a window before the next checkpoint)
  size_t kv_page_bytes_ = 0;         // one page of one layer's K (or V)
  // The KV pool is block-major (P7-13): pages come in blocks of kv_block_; a block holds the K and V slices
  // (kKvSlices: layer l's K is slice 2 l, its V 2 l + 1) of its pages, slice by slice, each slice the block's
  // pages in a row. The device page tables hold page_ref(p), so the kernels address a page of a layer's K as
  // before (base + page_ref * kv_page_bytes_, base = that slice of block 0). Whole blocks in a row are
  // contiguous memory (a parking area), and a slice of a block spans >= 2 MiB, so a long context's pages of
  // one layer do not each touch another GPU page (one page per block costs 1.5% at 131K).
  static constexpr int kKvSlices = 2 * dims::kAttnLayers;
  char* kv_base_ = nullptr;
  int kv_block_ = 1;
  int park_pages_ = 0;               // KV pages of a parking area (whole blocks)
  int32_t page_ref(int p) const { return (p / kv_block_) * kKvSlices * kv_block_ + p % kv_block_; }
  char* kv_page(int p, int slice) const {  // slice `slice` of page p
    return kv_base_ + (size_t(page_ref(p)) + size_t(slice) * kv_block_) * kv_page_bytes_;
  }
  std::vector<int32_t> pt_up_;       // upload_pages staging

  // Prefix cache.
  bool cache_on_ = false;
  std::vector<Checkpoint> ckpts_;
  size_t ckpt_bytes_ = 0;            // ssm + conv + draft ring
  std::vector<char*> slot_mem_;      // pinned [ckpt_bytes_] per slot, allocated as needed (grow_state_cache)
  std::vector<bool> slot_used_;
  char* stage_ = nullptr;            // device copy of one checkpoint on its way to the host
  cudaEvent_t stage_ev_ = nullptr, stage_in_ev_ = nullptr;  // staging free again / filled
  cudaStream_t cs_ = nullptr;        // device -> host copies of checkpoints
  // SPILL_ASYNC_IO only (null otherwise): the I/O stream of the tiered cache (KV pages <-> host entries, both
  // directions on this one stream, so they run in the order they were queued) and the event that orders it behind s_.
  // Not cs_: prefix_begin / spill_copy synchronize cs_ for the state slots and must not wait behind GiBs of KV.
  cudaStream_t io_s_ = nullptr;
  cudaEvent_t io_ev_ = nullptr;
  PinnedPool pin_pool_;              // declared before spill_ / retired_: their buffers release into it
  bool pin_failed_ = false;          // cudaHostAlloc of a slab failed: no more attempts
  std::vector<cudaEvent_t> ev_free_;  // recycled events
  struct Retired {                    // a promoted entry's host memory, until its host -> device copy is done
    HostKv kv;
    cudaEvent_t done;
    bool async = false;               // an async restore's: s_ did NOT wait for it, so spill_reap only frees it once complete
  };
  std::vector<Retired> retired_;
  // SPILL_ASYNC_PROMOTE: restores in progress (entry id -> state). disk: the thread that reads the file (until it ends);
  // issued: the pages and the host -> device copy exist (the checkpoint is loading). Engine thread only; the thread
  // inside `disk` touches nothing but its buffers (DiskRead in host_kv.h).
  struct Restore {
    std::unique_ptr<DiskRead> disk;
    std::chrono::steady_clock::time_point t0;
    size_t bytes = 0;
    bool issued = false;
    bool orphan = false;  // spill_release came before the checkpoint existed / completed: do not hold it
  };
  std::map<uint64_t, Restore> restoring_;
  // Restores that ended outside spill_restore_poll (inside begin, or finished by evict_checkpoint), with their outcome;
  // reported by the next poll. Capacity is reserved when a restore starts (<= one result per restoring entry), so
  // pushing never throws after state has changed.
  std::vector<std::pair<uint64_t, bool>> restore_results_;
  bool spill_sync_lookup_ = true;
  std::vector<int32_t> plan_;        // checkpoint positions of the running prefill (ascending)
  uint64_t use_clock_ = 0;
  CacheStats cstats_;
  std::map<uint64_t, Spilled> spill_;  // (a map: entries stay put while others come and go)
  uint64_t spill_next_ = 1;
  mutable uint64_t fingerprint_ = 0;   // spill_fingerprint(), once computed
  mutable bool fingerprint_set_ = false;
  size_t spill_ram_ = 0;               // host bytes the entries in memory hold
  SpillStats sstats_;
  int32_t* pos_dev_ = nullptr;       // == pos_ between calls; read by the attention kernels
  float* conv_ = nullptr;            // [48][10240][3]
  float* ssm_ = nullptr;             // [48][48][128][128]: the current buffer (ssm_buf_[sbuf_host_])
  float* ssm_buf_[2] = {nullptr, nullptr};  // speculative verify moves the committed state between them
  int sbuf_host_ = 0;                // which buffer holds the committed state (mirrors the device's)
  int32_t* sbuf_ = nullptr;          // device copy (flipped by every speculative step)
  size_t conv_bytes_ = 0, ssm_bytes_ = 0;

  // Images of the running request (P4-12).
  struct MmSpan {
    int start, gh, gw, row;  // positions start .. start + gh * gw - 1, rows of mm_ from row
    int len() const { return gh * gw; }
  };
  std::vector<MmSpan> mm_spans_;
  bf16* mm_ = nullptr;               // pinned [image_tokens][5120] image embeddings (UVA)
  int32_t* rope_delta_ = nullptr;    // device: rotary position - index after the last image
  int32_t* mpos_ = nullptr;          // device [prefill_chunk][3]: M-RoPE positions of the chunk's rows
  bool mpos_on_ = false;             // the current prefill chunk rotates by mpos_
  void* vis_ws_ = nullptr;           // vision workspace (overlays x_ .. big2_ unless larger)
  size_t vis_ws_bytes_ = 0;

  // Parked sequences (Phase 6). An area holds ssm | conv | draft ring K | V (the checkpoint layout).
  static constexpr int kSeqScalars = 1 + 1 + 4 + kSpecBlock + kTreeMax;  // rope delta, budget, rng, block / verify ids
  struct SeqState {
    char* dev = nullptr;  // the device area
    bool used = false;
    int pos = 0;
    std::vector<int32_t> pages, ids, plan;
    std::vector<MmSpan> mm;
    int32_t spec_last = -1;
    bool ring_valid = false;
    int ring_from = 0;
    SpecSampling smp{};
    int32_t scalars[kSeqScalars] = {};
  };
  std::vector<SeqState> parked_;

  // Batched steps (P6-10). While a batch is on, its lanes' step scalars live in the b_* arrays (lane z at
  // offset z) and their page tables in the rows of b_pt_; the running sequence's own copies are stale.
  bool batch_on_ = false;
  int batch_n_ = 0;                  // set while chunk() runs a batched verify
  std::vector<int32_t> blanes_;      // the lanes of the batch (-1: the running sequence)
  bool batch_pending_ = false;       // a step ran: its commits are replayed when the batch ends
  bool b_drafted_ = false;           // spec_draft_lanes drafted the step spec_step_lanes verifies next
  int bpar_ = 0;                     // the lanes' step parity (they step together)
  bool bwarm_ = false;
  AcceptLanes bacc_;                 // the lanes' sampling (chains) and top-k list lengths
  SeqLanes bsl_;                     // positions, rotary deltas and page tables of the lanes
  float *bconv_[kMaxLanes] = {}, *bssm_[kMaxLanes] = {};  // their GDN states (replayed in place)
  bf16 *bring_k_[kMaxLanes] = {}, *bring_v_[kMaxLanes] = {};  // their draft context rings
  int32_t *b_pos_ = nullptr, *b_delta_ = nullptr, *b_par_ = nullptr, *b_nc_ = nullptr, *b_budget_ = nullptr;
  int32_t *b_ids_ = nullptr, *b_blk_ = nullptr, *b_cpath_ = nullptr, *b_pt_ = nullptr;  // [lanes][8] / [lanes][max_pages_]
  uint64_t* b_rng_ = nullptr;        // [lanes][2]
  SpecTree* b_tree_ = nullptr;       // [lanes]
  SpecOut *b_out_ = nullptr, *b_out_dev_ = nullptr;  // [lanes][2] (by step parity), mapped pinned
  struct BatchGraph {
    std::vector<int32_t> lanes;
    int sbuf;
    AcceptLanes acc;
    cudaGraphExec_t exec;  // (masked lanes: the draft part)
    uint64_t used;
    cudaGraphExec_t exec2 = nullptr;  // masked lanes: the rest
  };
  std::vector<BatchGraph> bgraphs_;
  struct LaneRef {  // the host state of a lane's sequence
    int* pos;
    std::vector<int32_t>* pages;
    std::vector<int32_t>* ids;
    int32_t* spec_last;
  };
  LaneRef lane_ref(int z);
  void batch_enter(const int32_t* lanes, int n);
  void batch_exit();                 // replays the pending commits, gives the lanes their step scalars back
  void batch_body(int n, int part = 0);  // one step of the lanes (captured; part: as spec_body's)
  bool batch_masked() const;
  struct BatchGraph;
  BatchGraph& batch_graph(int n);
  void batch_prepare(const int32_t* lanes, int n);
  bool batch_part(int n, int part);
  GdnSpec gdn_spec_lanes(int gdn_layer) const;
  struct Region {
    void* dev;
    char* area;
    size_t bytes;
  };
  std::vector<Region> seq_regions(char* area) const;  // the running sequence's device state and its place in an area
  void seq_scalars(int32_t* host, bool save);          // device scalars of the running sequence <-> host
  int area_first(int a) const;                         // the first KV page of parking area a
  void reserve_area(int a);                            // moves the KV pages out of area a and takes it
  void release_area(int a);                            // gives area a's pages to the KV pool
  void seq_take(SeqState& st);                         // the running sequence's host state -> st (none left)
  void seq_give(SeqState& st);                         // st -> the running sequence (after reset / seq_take)

  // Activations (sized for prefill_chunk tokens).
  float *x_ = nullptr, *h_ = nullptr, *big1_ = nullptr, *big2_ = nullptr;
  float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *o_ = nullptr, *n_ = nullptr;
  float *a_ = nullptr, *b_ = nullptr, *g_ = nullptr, *beta_ = nullptr;
  uint8_t *aq_ = nullptr, *asf_ = nullptr;
  float* ascale_ = nullptr;  // per-token scales of aq_ (FP8 layers with per-channel scales: Weights::dyn8)
  int32_t* ids_ = nullptr;
  float* logits_ = nullptr;
  int32_t* topk_ids_ = nullptr;
  float* topk_logp_ = nullptr;
  float* uniform_ = nullptr;
  int32_t* sampled_ = nullptr;
  bool mlp_bf16_ = true;                              // prefill MLP gate / up in BF16 (Q27_MLP_BF16)
  bool mlp_fused_ = true;   // prefill MLP gate | up + silu * up + NVFP4 quantization in one GEMM (Q27_MLP_FUSED)
  static constexpr int kSkinnyMaxM = 64;              // decode and verify-sized GEMMs use skinny.cu
  static constexpr size_t kSkinnyWsBytes = 32 << 20;  // split-K partials (up to 64 rows) and counters
  void* skinny_ws_ = nullptr;
  void* attn_ws_ = nullptr;  // split-KV partials of attention_decode
  void* fa_stage_ = nullptr;  // NVFP4 KV: the context rounded to E4M3 for the prefill attention (P7-8)
  size_t fa_stage_bytes_ = 0;
  bool skinny_ = true;         // Q27_NO_SKINNY=1: the many-row GEMMs for every M (Phase 1 path)
  bool skinny_check_ = false;
  std::string skip_;  // Q27_SKIP (debug timing)
  bool skip(const char* kind) const;
  // Decode step as a CUDA graph (Q27_NO_GRAPH=1: eager). Not used while profiling.
  bool graphs_ = true, graph_warm_ = false;
  struct GraphKey {
    float temperature;
    int top_k;
    float top_p;
    float min_p, presence, frequency;
    int logprobs;
    bool masked, biased;
    bool operator==(const GraphKey& o) const {
      return temperature == o.temperature && top_k == o.top_k && top_p == o.top_p && min_p == o.min_p &&
             presence == o.presence && frequency == o.frequency && logprobs == o.logprobs && masked == o.masked &&
             biased == o.biased;
    }
  } graph_key_{};
  cudaGraphExec_t graph_exec_ = nullptr;
  int graph_sbuf_ = -1;  // the GDN state buffer the decode graph was captured with
  // Captured graphs by sampling settings (sequences with different settings take turns, Phase 6); the
  // least recently used one is destroyed beyond kGraphCache. graph_exec_ / spec_graph_ point into them.
  static constexpr int kGraphCache = 4;
  struct CachedGraph {
    GraphKey key;
    SpecSampling smp;
    int sbuf;
    cudaGraphExec_t exec;  // (masked steps: the draft part)
    uint64_t used;
    cudaGraphExec_t exec2 = nullptr;  // masked steps: the rest
  };
  std::vector<CachedGraph> dgraphs_, sgraphs_;
  uint64_t graph_clock_ = 0;
  void drop_graphs();
  struct HostIo {  // pinned, mapped: the decode step's input id, uniform and sampled token
    int32_t id, out;
    float uniform;
  }* host_io_ = nullptr;
  HostIo* host_io_dev_ = nullptr;  // its device address
  bool decode_memcpy_ = false;  // Q27_DECODE_MEMCPY=1: copy the step's input / output instead

  // Speculative decoding.
  std::unique_ptr<Draft> draft_;
  bool verify_ = false;         // chunk() runs a verify block (GDN without state advance, replay first)
  int verify_paths_ = 1;        // draft tree paths of that block
  bf16* cap_ = nullptr;         // target hidden of layers 5 / 19 / 33 / 47 / 61: [rows][5][5120] (null: off)
  bf16* cap_buf_ = nullptr;     // [prefill_chunk][25600]
  float* rep_ = nullptr;        // GDN verify inputs: [2][48][rep_rows_ tokens]
  static constexpr size_t kRepTokenFloats =  // raw conv input, k, v, a | b per token
      size_t(dims::kConvDim) + dims::kKHeads * dims::kKDim + dims::kVHeads * dims::kVDim + 2 * dims::kVHeads;
  int rep_rows_ = kVerifyMaxT;  // verify rows (tree nodes, or the rows of a batched step: 8 per lane)
  int32_t *par_ = nullptr, *ncommit_ = nullptr;  // step parity, tokens committed by the last verify
  int32_t* budget_ = nullptr;                    // tokens the steps may still output (spec_begin's max_new)
  uint64_t* rng_ = nullptr;                      // seed, step counter
  int32_t *ids_ver_ = nullptr, *block_ids_ = nullptr;  // verify input [32] (tree nodes); draft block [8]
  float* tv_ = nullptr;
  int32_t* ti_ = nullptr;     // target top-k lists of the verify rows [32][64]
  SpecTree* tree_ = nullptr;  // the step's draft tree
  int32_t* cpath_ = nullptr;  // rows of the last accepted path [8]
  KvLayer* kv_layers_dev_ = nullptr;  // device copy of kv_layers_ (KV commit)
  SpecOut *spec_out_ = nullptr, *spec_out_dev_ = nullptr;  // [2] (by step parity), mapped pinned
  int32_t* hist_ = nullptr;   // committed tokens by position (lookup drafting)
  uint32_t* pen_ = nullptr;   // penalty counts [1 + parking areas][vocab] (pen_counts)
  BiasList* bias_ = nullptr;  // logit bias lists [1 + parking areas] (bias_list)
  // Logprobs (SpecSampling::logprobs): mapped pinned results [2] by step parity of the running sequence, [lanes][2]
  // of a batched step, [1] of the sampler (kLpSample), the raw top lists the penalties would overwrite (scratch,
  // [verify rows][kMaxTopLogprobs]), and the host copies of the last call's (logprobs()).
  static constexpr int kLpLanes = 2, kLpSample = 2 + 2 * kMaxLanes;
  Logprobs *lp_ = nullptr, *lp_dev_ = nullptr;
  float* lpv_ = nullptr;
  int32_t* lpi_ = nullptr;
  Logprobs last_lp_[kMaxLanes] = {};
  uint32_t* lk_best_ = nullptr;
  Lookup lookup_{};           // Q27_LOOKUP (default off), Q27_LOOKUP_MIN, Q27_LOOKUP_AGREE
  int64_t stat_steps_ = 0, stat_lookup_ = 0, stat_tokens_ = 0;
  int spec_paths_ = 1;        // Q27_SPEC_PATHS: draft walks merged into the verify tree (1: chain)
  bool spec_pending_ = false;  // a verify ran since the last flush / reset
  bool spec_pipeline_ = true;  // Q27_SPEC_PIPELINE: launch the next step before reading this one's result
  bool spec_inflight_ = false; // a step launched ahead whose result has not been returned
  int inflight_slot_ = 0;      // its SpecOut slot
  int par_host_ = 0;           // the device's step parity at the next launch (= its SpecOut slot)
  cudaEvent_t spec_ev_[2] = {};
  bool spec_warm_ = false;
  cudaGraphExec_t spec_graph_ = nullptr;
  SpecSampling spec_key_{};
  SpecSampling spec_smp_{};
  int32_t stop_[2] = {-1, -1};  // generation_config.json eos_token_id
  uint32_t* mask_host_ = nullptr;  // pinned [kMaskRows][kMaskWords] (mask_rows)
  uint32_t* mask_dev_ = nullptr;   // the device copy of the rows a step uses
  int32_t* draft_host_ = nullptr;  // pinned [kMaxLanes][8]: spec_draft's verify tokens
  bool spec_drafted_ = false;      // spec_draft drafted the step spec_step verifies next
};

}  // namespace q27
