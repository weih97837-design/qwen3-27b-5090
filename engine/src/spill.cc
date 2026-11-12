// Tiered cache (SPILL_RAM_GIB > 0, docs/tiered-cache.md): checkpoints the KV pool gives up go to host memory instead
// of being dropped, and from there to files (frontend/tiercache.py decides when; the RAM cap here is the backstop).
// prefix_begin brings the deepest spilled match back into the pool before its own lookup, so everything after that
// (the shared full pages, the copied partial page, the plan of the prefill) is the unchanged cache-hit path.
//
// All of this runs on the engine's thread, like the rest of Model (no locks).
//
// Two optional speed-ups, both off by default (Options::spill_pinned_gib, Options::spill_async_io):
//
// SPILL_PINNED_GIB: a piece of an entry's KV bytes lives in pinned memory (PinnedPool) when the pool can hold ALL of it, else
// in one pageable buffer as before. In a pool a HostKv is a list of segments (host_kv.h), each inside one slab and a
// multiple of the page size long (so no page straddles two segments): pieces larger than a slab are pinned too. One
// HostKv is fully pinned or fully pageable, but an entry of the RAM tier is several of them (its tail and its shared
// chunks, "RAM DEDUP" below), each allocated on its own (spill_alloc_kv): an entry may be partly pinned and partly
// pageable (the pool can be full, or fragmented, for a later piece; orphans kept inside an OrphanScope hold pinned
// memory too). The host -> device copies do not care; the memory estimates are per piece (spill_new_host_bytes,
// spill_freed). Slabs are pinned only while idle (spill_grow_pinned).
// Pinned memory roughly doubles the PCIe speed of the copies; the linear byte stream (and so the file format) is the
// same. The pool is part of the RAM tier's budget: entries in pinned segments count in spill_ram_ like any other.
//
// SPILL_ASYNC_IO: the KV copies run on io_s_ instead of s_ and the CPU does not wait for them. Invariants:
//  (1) io_s_ READS or WRITES pool pages only after io_begin(): an event recorded on s_ after all work queued so far
//      (which may have written the pages, or still read pages that were just freed and handed out again) and waited
//      on by io_s_.
//  (2) A D2H copy of pages that are then freed (eviction): spill_copy records `done` on io_s_ and makes s_ wait for it
//      (cudaStreamWaitEvent, GPU side only) before returning, so no s_ work queued later can overwrite those pages
//      before they were read. An H2D copy into newly allocated pages likewise: s_ waits for its `done` before the
//      resumed sequence's kernels. Both directions share io_s_, so a H2D into a page that an earlier D2H still reads
//      is ordered after that read by the stream itself.
//  (3) The CPU does not touch an in-flight entry (Spilled::inflight): spill_wait (cudaEventSynchronize on `done`) runs
//      first in spill_write (so export and the RAM cap), spill_erase (so drop, subsume, the RAM cap's victims; its own
//      buffers only), spill_promote, and for the idle backup (its tail and the chunks it filled; not a chunk that was
//      resident) before spill_backup_one returns. So an idle backup (live entry, its
//      pages stay in the pool and may be moved or freed later) is never in flight once it is visible to evict_checkpoint
//      or reserve_area; only entries of evicted checkpoints (pages already free, s_ already ordered) can be.
//      The promoted entry's host memory is read by the H2D copy after the entry is erased: it is parked in retired_
//      with its event and freed by spill_reap (the next spill call, and prefix_begin after its cudaStreamSynchronize(s_)
//      which also covers the copy, as s_ waited for it).
//  (4) The state part of an entry (ckpt_bytes_) stays a memcpy from the pinned slot after cudaStreamSynchronize(cs_).
//  Bytes of a checkpoint's last, partial page past its pos may change under a running D2H (the sequence that saved
//  it can keep writing there); they are never read (consumers copy the page and overwrite those rows).
//  With SPILL_ASYNC_IO=0 io_s_ is null, spill_async() is false, and every code path below is the synchronous one.
//
// SPILL_ASYNC_PROMOTE (Options::spill_async_promote, default off; delay-hit scheduling, docs/tiered-cache.md): the
// frontend restores a spilled entry in the background instead of prefix_begin promoting it synchronously
// (spill_match -> spill_restore_begin -> spill_restore_poll). With the flag off none of the code below runs, and
// spill_lookup / spill_promote behave exactly as before. Invariants:
//  (A) An entry being restored is `busy` from begin until the checkpoint is visible. busy entries are skipped by
//      spill_lookup / spill_promote / spill_enforce_ram / spill_checkpoint's and spill_backup_one's subsume loops, and
//      refused by spill_export / spill_drop (frontend/tiercache.py also skips them: spill_list reports busy).
//  (B) DISK entries: begin allocates the entry's HostKv + state on the engine thread (spill_alloc_kv, may be pinned),
//      then ONE thread per entry (DiskRead, spill_files.h) only fread()s the entry's manifest and chunk files into those
//      buffers (and checksums them). It touches nothing else (no spill_, ckpts_, pool_, PinnedPool bookkeeping, disk_ (the
//      chunk index), ChunkStore, CUDA: it gets copies of the paths, ids and names), and the engine thread does not touch the buffers
//      until it has read DiskRead::st with acquire (poll) and joined the thread. The destructor joins before anything
//      is freed (restoring_.clear() first). The buffers' pool regions are released only by the engine thread.
//  (C) The host -> device copy goes on io_s_ behind io_begin() (invariant (1)) and a `load_ev` is recorded after it.
//      Unlike a synchronous promote, s_ does NOT wait for the copy (that is the point): the new Checkpoint is `loading`
//      and therefore (i) invisible to prefix_begin's best-match search, spill_best / spill_lookup's "deepest pool match",
//      save_checkpoint_from's duplicate check, spill_backup_one and cache_stats (its pages are not counted as room
//      for new requests and not in the checkpoint count: they are not evicted before the last resort in (ii), and soon after they will be counted
//      as any checkpoint's), (ii) evicted only as a last resort: evict_checkpoint takes non-held, non-loading checkpoints
//      first, then the oldest held one, and only when just loading ones remain it finishes the oldest of them
//      synchronously (cudaEventSynchronize(load_ev), as in (E)) and evicts it, (iii) not touched by any page move:
//      reserve_area makes
//      s_ wait for load_ev before it copies a loading checkpoint's pages. Its pages have one reference (alloc) and no
//      sequence maps them, so nothing else reads or writes them. spill_subsumed still counts a loading checkpoint as
//      covering a shallower one (it will exist; its entry is subsumed identically while busy).
//  (D) The entry's own host memory (its tail) is read by that copy after it is parked in retired_ with its own event
//      (async = true): spill_reap frees such an item only once cudaEventQuery says it completed (never blocks on it),
//      since s_ did not wait for it. The entry's CHUNKS are not parked: the busy entry keeps listing them (their
//      references held) until spill_loading_done erases it (spill_restore_finish -> spill_erase), so a chunk the copy
//      is reading stays in the store, findable by a spill of another entry that shares it, instead of sitting at 0
//      refs, invisible, until it is reaped. (spill_loading_done first clears `inflight` of each chunk whose latest
//      event completed.) A busy entry in that state has no state buffer (in_ram() false), only `chunks`.
//  (E) spill_restore_poll makes a loading checkpoint visible (loading = false, last_use refreshed) only after
//      cudaEventQuery(load_ev) == cudaSuccess (so a later D2D copy of its pages on s_ cannot run before the
//      host -> device copy: the event is complete before anything is queued on s_ that reads them), then erases the
//      entry. A pageable entry, or SPILL_ASYNC_IO=0, restores synchronously (still correct, the copy is complete when
//      begin returns; the finished id is reported by the next poll).
//  (F) HELD: a restored checkpoint is `held` from when it becomes visible until the request that waited for it runs
//      prefix_begin (which clears it when it uses the checkpoint) or the frontend calls spill_release (timeout, cancel).
//      evict_checkpoint takes every non-held, non-loading checkpoint first; then the oldest held one (better than
//      KvFull); then, when only loading ones remain, it FINISHES the oldest loading one synchronously
//      (cudaEventSynchronize(load_ev), the same completion path as spill_restore_poll, reported ok by the next poll) and
//      evicts that (it is the only candidate). The stall is rare: the pool is so full that nothing else can go. cache_stats
//      and spill_room do not count a held checkpoint's pages as room (so neither the scheduler nor a second restore
//      plans on evicting it). There is no age-based auto-clear: every path that stops waiting calls spill_release
//      (worker: timeout, cancel, finish, and after begin), and the oldest-held fallback bounds the damage of a leak.
//  (G) Exception safety: the map nodes (restoring_, restore_results_ capacity, ckpts_ capacity) are reserved before a
//      disk thread starts or any page / slot / event is taken, so no step after one can throw bad_alloc; a failed
//      reservation is "no restore" (0), never a half-registered one.
//  (G') A throw after spill_acquire (a CUDA error) in spill_promote / spill_restore_issue: an Undo scope guard waits for
//      the stream (a queued copy may still write the pages), then gives the pages and the slot back (spill_unacquire)
//      and recycles the load event; the entry is not left busy (spill_promote's own guard; spill_restore_begin /
//      spill_restore_poll for a restore, which also drop its restoring_ record and report it failed).
//
// RETENTION (retention.h, docs/tiered-cache.md "Retention"). Which entry spill_enforce_ram evicts is value based: the agent using the
// most RAM (Options::spill_fair_agents), then superseded entries, never-hit before hit ones, then the old rule (chunk_store.h
// pick_victim); a superseded sole copy is dropped, never written to the SSD. An entry's `owner` (Model::set_owner, carried by the
// sequence through park / resume), `hits` (a prefix_begin match, or a restore) and the manifest's copy of both (version 2) follow the
// entry through spill, the SSD, import and restore. The frontend applies the same order to the SSD (frontend/tiercache.py) from
// spill_list + spill_meta + spill_agents. All of it is bookkeeping on the engine thread: no stream, event or buffer is touched.
//
// RAM DEDUP (Options::spill_dedup, default on; SPILL_DEDUP=0 turns it off; chunk_store.h). The RAM tier stores the full
// kChunkPages-page (4096-token) blocks of an entry once, content-addressed, and entries reference them; only the
// remainder (the "tail": < kChunkPages pages, with the partial last page) and the recurrent state are per entry. The SSD
// file format is untouched: spill_write serializes chunks + tail back into the old linear slice-major stream, and an
// entry loaded from a file or imported is "private" (no chunks, all its pages in kv). Invariants:
//  (H) Chunk identity. A chunk covers whole, full pages only: nchunks = pos / 4096 by TOKENS (never pages / 64: a partial
//      page's trailing rows are not defined). Its key chains the keys of all earlier chunks (a chunk's KV depends on the
//      whole prefix), starting from the model / KV-type fingerprint; a hit is verified against the stored token ids AND
//      the parent chunk (its uid), so by induction the whole prefix is verified: a 128-bit collision never gives wrong
//      KV. Colliding chunks coexist in the store (a multimap; find checks every candidate), nothing is probed. Images need no extra
//      mixing: an image's tokens are ids >= kVocab, consecutive from a hash of the image's bytes and its grid
//      (reference/qwen27b/image_tokens.py; Checkpoint::ids holds them raw), so a chunk that ends inside an image run
//      still pins the image and its grid through the ids before it, and no two prompts agree on a partial run of them.
//      That is the identity the pool's prefix cache already relies on. Like pool pages shared by checkpoints, a shared chunk is
//      "the KV some entry computed for these exact ids".
//  (I) Refcounts are explicit (Chunk::refs): one per entry that lists the chunk (and one for an entry under construction,
//      taken in spill_copy before anything is copied; spill_copy gives them back if anything throws, and frees a chunk it
//      created whose bytes were not completely written). A chunk reaching 0 refs leaves the store in spill_chunk_free,
//      and its memory is released only when nothing can still touch it: a chunk that is not inflight goes at once; an
//      INFLIGHT one (a D2H still filling it, or a host -> device copy of ANY entry's restore still reading it) is
//      parked in retired_ with its own event (async = true: s_ never waited for it) and freed by spill_reap once
//      cudaEventQuery says it completed. The engine thread never waits for a chunk's copy on the free path (so dropping,
//      subsuming, evicting or writing an entry cannot stall behind a background restore's H2D); the parking capacity is
//      reserved before the chunk leaves the store (G), with the old synchronous wait only as the out-of-memory fallback.
//      The synchronous waits are where a CPU reader needs the bytes: spill_wait (promote, write, restore_issue).
//      An async restore holds its chunk references until it completes: (D).
//  (J) A chunk has its own `done` / `inflight` like an entry: recorded on io_s_ behind the D2H copy that fills it and
//      re-recorded behind every H2D copy that reads it (the CPU waits for the latest, a superset). spill_wait(Spilled&)
//      waits for the entry's chunks too, so every place that waited for an entry (3) still holds. s_ waits (2) for ONE
//      event per entry (s.done, recorded after the last queued copy); when every chunk was resident and the tail is empty
//      nothing is read from the pool and nothing is recorded or waited for. A resident chunk that is still being filled by
//      another entry's D2H is only read later (promote / write), after spill_wait, or by the stream order on io_s_.
//  (K) Accounting. spill_ram_ = the entries' OWN bytes (tails + states); spill_phys() = that + each resident chunk once =
//      the physical bytes the RAM cap and spill_host_use() are about. The logical size (kv_bytes + state) is what
//      spill_list reports and what the SSD budget needs; the stats report both. A chunk with 0 refs (an "orphan") stays
//      resident only inside an OrphanScope (spill_checkpoint / spill_backup_one erase the shallower entries they replace
//      and then copy the deeper one, which re-finds those chunks instead of copying them again); the scope trims the
//      orphans when it ends, and spill_enforce_ram trims any it finds first. (The orphans of a scope are the chunks of
//      entries that the new one extends, so it re-references all of them: trimming "those it will not re-reference"
//      beforehand would trim nothing, and is not done.) spill_enforce_ram re-reads spill_host_use()
//      every iteration and picks the victim that frees the most host memory (spill_freed: an entry whose chunks others
//      share frees little). Host-use estimates are per piece: spill_backup_one asks spill_new_host_bytes (the tail and the
//      missing chunks each against the free part of the pinned pool, state added), spill_enforce_ram asks spill_freed.
//      Its victim class (choose_victim, chunk_store.h): the idle backups first (most bytes freed first, the oldest within a
//      factor 2), except that when the best backup frees only its state and a sole copy frees >= 4x as much, the sole
//      copy goes (to the SSD). Each pass takes one entry's memory, so the loop terminates.
//  (L) Linear stream. The RAM side of the old file format: spill_h2d copies chunks + tail into pool pages, and a PRIVATE entry (no
//      chunks: SPILL_DEDUP=0, or read back from the SSD, (P)) holds the whole slice-major stream in kv; linear_pieces
//      (chunk_store.h) still puts chunks + tail back into that order (the host tests compare the files against it).
//
// SSD DEDUP (spill_files.h; layout, formats and checksums are described there). An entry on the SSD is a MANIFEST (entries/<id>.q27m:
// tokens, state, tail, the list of chunk names) plus the chunk files (chunks/<name>.kvc) it lists, shared with every other
// manifest that lists them. Invariants:
//  (M) Identity. A chunk file's name is derived from its content (the chain key over the fingerprint and every token up to the chunk's
//      end, plus a hash of its own ids and its parent's NAME), never from an in-memory uid, so the same prefix is the same file
//      across restarts and across entries. Skipping an existing file on export is VERIFIED (size, header, token ids, parent name:
//      chunk_file_matches); a restore verifies everything (and the Hash64 trailer) on the reader thread. A mismatch never shares:
//      the export fails (the entry is dropped by the caller) and a failed read forgets the chunk file (forget_file) and the entry.
//  (N) Write order (crash safety). spill_write: every missing chunk file (tmp, fflush + fsync, rename), fsync of chunks/, then the
//      manifest (tmp, fsync, rename, fsync of entries/), then the disk refs are taken and the entry's memory released. A crash
//      leaves only .tmp files and chunk files without a manifest (strays), never a manifest whose chunk is missing (the order),
//      and spill_import rejects such a manifest anyway (a chunk that is gone or the wrong size) so the frontend deletes it. An
//      export that fails removes the chunk files IT wrote that nobody lists (drop_unreferenced); replacing an entry's manifest
//      refs the new chunks before unreffing the old ones. Drop: the manifest is unlinked first, then its chunks are unreffed
//      (a crash in between leaves strays). Windows: _commit instead of fsync, no directory fsync (spill_files.h).
//  (O) Disk refcounts (DiskIndex disk_, engine thread only): a chunk file's refs = the entries with a manifest that list it
//      (Spilled::dnames). Taken in spill_write / spill_import, given back by spill_disk_forget (spill_erase: drop, a restore that ended,
//      a failed read), and the file is unlinked when they reach 0 and by nothing else. A busy (restoring) entry still lists its
//      chunks, so nothing the reader thread reads is ever unlinked by this process (the one exception: forget_file of a chunk that
//      failed verification; its readers hold an open file, and the entries that list it fail their own restore).
//  (P) A restore from the SSD reads the manifest and every chunk file into a PRIVATE entry (one linear buffer from spill_alloc_kv, the
//      reader's destination pieces cut by run_pieces): no half-read chunk is ever registered in the ChunkStore, so (H)-(K) are
//      untouched and a concurrent spill never finds unverified bytes. (RAM dedup does not continue for a restored entry: its KV is
//      in the pool the moment the restore ends. Reusing resident chunks to skip their reads is not done.)
//  (Q) Start-up: the frontend imports every manifest (the disk index is built from them), then calls spill_sweep, which deletes
//      .tmp files and chunk files nobody lists. Sweeping before the imports are done would delete live chunks: the engine cannot know,
//      so it never sweeps by itself.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <utility>

#include "model.h"
#include "retention.h"

namespace q27 {

using namespace dims;

namespace {

uint64_t fnv(uint64_t h, const void* p, size_t n) {
  const unsigned char* c = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; i++) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}

// Short checkpoints (a system prompt and tools) are shared by new conversations and cheap to keep, so they are
// never subsumed by a deeper one (only by an equal one).
constexpr int kKeepShort = kKeepShortTokens;

// d holds the tokens of c (at least as many, the same prefix) and, when c has the draft's ring, d has it too.
bool covers(const std::vector<int32_t>& d_ids, bool d_ring, const std::vector<int32_t>& c_ids, bool c_ring) {
  return d_ids.size() >= c_ids.size() && std::equal(c_ids.begin(), c_ids.end(), d_ids.begin()) && d_ring >= c_ring;
}

}  // namespace

uint64_t Model::spill_fingerprint() const {
  if (fingerprint_set_) return fingerprint_;  // computed once: the directory scan and config.json read are not free
  uint64_t h = 1469598103934665603ull;
  h = fnv(h, opt_.model_dir.data(), opt_.model_dir.size());
  h = fnv(h, "\0", 1);
  h = fnv(h, opt_.draft_dir.data(), opt_.draft_dir.size());
  const int kv = int(opt_.kv_type);
  h = fnv(h, &kv, sizeof kv);
  h = fnv(h, &kv_page_bytes_, sizeof kv_page_bytes_);
  h = fnv(h, &ckpt_bytes_, sizeof ckpt_bytes_);
  // The weights themselves: replacing the files at the same path must not reuse KV files computed with other
  // weights. The names and sizes of the files in each directory, and the bytes of the model's config.json.
  for (const std::string& dir : {opt_.model_dir, opt_.draft_dir}) {
    if (dir.empty()) continue;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec), end;
    if (ec) continue;
    std::vector<std::pair<std::string, uintmax_t>> files;
    for (; !ec && it != end; it.increment(ec)) {
      std::error_code e2;
      if (!it->is_regular_file(e2) || e2) continue;
      const uintmax_t size = it->file_size(e2);
      if (e2) continue;
      files.emplace_back(it->path().filename().string(), size);
    }
    std::sort(files.begin(), files.end());
    for (const auto& [name, size] : files) {
      h = fnv(h, name.data(), name.size());
      h = fnv(h, "\0", 1);
      h = fnv(h, &size, sizeof size);
    }
    h = fnv(h, "\0", 1);
  }
  if (!opt_.model_dir.empty()) {
    std::ifstream in(std::filesystem::path(opt_.model_dir) / "config.json", std::ios::binary);
    if (in) {
      const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      h = fnv(h, bytes.data(), bytes.size());
    }
  }
  fingerprint_ = h;
  fingerprint_set_ = true;
  return h;
}

void Model::copy_pages(const std::vector<int32_t>& pages, const HostKv& host, bool to_host) {
  copy_pages_on(pages, host, to_host, s_);
  Q27_CUDA(cudaStreamSynchronize(s_));
}

void Model::copy_pages_on(const std::vector<int32_t>& pages, const HostKv& host, bool to_host, cudaStream_t st) {
  Q27_CHECK(host.total == pages.size() * size_t(kKvSlices) * kv_page_bytes_, "tiered cache: the entry's size does not match its pages");
  copy_runs(pages, kKvSlices, kv_page_bytes_, kv_block_,
            [&](int p, int sl) { return kv_page(p, sl); }, host,
            [&](char* h, char* dev, size_t bytes) {
              if (to_host)
                Q27_CUDA(cudaMemcpyAsync(h, dev, bytes, cudaMemcpyDeviceToHost, st));
              else
                Q27_CUDA(cudaMemcpyAsync(dev, h, bytes, cudaMemcpyHostToDevice, st));
            });
}

void Model::io_begin() {
  Q27_CUDA(cudaEventRecord(io_ev_, s_));
  Q27_CUDA(cudaStreamWaitEvent(io_s_, io_ev_));
}

void Model::spill_wait(Spilled& s, bool with_chunks) {
  if (with_chunks)
    for (SpillChunk* c : s.chunks)  // (J) a shared chunk has its own copy in flight, maybe another entry's
      if (c->inflight) {
        Q27_CUDA(cudaEventSynchronize(c->done));
        c->inflight = false;
      }
  if (!s.inflight) return;
  Q27_CUDA(cudaEventSynchronize(s.done));
  s.inflight = false;
}

cudaEvent_t Model::spill_event() {
  if (!ev_free_.empty()) {
    cudaEvent_t e = ev_free_.back();
    ev_free_.pop_back();
    return e;
  }
  cudaEvent_t e;
  Q27_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  return e;
}

// (Only events whose last record has completed: re-recording one later does not affect a cudaStreamWaitEvent that
// was queued before, which captured the earlier record.)
void Model::spill_event_put(cudaEvent_t ev) {
  if (ev) ev_free_.push_back(ev);
}

// ---------------------------------------------------------------- RAM dedup: chunks (invariants (H)-(K) at the top)

size_t Model::spill_chunk_count(size_t ntokens) const {
  return opt_.spill_dedup ? ntokens / (size_t(kChunkPages) * kPageTokens) : 0;  // by TOKENS: a chunk holds only full pages (H)
}

namespace {
std::vector<int32_t> chunk_pages_of(const std::vector<int32_t>& pages, size_t first, size_t n) {
  return std::vector<int32_t>(pages.begin() + std::ptrdiff_t(first), pages.begin() + std::ptrdiff_t(first + n));
}
}  // namespace

// By how much spilling c would grow spill_host_use() (a dry run), without the state: its tail and the chunks that are not
// resident, each piece on its own, in the order spill_copy allocates them. A piece the free part of the pinned pool can
// still hold is already counted (its bytes are the pool's) and costs nothing; one that does not fit would be pageable
// and adds its bytes. (An entry can be partly pinned and partly pageable. Fragmentation can make a piece that "fits"
// pageable all the same: a brief overshoot the next spill_enforce_ram corrects.)
size_t Model::spill_new_host_bytes(const Checkpoint& c) {
  const size_t nfull = spill_chunk_count(c.ids.size());
  const size_t tail = (c.pages.size() - nfull * kChunkPages) * size_t(kKvSlices) * kv_page_bytes_;
  const size_t chunk_bytes = size_t(kChunkPages) * kKvSlices * kv_page_bytes_;
  size_t nmiss = 0;
  if (nfull > 0)
    nmiss = chunk_missing_bytes(spill_chunks_, chunk_seed(spill_fingerprint(), kChunkPages), c.ids.data(), nfull,
                                size_t(kChunkPages) * kPageTokens, chunk_bytes) / chunk_bytes;
  size_t pool_free = pin_pool_.total() - pin_pool_.used(), need = 0;
  auto piece = [&](size_t bytes) {
    if (bytes == 0) return;
    if (pool_free >= bytes) pool_free -= bytes;
    else need += bytes;
  };
  piece(tail);
  for (size_t i = 0; i < nmiss; i++) piece(chunk_bytes);
  return need;
}

// refs == 0: the chunk leaves the store (I). A chunk whose copy may still be running (a D2H that fills it, or an H2D that
// reads it: another entry's restore) is NEVER waited for here: its memory moves to retired_ with the chunk's own event
// (async: s_ did not wait for it) and spill_reap frees it once cudaEventQuery says it completed. So the engine thread
// does not stall behind a background restore's copy when an entry that listed the chunk is dropped, subsumed, evicted
// or written. (G) the parking capacity is reserved before the chunk leaves the store; when that fails (no memory) the
// old synchronous wait is the fallback, so this never throws bad_alloc.
void Model::spill_chunk_free(SpillChunk* c) {
  chunk_free(spill_chunks_, c, retired_, [&](SpillChunk* x) { Q27_CUDA(cudaEventSynchronize(x->done)); },
             [&](cudaEvent_t e) { spill_event_put(e); });
  // (the chunk's HostKv goes back to the pool / delete[] when its local in chunk_free ends)
}

void Model::spill_chunk_unref(SpillChunk* c, bool now) {
  if (!spill_chunks_.unref(c)) return;
  if (defer_orphans_ && !now) return;  // (K) an orphan until the OrphanScope ends
  spill_chunk_free(c);
}

void Model::spill_trim_orphans() {
  for (SpillChunk* c : spill_chunks_.orphans()) spill_chunk_free(c);
}

// The entry's memory goes: its chunk references and its own bytes. (The caller waited for the entry, (3), or parked
// what a copy still reads.) The entry itself stays (it may have a file).
void Model::spill_release_mem(Spilled& s, bool park) {
  spill_ram_ -= s.own_bytes();
  std::vector<SpillChunk*> cs = std::move(s.chunks);
  s.chunks.clear();
  s.kv.reset(), s.state.reset();
  for (SpillChunk* c : cs) spill_chunk_unref(c, park);
}

bool Model::spill_parts_ok(const Spilled& s) const {
  const size_t chunk_bytes = size_t(kChunkPages) * kKvSlices * kv_page_bytes_;
  size_t sum = s.kv.total;
  for (const SpillChunk* c : s.chunks) {
    if (c->kv.total != chunk_bytes || c->refs <= 0) return false;
    sum += c->kv.total;
  }
  return sum == s.kv_bytes;
}

size_t Model::spill_freed(const Spilled& s) const {
  // What spill_host_use drops when the entry's memory goes: the state, and the pageable pieces nobody else lists. A
  // pinned piece goes back to the pool, which stays counted whole.
  size_t n = s.state_bytes + (s.kv.pinned() ? 0 : s.kv.total);
  for (const SpillChunk* c : s.chunks)
    if (c->refs == 1 && !c->kv.pinned()) n += c->kv.total;
  return n;
}

// Queues the host -> device copies of the entry's chunks and tail into `pages` (all of the entry's pages) on st. record
// (async I/O): each chunk's own event is recorded behind its copy, so the CPU (spill_wait) and a later parking know when
// the chunk is no longer read (J).
void Model::spill_h2d(Spilled& s, const std::vector<int32_t>& pages, cudaStream_t st, bool record) {
  const size_t nc = s.chunks.size();
  for (size_t ci = 0; ci < nc; ci++) {
    SpillChunk* c = s.chunks[ci];
    copy_pages_on(chunk_pages_of(pages, ci * kChunkPages, kChunkPages), c->kv, false, st);
    if (record) {
      if (!c->done) c->done = spill_event();
      Q27_CUDA(cudaEventRecord(c->done, st));
      c->inflight = true;
    }
  }
  if (pages.size() > nc * kChunkPages)
    copy_pages_on(chunk_pages_of(pages, nc * kChunkPages, pages.size() - nc * kChunkPages), s.kv, false, st);
}

void Model::spill_reap() {
  // Exception-safe: an item is removed from retired_ only once its memory and event are given back (r.done is cleared
  // at once, so an item kept after a throw never returns its event twice), and keep never grows past its reserve.
  std::vector<Retired> keep;
  keep.reserve(retired_.size());
  size_t i = 0;
  try {
    for (; i < retired_.size(); ++i) {
      Retired& r = retired_[i];
      if (r.async) {  // (D) s_ did not wait for this copy: only a completed one is freed here
        const cudaError_t e = cudaEventQuery(r.done);
        if (e == cudaErrorNotReady) {
          cudaGetLastError();
          keep.push_back(std::move(r));
          continue;
        }
        Q27_CUDA(e);
      } else {
        Q27_CUDA(cudaEventSynchronize(r.done));  // (done already, in practice: see the header comment)
      }
      r.kv.reset();  // (back into the pinned pool, or delete[])
      cudaEvent_t ev = r.done;
      r.done = nullptr;
      spill_event_put(ev);
    }
  } catch (...) {
    // keep what was kept and what was not looked at yet (from i on: item i still holds its event if it threw before put)
    for (; i < retired_.size(); ++i)
      if (retired_[i].done) keep.push_back(std::move(retired_[i]));
    retired_ = std::move(keep);
    throw;
  }
  retired_ = std::move(keep);
}

HostKv Model::spill_alloc_kv(size_t bytes) {
  HostKv kv;
  if (opt_.spill_pinned_gib > 0 && bytes > 0) {
    if (alloc_pinned_kv(pin_pool_, bytes, kv_page_bytes_, kv)) {
      sstats_.pin_hits++;
      return kv;
    }
    sstats_.pin_misses++;  // (the pool is not grown yet, or too full / fragmented for the whole entry)
  }
  kv.push(KvBuf(new char[bytes], KvDel{}), bytes);  // (may throw bad_alloc)
  return kv;
}

int Model::spill_grow_pinned(int chunks) {
  if (!spill_on() || opt_.spill_pinned_gib <= 0 || pin_failed_) return 0;
  const size_t cap = size_t(opt_.spill_pinned_gib * double(size_t(1) << 30));
  const size_t slab = size_t(std::max(opt_.spill_pinned_slab_mib, 16)) << 20;
  int added = 0;
  while (added < chunks && pin_pool_.total() < cap) {
    const size_t bytes = std::min(slab, cap - pin_pool_.total()) >> 20 << 20;  // (whole MiB)
    if (bytes == 0) break;
    void* p = nullptr;
    const cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocPortable);
    if (e != cudaSuccess) {  // (host memory is short: not fatal, the entries use pageable memory)
      cudaGetLastError();
      pin_failed_ = true;
      log("tiered cache: could not pin %zu MiB (%s); the pinned pool stays at %zu MiB", bytes >> 20,
          cudaGetErrorString(e), pin_pool_.total() >> 20);
      break;
    }
    pin_pool_.add_slab(static_cast<char*>(p), bytes);
    added++;
  }
  if (added > 0) spill_enforce_ram();  // (the new slab counts toward the RAM tier at once)
  return added;
}

// Keeps only the deepest checkpoint of a conversation: the checkpoints of one long prompt share most of their
// pages, and spilling each would store those pages several times. Short checkpoints (up to kKeepShort tokens)
// are only subsumed by an equal one, since a new conversation often starts from them.
bool Model::spill_subsumed(const Checkpoint& c, uint64_t ignore_id) const {
  for (const auto& o : ckpts_)
    if (&o != &c && o.ids.size() > c.ids.size() && c.pos() > kKeepShort && covers(o.ids, o.ring, c.ids, c.ring))
      return true;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    if (kv.first != ignore_id && covers(s.ids, s.ring, c.ids, c.ring) &&
        (c.pos() > kKeepShort || s.ids.size() == c.ids.size()))
      return true;
  }
  return false;
}

// Whether evicting c loses little: one deeper checkpoint of the same chain (same tokens up to c's) is in the pool and
// usable and every other checkpoint extending c lies on that same chain (a branch point, where two conversations
// diverge, is kept), and c is past kKeepShort (shorter ones are shared by new conversations, kept like in
// spill_subsumed). The deeper one resumes everything c would, except a retry that branches off before c's end. Not
// loading ones: they are not usable yet.
bool Model::ckpt_covered(const Checkpoint& c) const {
  if (c.pos() <= kKeepShort || c.share_until > use_clock_) return false;  // (a sibling-share one: its waiter has not used it)
  const Checkpoint* deepest = nullptr;
  for (const auto& o : ckpts_)
    if (&o != &c && !o.loading && o.ids.size() > c.ids.size() && covers(o.ids, o.ring, c.ids, c.ring) &&
        (!deepest || o.ids.size() > deepest->ids.size()))
      deepest = &o;
  if (!deepest) return false;
  for (const auto& o : ckpts_)
    if (&o != &c && !o.loading && o.ids.size() > c.ids.size() && covers(o.ids, o.ring, c.ids, c.ring) &&
        !covers(deepest->ids, deepest->ring, o.ids, o.ring))
      return false;
  return true;
}

void Model::spill_checkpoint(const Checkpoint& c) {
  if (spill_subsumed(c, 0)) {
    sstats_.subsumed++;
    return;
  }
  OrphanScope scope(this);  // (K) the chunks of the shallower entries erased below are found again by the copy of c
  for (auto it = spill_.begin(); it != spill_.end();) {
    auto next = std::next(it);
    const Spilled& e = it->second;
    if (!e.busy && covers(c.ids, c.ring, e.ids, e.ring) && (e.pos() > kKeepShort || e.ids.size() == c.ids.size())) {
      spill_erase(it, true);
      sstats_.subsumed++;
    }
    it = next;
  }

  const uint64_t copied = spill_copy(c, false);
  scope.end();
  if (!copied) return;  // (dropped, as without the tiered cache)
  sstats_.spilled++;
  spill_enforce_ram();
}

uint64_t Model::spill_copy(const Checkpoint& c, bool live) {
  spill_reap();
  Spilled s;
  const size_t npages = c.pages.size();
  s.kv_bytes = npages * size_t(kKvSlices) * kv_page_bytes_;
  s.state_bytes = ckpt_bytes_;
  const size_t nfull = spill_chunk_count(c.ids.size());
  const size_t ctok = size_t(kChunkPages) * kPageTokens;
  const size_t chunk_bytes = size_t(kChunkPages) * kKvSlices * kv_page_bytes_;
  const size_t tail_pages = npages - nfull * kChunkPages;  // (nfull * ctok <= pos: every chunk page is full)
  // (H) Phase A: the references and the memory, nothing queued on a stream yet. refd: every chunk this entry will list
  // (a reference taken), fresh: the ones that were not resident (their bytes are copied below).
  std::vector<SpillChunk*> refd;
  std::vector<std::pair<size_t, SpillChunk*>> fresh;
  // Drops what the entry in construction holds, after any copy that may write it has finished. A fresh chunk is freed
  // whatever its refs (its bytes are not valid yet: it must never be found).
  auto rollback = [&] {
    for (SpillChunk* ch : refd) {
      const bool is_fresh = std::any_of(fresh.begin(), fresh.end(), [&](const auto& f) { return f.second == ch; });
      if (is_fresh) {
        spill_chunks_.unref(ch);
        spill_chunk_free(ch);
      } else {
        spill_chunk_unref(ch, false);
      }
    }
    refd.clear(), fresh.clear();
  };
  try {
    if (tail_pages) s.kv = spill_alloc_kv(tail_pages * size_t(kKvSlices) * kv_page_bytes_);
    s.state.reset(new char[s.state_bytes]);
    if (nfull)
      chunk_acquire(spill_chunks_, chunk_seed(spill_fingerprint(), kChunkPages), c.ids.data(), nfull, ctok, chunk_bytes,
                    [&](size_t bytes) { return spill_alloc_kv(bytes); }, refd, fresh);
    s.chunks = refd;  // (the entry lists them from here: spill_wait below covers them; refd keeps the rollback list)
  } catch (const std::bad_alloc&) {
    rollback();
    sstats_.failed++;
    return 0;
  }
  uint64_t id = 0;
  try {
    if (!spill_async()) {
      for (auto& [ci, ch] : fresh) copy_pages_on(chunk_pages_of(c.pages, ci * kChunkPages, kChunkPages), ch->kv, true, s_);
      if (tail_pages) copy_pages_on(chunk_pages_of(c.pages, nfull * kChunkPages, tail_pages), s.kv, true, s_);
      Q27_CUDA(cudaStreamSynchronize(s_));
      Q27_CUDA(cudaStreamSynchronize(cs_));  // (the state's copy into its host slot may still be on its way)
      std::memcpy(s.state.get(), slot_mem_[c.slot], ckpt_bytes_);
    } else {
      io_begin();  // (1)
      for (auto& [ci, ch] : fresh) {
        copy_pages_on(chunk_pages_of(c.pages, ci * kChunkPages, kChunkPages), ch->kv, true, io_s_);
        ch->done = spill_event();
        Q27_CUDA(cudaEventRecord(ch->done, io_s_));
        ch->inflight = true;
      }
      if (tail_pages) copy_pages_on(chunk_pages_of(c.pages, nfull * kChunkPages, tail_pages), s.kv, true, io_s_);
      if (!fresh.empty() || tail_pages) {  // (nothing queued when every chunk was resident and there is no tail)
        s.done = spill_event();
        Q27_CUDA(cudaEventRecord(s.done, io_s_));
        s.inflight = true;
        // (2) An eviction frees these pages right after this returns: nothing queued on s_ from now on may run before the
        // copy has read them. (A live backup keeps its pages, but is waited for below instead.)
        if (!live) Q27_CUDA(cudaStreamWaitEvent(s_, s.done));
      }
      Q27_CUDA(cudaStreamSynchronize(cs_));  // (4) the state, meanwhile, as before
      std::memcpy(s.state.get(), slot_mem_[c.slot], ckpt_bytes_);
      if (live) {  // (3) an idle backup completes before it is visible to evict_checkpoint / reserve_area
        // What reads the pool's pages is the tail's copy (s.done) and the fresh chunks' copies. A chunk that was resident
        // is not read from the pool by this entry: if it is in flight (another entry's restore reads it, or an older
        // spill fills it) that is no reason for the engine thread to wait here.
        for (auto& f : fresh)
          if (f.second->inflight) {
            Q27_CUDA(cudaEventSynchronize(f.second->done));
            f.second->inflight = false;
          }
        spill_wait(s, false);
      }
    }
    const size_t nfresh = fresh.size(), nrefd = refd.size();
    id = spill_add(c, std::move(s), live);
    sstats_.chunk_hits += int64_t(nrefd - nfresh), sstats_.chunk_misses += int64_t(nfresh);
  } catch (...) {
    // From the first queued copy to the emplace, a throw must not free kv (or a fresh chunk) while the copy may still write
    // it: wait for io_s_ first, and give the events back.
    cudaStreamSynchronize(spill_async() ? io_s_ : s_);  // (the copy has finished reading / writing now; errors are not reported here)
    cudaGetLastError();
    spill_event_put(s.done);  // (null if it was never created; complete after the synchronize)
    s.done = nullptr, s.inflight = false;
    for (auto& f : fresh) f.second->inflight = false;  // (complete too; spill_chunk_free then recycles their events)
    rollback();
    throw;  // (s, and with it kv, is destroyed during the unwinding)
  }
  return id;
}

// Adds the finished copy as an entry; s.chunks already lists the chunks (references taken by spill_copy, which gives them
// back if this throws). The only throwing step (the map node) comes first, so a failure leaves spill_ram_ and the
// counters untouched.
uint64_t Model::spill_add(const Checkpoint& c, Spilled&& s, bool live) {
  s.ids = c.ids;
  s.ring = c.ring;
  s.last_use = c.last_use;
  s.owner = c.owner, s.hits = c.hits, s.last_hit = c.last_hit;
  s.live = live;
  const size_t own = s.own_bytes();
  const uint64_t id = spill_next_;
  spill_.emplace(id, std::move(s));
  spill_next_++;
  spill_ram_ += own;
  return id;
}

int Model::spill_backup_one(int victims) {
  if (!spill_on()) return 0;
  // The pool checkpoint to copy: no valid backup, and one spill_checkpoint would not subsume. victims > 0 (the idle
  // backup): only among the first `victims` of those in the order evict_checkpoint takes them (not held, then the ones
  // a deeper checkpoint covers (never needing a copy: subsumed), then least recently used), so the RAM tier holds
  // copies of what is about to leave the pool, not of everything. victims <= 0 (the stop flush): all of them.
  // (Page-pressure eviction, evict_checkpoint without for_slot, is plain LRU; this order puts covered ones first. The two
  // nearly agree all the same: a covered checkpoint is mostly skipped below as subsumed, never needing a copy, so the
  // ranking among the rest is LRU in both.)
  std::vector<Checkpoint*> order;
  for (auto& c : ckpts_)
    if (!c.loading) order.push_back(&c);  // (C) a loading one's pages are still being filled
  std::vector<char> covered(order.size());
  for (size_t i = 0; i < order.size(); i++) covered[i] = ckpt_covered(*order[i]);
  std::vector<size_t> idx(order.size());
  for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    return std::make_tuple(order[a]->held, !covered[a], order[a]->last_use) <
           std::make_tuple(order[b]->held, !covered[b], order[b]->last_use);
  });
  Checkpoint* pick = nullptr;
  int ranked = 0;
  for (size_t i : idx) {
    Checkpoint& c = *order[i];
    const bool backed = c.backup && spill_.find(c.backup) != spill_.end();  // (a)
    // (b): as spill_checkpoint's rules. (A backed-up one counts, though its own copy "subsumes" it: it is a victim.)
    if (!backed && spill_subsumed(c, 0)) continue;
    if (victims > 0 && ++ranked > victims) break;
    if (backed) continue;
    pick = &c;
    break;
  }
  if (!pick) return 0;
  // The idle backup also leaves room: its live copies together stay under half the RAM cap (the other half is for
  // checkpoints the pool really gave up, whose only copy this is).
  if (victims > 0) {
    size_t live = 0;
    for (const auto& kv : spill_)
      if (kv.second.live) live += kv.second.ram_bytes();
    if (live + ckpt_bytes_ + pick->pages.size() * size_t(kKvSlices) * kv_page_bytes_ >
        size_t(opt_.spill_ram_gib * double(size_t(1) << 30)) / 2)
      return 0;
  }
  // A backup never makes spill_enforce_ram evict anything: the copy must fit under the RAM cap as it is. (Otherwise
  // the enforce would drop another backup, which the next call picks again: an endless loop.)
  const size_t cap = size_t(opt_.spill_ram_gib * double(size_t(1) << 30));
  // A pinned piece is already counted (its bytes are the pool's, spill_host_use); a pageable one adds its bytes, and so
  // does the state. Piece by piece (RAM dedup: its tail and the chunks not resident; an entry can be partly pinned): see
  // spill_new_host_bytes. (Fragmentation may force a pageable piece: then this may briefly overshoot, and the next
  // enforce corrects it. No loop: backup_one never calls enforce.)
  const size_t need = ckpt_bytes_ + spill_new_host_bytes(*pick);
  if (spill_host_use() + need > cap) return 0;
  // (c): as spill_checkpoint, the shallower non-busy entries that the copy covers go (live ones too: their
  // checkpoints are in the pool, so nothing is lost).
  OrphanScope scope(this);  // (K)
  for (auto it = spill_.begin(); it != spill_.end();) {
    auto next = std::next(it);
    const Spilled& e = it->second;
    if (!e.busy && covers(pick->ids, pick->ring, e.ids, e.ring) && (e.pos() > kKeepShort || e.ids.size() == pick->ids.size())) {
      spill_erase(it, true);
      sstats_.subsumed++;
    }
    it = next;
  }
  const uint64_t id = spill_copy(*pick, true);
  scope.end();
  if (id == 0) return 0;
  pick->backup = id;
  sstats_.backed_up++;
  return 1;
}

uint64_t Model::spill_best(const int32_t* ids, int n, bool spec, bool with_busy, int* depth) const {
  int deepest = 0;
  for (const auto& c : ckpts_)
    if (!c.loading && c.pos() < n && (!spec || c.ring) && c.pos() > deepest && std::equal(c.ids.begin(), c.ids.end(), ids))
      deepest = c.pos();  // (C) a loading checkpoint is not a match yet
  uint64_t best = 0;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    if (s.live) continue;  // (an idle backup: its checkpoint is in the pool, and the pool's lookup has it)
    if (s.busy && !with_busy) continue;  // (A)
    if (s.pos() < n && (!spec || s.ring) && s.pos() > deepest && std::equal(s.ids.begin(), s.ids.end(), ids))
      best = kv.first, deepest = s.pos();
  }
  *depth = deepest;
  return best;
}

void Model::spill_lookup(const int32_t* ids, int n, bool spec) {
  if (!spill_sync_lookup_) return;  // (the frontend chose to compute this prompt rather than load a match: SPILL_ASYNC_PROMOTE)
  int deepest = 0;
  const uint64_t best = spill_best(ids, n, spec, false, &deepest);
  if (best && !spill_promote(best))
    log("tiered cache: a spilled checkpoint of %d tokens could not be brought back (its prompt is computed again)",
        deepest);
}

// Whether the free pages plus the pages only (visible) checkpoints hold (counted as cache_stats does) cover npages, or
// the eviction would drop checkpoints and still fail. A loading checkpoint is never evicted: not counted.
bool Model::spill_room(int npages) const {
  int freeable = pool_.free_pages();
  std::vector<int32_t> n(pool_.pages(), 0);
  for (const auto& c : ckpts_)
    if (!c.loading && !c.held)  // (F)
      for (int32_t p : c.pages) freeable += ++n[p] == pool_.refs(p);
  return freeable >= npages;
}

bool Model::spill_acquire(int npages, int* slot_out, std::vector<int32_t>* out) {
  // A state slot, then the pages; a full pool spills its least recently used checkpoints (a busy entry is kept).
  int slot = -1;
  for (;;) {
    for (size_t i = 0; i < slot_used_.size() && slot < 0; i++)
      if (slot_mem_[i] && !slot_used_[i]) slot = int(i);
    if (slot >= 0 || (!grow_state_cache(1) && !evict_checkpoint())) break;
  }
  if (slot < 0) return false;
  slot_used_[slot] = true;
  std::vector<int32_t> pages;
  while (int(pages.size()) < npages) {
    int p = pool_.alloc();
    while (p < 0 && evict_checkpoint()) p = pool_.alloc();
    if (p < 0) break;
    pages.push_back(p);
  }
  if (int(pages.size()) < npages) {
    for (int32_t p : pages) pool_.unref(p);
    slot_used_[slot] = false;
    return false;
  }
  *slot_out = slot;
  *out = std::move(pages);
  return true;
}

// A throw between spill_acquire and the checkpoint taking the pages: copies queued on the pages (spill_h2d) may still be
// writing them, and a page freed now can be handed out and written by s_ meanwhile: wait for the stream first (errors are
// not reported here: we are already unwinding), then the references and the slot go back.
void Model::spill_unacquire(std::vector<int32_t>& pages, int slot) {
  cudaStreamSynchronize(spill_async() ? io_s_ : s_);
  cudaGetLastError();
  for (int32_t p : pages) pool_.unref(p);
  pages.clear();
  slot_used_[slot] = false;
}

bool Model::spill_promote(uint64_t id) {
  auto it = spill_.find(id);
  if (it == spill_.end() || it->second.live) return false;  // (a live backup is never restored: its original is here)
  if (it->second.busy) return false;  // (A) a background restore owns it
  spill_reap();
  Spilled& s = it->second;
  s.busy = true;
  // A throw below (a CUDA error) must not leave the entry busy: nothing would ever restore, write or drop it again.
  Undo undo_busy([&] { s.busy = false; });
  spill_wait(s);  // (3) an entry evicted moments ago may still be on its way to host memory
  const bool from_disk = !s.in_ram();
  auto give_up = [&](bool forget) {
    undo_busy.dismiss();  // (it handles busy itself; and the entry may be erased here)
    s.busy = false;
    if (forget) {
      spill_erase(it, true);
      sstats_.failed++;
    } else if (from_disk) {  // (back to disk only)
      spill_release_mem(s, false);
    }
    return false;
  };
  if (from_disk) {
    const int l = spill_load(s);
    if (l == 0) return give_up(true);
    if (l < 0) return give_up(spill_io_failed(s));  // an I/O error: this restore only (the file and the entry stay) until it repeats
  }
  const int npages = int(ceil_div(s.pos(), kPageTokens));
  if (size_t(npages) * kKvSlices * kv_page_bytes_ != s.kv_bytes || s.state_bytes != ckpt_bytes_ || !spill_parts_ok(s))
    return give_up(true);

  // Feasibility first (spill_room): nothing is evicted when it cannot work.
  if (!spill_room(npages)) return give_up(false);

  try {  // (G) the checkpoint's push_back below cannot throw
    ckpts_.reserve(ckpts_.size() + 1);
  } catch (const std::bad_alloc&) {
    return give_up(false);
  }
  int slot = -1;
  std::vector<int32_t> pages;
  if (!spill_acquire(npages, &slot, &pages)) return give_up(false);
  // The slot and pages go back (after the copies that may write them finished) if anything throws before the checkpoint
  // owns them.
  Undo undo_acq([&] { spill_unacquire(pages, slot); });
  try {  // (G) the parking below cannot throw: its capacity is reserved before a copy is queued (and after the acquire above:
         // evicting checkpoints can run spill_reap, which replaces retired_). The entry's tail only: a chunk parks itself.
    retired_.reserve(retired_.size() + 1);
  } catch (const std::bad_alloc&) {
    return give_up(false);
  }
  if (!spill_async()) {
    spill_h2d(s, pages, s_, false);
    Q27_CUDA(cudaStreamSynchronize(s_));
  } else {
    // The pages may have been freed by a sequence whose kernels are still queued, or by an eviction whose D2H is
    // queued on io_s_ (ordered by the stream): io_begin covers the first, the single I/O stream the second.
    io_begin();  // (1)
    spill_h2d(s, pages, io_s_, true);
    if (!s.done) s.done = spill_event();
    Q27_CUDA(cudaEventRecord(s.done, io_s_));
    s.inflight = true;
    Q27_CUDA(cudaStreamWaitEvent(s_, s.done));  // (2) the resumed sequence's kernels wait on the GPU, not the CPU
  }
  Q27_CUDA(cudaStreamSynchronize(cs_));
  std::memcpy(slot_mem_[slot], s.state.get(), ckpt_bytes_);
  undo_acq.dismiss();  // (nothing below can throw until the entry is erased: the checkpoint's capacity is reserved)
  Checkpoint ck;
  ck.ids = std::move(s.ids);
  ck.pages = std::move(pages);  // (one reference each, from alloc)
  ck.slot = slot;
  ck.ring = s.ring;
  ck.owner = s.owner, ck.hits = s.hits, ck.last_hit = s.last_hit;  // (the use that follows, prefix_begin's match, counts the hit)
  ck.last_use = ++use_clock_;
  ckpts_.push_back(std::move(ck));
  if (s.inflight) {  // (3) the H2D copy still reads kv: park it until the copy is done (state was copied above)
    spill_ram_ -= s.own_bytes();
    retired_.push_back({std::move(s.kv), s.done});
    s.done = nullptr, s.inflight = false;
    s.state.reset();
  }
  // (I) the chunks: the references go; one nobody else lists is parked with its own event while its copy may be reading it
  for (SpillChunk* c : std::exchange(s.chunks, {})) spill_chunk_unref(c, true);
  undo_busy.dismiss();  // (the entry goes; nothing waits any more: inflight is clear)
  spill_erase(it, true);  // (the pool's copy is the one now; it is spilled again when the pool gives it up)
  sstats_.restored++;
  (from_disk ? sstats_.restored_disk : sstats_.restored_ram)++;
  return true;
}

// ---------------------------------------------------------------- SPILL_ASYNC_PROMOTE (invariants (A)-(E) at the top)

int Model::spill_match(const int32_t* ids, int n, bool spec, int64_t* out4) const {
  if (!spill_on() || !opt_.spill_async_promote) return 0;
  int depth = 0;
  const uint64_t id = spill_best(ids, n, spec, true, &depth);
  if (!id) return 0;
  const Spilled& s = spill_.at(id);
  out4[0] = int64_t(id), out4[1] = s.pos(), out4[2] = int64_t(s.kv_bytes + s.state_bytes);
  out4[3] = (!s.in_ram() && !s.path.empty()) ? 1 : 0;
  return 1;
}

int Model::spill_restore_begin(uint64_t id) {
  if (!spill_on() || !opt_.spill_async_promote) {
    log("tiered cache: restore of entry %llu refused: %s", (unsigned long long)id,
        !spill_on() ? "spill is off" : "SPILL_ASYNC_PROMOTE is off");
    return 0;
  }
  auto it = spill_.find(id);
  if (it == spill_.end() || it->second.live || it->second.busy || restoring_.count(id)) {
    log("tiered cache: restore of entry %llu refused: %s", (unsigned long long)id,
        it == spill_.end() ? "unknown entry" : it->second.live ? "live backup" : it->second.busy ? "busy" : "already restoring");
    return 0;
  }
  Spilled& s = it->second;
  spill_reap();
  const int npages = int(ceil_div(s.pos(), kPageTokens));
  if (size_t(npages) * kKvSlices * kv_page_bytes_ != s.kv_bytes || s.state_bytes != ckpt_bytes_ ||
      (s.in_ram() && !spill_parts_ok(s))) {  // (damaged: as promote)
    spill_erase(it, true);
    sstats_.failed++;
    sstats_.restore_failed++;
    return 0;
  }
  if (!spill_room(npages)) {  // (nothing changed; checked again when a file read ends)
    log("tiered cache: restore of entry %llu refused: no room (freeable < needed %d pages; %d free now, the rest is held by sequences)",
        (unsigned long long)id, npages, pool_.free_pages());
    return 0;
  }
  if (!s.in_ram() && s.path.empty()) {
    log("tiered cache: restore of entry %llu refused: no memory copy and no file", (unsigned long long)id);
    return 0;
  }
  if (!s.in_ram() && !spill_disk_ready(s)) {  // (a chunk file of its manifest is gone or the manifest is inconsistent: as a damaged file)
    log("tiered cache: restore of entry %llu: a chunk file of its manifest is missing (its prompt is computed again)", (unsigned long long)id);
    spill_erase(it, true);
    sstats_.failed++;
    sstats_.restore_failed++;
    return 0;
  }
  // (G) every allocation that can throw comes before any state changes or thread starts.
  Restore r;
  r.t0 = std::chrono::steady_clock::now();
  r.bytes = s.kv_bytes + s.state_bytes;
  try {
    restore_results_.reserve(restoring_.size() + restore_results_.size() + 1);
    ckpts_.reserve(ckpts_.size() + 1);
    restoring_.emplace(id, std::move(r));
  } catch (const std::bad_alloc&) {
    log("tiered cache: restore of entry %llu refused: out of host memory", (unsigned long long)id);
    return 0;
  }
  if (s.in_ram()) {
    s.busy = true;
    int res;
    try {
      res = spill_restore_issue(id);
    } catch (...) {  // (a CUDA error: the pages and slot went back inside; the entry is idle again, not stuck busy)
      restoring_.erase(id);
      s.busy = false;
      throw;
    }
    if (res == 0) {
      restoring_.erase(id);
      s.busy = false;
      log("tiered cache: restore of entry %llu refused: no state slot or pages could be taken", (unsigned long long)id);
      return 0;
    }
    if (res == 2) restore_results_.emplace_back(id, true);  // (complete already)
    return 1;
  }
  // Only on disk: the buffers here (a private entry: P), the reads of the manifest and its chunk files on a thread (B).
  auto d = std::make_unique<DiskRead>();
  try {
    s.state.reset(new char[s.state_bytes]);
    s.kv = spill_alloc_kv(s.kv_bytes);
    if (!spill_read_plan(s, *d)) throw std::runtime_error("inconsistent entry");
    d->start();  // (the last step that can throw: if it did, no thread runs)
  } catch (const std::exception&) {  // (bad_alloc, or no thread: the entry stays on disk)
    s.kv.reset(), s.state.reset();
    restoring_.erase(id);
    log("tiered cache: restore of entry %llu refused: the file read could not start", (unsigned long long)id);
    return 0;
  }
  // From here nothing throws: the thread is owned by the registered restore (its destructor joins).
  spill_ram_ += s.own_bytes();  // (a private entry: all its pages are in kv)
  s.busy = true;
  restoring_.at(id).disk = std::move(d);
  restoring_.at(id).from_disk = true;
  return 2;
}

// The entry (in memory, busy, registered in restoring_) gets a state slot and pages and its copy. 0: no room (nothing
// changed). 1: queued (async I/O + pinned): the checkpoint is loading, the entry's memory is parked in retired_.
// 2: the copy is complete (finished and counted).
int Model::spill_restore_issue(uint64_t id) {
  Spilled& s = spill_.at(id);
  const int npages = int(ceil_div(s.pos(), kPageTokens));
  if (!spill_room(npages)) return 0;
  try {  // (G) before a slot, pages or an event are taken: push_back below cannot throw
    ckpts_.reserve(ckpts_.size() + 1);
  } catch (const std::bad_alloc&) {
    return 0;
  }
  std::vector<int32_t> ids = s.ids;  // (copied before a slot, pages or an event are taken: this one can throw bad_alloc)
  int slot = -1;
  std::vector<int32_t> pages;
  if (!spill_acquire(npages, &slot, &pages)) return 0;
  cudaEvent_t ev = nullptr;
  // A throw from here to the checkpoint's registration (a CUDA error): the pages and slot go back, after the copies that
  // may write them finished, and the load event is recycled.
  Undo undo_acq([&] {
    spill_unacquire(pages, slot);
    if (ev) spill_event_put(ev);  // (complete: spill_unacquire waited for the stream)
  });
  try {  // (G) parking capacity for the tail, after the acquire (evicting checkpoints can run spill_reap, which replaces
         // retired_). The chunks stay listed (4: held until the restore completes); one freed later parks itself.
    retired_.reserve(retired_.size() + 1);
  } catch (const std::bad_alloc&) {
    return 0;
  }
  // A pageable entry is read by the driver while staging: an earlier D2H into it must have completed first. A pinned
  // one is ordered by io_s_ itself.
  bool overlap = spill_async() && (s.kv.pinned() || !s.kv.in_ram());
  for (const SpillChunk* c : s.chunks) overlap = overlap && c->kv.pinned();  // (a mixed entry waits for its pageable pieces)
  if (!overlap) spill_wait(s);
  if (!spill_async()) {
    spill_h2d(s, pages, s_, false);
    Q27_CUDA(cudaStreamSynchronize(s_));
  } else {
    io_begin();  // (1)
    spill_h2d(s, pages, io_s_, true);
    if (!s.done) s.done = spill_event();
    Q27_CUDA(cudaEventRecord(s.done, io_s_));
    s.inflight = true;
    ev = spill_event();
    Q27_CUDA(cudaEventRecord(ev, io_s_));
    // (C) no cudaStreamWaitEvent(s_, ...): s_ goes on; the checkpoint is invisible until ev completes.
  }
  Q27_CUDA(cudaStreamSynchronize(cs_));
  std::memcpy(slot_mem_[slot], s.state.get(), ckpt_bytes_);
  undo_acq.dismiss();  // (nothing below throws: ckpts_ capacity was reserved above)
  Checkpoint ck;
  ck.ids = std::move(ids);  // (the busy entry is still looked at by spill_best / spill_subsumed: it was copied)
  ck.pages = std::move(pages);
  ck.slot = slot;
  ck.ring = s.ring;
  ck.owner = s.owner, ck.hits = s.hits, ck.last_hit = s.last_hit;
  ck.last_use = ++use_clock_;
  ck.load_id = id;
  if (ev) ck.loading = true, ck.load_ev = ev;
  else ck.held = !restoring_.at(id).orphan;  // (F) complete already
  ckpts_.push_back(std::move(ck));  // (capacity reserved above: no throw)
  if (!ev) {
    spill_restore_finish(id);
    return 2;
  }
  spill_ram_ -= s.own_bytes();  // (D) the copy still reads kv: parked until it is complete
  retired_.push_back({std::move(s.kv), s.done, true});
  s.done = nullptr, s.inflight = false;
  s.state.reset();
  // (4) The chunks stay listed (their references held) until the restore completes (spill_loading_done -> spill_erase
  // releases them): a chunk still being read by the copy stays in the store, findable by other spills, instead of being
  // parked at 0 refs where nothing can re-find it.
  restoring_.at(id).issued = true;
  return 1;
}

// A restore ended well: counters, and the entry goes (its file too; the pool's copy is the one now; it is spilled again
// when the pool gives it up).
void Model::spill_restore_finish(uint64_t id) {
  if (auto r = restoring_.find(id); r != restoring_.end()) {
    sstats_.restore_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r->second.t0).count();
    sstats_.restore_bytes += int64_t(r->second.bytes);
    (r->second.from_disk ? sstats_.restored_disk : sstats_.restored_ram)++;
    restoring_.erase(r);
  } else {
    sstats_.restored_ram++;
  }
  sstats_.restored++;
  sstats_.restored_async++;
  if (auto it = spill_.find(id); it != spill_.end()) spill_erase(it, true);
}

// A loading checkpoint's copy is complete (the caller saw load_ev done, or waited for it): it becomes visible, and held
// for the request that waited for it unless that one gave up already (F). Reported ok by the next poll.
void Model::spill_loading_done(Checkpoint& c) {
  const uint64_t id = c.load_id;
  const auto r = restoring_.find(id);
  c.loading = false;
  c.held = r == restoring_.end() || !r->second.orphan;
  c.last_use = ++use_clock_;  // (fresh: not the next victim before the request that waited for it is admitted)
  spill_event_put(c.load_ev);
  c.load_ev = nullptr;
  restore_results_.emplace_back(id, true);  // (capacity reserved at begin (G))
  // load_ev completed, and it was recorded after the copy of each of the entry's chunks (same stream): a chunk whose own
  // latest event is complete needs no waiting or parking when its last reference goes (a later restore of another entry
  // that re-records it sets `inflight` again).
  if (auto it = spill_.find(id); it != spill_.end())
    for (SpillChunk* ch : it->second.chunks)
      if (ch->inflight && cudaEventQuery(ch->done) == cudaSuccess) ch->inflight = false;
  cudaGetLastError();  // (a not-ready query is not an error)
  spill_restore_finish(id);
}

int Model::spill_release(uint64_t id) {
  if (!spill_on() || !opt_.spill_async_promote || id == 0) return 0;
  if (auto r = restoring_.find(id); r != restoring_.end()) {
    r->second.orphan = true;  // (its checkpoint will not be held)
    return 1;
  }
  for (auto& c : ckpts_)
    if (c.held && c.load_id == id) {
      c.held = false;
      return 1;
    }
  return 0;
}

int Model::spill_restore_poll(int64_t* done_ids, int32_t* done_ok, int cap) {
  if (!spill_on() || !opt_.spill_async_promote || !done_ids || cap <= 0) return 0;
  // (results are queued in restore_results_ (capacity reserved at begin) and handed out at the end, oldest first)
  // File reads that ended: failed -> forget the entry; ok -> pages and the copy (as for an entry in memory).
  std::vector<uint64_t> ended;
  for (auto& [id, r] : restoring_)
    if (r.disk && r.disk->state_now() != 0) ended.push_back(id);
  for (uint64_t id : ended) {
    auto rit = restoring_.find(id);
    if (rit == restoring_.end()) continue;  // (finished meanwhile by evict_checkpoint? only issued ones: cannot be, kept for safety)
    Restore& r = rit->second;
    r.disk->join();
    const bool ok = r.disk->state_now() > 0;
    const int bad = r.disk->bad.load();  // (the chunk file that failed verification or is missing; -1: the manifest, or an I/O error)
    const bool io_error = r.disk->io_error.load();
    r.disk.reset();
    auto it = spill_.find(id);
    if (!ok && io_error && it != spill_.end() && !spill_io_failed(it->second)) {
      // An I/O error (a lock, EIO ...) says nothing about the files: only this restore fails; the entry, its manifest and the chunk
      // files (which other entries share) stay, and the next prefix_begin / restore may try again.
      log("tiered cache: a file of a spilled checkpoint could not be read (I/O error; kept, its prompt is computed again)");
      Spilled& s = it->second;
      spill_release_mem(s, false);
      s.busy = false;
      restoring_.erase(id);
      sstats_.restore_failed++;
      restore_results_.emplace_back(id, false);
      continue;
    }
    if (!ok) {
      log("tiered cache: the file of a spilled checkpoint could not be read (its prompt is computed again)");
      if (bad >= 0 && it != spill_.end() && size_t(bad) < it->second.dnames.size())
        disk_.forget_file(it->second.dnames[size_t(bad)]);  // (O) not served again: a later export writes it afresh
      if (it != spill_.end()) spill_erase(it, true);
      restoring_.erase(id);
      sstats_.failed++;
      sstats_.restore_failed++;
      restore_results_.emplace_back(id, false);
      continue;
    }
    int res;
    try {
      res = spill_restore_issue(id);
    } catch (...) {  // (a CUDA error: the pages and slot went back inside; back to disk only, not stuck busy)
      if (it != spill_.end()) {
        spill_release_mem(it->second, false);
        it->second.busy = false;
      }
      restoring_.erase(id);
      sstats_.restore_failed++;
      restore_results_.emplace_back(id, false);  // (capacity reserved at begin (G))
      throw;
    }
    if (res == 0) {  // (the pool filled up meanwhile: back to disk only, the next prefix_begin may still try)
      Spilled& s = it->second;
      spill_release_mem(s, false);
      s.busy = false;
      restoring_.erase(id);
      sstats_.restore_failed++;
      restore_results_.emplace_back(id, false);
    } else if (res == 2) {
      restore_results_.emplace_back(id, true);
    }
  }
  // Copies that completed: the checkpoint becomes visible (E).
  for (size_t i = 0; i < ckpts_.size(); i++) {
    Checkpoint& c = ckpts_[i];
    if (!c.loading) continue;
    const cudaError_t e = cudaEventQuery(c.load_ev);
    if (e == cudaErrorNotReady) {
      cudaGetLastError();
      continue;
    }
    Q27_CUDA(e);
    spill_loading_done(c);
  }
  spill_reap();  // (non-blocking for async items)
  const int n = int(std::min<size_t>(size_t(cap), restore_results_.size()));
  for (int i = 0; i < n; i++) {
    done_ids[i] = int64_t(restore_results_[i].first);
    if (done_ok) done_ok[i] = restore_results_[i].second ? 1 : 0;
  }
  restore_results_.erase(restore_results_.begin(), restore_results_.begin() + n);
  return n;
}

// ---------------------------------------------------------------- retention (retention.h)

// Superseded flags of all spilled entries: nodes = the entries (checked), then the pool's checkpoints (extenders only). Cached until an
// entry or a checkpoint was added or removed (the counters below change with each of those).
void Model::spill_retention_refresh() const {
  const int64_t key[6] = {int64_t(spill_next_), int64_t(spill_.size()), cstats_.saved, cstats_.evicted, int64_t(ckpts_.size()),
                          sstats_.restored};
  if (std::equal(std::begin(key), std::end(key), std::begin(sup_key_))) return;
  std::vector<RetNode> nodes;
  std::vector<uint64_t> ids;
  nodes.reserve(spill_.size() + ckpts_.size()), ids.reserve(spill_.size());
  auto bh = [](const std::vector<int32_t>& v, std::vector<uint64_t>& h) -> const std::vector<uint64_t>* {
    if (h.size() != v.size() / kBlockTokens) h = block_hashes(v);  // (once per entry / checkpoint: its tokens never change)
    return &h;
  };
  for (const auto& kv : spill_) {
    nodes.push_back({&kv.second.ids, bh(kv.second.ids, kv.second.bh), kv.second.ring, kv.second.owner});
    ids.push_back(kv.first);
  }
  const size_t n_check = nodes.size();
  for (const auto& c : ckpts_) nodes.push_back({&c.ids, bh(c.ids, c.bh), c.ring, c.owner});
  std::vector<char> flags;
  superseded_flags(nodes, n_check, flags);
  sup_.clear();
  for (size_t i = 0; i < n_check; i++) sup_[ids[i]] = flags[i];
  std::copy(std::begin(key), std::end(key), std::begin(sup_key_));
}

bool Model::spill_superseded(uint64_t id) const {
  spill_retention_refresh();
  const auto it = sup_.find(id);
  return it != sup_.end() && it->second;
}

// Bytes per agent in a tier (retention.h agent_use): the RAM tier's entries' own bytes and the resident chunks they list; the SSD
// tier's manifests and the chunk files their manifests list. A chunk shared by entries of several agents is split between the agents.
std::map<uint64_t, double> Model::spill_use(bool ssd) const {
  std::vector<UseEntry> use;
  use.reserve(spill_.size());
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    UseEntry u;
    u.owner = s.owner;
    if (!ssd) {
      if (!s.in_ram() && s.chunks.empty()) continue;
      u.own = double(s.own_bytes());
      for (const SpillChunk* c : s.chunks) u.shared.emplace_back(c, double(c->bytes()));
    } else {
      if (s.path.empty()) continue;
      u.own = double(s.manifest_bytes);
      for (const std::string& n : s.dnames)
        if (const DiskIndex::Chunk* c = disk_.find(n); c && c->present) u.shared.emplace_back(c, double(c->bytes));
    }
    use.push_back(std::move(u));
  }
  return agent_use(use);
}

std::string Model::spill_meta() const {
  spill_retention_refresh();
  std::ostringstream out;
  for (const auto& kv : spill_) {
    const auto it = sup_.find(kv.first);
    out << kv.first << '\t' << kv.second.owner << '\t' << kv.second.hits << '\t' << (it != sup_.end() && it->second ? 1 : 0) << '\n';
  }
  return out.str();
}

std::string Model::spill_agents() const {
  struct Row {
    double ram = 0, ssd = 0;
    int ram_n = 0, ssd_n = 0;
  };
  std::map<uint64_t, Row> rows;
  for (const auto& [o, b] : spill_use(false)) rows[o].ram = b;
  for (const auto& [o, b] : spill_use(true)) rows[o].ssd = b;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    if (s.in_ram() || (s.busy && !s.chunks.empty())) rows[s.owner].ram_n++;
    if (!s.path.empty()) rows[s.owner].ssd_n++;
  }
  std::ostringstream out;
  for (const auto& [o, r] : rows)
    out << o << '\t' << int64_t(r.ram + 0.5) << '\t' << int64_t(r.ssd + 0.5) << '\t' << r.ram_n << '\t' << r.ssd_n << '\n';
  return out.str();
}

void Model::spill_enforce_ram() {
  const size_t cap = size_t(opt_.spill_ram_gib * double(size_t(1) << 30));
  // (K) host use is re-read every iteration: an entry whose chunks others share frees little (or nothing).
  while (spill_host_use() > cap) {
    // Chunks nobody lists are the cheapest memory to give back.
    if (spill_chunks_.orphan_count() > 0) {
      spill_trim_orphans();
      continue;
    }
    // The victim: the entry that gives the most host memory back (spill_freed), the least recently used among those
    // within a factor 2 of the best (pick_victim). An idle backup goes first: its checkpoint is still in the pool, so
    // dropping it loses nothing, and it is never written to the SSD (it is not the only copy). Otherwise only sole copies
    // are left: written to the directory, or dropped without one. Pinned pieces free only the state (their bytes stay in
    // the pool, counted whole), so pageable ones weigh more.
    // choose_victim (chunk_store.h): the backups first, most bytes freed first, unless the best one frees only its state
    // while a sole copy frees >= 4x that (then the sole copy goes to the SSD instead of backups being dropped one by one).
    // Within a class the order is the retention policy's (retention.h; chunk_store.h pick_victim): the agent using the most RAM,
    // then superseded entries, never-hit before hit ones, and the old "frees the most / least recently used".
    auto victim = spill_.end();
    bool victim_superseded = false;
    {
      std::vector<VictimCand> cand[2];  // [0] the idle backups, [1] the sole copies
      std::vector<std::map<uint64_t, Spilled>::iterator> its[2];
      const std::map<uint64_t, double> use = opt_.spill_fair_agents ? spill_use(false) : std::map<uint64_t, double>();
      for (auto it = spill_.begin(); it != spill_.end(); ++it)
        if (it->second.in_ram() && !it->second.busy) {
          const Spilled& s = it->second;
          const int cls = s.live ? 0 : 1;
          VictimCand vc{s.last_use, spill_freed(s), s.state_bytes};
          vc.agent = s.owner, vc.hits = s.hits, vc.superseded = spill_superseded(it->first);
          if (const auto u = use.find(s.owner); u != use.end()) vc.agent_use = u->second;
          cand[cls].push_back(vc);
          its[cls].push_back(it);
        }
      if (const VictimChoice v = choose_victim(cand[0], cand[1], opt_.spill_fair_agents); v.pass >= 0) {
        victim = its[v.pass][size_t(v.idx)];
        victim_superseded = cand[v.pass][size_t(v.idx)].superseded;
      }
    }
    // (A victim may be an entry whose D2H copy is still in flight: spill_erase / spill_write wait for it, (3).)
    if (victim == spill_.end()) return;
    if (victim->second.live) {
      spill_erase(victim, true);
      sstats_.dropped++;
      continue;
    }
    // A superseded entry is never written to the SSD (a deeper checkpoint of its conversation exists): dropped, as the frontend's
    // retention does.
    if (!opt_.spill_dir.empty() && !victim_superseded) {
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();
      const std::string path =
          opt_.spill_dir + "/entries/" + std::to_string(ns) + "-" + std::to_string(victim->first) + ".q27m";
      try {
        spill_write(victim->second, path);
        continue;
      } catch (const std::exception& e) {
        log("tiered cache: %s", e.what());
      }
    }
    spill_erase(victim, true);
    sstats_.dropped++;
  }
}

// ---------------------------------------------------------------- SSD dedup: manifests + chunk files ((M)-(Q) at the top)

Geom Model::spill_geom() const {
  Geom g;
  g.slices = uint32_t(kKvSlices), g.chunk_pages = uint32_t(kChunkPages), g.page_tokens = uint32_t(kPageTokens);
  g.page_bytes = kv_page_bytes_;
  return g;
}

// One chunk directory per Model: <SPILL_DIR>/chunks, or (no SPILL_DIR) next to the manifest's directory.
bool Model::spill_disk_dir(const std::string& manifest_path) {
  if (!disk_.dir().empty()) return true;
  std::filesystem::path d;
  if (!opt_.spill_dir.empty()) d = std::filesystem::path(opt_.spill_dir) / "chunks";
  else if (!manifest_path.empty()) d = std::filesystem::path(manifest_path).parent_path().parent_path() / "chunks";
  else return false;
  disk_.set_dir(d.string());
  return true;
}

// (O) The manifest goes first (a crash in between leaves chunks nobody lists: swept at the next start), then each chunk file loses
// one reference and is deleted at 0.
void Model::spill_disk_forget(Spilled& s) {
  if (s.path.empty()) return;
  std::error_code ec;
  std::filesystem::remove(s.path, ec);
  for (const std::string& n : s.dnames) disk_.unref(n);
  s.path.clear(), s.dnames.clear();
  s.manifest_bytes = 0;
}

const std::vector<std::string>& Model::spill_chain_names(const Spilled& s, size_t n) const {
  if (s.cnames.size() < n) {
    std::vector<ChunkId> chain =
        chunk_chain(spill_fingerprint(), kChunkPages, s.ids.data(), n, size_t(kChunkPages) * kPageTokens);
    s.cnames.clear();
    for (ChunkId& c : chain) s.cnames.push_back(std::move(c.name));
  }
  return s.cnames;
}

// The entry's manifest is consistent with what this engine can restore and every chunk file it lists is believed to be there.
bool Model::spill_disk_ready(const Spilled& s) const {
  if (s.path.empty() || disk_.dir().empty()) return false;
  const size_t nch = s.dnames.size(), npages = size_t(ceil_div(s.pos(), kPageTokens));
  if (nch * kChunkPages > npages || nch * size_t(kChunkPages) * kPageTokens > s.ids.size()) return false;
  for (const std::string& n : s.dnames) {
    const DiskIndex::Chunk* c = disk_.find(n);
    if (!c || !c->present) return false;
  }
  return true;
}

// The reader's inputs for an entry on disk whose private buffers (s.kv: the whole stream, s.state) are allocated: the tail and each
// chunk's payload are cut out of that one linear stream (run_pieces), the manifest / chunk paths and the expectations copied. May
// throw bad_alloc (nothing started yet).
bool Model::spill_read_plan(Spilled& s, DiskRead& d) {
  const size_t nch = s.dnames.size(), npages = size_t(ceil_div(s.pos(), kPageTokens));
  if (!spill_disk_ready(s) || s.kv.total != s.kv_bytes || s.kv_bytes != npages * kKvSlices * kv_page_bytes_) return false;
  const size_t tail_pages = npages - nch * kChunkPages;
  d.manifest = s.path;
  d.info.fingerprint = spill_fingerprint(), d.info.g = spill_geom(), d.info.ring = s.ring;
  d.info.version = s.manifest_version;  // (the file's own: a version 1 manifest of an older build is read as it was written)
  d.info.state_bytes = s.state_bytes, d.info.tail_bytes = tail_pages * kKvSlices * kv_page_bytes_;
  d.ids = s.ids;
  d.names = s.dnames;
  d.state = s.state.get();
  run_pieces(s.kv, kKvSlices, kv_page_bytes_, npages, nch * kChunkPages, tail_pages, d.tail);
  d.chunk.resize(nch);
  d.chunk_paths.reserve(nch);
  for (size_t i = 0; i < nch; i++) {
    d.chunk_paths.push_back(disk_.path(s.dnames[i]));
    run_pieces(s.kv, kKvSlices, kv_page_bytes_, npages, i * kChunkPages, kChunkPages, d.chunk[i]);
  }
  return true;
}

// What dropping the entry frees on the SSD: its manifest and the chunk files only it lists.
uint64_t Model::spill_unique_disk(const Spilled& s) const {
  if (s.path.empty()) return 0;
  uint64_t n = s.manifest_bytes;
  for (const std::string& name : s.dnames) {
    const DiskIndex::Chunk* c = disk_.find(name);
    if (c && c->present && c->refs == 1) n += c->bytes;
  }
  return n;
}

// What exporting an entry in memory writes now: its manifest and the chunk files that are not on disk yet.
uint64_t Model::spill_new_disk(const Spilled& s) const {
  if (!s.in_ram()) return 0;
  const size_t nch = s.chunks.size(), ct = size_t(kChunkPages) * kPageTokens;
  uint64_t n = manifest_file_bytes(s.state_bytes, s.kv.total, s.ids.size(), nch);
  if (nch && nch * ct <= s.ids.size()) {
    const std::vector<std::string>& names = spill_chain_names(s, nch);
    const uint64_t cb = chunk_file_bytes(spill_geom(), ct);
    for (size_t i = 0; i < nch; i++) {
      const DiskIndex::Chunk* c = disk_.find(names[i]);
      if (!c || !c->present) n += cb;
    }
  }
  return n;
}

void Model::spill_write(Spilled& s, const std::string& path) {
  Q27_CHECK(s.in_ram(), "tiered cache: the entry is not in memory");
  Q27_CHECK(spill_disk_dir(path), "tiered cache: no directory for the chunk files");
  spill_wait(s);  // (3) its D2H copy (and its chunks') must be complete before the bytes are written
  const size_t nch = s.chunks.size();
  const size_t npages = size_t(ceil_div(s.pos(), kPageTokens));
  Q27_CHECK(nch * kChunkPages <= npages && nch * size_t(kChunkPages) * kPageTokens <= s.ids.size() &&
                s.kv.total == (npages - nch * kChunkPages) * size_t(kKvSlices) * kv_page_bytes_,
            "tiered cache: the entry's parts do not add up");
  {  // (M) each RAM chunk is the chunk its position in the entry's token chain names
    const std::vector<ChunkId> chain =
        chunk_chain(spill_fingerprint(), kChunkPages, s.ids.data(), nch, size_t(kChunkPages) * kPageTokens);
    for (size_t i = 0; i < nch; i++)
      Q27_CHECK(s.chunks[i]->key == chain[i].key, "tiered cache: a chunk's key does not match the entry's tokens");
  }
  // (N) the chunk files, the manifest, the references: spill_files.h export_entry_files (tested on the host, tests/test_spill_files.cc).
  // A throw leaves the entry as it was (its memory, its old manifest) and removes the chunk files it wrote that nobody lists.
  const uint64_t mbytes = export_entry_files(
      disk_, spill_fingerprint(), spill_geom(), path, s.ring, s.ids, s.state.get(), s.state_bytes, s.kv, nch,
      [&](size_t i) -> const HostKv& { return s.chunks[i]->kv; }, s.path, s.dnames, EntryMeta{s.owner, s.hits});
  s.manifest_bytes = mbytes;
  s.manifest_version = kManifestVersion;
  spill_release_mem(s, false);  // (its chunk references too: a chunk nobody else lists is freed)
  sstats_.written++;
}

// 1: loaded. 0: the files are damaged / missing / do not fit (the caller drops the entry; a damaged chunk file is forgotten).
// -1: an I/O error (the files may be fine: the caller keeps the entry, spill_io_failed counts it).
int Model::spill_load(Spilled& s) {
  const size_t npages = size_t(ceil_div(s.pos(), kPageTokens));
  if (s.state_bytes != ckpt_bytes_ || s.kv_bytes != npages * kKvSlices * kv_page_bytes_ || !spill_disk_ready(s)) return 0;
  DiskRead d;
  try {
    spill_reap();
    s.state.reset(new char[s.state_bytes]);
    s.kv = spill_alloc_kv(s.kv_bytes);
    if (!spill_read_plan(s, d)) throw std::bad_alloc();
  } catch (const std::bad_alloc&) {
    s.kv.reset(), s.state.reset();
    return -1;  // (out of host memory: nothing is known about the files)
  }
  if (!d.run_sync()) {
    s.kv.reset(), s.state.reset();
    if (d.io_error.load()) return -1;
    const int bad = d.bad.load();
    if (bad >= 0 && size_t(bad) < s.dnames.size()) disk_.forget_file(s.dnames[size_t(bad)]);  // (O)
    return 0;
  }
  spill_ram_ += s.own_bytes();  // (a private entry: all its pages in kv)
  return 1;
}

// A restore of the entry failed on an I/O error: counts it; true (drop the entry) once kMaxIoFails in a row failed (a persistent
// error is as good as a damaged file, and the entry would otherwise be retried for ever).
bool Model::spill_io_failed(Spilled& s) {
  log("tiered cache: reading the files of a spilled checkpoint failed with an I/O error (%d in a row)", s.io_fails + 1);
  return ++s.io_fails >= kMaxIoFails;
}

// remove_file == false is not used: an erase always gives the entry's disk references back (a leaked one would keep a chunk file for good).
void Model::spill_erase(std::map<uint64_t, Spilled>::iterator it, bool remove_file) {
  // (3) the entry's own buffers are freed below: an in-flight D2H must have finished writing them. Its chunks are waited for
  // only if they go (spill_chunk_free): an erase does not stall behind another entry's copy into a chunk that stays.
  spill_wait(it->second, false);
  spill_event_put(it->second.done);
  it->second.done = nullptr;
  spill_release_mem(it->second, false);  // (the entry's memory and its chunk references)
  (void)remove_file;
  spill_disk_forget(it->second);  // (O) its manifest and its share of the chunk files
  spill_.erase(it);
}

std::string Model::spill_list() const {
  std::ostringstream out;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    out << kv.first << '\t' << s.pos() << '\t' << s.ram_bytes() << '\t' << (s.path.empty() ? 0 : 1) << '\t'
        << s.last_use << '\t' << (s.live ? 1 : s.busy ? 2 : 0) << '\t' << s.path << '\n';  // (busy: a restore owns it)
  }
  return out.str();
}

std::string Model::spill_list_ex() const {
  std::ostringstream out;
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    out << kv.first << '\t' << s.pos() << '\t' << s.ram_bytes() << '\t' << (s.path.empty() ? 0 : 1) << '\t'
        << s.last_use << '\t' << (s.live ? 1 : s.busy ? 2 : 0) << '\t' << (s.kv_bytes + s.state_bytes) << '\t'
        << spill_unique_disk(s) << '\t' << spill_new_disk(s) << '\t' << s.path << '\n';
  }
  return out.str();
}

void Model::spill_disk_usage(int64_t* out, int n) const {
  // [4]: chunk files whose removal failed and is retried (their bytes are in [0]).
  int64_t v[5] = {int64_t(disk_.physical_bytes()), 0, int64_t(disk_.files()), 0, int64_t(disk_.pending_files())};
  for (const auto& kv : spill_) {
    const Spilled& s = kv.second;
    if (s.path.empty()) continue;
    v[0] += int64_t(s.manifest_bytes), v[1] += int64_t(s.kv_bytes + s.state_bytes), v[3]++;
  }
  for (int i = 0; i < n && i < 5; i++) out[i] = v[i];
}

int Model::spill_sweep() {
  if (!spill_on() || !spill_disk_dir("")) return 0;
  return int(disk_.sweep());
}

void Model::spill_export(uint64_t id, const std::string& path) {
  auto it = spill_.find(id);
  Q27_CHECK(it != spill_.end(), "tiered cache: no such entry");
  Q27_CHECK(!it->second.busy, "tiered cache: the entry is being restored");
  spill_write(it->second, path);
}

void Model::spill_drop(uint64_t id) {
  auto it = spill_.find(id);
  if (it == spill_.end() || it->second.busy) return;
  spill_erase(it, true);
  sstats_.dropped++;
}

// Indexes a manifest an earlier run wrote (0: not one of this model and KV type, damaged, or a chunk file it lists is missing or the
// wrong size: the caller deletes it). Reads the manifest's head only (tokens, chunk names); the checksums are a restore's job.
uint64_t Model::spill_import(const std::string& path) {
  if (!spill_on()) return 0;
  for (const auto& kv : spill_)
    if (kv.second.path == path) return kv.first;  // (already indexed: its id, so a second start() does not take it for a rejected file)
  if (!spill_disk_dir(path)) return 0;
  ManifestHead h;
  std::vector<std::string> unknown;  // (listed, present on disk, not in the index yet)
  if (!validate_manifest(disk_, path, spill_fingerprint(), spill_geom(), ckpt_bytes_, size_t(opt_.max_tokens) + 64, h, unknown))
    return 0;
  const size_t npages = size_t(ceil_div(h.ids.size(), kPageTokens));
  Spilled s;
  s.ids = std::move(h.ids);
  s.kv_bytes = npages * kKvSlices * kv_page_bytes_;
  s.state_bytes = h.info.state_bytes;
  s.ring = h.info.ring;
  s.owner = h.info.owner, s.hits = h.info.hits, s.manifest_version = h.info.version;
  s.path = path;
  s.dnames = h.names;
  s.manifest_bytes = h.file_bytes;
  s.last_use = 0;  // (older than anything this run used)
  const uint64_t cb = chunk_file_bytes(spill_geom(), size_t(kChunkPages) * kPageTokens);
  for (const std::string& n : unknown) disk_.add(n, cb);  // (may throw: nothing references them yet, the sweep deletes strays)
  const uint64_t id = spill_next_;
  spill_.emplace(id, std::move(s));  // (the last step that can throw)
  spill_next_++;
  for (const std::string& n : h.names) disk_.ref(n);
  return id;
}

void Model::spill_stats(int64_t* out, int n) const {
  int64_t v[kSpillStats], ram = 0, disk = 0, logical = 0;
  int64_t* const dst = out;
  out = v;
  for (const auto& kv : spill_) {
    ram += kv.second.in_ram() || (kv.second.busy && !kv.second.chunks.empty());  // (an issued restore holds its chunks)
    disk += !kv.second.path.empty();
    logical += int64_t(kv.second.ram_bytes());
  }
  const int64_t phys = int64_t(spill_phys());
  out[0] = int64_t(spill_.size()), out[1] = ram, out[2] = disk, out[3] = phys;
  out[4] = sstats_.spilled, out[5] = sstats_.restored, out[6] = sstats_.written, out[7] = sstats_.dropped;
  out[8] = sstats_.subsumed, out[9] = sstats_.failed, out[10] = sstats_.backed_up;
  out[11] = int64_t(pin_pool_.total()), out[12] = int64_t(pin_pool_.used());
  out[13] = sstats_.pin_hits, out[14] = sstats_.pin_misses;
  out[15] = int64_t(restoring_.size()), out[16] = sstats_.restored_async, out[17] = sstats_.restore_failed;
  out[18] = sstats_.restore_ns / 1000000, out[19] = sstats_.restore_bytes;
  out[20] = logical, out[21] = phys, out[22] = int64_t(spill_chunks_.size());
  out[23] = sstats_.chunk_hits, out[24] = sstats_.chunk_misses, out[25] = std::max<int64_t>(0, logical - phys);
  out[26] = sstats_.restored_ram, out[27] = sstats_.restored_disk;
  for (int i = 0; i < n && i < kSpillStats; i++) dst[i] = v[i];  // (an older caller's shorter buffer is never overrun)
}

}  // namespace q27
