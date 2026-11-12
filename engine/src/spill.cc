// Tiered cache (SPILL_RAM_GIB > 0, docs/tiered-cache.md): checkpoints the KV pool gives up go to host memory instead
// of being dropped, and from there to files (frontend/tiercache.py decides when; the RAM cap here is the backstop).
// prefix_begin brings the deepest spilled match back into the pool before its own lookup, so everything after that
// (the shared full pages, the copied partial page, the plan of the prefill) is the unchanged cache-hit path.
//
// All of this runs on the engine's thread, like the rest of Model (no locks).
//
// Two optional speed-ups, both off by default (Options::spill_pinned_gib, Options::spill_async_io):
//
// SPILL_PINNED_GIB: an entry's KV bytes live in pinned memory (PinnedPool) when the pool can hold ALL of them, else
// in one pageable buffer as before. In a pool an entry is a list of segments (HostKv, host_kv.h), each inside one slab
// and a multiple of the page size long (so no page straddles two segments): entries larger than a slab are pinned
// too. An entry is fully pinned or fully pageable, never a mix. Slabs are pinned only while idle (spill_grow_pinned).
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
//      first in spill_write (so export and the RAM cap), spill_erase (so drop, subsume, the RAM cap's victims),
//      spill_promote, and for the idle backup before spill_backup_one returns. So an idle backup (live entry, its
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
//      then ONE thread per entry (DiskRead, host_kv.h) only fread()s the file into those buffers. It touches nothing
//      else (no spill_, ckpts_, pool_, PinnedPool bookkeeping, CUDA), and the engine thread does not touch the buffers
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
//  (D) The entry's host memory is read by that copy after the entry's buffers are parked in retired_ with their own
//      event (async = true): spill_reap frees such an item only once cudaEventQuery says it completed (never blocks
//      on it), since s_ did not wait for it.
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
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <new>
#include <sstream>
#include <system_error>
#include <tuple>

#include "model.h"

namespace q27 {

using namespace dims;

namespace {

// (the file format, kIoChunk and read_all: host_kv.h, shared with the background reader DiskRead)
constexpr const char (&kMagic)[8] = kSpillMagic;
constexpr uint32_t kVersion = kSpillVersion;

bool write_all(std::FILE* f, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    const size_t k = std::min(n, kIoChunk);
    if (std::fwrite(c, 1, k, f) != k) return false;
    c += k, n -= k;
  }
  return true;
}

struct File {
  std::FILE* f = nullptr;
  File(const std::string& path, const char* mode) : f(std::fopen(path.c_str(), mode)) {}
  ~File() {
    if (f) std::fclose(f);
  }
  bool close() {  // (fclose reports a write that failed late, e.g. a full disk)
    const bool ok = std::fclose(f) == 0;
    f = nullptr;
    return ok;
  }
};

uint64_t fnv(uint64_t h, const void* p, size_t n) {
  const unsigned char* c = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; i++) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}

// Short checkpoints (a system prompt and tools) are shared by new conversations and cheap to keep, so they are
// never subsumed by a deeper one (only by an equal one).
constexpr int kKeepShort = 16384;

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

void Model::spill_wait(Spilled& s) {
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

void Model::spill_reap() {
  std::vector<Retired> keep;
  for (Retired& r : retired_) {
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
    spill_event_put(r.done);
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
  for (auto it = spill_.begin(); it != spill_.end();) {
    auto next = std::next(it);
    const Spilled& e = it->second;
    if (!e.busy && covers(c.ids, c.ring, e.ids, e.ring) && (e.pos() > kKeepShort || e.ids.size() == c.ids.size())) {
      spill_erase(it, true);
      sstats_.subsumed++;
    }
    it = next;
  }

  if (!spill_copy(c, false)) return;  // (dropped, as without the tiered cache)
  sstats_.spilled++;
  spill_enforce_ram();
}

uint64_t Model::spill_copy(const Checkpoint& c, bool live) {
  spill_reap();
  Spilled s;
  s.kv_bytes = c.pages.size() * size_t(kKvSlices) * kv_page_bytes_;
  s.state_bytes = ckpt_bytes_;
  try {
    s.kv = spill_alloc_kv(s.kv_bytes);
    s.state.reset(new char[s.state_bytes]);
  } catch (const std::bad_alloc&) {
    sstats_.failed++;
    return 0;
  }
  uint64_t id = 0;
  if (!spill_async()) {
    copy_pages(c.pages, s.kv, true);
    Q27_CUDA(cudaStreamSynchronize(cs_));  // (the state's copy into its host slot may still be on its way)
    std::memcpy(s.state.get(), slot_mem_[c.slot], ckpt_bytes_);
    id = spill_add(c, std::move(s), live);
  } else {
    // From the first queued copy to the emplace, a throw must not free kv while the copy may still write it (s's
    // destructor would release it into the pool or delete[] it): wait for io_s_ first, and give the event back.
    try {
      io_begin();  // (1)
      copy_pages_on(c.pages, s.kv, true, io_s_);
      s.done = spill_event();
      Q27_CUDA(cudaEventRecord(s.done, io_s_));
      s.inflight = true;
      // (2) An eviction frees these pages right after this returns: nothing queued on s_ from now on may run before the
      // copy has read them. (A live backup keeps its pages, but is waited for below instead.)
      if (!live) Q27_CUDA(cudaStreamWaitEvent(s_, s.done));
      Q27_CUDA(cudaStreamSynchronize(cs_));  // (4) the state, meanwhile, as before
      std::memcpy(s.state.get(), slot_mem_[c.slot], ckpt_bytes_);
      if (live) spill_wait(s);  // (3) an idle backup completes before it is visible to evict_checkpoint / reserve_area
      id = spill_add(c, std::move(s), live);
    } catch (...) {
      cudaStreamSynchronize(io_s_);  // (the copy has finished reading / writing now; errors are not reported here)
      cudaGetLastError();
      spill_event_put(s.done);  // (null if it was never created; complete after the synchronize)
      s.done = nullptr, s.inflight = false;
      throw;  // (s, and with it kv, is destroyed during the unwinding)
    }
  }
  return id;
}

// Adds the finished copy as an entry. The only throwing step (the map node) comes first, so a failure leaves
// spill_ram_ and the counters untouched.
uint64_t Model::spill_add(const Checkpoint& c, Spilled&& s, bool live) {
  s.ids = c.ids;
  s.ring = c.ring;
  s.last_use = c.last_use;
  s.live = live;
  const size_t ram = s.ram_bytes();
  const uint64_t id = spill_next_;
  spill_.emplace(id, std::move(s));
  spill_next_++;
  spill_ram_ += ram;
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
  const size_t kv_bytes = pick->pages.size() * size_t(kKvSlices) * kv_page_bytes_;
  // A pinned copy is already counted (its bytes are the pool's, spill_host_use); only its state adds. A pageable copy
  // adds its KV too. (Fragmentation may force a pageable copy: then this may briefly overshoot, and the next enforce
  // corrects it. No loop: backup_one never calls enforce.)
  const size_t pool_free = pin_pool_.total() - pin_pool_.used();
  const size_t need = ckpt_bytes_ + (pool_free >= kv_bytes ? 0 : kv_bytes);
  if (spill_host_use() + need > cap) return 0;
  // (c): as spill_checkpoint, the shallower non-busy entries that the copy covers go (live ones too: their
  // checkpoints are in the pool, so nothing is lost).
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

bool Model::spill_promote(uint64_t id) {
  auto it = spill_.find(id);
  if (it == spill_.end() || it->second.live) return false;  // (a live backup is never restored: its original is here)
  if (it->second.busy) return false;  // (A) a background restore owns it
  spill_reap();
  Spilled& s = it->second;
  s.busy = true;
  spill_wait(s);  // (3) an entry evicted moments ago may still be on its way to host memory
  const bool from_disk = !s.in_ram();
  auto give_up = [&](bool forget) {
    s.busy = false;
    if (forget) {
      spill_erase(it, true);
      sstats_.failed++;
    } else if (from_disk) {  // (back to disk only)
      spill_ram_ -= s.ram_bytes();
      s.kv.reset(), s.state.reset();
    }
    return false;
  };
  if (from_disk && !spill_load(s)) return give_up(true);
  const int npages = int(ceil_div(s.pos(), kPageTokens));
  if (size_t(npages) * kKvSlices * kv_page_bytes_ != s.kv_bytes || s.state_bytes != ckpt_bytes_) return give_up(true);

  // Feasibility first (spill_room): nothing is evicted when it cannot work.
  if (!spill_room(npages)) return give_up(false);

  int slot = -1;
  std::vector<int32_t> pages;
  if (!spill_acquire(npages, &slot, &pages)) return give_up(false);
  if (!spill_async()) {
    copy_pages(pages, s.kv, false);
  } else {
    // The pages may have been freed by a sequence whose kernels are still queued, or by an eviction whose D2H is
    // queued on io_s_ (ordered by the stream): io_begin covers the first, the single I/O stream the second.
    io_begin();  // (1)
    copy_pages_on(pages, s.kv, false, io_s_);
    if (!s.done) s.done = spill_event();
    Q27_CUDA(cudaEventRecord(s.done, io_s_));
    s.inflight = true;
    Q27_CUDA(cudaStreamWaitEvent(s_, s.done));  // (2) the resumed sequence's kernels wait on the GPU, not the CPU
  }
  Q27_CUDA(cudaStreamSynchronize(cs_));
  std::memcpy(slot_mem_[slot], s.state.get(), ckpt_bytes_);
  Checkpoint ck;
  ck.ids = std::move(s.ids);
  ck.pages = std::move(pages);  // (one reference each, from alloc)
  ck.slot = slot;
  ck.ring = s.ring;
  ck.last_use = ++use_clock_;
  ckpts_.push_back(std::move(ck));
  if (s.inflight) {  // (3) the H2D copy still reads kv: park it until the copy is done (state was copied above)
    spill_ram_ -= s.ram_bytes();
    retired_.push_back({std::move(s.kv), s.done});
    s.done = nullptr, s.inflight = false;
  }
  spill_erase(it, true);  // (the pool's copy is the one now; it is spilled again when the pool gives it up)
  sstats_.restored++;
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
  if (size_t(npages) * kKvSlices * kv_page_bytes_ != s.kv_bytes || s.state_bytes != ckpt_bytes_) {  // (damaged: as promote)
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
    const int res = spill_restore_issue(id);
    if (res == 0) {
      restoring_.erase(id);
      s.busy = false;
      log("tiered cache: restore of entry %llu refused: no state slot or pages could be taken", (unsigned long long)id);
      return 0;
    }
    if (res == 2) restore_results_.emplace_back(id, true);  // (complete already)
    return 1;
  }
  // Only on disk: the buffers here, the fread on a thread (B).
  auto d = std::make_unique<DiskRead>();
  try {
    s.state.reset(new char[s.state_bytes]);
    s.kv = spill_alloc_kv(s.kv_bytes);
    d->path = s.path;
    d->fingerprint = spill_fingerprint();
    d->state_bytes = s.state_bytes, d->kv_bytes = s.kv_bytes;
    d->ids = s.ids;
    d->state = s.state.get();
    for (size_t i = 0; i < s.kv.seg.size(); i++) d->segs.emplace_back(s.kv.seg[i].get(), s.kv.len[i]);
    d->start();  // (the last step that can throw: if it did, no thread runs)
  } catch (const std::exception&) {  // (bad_alloc, or no thread: the entry stays on disk)
    s.kv.reset(), s.state.reset();
    restoring_.erase(id);
    log("tiered cache: restore of entry %llu refused: the file read could not start", (unsigned long long)id);
    return 0;
  }
  // From here nothing throws: the thread is owned by the registered restore (its destructor joins).
  spill_ram_ += s.ram_bytes();
  s.busy = true;
  restoring_.at(id).disk = std::move(d);
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
  // A pageable entry is read by the driver while staging: an earlier D2H into it must have completed first. A pinned
  // one is ordered by io_s_ itself.
  const bool overlap = spill_async() && s.kv.pinned();
  if (!overlap) spill_wait(s);
  cudaEvent_t ev = nullptr;
  if (!spill_async()) {
    copy_pages(pages, s.kv, false);
  } else {
    io_begin();  // (1)
    copy_pages_on(pages, s.kv, false, io_s_);
    if (!s.done) s.done = spill_event();
    Q27_CUDA(cudaEventRecord(s.done, io_s_));
    s.inflight = true;
    ev = spill_event();
    Q27_CUDA(cudaEventRecord(ev, io_s_));
    // (C) no cudaStreamWaitEvent(s_, ...): s_ goes on; the checkpoint is invisible until ev completes.
  }
  Q27_CUDA(cudaStreamSynchronize(cs_));
  std::memcpy(slot_mem_[slot], s.state.get(), ckpt_bytes_);
  Checkpoint ck;
  ck.ids = std::move(ids);  // (the busy entry is still looked at by spill_best / spill_subsumed: it was copied)
  ck.pages = std::move(pages);
  ck.slot = slot;
  ck.ring = s.ring;
  ck.last_use = ++use_clock_;
  ck.load_id = id;
  if (ev) ck.loading = true, ck.load_ev = ev;
  else ck.held = !restoring_.at(id).orphan;  // (F) complete already
  ckpts_.push_back(std::move(ck));  // (capacity reserved above: no throw)
  if (!ev) {
    spill_restore_finish(id);
    return 2;
  }
  spill_ram_ -= s.ram_bytes();  // (D) the copy still reads kv: parked until it is complete
  retired_.push_back({std::move(s.kv), s.done, true});
  s.done = nullptr, s.inflight = false;
  s.state.reset();
  restoring_.at(id).issued = true;
  return 1;
}

// A restore ended well: counters, and the entry goes (its file too; the pool's copy is the one now; it is spilled again
// when the pool gives it up).
void Model::spill_restore_finish(uint64_t id) {
  if (auto r = restoring_.find(id); r != restoring_.end()) {
    sstats_.restore_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - r->second.t0).count();
    sstats_.restore_bytes += int64_t(r->second.bytes);
    restoring_.erase(r);
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
    r.disk.reset();
    auto it = spill_.find(id);
    if (!ok) {
      log("tiered cache: the file of a spilled checkpoint could not be read (its prompt is computed again)");
      if (it != spill_.end()) spill_erase(it, true);
      restoring_.erase(id);
      sstats_.failed++;
      sstats_.restore_failed++;
      restore_results_.emplace_back(id, false);
      continue;
    }
    const int res = spill_restore_issue(id);
    if (res == 0) {  // (the pool filled up meanwhile: back to disk only, the next prefix_begin may still try)
      Spilled& s = it->second;
      spill_ram_ -= s.ram_bytes();
      s.kv.reset(), s.state.reset();
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

void Model::spill_enforce_ram() {
  const size_t cap = size_t(opt_.spill_ram_gib * double(size_t(1) << 30));
  while (spill_host_use() > cap) {
    // An idle backup goes first: its checkpoint is still in the pool, so dropping it loses nothing, and it is never
    // written to the SSD (it is not the only copy).
    auto victim = spill_.end();
    for (auto it = spill_.begin(); it != spill_.end(); ++it)
      if (it->second.in_ram() && !it->second.busy && it->second.live &&
          (victim == spill_.end() || it->second.last_use < victim->second.last_use))
        victim = it;
    // (A victim may be an entry whose D2H copy is still in flight: spill_erase / spill_write wait for it, (3).)
    if (victim != spill_.end()) {
      spill_erase(victim, true);
      sstats_.dropped++;
      continue;
    }
    // Only sole copies are left: written to the directory, or dropped without one. Pageable ones first: evicting a
    // pinned entry frees only its state bytes from host use (its KV stays in the pool, counted whole).
    for (int pass = 0; pass < 2 && victim == spill_.end(); pass++)
      for (auto it = spill_.begin(); it != spill_.end(); ++it)
        if (it->second.in_ram() && !it->second.busy && !it->second.live && it->second.kv.pinned() == (pass == 1) &&
            (victim == spill_.end() || it->second.last_use < victim->second.last_use))
          victim = it;
    if (victim == spill_.end()) return;
    if (!opt_.spill_dir.empty()) {
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();
      const std::string path =
          opt_.spill_dir + "/" + std::to_string(ns) + "-" + std::to_string(victim->first) + ".q27kv";
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

void Model::spill_write(Spilled& s, const std::string& path) {
  Q27_CHECK(s.in_ram(), "tiered cache: the entry is not in memory");
  spill_wait(s);  // (3) its D2H copy must be complete before the bytes are written
  const std::string tmp = path + ".tmp";
  SpillHeader h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.version = kVersion;
  h.kv_slices = kKvSlices;
  h.fingerprint = spill_fingerprint();
  h.kv_page_bytes = kv_page_bytes_;
  h.state_bytes = s.state_bytes;
  h.page_tokens = kPageTokens;
  h.ring = s.ring ? 1 : 0;
  h.n_ids = uint32_t(s.ids.size());
  h.kv_bytes = s.kv_bytes;
  bool ok;
  {
    File f(tmp, "wb");
    ok = f.f && write_all(f.f, &h, sizeof h) && write_all(f.f, s.ids.data(), s.ids.size() * sizeof(int32_t)) &&
         write_all(f.f, s.state.get(), s.state_bytes);
    for (size_t i = 0; ok && i < s.kv.seg.size(); i++) ok = write_all(f.f, s.kv.seg[i].get(), s.kv.len[i]);  // (the linear stream)
    ok = f.f && f.close() && ok;
  }
  if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    throw std::runtime_error("could not write " + path);
  }
  if (!s.path.empty() && s.path != path) std::remove(s.path.c_str());
  s.path = path;
  spill_ram_ -= s.ram_bytes();
  s.kv.reset(), s.state.reset();
  sstats_.written++;
}

bool Model::spill_load(Spilled& s) {
  File f(s.path, "rb");
  SpillHeader h{};
  if (!f.f || !read_all(f.f, &h, sizeof h) || std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 ||
      h.version != kVersion || h.fingerprint != spill_fingerprint() || h.n_ids != s.ids.size() ||
      h.state_bytes != ckpt_bytes_ || h.kv_bytes != s.kv_bytes)
    return false;  // (checked before anything is allocated)
  std::vector<int32_t> ids(h.n_ids);
  if (!read_all(f.f, ids.data(), ids.size() * sizeof(int32_t)) || ids != s.ids) return false;
  try {
    spill_reap();
    s.state.reset(new char[h.state_bytes]);
    s.kv = spill_alloc_kv(h.kv_bytes);
  } catch (const std::bad_alloc&) {
    s.kv.reset(), s.state.reset();
    return false;
  }
  s.state_bytes = h.state_bytes, s.kv_bytes = h.kv_bytes;
  bool ok = read_all(f.f, s.state.get(), s.state_bytes);
  for (size_t i = 0; ok && i < s.kv.seg.size(); i++) ok = read_all(f.f, s.kv.seg[i].get(), s.kv.len[i]);
  if (!ok) {
    s.kv.reset(), s.state.reset();
    return false;
  }
  spill_ram_ += s.ram_bytes();
  return true;
}

void Model::spill_erase(std::map<uint64_t, Spilled>::iterator it, bool remove_file) {
  spill_wait(it->second);  // (3) the buffers are freed below: an in-flight D2H must have finished writing them
  spill_event_put(it->second.done);
  spill_ram_ -= it->second.ram_bytes();
  if (remove_file && !it->second.path.empty()) std::remove(it->second.path.c_str());
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

uint64_t Model::spill_import(const std::string& path) {
  if (!spill_on()) return 0;
  for (const auto& kv : spill_)
    if (kv.second.path == path) return 0;  // (already indexed)
  File f(path, "rb");
  SpillHeader h{};
  if (!f.f || !read_all(f.f, &h, sizeof h)) return 0;
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.version != kVersion || h.kv_slices != uint32_t(kKvSlices) ||
      h.fingerprint != spill_fingerprint() || h.kv_page_bytes != kv_page_bytes_ || h.state_bytes != ckpt_bytes_ ||
      h.page_tokens != uint32_t(kPageTokens) || h.n_ids == 0 || h.n_ids > uint32_t(opt_.max_tokens) + 64 ||
      h.kv_bytes != size_t(ceil_div(h.n_ids, kPageTokens)) * kKvSlices * kv_page_bytes_)
    return 0;
  // The file is exactly its header, the tokens, the state and the pages (a truncated or padded file is rejected).
  std::error_code ec;
  const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
  if (ec || file_bytes != sizeof h + uintmax_t(h.n_ids) * sizeof(int32_t) + uintmax_t(h.state_bytes) +
                              uintmax_t(h.kv_bytes))
    return 0;
  Spilled s;
  s.ids.resize(h.n_ids);
  if (!read_all(f.f, s.ids.data(), s.ids.size() * sizeof(int32_t))) return 0;
  s.kv_bytes = h.kv_bytes;
  s.state_bytes = h.state_bytes;
  s.ring = h.ring != 0;
  s.path = path;
  s.last_use = 0;  // (older than anything this run used)
  const uint64_t id = spill_next_++;
  spill_.emplace(id, std::move(s));
  return id;
}

void Model::spill_stats(int64_t* out, int n) const {
  int64_t v[kSpillStats], ram = 0, disk = 0;
  int64_t* const dst = out;
  out = v;
  for (const auto& kv : spill_) {
    ram += kv.second.in_ram();
    disk += !kv.second.path.empty();
  }
  out[0] = int64_t(spill_.size()), out[1] = ram, out[2] = disk, out[3] = int64_t(spill_ram_);
  out[4] = sstats_.spilled, out[5] = sstats_.restored, out[6] = sstats_.written, out[7] = sstats_.dropped;
  out[8] = sstats_.subsumed, out[9] = sstats_.failed, out[10] = sstats_.backed_up;
  out[11] = int64_t(pin_pool_.total()), out[12] = int64_t(pin_pool_.used());
  out[13] = sstats_.pin_hits, out[14] = sstats_.pin_misses;
  out[15] = int64_t(restoring_.size()), out[16] = sstats_.restored_async, out[17] = sstats_.restore_failed;
  out[18] = sstats_.restore_ns / 1000000, out[19] = sstats_.restore_bytes;
  for (int i = 0; i < n && i < kSpillStats; i++) dst[i] = v[i];  // (an older caller's shorter buffer is never overrun)
}

}  // namespace q27
