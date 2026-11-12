// Prefix cache bookkeeping (host side, docs/architecture.md §9): the KV page pool and the state checkpoints.
//
// A GDN layer can only resume from a saved state, so the cache unit is a checkpoint: the token prefix
// [0, pos), the KV pages covering it and a copy of the recurrent state (GDN + conv + the draft's
// context ring) in a pinned host slot. KV pages past the deepest checkpoint of a prefix are useless
// (the layers have to run over those tokens again anyway), so pages are kept only through the
// checkpoints that reference them. Checkpoints of one conversation share their full pages by
// reference count; the last, partial page is copied before the resumed sequence writes into it.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "chunk_store.h"
#include "host_kv.h"

namespace q27 {

class PagePool {
 public:
  // Pages 1 .. pages - 1 are handed out; page 0 is the scratch page of unmapped page table entries. The
  // lowest free page goes first, so the parking areas at the top of the pool stay free while there is room
  // (P7-13); high (Q27_PAGES_HIGH, tests): the highest, so parking moves pages.
  void init(int pages, bool high = false) {
    ref_.assign(pages, 0);
    ref_[0] = 1;
    free_.clear();
    for (int p = 1; p < pages; p++) free_.insert(free_.end(), p);
    high_ = high;
  }
  int pages() const { return int(ref_.size()); }
  int free_pages() const { return int(free_.size()); }
  int refs(int p) const { return ref_[p]; }
  // -1 when the pool is empty.
  int alloc() { return alloc_outside(0, 0); }
  // A free page outside [lo, hi); -1 when there is none.
  int alloc_outside(int lo, int hi) {
    int p = -1;
    if (!high_) {
      auto it = free_.begin();
      if (it != free_.end() && *it >= lo && *it < hi) it = free_.lower_bound(hi);
      if (it != free_.end()) p = *it;
    } else {
      auto it = free_.lower_bound(hi);
      if (it != free_.end()) p = *free_.rbegin();
      else if ((it = free_.lower_bound(lo)) != free_.begin()) p = *std::prev(it);
    }
    if (p < 0) return -1;
    free_.erase(p);
    ref_[p] = 1;
    return p;
  }
  // Free pages outside [lo, hi).
  int free_outside(int lo, int hi) const {
    return int(free_.size()) - int(std::distance(free_.lower_bound(lo), free_.lower_bound(hi)));
  }
  // Takes the free page p (a parking area's); false when it is in use.
  bool take(int p) {
    if (!free_.erase(p)) return false;
    ref_[p] = 1;
    return true;
  }
  // The references of page p move to page q (from alloc_outside); p is free then.
  void move(int p, int q) {
    ref_[q] = ref_[p];
    ref_[p] = 0;
    free_.insert(p);
  }
  void ref(int p) { ref_[p]++; }
  void unref(int p) {
    if (--ref_[p] == 0) free_.insert(p);
  }

 private:
  std::vector<int32_t> ref_;
  std::set<int32_t> free_;
  bool high_ = false;
};

struct Checkpoint {
  std::vector<int32_t> ids;    // the tokens [0, pos)
  std::vector<int32_t> pages;  // physical KV pages of [0, pos) (one reference each)
  int slot = -1;               // host state slot
  bool ring = false;           // the draft's context ring is valid for [pos - 2048, pos)
  uint64_t last_use = 0;
  uint64_t backup = 0;         // the spill_ id of its host copy while the pool keeps it (idle backup); 0: none
  // SPILL_ASYNC_PROMOTE: a checkpoint whose pages are still being filled by a host -> device copy on the I/O stream
  // (Model::spill_restore_begin). load_ev is recorded behind that copy and load_id is the spill_ entry it comes from.
  // INVARIANTS (docs/tiered-cache.md; enforced in model.cc / spill.cc, see the comment at the top of spill.cc):
  // a loading checkpoint is invisible to every lookup / dedupe / backup / room count, never evicted, and any page
  // move or copy that reads its pages first makes s_ wait for load_ev. Poll clears `loading` once load_ev completed.
  bool loading = false;
  cudaEvent_t load_ev = nullptr;
  uint64_t load_id = 0;       // the spill_ entry it was restored from (kept after the copy completes: spill_release finds it)
  // Set when a restore completes: the request that waited for it has not run prefix_begin yet, so evict_checkpoint takes
  // any other checkpoint first (and cache_stats does not count its pages as room). Cleared by prefix_begin using it,
  // by spill_release, or implicitly when only held / loading checkpoints remain (the oldest held one is then evicted).
  bool held = false;
  int pos() const { return int(ids.size()); }
};

// Tiered cache (SPILL_RAM_GIB > 0): a checkpoint the KV pool had to give up is copied to host memory instead of
// being dropped, and later written to a file (the frontend's policy, frontend/tiercache.py, or the RAM cap). Its
// tokens stay in memory for the lookup; prefix_begin brings a spilled checkpoint back into the pool when it is
// the deepest match. The KV bytes are split (RAM dedup, chunk_store.h, spill.cc invariants (H)-(K)): the full
// kChunkPages-page blocks of the prefix are shared, content-addressed SpillChunks listed in `chunks` (one reference
// each), and `kv` holds only the rest: pages [chunks.size() * kChunkPages, pages), slice-major over those pages
// ([kKvSlices][tail pages][kv_page_bytes]). An entry with NO chunks ("private": loaded from a file, imported, or
// SPILL_DEDUP=0) keeps all its pages in kv, which is then the old whole-entry layout. kv_bytes is always the LOGICAL
// total (the file's KV payload); state the host slot's bytes (ssm | conv | draft ring), never shared.
using SpillChunk = Chunk<cudaEvent_t>;
using SpillChunkStore = ChunkStore<cudaEvent_t>;

struct Spilled {
  std::vector<int32_t> ids;
  std::vector<SpillChunk*> chunks;    // shared full chunks, in order (empty once the entry is only on disk)
  HostKv kv;                          // the tail (all pages when private); empty once only on disk, or when the tail is 0 pages
  std::unique_ptr<char[]> state;      // (null once the entry is only on disk: this is what in_ram() means)
  size_t kv_bytes = 0, state_bytes = 0;
  bool ring = false;
  std::string path;     // the file (empty: in memory only)
  uint64_t last_use = 0;
  bool busy = false;    // being restored: neither dropped nor written out meanwhile
  bool live = false;    // an idle backup whose checkpoint is still in the pool (never restored while live)
  // SPILL_ASYNC_IO: the device -> host copy that fills kv (or the host -> device copy that reads it) is queued on the
  // I/O stream and `done` was recorded behind it. While inflight the CPU must not touch kv / state / free them:
  // Model::spill_wait first (cudaEventSynchronize; it also waits for the entry's chunks). Never set with SPILL_ASYNC_IO=0.
  cudaEvent_t done = nullptr;
  bool inflight = false;
  int pos() const { return int(ids.size()); }
  bool in_ram() const { return bool(state); }
  size_t ram_bytes() const { return in_ram() ? kv_bytes + state_bytes : 0; }  // logical (the file's size); spill_list column 3
  size_t own_bytes() const { return in_ram() ? kv.total + state_bytes : 0; }  // physical, without the shared chunks
};

struct SpillStats {
  int64_t spilled = 0, restored = 0, written = 0, dropped = 0, subsumed = 0, failed = 0, backed_up = 0;
  int64_t chunk_hits = 0, chunk_misses = 0;  // RAM dedup: full chunks of a spilled entry found resident / copied
  int64_t pin_hits = 0, pin_misses = 0;  // entries given pinned memory / pageable memory although SPILL_PINNED_GIB > 0
  // SPILL_ASYNC_PROMOTE: restores finished (also counted in `restored`), failed ones (also in `failed` when the entry
  // was dropped), and the wall time / bytes of the finished ones (begin -> visible; bytes = kv + state).
  int64_t restored_async = 0, restore_failed = 0, restore_ns = 0, restore_bytes = 0;
};

struct CacheStats {
  int64_t lookups = 0, hits = 0, prompt_tokens = 0, cached_tokens = 0, saved = 0, evicted = 0;
};

}  // namespace q27
