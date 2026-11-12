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
  int pos() const { return int(ids.size()); }
};

// Tiered cache (SPILL_RAM_GIB > 0): a checkpoint the KV pool had to give up is copied to host memory instead of
// being dropped, and later written to a file (the frontend's policy, frontend/tiercache.py, or the RAM cap). Its
// tokens stay in memory for the lookup; prefix_begin brings a spilled checkpoint back into the pool when it is
// the deepest match. kv holds the pages slice-major ([kKvSlices][pages][kv_page_bytes]), state the host slot's
// bytes (ssm | conv | draft ring).
struct Spilled {
  std::vector<int32_t> ids;
  HostKv kv;                          // empty once the entry is only on disk (pinned segments, or one pageable segment)
  std::unique_ptr<char[]> state;      // (same)
  size_t kv_bytes = 0, state_bytes = 0;
  bool ring = false;
  std::string path;     // the file (empty: in memory only)
  uint64_t last_use = 0;
  bool busy = false;    // being restored: neither dropped nor written out meanwhile
  bool live = false;    // an idle backup whose checkpoint is still in the pool (never restored while live)
  // SPILL_ASYNC_IO: the device -> host copy that fills kv (or the host -> device copy that reads it) is queued on the
  // I/O stream and `done` was recorded behind it. While inflight the CPU must not touch kv / state / free them:
  // Model::spill_wait first (cudaEventSynchronize). Never set with SPILL_ASYNC_IO=0.
  cudaEvent_t done = nullptr;
  bool inflight = false;
  int pos() const { return int(ids.size()); }
  bool in_ram() const { return kv.in_ram(); }
  size_t ram_bytes() const { return in_ram() ? kv_bytes + state_bytes : 0; }
};

struct SpillStats {
  int64_t spilled = 0, restored = 0, written = 0, dropped = 0, subsumed = 0, failed = 0, backed_up = 0;
  int64_t pin_hits = 0, pin_misses = 0;  // entries given pinned memory / pageable memory although SPILL_PINNED_GIB > 0
};

struct CacheStats {
  int64_t lookups = 0, hits = 0, prompt_tokens = 0, cached_tokens = 0, saved = 0, evicted = 0;
};

}  // namespace q27
