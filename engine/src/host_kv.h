// Host-side storage of a spill entry's KV bytes (docs/tiered-cache.md). Pure C++ (no CUDA), so tests/test_spill_segments.cc
// builds it with g++ alone.
//
// An entry's KV bytes are one linear stream, slice-major: the page (slice sl, index i) of an n-page entry is at
// byte (sl * n + i) * page_bytes. In host memory the stream is stored in segments (HostKv): either ONE pageable
// segment (new char[total]), or segments carved out of the PinnedPool, each inside one slab and each a multiple of
// page_bytes long, so no page ever straddles two segments. An entry is fully pinned or fully pageable, never a
// mix. The file format is the same linear stream (spill_write / spill_load go segment by segment).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

namespace q27 {

// Pinned host memory for spill entries (SPILL_PINNED_GIB > 0). Bookkeeping only: the engine pins whole slabs with
// cudaHostAlloc, never on the hot path (Model::spill_grow_pinned, called while idle), and hands them to add_slab.
// Regions are 4 KiB aligned and sized; the extents are first-fit (lowest slab, lowest offset), freed extents are
// coalesced. alloc returns nullptr when no extent is that large. take_largest hands out the front of the largest
// extent (a segment of a big entry). Engine thread only, no locks.
class PinnedPool {
 public:
  static constexpr size_t kAlign = 4096;
  void add_slab(char* base, size_t bytes) {
    slabs_.push_back({base, bytes, {{0, bytes}}});
    total_ += bytes;
  }
  char* alloc(size_t bytes) {
    bytes = round(bytes);
    if (bytes == 0) return nullptr;
    for (Slab& s : slabs_)
      for (auto it = s.free.begin(); it != s.free.end(); ++it)
        if (it->second >= bytes) return carve(s, it, bytes);
    return nullptr;
  }
  // The front of the largest free extent: *len = min(max_len, the extent) rounded down to a multiple of `unit` (so
  // round(*len) still fits the extent: extents are multiples of kAlign). nullptr when no extent holds one unit.
  char* take_largest(size_t max_len, size_t unit, size_t* len) {
    Slab* bs = nullptr;
    std::map<size_t, size_t>::iterator bit;
    for (Slab& s : slabs_)
      for (auto it = s.free.begin(); it != s.free.end(); ++it)
        if (!bs || it->second > bit->second) bs = &s, bit = it;
    if (!bs || unit == 0) return nullptr;
    const size_t l = std::min(max_len, bit->second) / unit * unit;
    if (l == 0) return nullptr;
    *len = l;
    return carve(*bs, bit, round(l));
  }
  // p / bytes: exactly what alloc returned / was asked for (or what take_largest returned / *len).
  void release(char* p, size_t bytes) {
    bytes = round(bytes);
    for (Slab& s : slabs_) {
      if (p < s.base || p >= s.base + s.bytes) continue;
      auto it = s.free.emplace(size_t(p - s.base), bytes).first;
      auto next = std::next(it);
      if (next != s.free.end() && it->first + it->second == next->first) {
        it->second += next->second;
        s.free.erase(next);
      }
      if (it != s.free.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second == it->first) {
          prev->second += it->second;
          s.free.erase(it);
        }
      }
      used_ -= bytes;
      return;
    }
  }
  size_t total() const { return total_; }
  size_t used() const { return used_; }
  std::vector<char*> slab_bases() const {
    std::vector<char*> v;
    for (const Slab& s : slabs_) v.push_back(s.base);
    return v;
  }
  void forget_slabs() { slabs_.clear(), total_ = used_ = 0; }

 private:
  struct Slab {
    char* base;
    size_t bytes;
    std::map<size_t, size_t> free;  // offset -> length of the free extents
  };
  static size_t round(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
  char* carve(Slab& s, std::map<size_t, size_t>::iterator it, size_t bytes) {
    const size_t off = it->first, len = it->second;
    s.free.erase(it);
    if (len > bytes) s.free.emplace(off + bytes, len - bytes);
    used_ += bytes;
    return s.base + off;
  }
  std::vector<Slab> slabs_;
  size_t total_ = 0, used_ = 0;
};

// Owner of one segment: a region of the PinnedPool (pool != nullptr, released back to it) or a new char[]
// (pool == nullptr). The deleter is part of the pointer's state and survives reset(), so a KvBuf is only ever
// (re)filled by assignment, never by reset(new char[...]).
struct KvDel {
  PinnedPool* pool = nullptr;
  size_t bytes = 0;
  void operator()(char* p) const {
    if (!p) return;
    if (pool)
      pool->release(p, bytes);
    else
      delete[] p;
  }
};
using KvBuf = std::unique_ptr<char[], KvDel>;

// The KV bytes of one entry: seg[i] holds len[i] bytes of the linear stream, in order; total == sum(len). Empty
// (no segment): the entry is only on disk. Segments of a pinned entry are multiples of the page size; a pageable
// entry is the single segment [0, total).
struct HostKv {
  std::vector<KvBuf> seg;
  std::vector<size_t> len;
  size_t total = 0;
  bool in_ram() const { return !seg.empty(); }
  bool pinned() const { return !seg.empty() && seg[0].get_deleter().pool != nullptr; }  // (all segments are pool regions)
  void reset() { seg.clear(), len.clear(), total = 0; }  // (each segment goes back to the pool / delete[])
  void push(KvBuf b, size_t n) { seg.push_back(std::move(b)), len.push_back(n), total += n; }
};

// The segment holding linear byte `off` (< total): its address at off, and the bytes from there to the segment's end.
template <class H>
char* seg_at(const H& kv, size_t off, size_t* avail) {
  for (size_t i = 0; i < kv.seg.size(); i++) {
    if (off < kv.len[i]) {
      *avail = kv.len[i] - off;
      return kv.seg[i].get() + off;
    }
    off -= kv.len[i];
  }
  *avail = 0;
  return nullptr;
}

// fn(done, host, bytes) for each piece of the linear range [off, off + bytes), clipped at segment boundaries; done
// is the piece's offset in the range (so the device side of a contiguous run advances by the same amount).
template <class H, class F>
void clip_run(const H& kv, size_t off, size_t bytes, F&& fn) {
  size_t done = 0;
  while (done < bytes) {
    size_t avail = 0;
    char* h = seg_at(kv, off + done, &avail);
    const size_t k = std::min(avail, bytes - done);
    if (!h || k == 0) return;  // (out of range: the caller's bug, caught by its own size check)
    fn(done, h, k);
    done += k;
  }
}

// The runs of pages[0..n): fn(i, j) for each maximal run pages[i..j] of consecutive page numbers within one block
// (pages p, p + 1 of one block are adjacent in each slice, so a run is one device range).
template <class F>
void for_each_run(const std::vector<int32_t>& pages, int block, F&& fn) {
  const size_t n = pages.size();
  for (size_t i = 0; i < n;) {
    size_t j = i;
    while (j + 1 < n && pages[j + 1] == pages[j] + 1 && pages[j + 1] / block == pages[i] / block) j++;
    fn(i, j);
    i = j + 1;
  }
}

// The copies of pages[0..n) between the device KV pool and `host` (linear layout: slice-major, page i of slice sl at
// (sl * n + i) * page_bytes): for each slice and each run of adjacent pages, copy(host_ptr, dev_ptr, bytes) once per
// piece that clip_run cuts at the segment boundaries. dev_ptr(page, slice) -> char*. Model::copy_pages_on issues
// cudaMemcpyAsync from it; the host test issues memcpy.
template <class PtrOf, class Copy>
void copy_runs(const std::vector<int32_t>& pages, int slices, size_t page_bytes, int kv_block, PtrOf dev_ptr,
               const HostKv& host, Copy copy) {
  const size_t n = pages.size();
  for (int sl = 0; sl < slices; sl++)
    // Pages p, p + 1 of one block are adjacent in each slice, so a run of them is one device range; its host side is
    // the linear range [(sl * n + i) * page, ...), which clip_run cuts at the segment boundaries.
    for_each_run(pages, kv_block, [&](size_t i, size_t j) {
      char* dev = dev_ptr(pages[i], sl);
      clip_run(host, (size_t(sl) * n + i) * page_bytes, (j - i + 1) * page_bytes,
               [&](size_t done, char* h, size_t bytes) { copy(h, dev + done, bytes); });
    });
}

// Fills `out` with `total` bytes (a multiple of `page`) from the pool: one region if an extent holds it all,
// else the largest extents one after the other (each segment inside one slab, a multiple of `page`). All or
// nothing: false (out empty, everything released) when the pool cannot supply the whole entry.
inline bool alloc_pinned_kv(PinnedPool& pool, size_t total, size_t page, HostKv& out) {
  out.reset();
  size_t rem = total;
  while (rem) {
    if (char* p = pool.alloc(rem)) {
      out.push(KvBuf(p, KvDel{&pool, rem}), rem);
      break;
    }
    size_t len = 0;
    char* p = pool.take_largest(rem, page, &len);
    if (!p) {
      out.reset();
      return false;
    }
    out.push(KvBuf(p, KvDel{&pool, len}), len);
    rem -= len;
  }
  return true;
}

}  // namespace q27
