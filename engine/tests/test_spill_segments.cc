// Host-only test of the segmented spill entries (src/host_kv.h): no CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Isrc tests/test_spill_segments.cc -o /tmp/test_spill_segments && /tmp/test_spill_segments
// Calls the same copy_runs template as Model::copy_pages_on, with memcpy standing in for cudaMemcpyAsync, on a fake device.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>

#include "host_kv.h"

using namespace q27;

static int fails = 0, n_pinned = 0, n_multi = 0, n_pageable = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
      if (++fails > 20) std::exit(1);                                          \
    }                                                                          \
  } while (0)

constexpr int kSlices = 32;

struct Dev {  // the KV pool: [slice][phys page][page bytes]
  int phys;
  size_t P;
  std::vector<char> mem;
  Dev(int phys_, size_t P_) : phys(phys_), P(P_), mem(size_t(kSlices) * phys_ * P_) {}
  char* page(int p, int sl) { return mem.data() + (size_t(sl) * phys + p) * P; }
};

static void copy_pages(Dev& d, const std::vector<int32_t>& pages, const HostKv& host, bool to_host, int block, long* pieces) {
  CHECK(host.total == pages.size() * kSlices * d.P);
  copy_runs(pages, kSlices, d.P, block, [&](int p, int sl) { return d.page(p, sl); }, host,
            [&](char* h, char* dev, size_t bytes) {
              CHECK(bytes % d.P == 0);  // a page never straddles two segments
              ++*pieces;
              if (to_host) std::memcpy(h, dev, bytes);
              else std::memcpy(dev, h, bytes);
            });
}

// A pool of `nslab` slabs of `slab` bytes (malloc'ed, 4096-aligned), partly occupied by random other regions.
struct Pool {
  PinnedPool pool;
  std::vector<char*> bases;
  std::vector<KvBuf> held;
  Pool(std::mt19937& rng, int nslab, size_t slab, int holes) {
    for (int i = 0; i < nslab; i++) {
      char* b = static_cast<char*>(std::aligned_alloc(4096, slab));
      bases.push_back(b);
      pool.add_slab(b, slab);
    }
    for (int i = 0; i < holes; i++) {  // fragment: take many small regions, keep a random half
      size_t len = (rng() % 8 + 1) * 4096;
      if (char* p = pool.alloc(len)) {
        if (rng() % 2) held.emplace_back(p, KvDel{&pool, len});
        else pool.release(p, len);
      }
    }
  }
  ~Pool() {
    held.clear();
    for (char* b : bases) std::free(b);
  }
};

static void one_round(std::mt19937& rng, int round) {
  const size_t P_choices[] = {4096, 5000, 4096 * 2, 4104, 3 * 4096 + 17 * 8};
  const size_t P = P_choices[rng() % 5];
  const int block = 1 + int(rng() % 8);
  const int phys = 24 + int(rng() % 16);
  // random page list: runs of consecutive pages (some crossing a block boundary) and gaps, no duplicates
  std::vector<int32_t> pages;
  std::set<int32_t> used;
  for (int p = int(rng() % 4); p < phys && pages.size() < 20;) {
    const int run = 1 + int(rng() % 9);
    for (int k = 0; k < run && p < phys; k++, p++) pages.push_back(p), used.insert(p);
    p += int(rng() % 6);  // gap (possibly 0: a run broken only by the block boundary or contiguous)
  }
  if (rng() % 4 == 0) std::shuffle(pages.begin(), pages.end(), rng);  // out of order: runs of 1
  const size_t n = pages.size(), total = n * kSlices * P;

  Dev dev(phys, P);
  for (auto& c : dev.mem) c = char(rng());

  // the unsegmented layout: the linear stream, page (sl, i) at (sl * n + i) * P
  std::vector<char> ref(total);
  for (int sl = 0; sl < kSlices; sl++)
    for (size_t i = 0; i < n; i++) std::memcpy(&ref[(size_t(sl) * n + i) * P], dev.page(pages[i], sl), P);

  // a pool whose slabs hold a few segments, fragmented; sometimes too small (-> must fail cleanly)
  const size_t slab = (rng() % 6 + 1) * (1u << 20) / 4;  // 256 KiB .. 1.5 MiB, multiples of 4096
  const int nslab = 1 + int(rng() % 12);
  Pool pp(rng, nslab, slab, int(rng() % 40));
  const size_t used_before = pp.pool.used();
  HostKv kv;
  const bool ok = alloc_pinned_kv(pp.pool, total, P, kv);
  if (!ok) {
    n_pageable++;
    CHECK(!kv.in_ram() && pp.pool.used() == used_before);  // all or nothing
    kv.push(KvBuf(new char[total], KvDel{}), total);          // the pageable fallback: ONE segment
  } else {
    size_t sum = 0;
    for (size_t i = 0; i < kv.seg.size(); i++) {
      CHECK(kv.len[i] > 0 && kv.len[i] % P == 0);
      CHECK(kv.len[i] <= slab);
      CHECK((reinterpret_cast<uintptr_t>(kv.seg[i].get()) & 4095) == 0);
      bool inside = false;  // each segment inside one slab
      for (char* b : pp.bases) inside |= kv.seg[i].get() >= b && kv.seg[i].get() + kv.len[i] <= b + slab;
      CHECK(inside);
      sum += kv.len[i];
    }
    CHECK(sum == total && kv.total == total);
    n_pinned++, n_multi += kv.seg.size() > 1;
    // no overlap between the segments
    for (size_t a = 0; a < kv.seg.size(); a++)
      for (size_t b = a + 1; b < kv.seg.size(); b++) {
        char *x = kv.seg[a].get(), *y = kv.seg[b].get();
        CHECK(x + kv.len[a] <= y || y + kv.len[b] <= x);
      }
  }
  if (round < 3) std::printf("round %d: P=%zu n=%zu total=%zu slab=%zu -> %s, %zu segment(s)\n", round, P, n, total, slab,
                             ok ? "pinned" : "pageable", kv.seg.size());

  // device -> host: the concatenation of the segments equals the unsegmented layout
  long pieces = 0;
  copy_pages(dev, pages, kv, true, block, &pieces);
  std::vector<char> cat;
  for (size_t i = 0; i < kv.seg.size(); i++) cat.insert(cat.end(), kv.seg[i].get(), kv.seg[i].get() + kv.len[i]);
  CHECK(cat == ref);

  // seg_at agrees with the concatenation at random offsets, and reports the bytes up to the segment end
  for (int k = 0; k < 50; k++) {
    const size_t off = rng() % total;
    size_t avail = 0;
    char* h = seg_at(kv, off, &avail);
    CHECK(h && avail > 0 && off + avail <= total && std::memcmp(h, &ref[off], avail) == 0);
  }

  // host -> device into other pages (reversed mapping) reproduces the bytes
  std::vector<int32_t> other(pages.rbegin(), pages.rend());
  Dev dev2(phys, P);
  copy_pages(dev2, other, kv, false, block, &pieces);
  // dev2 page other[i] holds what the entry's page i holds, i.e. the original pages[i]
  for (int sl = 0; sl < kSlices; sl++)
    for (size_t i = 0; i < n; i++) CHECK(std::memcmp(dev2.page(other[i], sl), dev.page(pages[i], sl), P) == 0);

  // a file round trip: write the segments in order, read them into a differently segmented entry
  std::vector<char> file;
  for (size_t i = 0; i < kv.seg.size(); i++) file.insert(file.end(), kv.seg[i].get(), kv.seg[i].get() + kv.len[i]);
  HostKv kv2;
  const size_t cut = P * (1 + rng() % 7);
  for (size_t off = 0; off < total; off += cut) {
    const size_t l = std::min(cut, total - off);
    kv2.push(KvBuf(new char[l], KvDel{}), l);
  }
  size_t pos = 0;
  for (size_t i = 0; i < kv2.seg.size(); i++) std::memcpy(kv2.seg[i].get(), &file[pos], kv2.len[i]), pos += kv2.len[i];
  Dev dev3(phys, P);
  copy_pages(dev3, pages, kv2, false, block, &pieces);
  for (int sl = 0; sl < kSlices; sl++)
    for (size_t i = 0; i < n; i++) CHECK(std::memcmp(dev3.page(pages[i], sl), dev.page(pages[i], sl), P) == 0);

  // releasing the entry gives every byte back
  kv.reset();
  CHECK(pp.pool.used() == used_before);
}

static void test_runs() {
  std::vector<int32_t> pages = {0, 1, 2, 4, 5, 7, 8, 9, 10, 11};
  std::vector<std::pair<size_t, size_t>> got;
  for_each_run(pages, 4, [&](size_t i, size_t j) { got.emplace_back(i, j); });
  // block 4: 0-3 | 4-7 | 8-11 -> runs {0,1,2} {4,5} {7} {8,9,10,11}
  const std::vector<std::pair<size_t, size_t>> want = {{0, 2}, {3, 4}, {5, 5}, {6, 9}};
  CHECK(got == want);
}

int main() {
  test_runs();
  std::mt19937 rng(12345);
  for (int r = 0; r < 300; r++) one_round(rng, r);
  if (fails) {
    std::printf("%d check(s) FAILED\n", fails);
    return 1;
  }
  CHECK(n_pinned > 100 && n_multi > 50 && n_pageable > 5);
  std::printf("%d pinned (%d with several segments), %d pageable fallbacks\n", n_pinned, n_multi, n_pageable);
  if (fails) return 1;
  std::printf("OK: 300 random rounds\n");
  return 0;
}
