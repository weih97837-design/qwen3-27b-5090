// Host-only test of the RAM-tier KV chunk store (src/chunk_store.h): no CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Isrc tests/test_chunk_store.cc -o /tmp/test_chunk_store && /tmp/test_chunk_store
// A SimCache mirrors Model::spill_copy / spill_promote / spill_erase / spill_enforce_ram with memcpy on a fake device
// standing in for cudaMemcpyAsync, using the same header code (chunk_acquire, linear_pieces, copy_runs, pick_victim).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>

#include "chunk_store.h"

using namespace q27;

static int fails = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
      if (++fails > 20) std::exit(1);                                          \
    }                                                                          \
  } while (0)

constexpr int kSlices = 3;
constexpr size_t kP = 4096;          // page bytes
constexpr size_t kChunkTokens = size_t(kChunkPages) * 64;
constexpr size_t kChunkBytes = size_t(kSlices) * kChunkPages * kP;

using Store = ChunkStore<int*>;
using Ch = Store::C;

// ---------------------------------------------------------------- a fake device KV pool

struct Dev {
  int phys;
  std::vector<char> mem;
  explicit Dev(int phys_) : phys(phys_), mem(size_t(kSlices) * phys_ * kP) {}
  char* page(int p, int sl) { return mem.data() + (size_t(sl) * phys + p) * kP; }
};

// The device pages of a sequence `ids` (page i = tokens [64 i, 64 i + 64)): their bytes depend on the whole prefix up
// to the end of the page, as real KV does, so a shared chunk restored for another entry must carry the right bytes.
static void fill_dev(Dev& d, const std::vector<int32_t>& ids, const std::vector<int32_t>& pages) {
  for (size_t i = 0; i < pages.size(); i++) {
    const size_t upto = std::min(ids.size(), 64 * (i + 1));
    Hash128 h;
    h.ids(ids.data(), upto);
    const Key128 k = h.finish();
    for (int sl = 0; sl < kSlices; sl++) {
      std::mt19937_64 rng(k.a ^ (k.b * 31) ^ uint64_t(sl + 1));
      char* p = d.page(pages[i], sl);
      for (size_t b = 0; b < kP; b++) p[b] = char(rng());
    }
  }
}

static std::vector<int32_t> random_pages(std::mt19937& rng, int phys, size_t n) {
  std::vector<int32_t> all(phys);
  for (int i = 0; i < phys; i++) all[i] = i;
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(n);
  if (rng() % 2) std::sort(all.begin(), all.end());  // sometimes runs of adjacent pages
  return all;
}

static std::vector<int32_t> sub(const std::vector<int32_t>& v, size_t from, size_t n) {
  return std::vector<int32_t>(v.begin() + long(from), v.begin() + long(from + n));
}

static void copy_pages(Dev& d, const std::vector<int32_t>& pages, const HostKv& host, bool to_host, int block) {
  CHECK(host.total == pages.size() * kSlices * kP);
  copy_runs(pages, kSlices, kP, block, [&](int p, int sl) { return d.page(p, sl); }, host,
            [&](char* h, char* dev, size_t bytes) {
              CHECK(bytes % kP == 0);
              if (to_host) std::memcpy(h, dev, bytes);
              else std::memcpy(dev, h, bytes);
            });
}

// A pageable HostKv of `bytes`, or a pinned one carved from a (fragmented) PinnedPool when `pool` is given and has room.
static HostKv alloc_kv(PinnedPool* pool, size_t bytes) {
  HostKv kv;
  if (pool && bytes > 0 && alloc_pinned_kv(*pool, bytes, kP, kv)) return kv;
  kv.push(KvBuf(new char[bytes], KvDel{}), bytes);
  return kv;
}

// ---------------------------------------------------------------- the simulated tiered cache

struct Entry {
  std::vector<int32_t> ids;
  std::vector<Ch*> chunks;
  HostKv tail;
  size_t tail_pages = 0, kv_bytes = 0;
  uint64_t last_use = 0;
  bool live = false;
};

struct SimCache {
  Store store;
  PinnedPool* pool = nullptr;
  Key128 seed = chunk_seed(0x1234, kChunkPages);
  std::map<int, Entry> entries;
  size_t own = 0;  // tails (the state bytes are left out)
  long hits = 0, misses = 0;
  int next = 1;
  int block = 4;

  size_t phys() const { return own + store.bytes(); }
  size_t logical() const {
    size_t n = 0;
    for (auto& kv : entries) n += kv.second.kv_bytes;
    return n;
  }

  // Model::spill_copy
  int spill(Dev& dev, const std::vector<int32_t>& ids, const std::vector<int32_t>& pages) {
    Entry e;
    e.ids = ids;
    const size_t nfull = ids.size() / kChunkTokens;  // by TOKENS: a chunk's pages are all full
    CHECK(pages.size() == (ids.size() + 63) / 64);
    e.tail_pages = pages.size() - nfull * kChunkPages;
    e.kv_bytes = pages.size() * kSlices * kP;
    e.tail = alloc_kv(pool, e.tail_pages * kSlices * kP);
    std::vector<Ch*> refd;
    std::vector<std::pair<size_t, Ch*>> fresh;
    chunk_acquire(store, seed, ids.data(), nfull, kChunkTokens, kChunkBytes, [&](size_t b) { return alloc_kv(pool, b); },
                  refd, fresh);
    for (auto& [c, ch] : fresh) copy_pages(dev, sub(pages, c * kChunkPages, kChunkPages), ch->kv, true, block);
    misses += long(fresh.size()), hits += long(refd.size() - fresh.size());
    if (e.tail_pages) copy_pages(dev, sub(pages, nfull * kChunkPages, e.tail_pages), e.tail, true, block);
    e.chunks = std::move(refd);
    own += e.tail.total;
    const int id = next++;
    entries.emplace(id, std::move(e));
    return id;
  }

  void free_chunk(Ch* c) {
    CHECK(c->refs == 0);
    Ch out = store.extract(c);  // (its HostKv is released here)
    (void)out;
  }
  void unref(Ch* c) {
    if (store.unref(c)) free_chunk(c);
  }

  // Model::spill_erase
  void erase(int id) {
    Entry& e = entries.at(id);
    for (Ch* c : e.chunks) unref(c);
    own -= e.tail.total;
    entries.erase(id);
  }

  // Model::spill_promote: the entry's pages into `pages` of dev2; the entry is erased
  void promote(int id, Dev& dev2, const std::vector<int32_t>& pages) {
    Entry& e = entries.at(id);
    size_t sum = e.tail.total;
    for (Ch* c : e.chunks) sum += c->kv.total;
    CHECK(sum == e.kv_bytes);
    CHECK(pages.size() == e.chunks.size() * kChunkPages + e.tail_pages);
    for (size_t c = 0; c < e.chunks.size(); c++) copy_pages(dev2, sub(pages, c * kChunkPages, kChunkPages), e.chunks[c]->kv, false, block);
    if (e.tail_pages) copy_pages(dev2, sub(pages, e.chunks.size() * kChunkPages, e.tail_pages), e.tail, false, block);
    erase(id);
  }

  // The old linear stream of an entry (what spill_write puts in the file)
  std::vector<char> linear(int id) {
    Entry& e = entries.at(id);
    std::vector<char> out;
    linear_pieces(e.chunks, e.tail, kSlices, kP, kChunkPages, e.tail_pages,
                  [&](const char* p, size_t n) { out.insert(out.end(), p, p + n); });
    return out;
  }

  // The host bytes erasing the entry would free (pageable only here: no pinned pieces)
  size_t freed(const Entry& e) const {
    size_t n = e.tail.total;
    for (Ch* c : e.chunks)
      if (c->refs == 1) n += c->kv.total;
    return n;
  }
  // Model::spill_enforce_ram without the disk: orphans first, then the best victim, re-checked each iteration
  int enforce(size_t cap) {
    int evicted = 0;
    while (phys() > cap) {
      auto orph = store.orphans();
      if (!orph.empty()) {
        for (Ch* c : orph) free_chunk(c);
        continue;
      }
      std::vector<VictimCand> cand;
      std::vector<int> ids;
      for (int pass = 0; pass < 2 && cand.empty(); pass++)
        for (auto& kv : entries)
          if (kv.second.live == (pass == 0)) cand.push_back({kv.second.last_use, freed(kv.second)}), ids.push_back(kv.first);
      if (cand.empty()) return evicted;
      erase(ids[size_t(pick_victim(cand))]);
      evicted++;
    }
    return evicted;
  }
};

// ---------------------------------------------------------------- tests

static std::vector<int32_t> make_ids(uint32_t seed, size_t n, const std::vector<int32_t>* prefix = nullptr, size_t keep = 0) {
  std::mt19937 rng(seed);
  std::vector<int32_t> v;
  if (prefix) v.assign(prefix->begin(), prefix->begin() + long(keep));
  while (v.size() < n) v.push_back(int32_t(rng() % 150000));
  return v;
}

static void test_keys() {
  const auto a = make_ids(1, 3 * kChunkTokens + 100);
  auto b = a;
  b[2 * kChunkTokens + 5] ^= 1;  // differs inside chunk 2 only
  const Key128 s = chunk_seed(0x1234, kChunkPages), s2 = chunk_seed(0x1235, kChunkPages), s3 = chunk_seed(0x1234, 32);
  CHECK(!(s == s2) && !(s == s3));
  auto chain = [&](const std::vector<int32_t>& v, Key128 seed) {
    std::vector<Key128> k;
    Key128 prev = seed;
    for (size_t c = 0; c < 3; c++) k.push_back(prev = chunk_next(prev, v.data() + c * kChunkTokens, kChunkTokens));
    return k;
  };
  const auto ka = chain(a, s), kb = chain(b, s), ka2 = chain(a, s), kc = chain(a, s2);
  CHECK(ka[0] == ka2[0] && ka[1] == ka2[1] && ka[2] == ka2[2]);  // determinism
  CHECK(ka[0] == kb[0] && ka[1] == kb[1] && !(ka[2] == kb[2]));                  // prefix sharing, divergence at chunk 2
  CHECK(!(ka[0] == kc[0]));                                                      // another model / kv type: no sharing
  // a later chunk's key depends on the earlier ones: same tokens in chunk 1 after a different chunk 0
  auto d = a;
  d[7] ^= 1;
  const auto kd = chain(d, s);
  CHECK(!(kd[0] == ka[0]) && !(kd[1] == ka[1]) && !(kd[2] == ka[2]));
  // 128 bits: no collisions among many different chunks, and both lanes vary
  std::set<std::pair<uint64_t, uint64_t>> seen;
  std::set<uint64_t> as, bs;
  for (uint32_t i = 0; i < 20000; i++) {
    int32_t x[3] = {int32_t(i), int32_t(i * 7), int32_t(i >> 3)};
    const Key128 k = chunk_next(s, x, 3);
    seen.insert({k.a, k.b}), as.insert(k.a), bs.insert(k.b);
  }
  CHECK(seen.size() == 20000 && as.size() == 20000 && bs.size() == 20000);
}

static void test_collision() {
  Store st;
  const auto x = make_ids(5, 8), y = make_ids(6, 8);
  const Key128 nominal{42, 43};
  Key128 sk;
  CHECK(st.find(nominal, x.data(), 8, &sk) == nullptr && sk == nominal);
  Ch* cx = st.add(sk, x.data(), 8, alloc_kv(nullptr, 100));
  st.ref(cx);
  CHECK(st.find(nominal, x.data(), 8, &sk) == cx && sk == nominal);
  // the same key, other ids (a collision): a miss, and the new chunk goes to the next probe
  CHECK(st.find(nominal, y.data(), 8, &sk) == nullptr && !(sk == nominal));
  Ch* cy = st.add(sk, y.data(), 8, alloc_kv(nullptr, 50));
  st.ref(cy);
  CHECK(cy != cx && st.size() == 2 && st.bytes() == 150);
  Key128 sk2;
  CHECK(st.find(nominal, x.data(), 8, &sk2) == cx);
  CHECK(st.find(nominal, y.data(), 8, &sk2) == cy && sk2 == sk);
  // removing the first leaves a gap: y is then only a miss (never wrong data)
  CHECK(st.unref(cx));
  Ch out = st.extract(cx);
  CHECK(st.bytes() == 50 && st.orphan_count() == 0);
  CHECK(st.find(nominal, y.data(), 8, &sk2) == nullptr);
  CHECK(st.find(nominal, x.data(), 8, &sk2) == nullptr);
  st.unref(cy);
  st.extract(cy);
  CHECK(st.size() == 0 && st.bytes() == 0 && st.orphan_count() == 0);

  // through a whole chain: chunk 0 collides (forced: a chunk with other ids sits at the key of chunk 0), so chunk 1
  // must not hit the chunk 1 of the entry that owns the real chain
  Store s2;
  const Key128 seed = chunk_seed(7, kChunkPages);
  const auto a = make_ids(9, 2 * kChunkTokens);
  std::vector<Ch*> r1, r2, r3;
  std::vector<std::pair<size_t, Ch*>> f1, f2, f3;
  auto alloc = [](size_t b) { return alloc_kv(nullptr, b); };
  chunk_acquire(s2, seed, a.data(), 2, kChunkTokens, 64, alloc, r1, f1);
  CHECK(f1.size() == 2);
  // an impostor registered at the nominal key of chunk 0 of `b` (ids differ from b's)
  auto b = make_ids(10, 2 * kChunkTokens);
  Key128 sk0;
  const Key128 nom0 = chunk_next(seed, b.data(), kChunkTokens);
  CHECK(s2.find(nom0, b.data(), kChunkTokens, &sk0) == nullptr);
  s2.ref(s2.add(nom0, a.data(), kChunkTokens, alloc_kv(nullptr, 64)));  // wrong ids under b's key
  chunk_acquire(s2, seed, b.data(), 2, kChunkTokens, 64, alloc, r2, f2);
  CHECK(f2.size() == 2 && r2[0] != r1[0] && r2[1] != r1[1]);  // both miss
  CHECK(r2[0]->ids == std::vector<int32_t>(b.begin(), b.begin() + long(kChunkTokens)));
  // and b again hits its own chunks (the probe path is deterministic)
  chunk_acquire(s2, seed, b.data(), 2, kChunkTokens, 64, alloc, r3, f3);
  CHECK(f3.empty() && r3 == r2);
}

static void test_refcounts_accounting() {
  std::mt19937 rng(77);
  SimCache sc;
  std::vector<Dev> devs;  // one device pool per agent (their page numbers overlap)
  // system prompt of 2.5 chunks shared by 4 "agents", each with its own tail and a different length
  const auto sys = make_ids(100, size_t(2.5 * kChunkTokens));
  std::vector<int> ids;
  std::vector<std::vector<int32_t>> seqs, pgs;
  for (int a = 0; a < 4; a++) {
    seqs.push_back(make_ids(200 + a, 2 * kChunkTokens + 1000 + size_t(a) * 900 + 7, &sys, sys.size()));
  }
  // (agent 0 has fewer than 3 chunks' worth of tokens; agents 1..3 reach the third)
  for (auto& s : seqs) {
    pgs.push_back(random_pages(rng, 400, (s.size() + 63) / 64));
    devs.emplace_back(400);
    fill_dev(devs.back(), s, pgs.back());
  }
  size_t sum_logical = 0;
  for (size_t a = 0; a < seqs.size(); a++) {
    ids.push_back(sc.spill(devs[a], seqs[a], pgs[a]));
    sum_logical += pgs[a].size() * kSlices * kP;
  }
  CHECK(sc.logical() == sum_logical);
  CHECK(sc.phys() < sc.logical());
  // the first two chunks are shared by all four: 2 chunks stored once; chunk 2 only if the sequence has 3 full chunks
  size_t nchunks = 0;
  for (auto& s : seqs) nchunks += s.size() / kChunkTokens;
  CHECK(sc.hits + sc.misses == long(nchunks));
  CHECK(sc.misses == long(sc.store.size()));
  CHECK(sc.hits > 0);
  CHECK(sc.store.orphan_count() == 0);
  for (Ch* c : sc.entries.at(ids[0]).chunks) CHECK(c->refs >= 1);
  CHECK(sc.entries.at(ids[0]).chunks[0]->refs == 4 && sc.entries.at(ids[3]).chunks[0] == sc.entries.at(ids[0]).chunks[0]);

  // the linear stream of every entry equals what a whole-entry copy_runs gives (the unchanged file format)
  for (size_t a = 0; a < seqs.size(); a++) {
    const size_t n = pgs[a].size();
    HostKv whole = alloc_kv(nullptr, n * kSlices * kP);
    copy_pages(devs[a], pgs[a], whole, true, sc.block);
    std::vector<char> ref(whole.seg[0].get(), whole.seg[0].get() + whole.total);
    CHECK(sc.linear(ids[a]) == ref);
  }

  // promote each into other pages: the bytes equal the originals; refs fall and chunks go at 0 (no orphans left)
  const size_t phys_before = sc.phys();
  for (size_t a = 0; a < seqs.size(); a++) {
    auto other = random_pages(rng, 400, pgs[a].size());
    Dev dev2(400);
    sc.promote(ids[a], dev2, other);
    for (size_t i = 0; i < other.size(); i++)
      for (int sl = 0; sl < kSlices; sl++) CHECK(std::memcmp(dev2.page(other[i], sl), devs[a].page(pgs[a][i], sl), kP) == 0);
    CHECK(sc.phys() <= phys_before);  // promoting never grows the RAM tier
  }
  CHECK(sc.entries.empty() && sc.store.size() == 0 && sc.store.bytes() == 0 && sc.own == 0 && sc.phys() == 0);
}

// An entry whose pos is not a multiple of 4096 keeps its partial last page in the tail; one of exactly k chunks has no tail.
static void test_ragged() {
  std::mt19937 rng(5);
  const size_t lens[] = {10, 64, 4095, kChunkTokens - 1, kChunkTokens, kChunkTokens + 1, 2 * kChunkTokens, 2 * kChunkTokens + 65,
                         3 * kChunkTokens - 10};
  for (size_t len : lens) {
    SimCache sc;
    Dev dev(300);
    const auto ids = make_ids(uint32_t(len), len);
    const auto pages = random_pages(rng, 300, (len + 63) / 64);
    fill_dev(dev, ids, pages);
    const int id = sc.spill(dev, ids, pages);
    const Entry& e = sc.entries.at(id);
    CHECK(e.chunks.size() == len / kChunkTokens);
    CHECK(e.tail_pages == pages.size() - e.chunks.size() * kChunkPages);
    // a partial last page is never inside a chunk
    CHECK(e.chunks.size() * kChunkTokens <= len);
    HostKv whole = alloc_kv(nullptr, pages.size() * kSlices * kP);
    copy_pages(dev, pages, whole, true, sc.block);
    CHECK(sc.linear(id) == std::vector<char>(whole.seg[0].get(), whole.seg[0].get() + whole.total));
    const auto other = random_pages(rng, 300, pages.size());
    Dev dev2(300);
    sc.promote(id, dev2, other);
    for (size_t i = 0; i < other.size(); i++)
      for (int sl = 0; sl < kSlices; sl++) CHECK(std::memcmp(dev2.page(other[i], sl), dev.page(pages[i], sl), kP) == 0);
    CHECK(sc.phys() == 0 && sc.store.size() == 0);
  }
}

// Pinned pools hand out multi-segment, fragmented regions: the per-chunk layout and the linear stream still hold.
static void test_pinned() {
  std::mt19937 rng(11);
  const size_t slab = 3 * (1u << 20);
  std::vector<char*> bases;
  PinnedPool pool;
  for (int i = 0; i < 6; i++) {
    bases.push_back(static_cast<char*>(std::aligned_alloc(4096, slab)));
    pool.add_slab(bases.back(), slab);
  }
  std::vector<KvBuf> noise;  // fragment it
  for (int i = 0; i < 40; i++) {
    const size_t l = (rng() % 8 + 1) * 4096;
    if (char* p = pool.alloc(l)) {
      if (rng() % 2) noise.emplace_back(p, KvDel{&pool, l});
      else pool.release(p, l);
    }
  }
  const size_t used0 = pool.used();
  {
    SimCache sc;
    sc.pool = &pool;
    Dev dev(300), devb(300);
    const auto base = make_ids(1, 2 * kChunkTokens + 500);
    const auto b2 = make_ids(2, 2 * kChunkTokens + 900, &base, 2 * kChunkTokens);
    const auto p1 = random_pages(rng, 300, (base.size() + 63) / 64), p2 = random_pages(rng, 300, (b2.size() + 63) / 64);
    fill_dev(dev, base, p1);
    fill_dev(devb, b2, p2);
    const int i1 = sc.spill(dev, base, p1), i2 = sc.spill(devb, b2, p2);
    CHECK(sc.hits == 2 && sc.misses == 2);
    Dev dev2(300);
    const auto o2 = random_pages(rng, 300, p2.size());
    sc.promote(i2, dev2, o2);
    for (size_t i = 0; i < o2.size(); i++)
      for (int sl = 0; sl < kSlices; sl++) CHECK(std::memcmp(dev2.page(o2[i], sl), devb.page(p2[i], sl), kP) == 0);
    CHECK(sc.store.size() == 2);  // still held by entry 1
    sc.erase(i1);
    CHECK(sc.store.size() == 0 && sc.own == 0);
  }
  CHECK(pool.used() == used0);  // every pinned byte came back
  noise.clear();
  CHECK(pool.used() == 0);
  for (char* b : bases) std::free(b);
}

// Orphans (refs == 0) stay resident so a replacement entry re-finds them; enforce frees them first, then picks victims
// by freed bytes.
static void test_orphans_and_enforce() {
  std::mt19937 rng(3);
  SimCache sc;
  Dev dev(500);
  // conversation: a shallow checkpoint (1 chunk + tail) and the deeper one that replaces it
  const auto deep = make_ids(1, 3 * kChunkTokens + 300);
  const std::vector<int32_t> shallow(deep.begin(), deep.begin() + long(kChunkTokens + 200));
  const auto ps = random_pages(rng, 500, (shallow.size() + 63) / 64), pd = random_pages(rng, 500, (deep.size() + 63) / 64);
  fill_dev(dev, shallow, ps), fill_dev(dev, deep, pd);
  const int s = sc.spill(dev, shallow, ps);
  const size_t before = sc.phys();
  sc.erase(s);  // (simulates "erase first": here chunks go at once)
  CHECK(sc.phys() == 0);
  const int s2 = sc.spill(dev, shallow, ps);
  // the orphan-deferral flow: unref without freeing, then the deeper copy re-finds the chunk
  Entry& e = sc.entries.at(s2);
  Ch* c0 = e.chunks[0];
  for (Ch* c : e.chunks) sc.store.unref(c);
  sc.own -= e.tail.total;
  sc.entries.erase(s2);
  CHECK(sc.store.orphan_count() == 1 && sc.store.bytes() == kChunkBytes);
  sc.spill(dev, deep, pd);
  CHECK(sc.hits == 1 && sc.store.orphan_count() == 0);  // revived, not copied again
  CHECK(sc.store.size() == 3);
  (void)c0, (void)before;

  // enforce: orphan chunks first
  SimCache sc2;
  Dev d2(500);
  const auto a = make_ids(20, 2 * kChunkTokens + 100), b = make_ids(21, kChunkTokens + 100, &a, kChunkTokens);
  const auto pa = random_pages(rng, 500, (a.size() + 63) / 64), pb = random_pages(rng, 500, (b.size() + 63) / 64);
  fill_dev(d2, a, pa), fill_dev(d2, b, pb);
  const int ia = sc2.spill(d2, a, pa);
  sc2.entries.at(ia).last_use = 1;
  const int ib = sc2.spill(d2, b, pb);
  sc2.entries.at(ib).last_use = 2;
  CHECK(sc2.hits == 1 && sc2.store.size() == 2);  // chunk 0 shared; a has a second chunk of its own
  const size_t full = sc2.phys(), logical = sc2.logical();
  CHECK(full < logical);
  // an orphan appears
  Ch* second = sc2.entries.at(ia).chunks[1];
  sc2.store.unref(second);
  sc2.entries.at(ia).chunks.pop_back();
  CHECK(sc2.store.orphan_count() == 1);
  // a cap just below the physical use is met by freeing the orphan alone: nothing is evicted
  CHECK(sc2.enforce(sc2.phys() - 1) == 0);
  CHECK(sc2.store.orphan_count() == 0 && sc2.entries.size() == 2);
  // a cap that needs an entry gone: b (older a frees 'tail only' since chunk 0 is shared; both share) -> the one freeing most
  const size_t cap = sc2.phys() - 1;
  const size_t fa = sc2.freed(sc2.entries.at(ia)), fb = sc2.freed(sc2.entries.at(ib));
  CHECK(fa > 0 && fb > 0);
  // the victim is the entry freeing more (a: its own tail and chunk-1 are gone already, so compare)
  const bool expect_a = fa > fb || (fa * 2 >= fb && fb * 2 >= fa && sc2.entries.at(ia).last_use < sc2.entries.at(ib).last_use);
  const int ev = sc2.enforce(cap);
  CHECK(ev == 1 && sc2.phys() <= cap);
  CHECK(expect_a ? (sc2.entries.count(ia) == 0 && sc2.entries.count(ib) == 1)
                 : (sc2.entries.count(ib) == 0 && sc2.entries.count(ia) == 1));
  // the shared chunk survives while one holder remains
  CHECK(sc2.store.size() >= 1);
  sc2.enforce(0);
  CHECK(sc2.entries.empty() && sc2.store.size() == 0 && sc2.phys() == 0);
}

// An allocation failing in the middle of chunk_acquire (bad_alloc): the caller unrefs exactly what `refd` lists, and a
// chunk that was created is not left registered.
static void test_acquire_throws() {
  Store st;
  const Key128 seed = chunk_seed(1, kChunkPages);
  const auto ids = make_ids(31, 4 * kChunkTokens);
  std::vector<Ch*> r0;
  std::vector<std::pair<size_t, Ch*>> f0;
  chunk_acquire(st, seed, ids.data(), 1, kChunkTokens, 64, [](size_t b) { return alloc_kv(nullptr, b); }, r0, f0);  // chunk 0 resident
  CHECK(r0.size() == 1 && f0.size() == 1 && r0[0]->refs == 1);
  for (int fail_at = 0; fail_at < 3; fail_at++) {  // (chunks 1..3 are missing) the allocation of the (fail_at)th missing chunk throws
    std::vector<Ch*> refd;
    std::vector<std::pair<size_t, Ch*>> fresh;
    int calls = 0;
    bool threw = false;
    try {
      chunk_acquire(st, seed, ids.data(), 4, kChunkTokens, 64, [&](size_t b) {
        if (calls++ == fail_at) throw std::bad_alloc();
        return alloc_kv(nullptr, b);
      }, refd, fresh);
    } catch (const std::bad_alloc&) {
      threw = true;
    }
    CHECK(threw && fresh.size() == size_t(fail_at) && refd.size() == size_t(fail_at) + 1);  // chunk 0 hit + fail_at new ones
    CHECK(r0[0]->refs == 2);
    // the rollback (what Model::spill_copy does): fresh ones are freed, hits unref'd
    for (Ch* c : refd) {
      const bool is_fresh = std::any_of(fresh.begin(), fresh.end(), [&](auto& f) { return f.second == c; });
      if (is_fresh) {
        st.unref(c);
        st.extract(c);
      } else {
        st.unref(c);
      }
    }
    CHECK(st.size() == 1 && st.bytes() == 64 && r0[0]->refs == 1 && st.orphan_count() == 0);
  }
}

static void test_pick_victim() {
  // equal freed: LRU
  CHECK(pick_victim({{5, 100}, {3, 100}, {9, 100}}) == 1);
  // an entry freeing almost nothing (shared chunks) is not chosen over one that frees real memory, even if older
  CHECK(pick_victim({{1, 10}, {7, 1000}, {8, 900}}) == 1);
  // within a factor 2 of the best: the oldest
  CHECK(pick_victim({{4, 600}, {7, 1000}, {2, 400}}) == 0);
  CHECK(pick_victim({{4, 0}, {2, 0}}) == 1);
  CHECK(pick_victim({}) == -1);
}

// Random churn: entries from a few conversations come and go; the physical bytes always equal tails + resident chunks,
// every chunk's refs equals the number of entries listing it, and a drained cache is empty.
static void test_churn() {
  std::mt19937 rng(2024);
  SimCache sc;
  Dev dev(700);
  std::vector<std::vector<int32_t>> convs;
  for (int i = 0; i < 3; i++) convs.push_back(make_ids(300 + i, 5 * kChunkTokens));
  std::vector<int> live;
  for (int step = 0; step < 150; step++) {
    if (live.size() < 6 && rng() % 3) {
      const auto& c = convs[rng() % convs.size()];
      size_t len = 50 + rng() % (c.size() - 50);
      std::vector<int32_t> ids(c.begin(), c.begin() + long(len));
      if (rng() % 3 == 0) ids = make_ids(rng(), len);  // an unrelated one
      auto pages = random_pages(rng, 700, (len + 63) / 64);
      fill_dev(dev, ids, pages);
      const int id = sc.spill(dev, ids, pages);
      // check the content round trip now and then
      if (rng() % 4 == 0) {
        Dev d2(700);
        auto other = random_pages(rng, 700, pages.size());
        std::vector<char> lin = sc.linear(id);
        sc.promote(id, d2, other);
        for (size_t i = 0; i < other.size(); i++)
          for (int sl = 0; sl < kSlices; sl++) CHECK(std::memcmp(d2.page(other[i], sl), dev.page(pages[i], sl), kP) == 0);
      } else {
        live.push_back(id);
      }
    } else if (!live.empty()) {
      const size_t k = rng() % live.size();
      sc.erase(live[k]);
      live.erase(live.begin() + long(k));
    }
    std::map<Ch*, int> cnt;
    size_t tails = 0;
    for (auto& kv : sc.entries) {
      for (Ch* c : kv.second.chunks) cnt[c]++;
      tails += kv.second.tail.total;
    }
    size_t cb = 0;
    sc.store.for_each([&](Ch& c) {
      CHECK(cnt.count(&c) && cnt[&c] == c.refs);
      cb += c.kv.total;
    });
    CHECK(cnt.size() == sc.store.size() && cb == sc.store.bytes() && sc.own == tails);
    CHECK(sc.phys() <= sc.logical());
  }
  for (int id : live) sc.erase(id);
  CHECK(sc.store.size() == 0 && sc.phys() == 0 && sc.store.orphan_count() == 0);
}

int main() {
  test_keys();
  test_collision();
  test_refcounts_accounting();
  test_ragged();
  test_pinned();
  test_orphans_and_enforce();
  test_acquire_throws();
  test_pick_victim();
  test_churn();
  if (fails) {
    std::printf("%d check(s) FAILED\n", fails);
    return 1;
  }
  std::printf("OK\n");
  return 0;
}
