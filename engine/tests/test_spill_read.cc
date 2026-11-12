// Host-only test of the background spill-file reader (DiskRead in src/host_kv.h, SPILL_ASYNC_PROMOTE): no CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=thread -Isrc tests/test_spill_read.cc -o /tmp/test_spill_read -pthread && /tmp/test_spill_read
//   (and the same with -fsanitize=address,undefined)
// The reader reads a file in Model::spill_write's format into a segmented HostKv (pinned-pool regions or one pageable
// buffer) on a thread, as Model::spill_restore_begin / spill_restore_poll use it: the "engine thread" here polls the
// atomic state, joins, then checks the bytes. Failure cases: damaged header, fingerprint, ids, truncation, no file.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "host_kv.h"

using namespace q27;

static int fails = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
      if (++fails > 20) std::exit(1);                                          \
    }                                                                          \
  } while (0)

constexpr uint64_t kFp = 0x1234abcd5678ef01ull;

struct Entry {
  std::vector<int32_t> ids;
  std::vector<char> state, kv;
};

static Entry make_entry(std::mt19937& rng, size_t n_ids, size_t state_bytes, size_t kv_bytes) {
  Entry e;
  for (size_t i = 0; i < n_ids; i++) e.ids.push_back(int32_t(rng() % 100000));
  e.state.resize(state_bytes), e.kv.resize(kv_bytes);
  for (auto& c : e.state) c = char(rng());
  for (auto& c : e.kv) c = char(rng());
  return e;
}

static std::string write_file(const Entry& e, uint64_t fp, const char* dir, int idx) {
  std::string path = std::string(dir) + "/e" + std::to_string(idx) + ".q27kv";
  SpillHeader h{};
  std::memcpy(h.magic, kSpillMagic, sizeof kSpillMagic);
  h.version = kSpillVersion;
  h.fingerprint = fp;
  h.state_bytes = e.state.size();
  h.n_ids = uint32_t(e.ids.size());
  h.kv_bytes = e.kv.size();
  std::FILE* f = std::fopen(path.c_str(), "wb");
  CHECK(f);
  std::fwrite(&h, sizeof h, 1, f);
  std::fwrite(e.ids.data(), sizeof(int32_t), e.ids.size(), f);
  std::fwrite(e.state.data(), 1, e.state.size(), f);
  std::fwrite(e.kv.data(), 1, e.kv.size(), f);
  std::fclose(f);
  return path;
}

// Destination buffers: pinned-pool style segments (page multiples) or one pageable buffer.
struct Dest {
  HostKv kv;
  std::unique_ptr<char[]> state;
};

static void fill_read(DiskRead& d, const Entry& e, const std::string& path, uint64_t fp, Dest& dst, const std::vector<int32_t>* ids = nullptr) {
  d.path = path, d.fingerprint = fp, d.state_bytes = e.state.size(), d.kv_bytes = e.kv.size();
  d.ids = ids ? *ids : e.ids;
  d.state = dst.state.get();
  for (size_t i = 0; i < dst.kv.seg.size(); i++) d.segs.emplace_back(dst.kv.seg[i].get(), dst.kv.len[i]);
}

static int wait_state(DiskRead& d) {
  while (d.state_now() == 0) std::this_thread::yield();  // (the engine thread polls; here it spins)
  d.join();
  return d.state_now();
}

static bool same(const Entry& e, const Dest& dst) {
  if (std::memcmp(dst.state.get(), e.state.data(), e.state.size()) != 0) return false;
  size_t off = 0;
  for (size_t i = 0; i < dst.kv.seg.size(); i++) {
    if (std::memcmp(dst.kv.seg[i].get(), e.kv.data() + off, dst.kv.len[i]) != 0) return false;
    off += dst.kv.len[i];
  }
  return off == e.kv.size();
}

int main() {
  char tmpl[] = "/tmp/test_spill_read_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  CHECK(dir);
  std::mt19937 rng(7);
  constexpr size_t kPage = 4096 * 3;

  // 1. Several entries read at once into segments of a fragmented PinnedPool, and into pageable buffers.
  {
    constexpr size_t slab = size_t(1) << 20;
    std::vector<char*> bases;
    PinnedPool pool;
    for (int i = 0; i < 3; i++) {
      char* b = static_cast<char*>(std::aligned_alloc(4096, slab));
      bases.push_back(b);
      pool.add_slab(b, slab);
    }
    std::vector<Entry> es;
    std::vector<Dest> dsts(6);
    std::vector<std::string> paths;
    std::vector<std::unique_ptr<DiskRead>> reads;
    std::vector<std::pair<char*, size_t>> held;
    for (int i = 0; i < 24; i++) {  // fragment the pool: keep a random half of many small regions
      const size_t len = (rng() % 5 + 1) * 4096;
      if (char* p = pool.alloc(len)) {
        if (rng() % 2) held.emplace_back(p, len);
        else pool.release(p, len);
      }
    }
    for (int i = 0; i < 6; i++) {
      const size_t pages = 5 + rng() % 40;
      es.push_back(make_entry(rng, 100 + rng() % 5000, 777 + rng() % 100, pages * kPage));
      paths.push_back(write_file(es.back(), kFp, dir, i));
      if (i % 2 == 0) {
        CHECK(alloc_pinned_kv(pool, es.back().kv.size(), kPage, dsts[i].kv));
        CHECK(dsts[i].kv.pinned());
      } else {
        dsts[i].kv.push(KvBuf(new char[es.back().kv.size()], KvDel{}), es.back().kv.size());
      }
      dsts[i].state.reset(new char[es.back().state.size()]);
    }
    for (int i = 0; i < 6; i++) {
      reads.push_back(std::make_unique<DiskRead>());
      fill_read(*reads.back(), es[i], paths[i], kFp, dsts[i]);
      reads.back()->start();
    }
    for (int i = 0; i < 6; i++) {
      CHECK(wait_state(*reads[i]) == 1);
      CHECK(same(es[i], dsts[i]));
      CHECK(reads[i]->ns.load() > 0);
    }
    reads.clear();
    for (auto& d : dsts) d.kv.reset();  // regions go back to the pool
    for (auto& [p, len] : held) pool.release(p, len);
    CHECK(pool.used() == 0);
    for (char* b : bases) std::free(b);
  }

  // 2. Failures: each ends with st == -1 and never touches memory past the buffers.
  {
    Entry e = make_entry(rng, 300, 512, 4 * kPage);
    const std::string good = write_file(e, kFp, dir, 100);
    auto run = [&](const std::string& path, uint64_t fp, const std::vector<int32_t>* ids, size_t kv_override) {
      Dest d;
      Entry shape = e;
      if (kv_override) shape.kv.resize(kv_override);
      d.kv.push(KvBuf(new char[shape.kv.size()], KvDel{}), shape.kv.size());
      d.state.reset(new char[shape.state.size()]);
      DiskRead r;
      fill_read(r, shape, path, fp, d, ids);
      r.start();
      return wait_state(r);
    };
    CHECK(run(good, kFp, nullptr, 0) == 1);
    CHECK(run(good, kFp + 1, nullptr, 0) == -1);                   // another model
    std::vector<int32_t> other = e.ids;
    other[17]++;
    CHECK(run(good, kFp, &other, 0) == -1);                        // other tokens
    CHECK(run(std::string(dir) + "/missing.q27kv", kFp, nullptr, 0) == -1);
    CHECK(run(good, kFp, nullptr, 3 * kPage) == -1);               // kv_bytes differs from the header
    // truncated file
    std::string cut = std::string(dir) + "/cut.q27kv";
    {
      std::FILE* f = std::fopen(good.c_str(), "rb");
      std::vector<char> all(sizeof(SpillHeader) + 300 * 4 + 512 + 4 * kPage);
      CHECK(std::fread(all.data(), 1, all.size(), f) == all.size());
      std::fclose(f);
      f = std::fopen(cut.c_str(), "wb");
      std::fwrite(all.data(), 1, all.size() - 100, f);
      std::fclose(f);
    }
    CHECK(run(cut, kFp, nullptr, 0) == -1);
    // bad magic / version
    for (int which = 0; which < 2; which++) {
      std::string bad = std::string(dir) + "/bad" + std::to_string(which) + ".q27kv";
      std::FILE* f = std::fopen(good.c_str(), "rb");
      std::vector<char> all(sizeof(SpillHeader) + 300 * 4 + 512 + 4 * kPage);
      CHECK(std::fread(all.data(), 1, all.size(), f) == all.size());
      std::fclose(f);
      all[which == 0 ? 0 : 8] ^= 0x55;
      f = std::fopen(bad.c_str(), "wb");
      std::fwrite(all.data(), 1, all.size(), f);
      std::fclose(f);
      CHECK(run(bad, kFp, nullptr, 0) == -1);
    }
  }

  // 3. Destroying a running reader (the Model's destructor) cancels it between chunks and joins: no hang, no race.
  {
    Entry e = make_entry(rng, 50, 256, size_t(48) << 20);
    const std::string path = write_file(e, kFp, dir, 200);
    for (int round = 0; round < 5; round++) {
      Dest d;
      d.kv.push(KvBuf(new char[e.kv.size()], KvDel{}), e.kv.size());
      d.state.reset(new char[e.state.size()]);
      auto r = std::make_unique<DiskRead>();
      fill_read(*r, e, path, kFp, d);
      r->start();
      if (round % 2) std::this_thread::yield();
      r.reset();      // joins first,
      d.kv.reset();   // and only then are the buffers freed
    }
  }

  std::printf(fails ? "FAILED (%d)\n" : "ok\n", fails);
  return fails ? 1 : 0;
}
