// Host-only test of the background entry reader (DiskRead in src/spill_files.h, SPILL_ASYNC_PROMOTE): no CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=thread -Isrc tests/test_spill_read.cc -o /tmp/test_spill_read -pthread && /tmp/test_spill_read
//   (and the same with -fsanitize=address,undefined)
// The reader reads a manifest AND the chunk files it lists (several files, some shared by other entries being read at the same
// moment) into a segmented HostKv (pinned-pool regions or one pageable buffer) on a thread, as Model::spill_restore_begin /
// spill_restore_poll use it: the "engine thread" here polls the atomic state, joins, then checks the bytes. Failure cases: a
// missing / damaged chunk (named in `bad`), fingerprint, tokens, truncation. Cancelling by destruction joins before the buffers go.
#include "spill_test_util.h"

using namespace q27test;

int main() {
  const std::string dir = tmpdir("spill_read");
  std::mt19937 rng(7);
  DiskIndex idx;
  idx.set_dir(dir + "/chunks");

  // A family of entries over a shared 2-chunk prefix, plus unrelated ones.
  const Entry base = make_entry(rng, {}, 3, 90, 700);
  std::vector<int32_t> prefix(base.ids.begin(), base.ids.begin() + std::ptrdiff_t(2 * base.chunk_tokens()));
  std::vector<Entry> es;
  for (int i = 0; i < 6; i++) {
    Entry e = make_entry(rng, i < 4 ? prefix : std::vector<int32_t>{}, 2 + rng() % 3, 20 + rng() % 200, 300 + rng() % 400);
    if (i < 4)  // (the shared prefix has the same KV: that is what a chunk file shared between entries holds)
      for (uint32_t sl = 0; sl < e.g.slices; sl++)
        std::memcpy(e.linear.data() + size_t(sl) * e.n_pages() * e.g.page_bytes,
                    base.linear.data() + size_t(sl) * base.n_pages() * base.g.page_bytes, size_t(2) * e.g.chunk_pages * e.g.page_bytes);
    es.push_back(std::move(e));
  }
  std::vector<Exported> xs(es.size());
  for (size_t i = 0; i < es.size(); i++) CHECK(export_entry(es[i], idx, dir + "/entries/e" + std::to_string(i) + ".q27m", &xs[i]));
  CHECK(idx.find(xs[0].names[0])->refs == 4);  // the shared chunk file: once on disk, listed by four manifests

  // 1. All entries read at once (the four that share chunk files read the same files concurrently), into segments of a
  // fragmented PinnedPool and into pageable buffers.
  {
    constexpr size_t slab = size_t(1) << 20;
    std::vector<char*> bases;
    PinnedPool pool;
    for (int i = 0; i < 3; i++) {
      char* b = static_cast<char*>(std::aligned_alloc(4096, slab));
      bases.push_back(b);
      pool.add_slab(b, slab);
    }
    std::vector<std::pair<char*, size_t>> held;
    for (int i = 0; i < 24; i++) {  // fragment the pool: keep a random half of many small regions
      const size_t len = (rng() % 5 + 1) * 4096;
      if (char* p = pool.alloc(len)) {
        if (rng() % 2) held.emplace_back(p, len);
        else pool.release(p, len);
      }
    }
    std::vector<HostKv> dsts(es.size());
    std::vector<std::vector<char>> states(es.size());
    std::vector<std::unique_ptr<DiskRead>> reads;
    int pinned = 0;
    for (size_t i = 0; i < es.size(); i++) {
      if (i % 2 == 0 && alloc_pinned_kv(pool, es[i].linear.size(), es[i].g.page_bytes, dsts[i])) pinned++;
      else dsts[i] = empty_kv(es[i].linear.size());
    }
    CHECK(pinned >= 1);
    for (size_t i = 0; i < es.size(); i++) {
      reads.push_back(std::make_unique<DiskRead>());
      plan_read(*reads.back(), es[i], idx, xs[i].manifest, xs[i].names, &states[i], &dsts[i]);
      reads.back()->start();
    }
    for (size_t i = 0; i < es.size(); i++) {
      CHECK(wait_state(*reads[i]) == 1);
      CHECK(same(es[i], states[i], dsts[i]));
      CHECK(reads[i]->ns.load() > 0);
    }
    reads.clear();
    for (auto& d : dsts) d.reset();  // regions go back to the pool
    for (auto& [p, len] : held) pool.release(p, len);
    CHECK(pool.used() == 0);
    for (char* b : bases) std::free(b);
  }

  // 2. Failures end with st == -1 and never touch memory past the buffers; a chunk's failure is named.
  {
    auto run = [&](const Entry& want, const std::string& manifest, const std::vector<std::string>& names, int* bad) {
      HostKv dst = empty_kv(want.linear.size());
      std::vector<char> state;
      DiskRead r;
      plan_read(r, want, idx, manifest, names, &state, &dst);
      r.start();
      const int st = wait_state(r);
      *bad = r.bad.load();
      return st;
    };
    int bad = 0;
    CHECK(run(es[0], xs[0].manifest, xs[0].names, &bad) == 1);
    Entry w = es[0];
    w.fp++;
    CHECK(run(w, xs[0].manifest, xs[0].names, &bad) == -1);
    w = es[0];
    w.ids[17]++;  // other tokens
    CHECK(run(w, xs[0].manifest, xs[0].names, &bad) == -1);
    CHECK(run(es[0], dir + "/entries/missing.q27m", xs[0].names, &bad) == -1 && bad == -1);
    // a chunk file the entry shares with its siblings is damaged: every sibling fails on it, the entry that does not list it reads
    const std::string shared = idx.path(xs[0].names[1]);
    fs::copy_file(shared, shared + ".keep");
    flip(shared, long(sizeof(ChunkHeader) + es[0].chunk_tokens() * 4 + 100));
    for (int i = 0; i < 4; i++) CHECK(run(es[size_t(i)], xs[size_t(i)].manifest, xs[size_t(i)].names, &bad) == -1 && bad == 1);
    CHECK(run(es[4], xs[4].manifest, xs[4].names, &bad) == 1);
    fs::rename(shared + ".keep", shared);
    CHECK(run(es[1], xs[1].manifest, xs[1].names, &bad) == 1);
    // a chunk file that is gone
    fs::rename(shared, shared + ".away");
    CHECK(run(es[2], xs[2].manifest, xs[2].names, &bad) == -1 && bad == 1);
    fs::rename(shared + ".away", shared);
    // truncated chunk
    fs::copy_file(shared, shared + ".keep");
    truncate_by(shared, 1);
    CHECK(run(es[3], xs[3].manifest, xs[3].names, &bad) == -1 && bad == 1);
    fs::remove(shared);
    fs::rename(shared + ".keep", shared);
    CHECK(run(es[3], xs[3].manifest, xs[3].names, &bad) == 1);
  }

  // 3. Destroying a running reader (the Model's destructor) cancels it between blocks and joins: no hang, no race, and the
  // buffers outlive the thread. (A big entry: 3 slices x 48 pages x 64 KiB per page of tail and chunks.)
  {
    const Entry big = make_entry(rng, {}, 2, 40 * 64, 256, 3, 4, size_t(1) << 20);
    Exported bx;
    CHECK(export_entry(big, idx, dir + "/entries/big.q27m", &bx));
    for (int round = 0; round < 6; round++) {
      HostKv dst = empty_kv(big.linear.size());
      std::vector<char> state;
      auto r = std::make_unique<DiskRead>();
      plan_read(*r, big, idx, bx.manifest, bx.names, &state, &dst);
      r->start();
      if (round % 2) std::this_thread::yield();
      r.reset();      // joins first,
      dst.reset();    // and only then are the buffers freed
    }
  }

  // 4. SPILL_READ_THREADS (DiskRead::threads > 1: the chunk files are read by several threads). Each case gets its own chunk
  // directory (the files it damages are not shared with another case). threads 1 and 4 must give the same bytes, the damaged
  // chunk is named (the LOWEST index that failed verification), a missing file is named, a cancel joins cleanly, and an entry of
  // one chunk reads like before.
  {
    auto exported = [&](const std::string& tag, const Entry& e, DiskIndex& ix, Exported& x) {
      ix.set_dir(dir + "/" + tag + "/chunks");
      return export_entry(e, ix, dir + "/" + tag + "/e.q27m", &x);
    };
    // reads `e` from its manifest with `threads` threads: the status, the named chunk, and the bytes read
    auto read = [&](const Entry& e, const Exported& x, const DiskIndex& ix, int threads, int* bad, bool* io,
                    std::vector<char>* state, HostKv* dst) {
      *dst = empty_kv(e.linear.size());
      DiskRead r;
      plan_read(r, e, ix, x.manifest, x.names, state, dst);
      r.threads = threads;
      r.start();
      const int st = wait_state(r);
      *bad = r.bad.load(), *io = r.io_error.load();
      return st;
    };
    const Entry four = make_entry(rng, {}, 4, 100, 256);
    // a. threads 1 and 4: the same bytes, both equal to what was exported
    {
      DiskIndex ix;
      Exported x;
      CHECK(exported("a", four, ix, x));
      int bad1 = 0, bad4 = 0;
      bool io1 = false, io4 = false;
      std::vector<char> st1, st4;
      HostKv d1, d4;
      CHECK(read(four, x, ix, 1, &bad1, &io1, &st1, &d1) == 1);
      CHECK(read(four, x, ix, 4, &bad4, &io4, &st4, &d4) == 1);
      CHECK(bad1 == -1 && bad4 == -1 && !io1 && !io4);
      CHECK(same(four, st1, d1) && same(four, st4, d4));
      CHECK(st1 == st4);
    }
    // b. the chunk at index 2 is damaged: bad == 2 with 4 threads (and with 1); two damaged chunks (1 and 3): bad == 1, every
    // time (the race between the workers is run 50 times)
    {
      DiskIndex ix;
      Exported x;
      CHECK(exported("b", four, ix, x));
      const std::string c2 = ix.path(x.names[2]);
      fs::copy_file(c2, c2 + ".keep");
      flip(c2, long(sizeof(ChunkHeader) + four.chunk_tokens() * 4 + 100));
      int bad = 0;
      bool io = false;
      std::vector<char> state;
      HostKv dst;
      CHECK(read(four, x, ix, 1, &bad, &io, &state, &dst) == -1 && bad == 2 && !io);
      CHECK(read(four, x, ix, 4, &bad, &io, &state, &dst) == -1 && bad == 2 && !io);
      fs::rename(c2 + ".keep", c2);
      const std::string c1 = ix.path(x.names[1]), c3 = ix.path(x.names[3]);
      flip(c1, long(sizeof(ChunkHeader) + four.chunk_tokens() * 4 + 100));
      flip(c3, long(sizeof(ChunkHeader) + four.chunk_tokens() * 4 + 100));
      for (int round = 0; round < 50; round++) {
        CHECK(read(four, x, ix, 4, &bad, &io, &state, &dst) == -1 && bad == 1 && !io);
      }
      CHECK(read(four, x, ix, 1, &bad, &io, &state, &dst) == -1 && bad == 1);
      flip(c1, long(sizeof(ChunkHeader) + four.chunk_tokens() * 4 + 100));  // (the same flip again: undone)
      flip(c3, long(sizeof(ChunkHeader) + four.chunk_tokens() * 4 + 100));
      CHECK(read(four, x, ix, 4, &bad, &io, &state, &dst) == 1);
    }
    // c. a chunk file that is gone (index 2): named, and not an I/O error
    {
      DiskIndex ix;
      Exported x;
      CHECK(exported("c", four, ix, x));
      const std::string c2 = ix.path(x.names[2]);
      fs::rename(c2, c2 + ".away");
      int bad = 0;
      bool io = false;
      std::vector<char> state;
      HostKv dst;
      CHECK(read(four, x, ix, 4, &bad, &io, &state, &dst) == -1 && bad == 2 && !io);
    }
    // d. a cancel in the middle of the reads (the big entry, 4 threads): the destructor joins, no hang, and a cancelled read is
    // no verdict (bad -1, not an I/O error). Repeated, so that the cancel lands at different points.
    {
      DiskIndex ix;
      Exported x;
      const Entry big = make_entry(rng, {}, 6, 40 * 64, 256, 3, 4, size_t(1) << 20);
      CHECK(exported("d", big, ix, x));
      for (int round = 0; round < 20; round++) {
        HostKv dst = empty_kv(big.linear.size());
        std::vector<char> state;
        auto r = std::make_unique<DiskRead>();
        plan_read(*r, big, ix, x.manifest, x.names, &state, &dst);
        r->threads = 4;
        r->start();
        for (int k = 0; k < round; k++) std::this_thread::yield();
        r->cancel.store(true);
        const int st = wait_state(*r);
        CHECK(st == 1 || (st == -1 && r->bad.load() == -1 && !r->io_error.load()));
        r.reset();
        dst.reset();
      }
    }
    // e. an entry of one chunk, with 4 threads (the serial path): the same as before
    {
      const Entry one = make_entry(rng, {}, 1, 50, 256);
      DiskIndex ix;
      Exported x;
      CHECK(exported("e", one, ix, x));
      int bad = 0;
      bool io = false;
      std::vector<char> state;
      HostKv dst;
      CHECK(read(one, x, ix, 4, &bad, &io, &state, &dst) == 1 && same(one, state, dst));
    }
  }

  fs::remove_all(dir);
  std::printf(fails ? "FAILED (%d)\n" : "ok\n", fails);
  return fails ? 1 : 0;
}
