// Host-only test of the SSD tier's files (src/spill_files.h): no CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Isrc tests/test_spill_files.cc -o /tmp/test_spill_files && /tmp/test_spill_files
// Hash64, the chunk names (= chunk_walk's keys), the .kvc / .q27m round trip against linear_pieces (the old linear stream), damage and
// truncation detection, crash leftovers (.tmp, a manifest whose chunk is missing, a stray chunk), and the disk refcounts.
#include "spill_test_util.h"

using namespace q27test;

static void test_hash() {
  std::mt19937 rng(1);
  std::vector<unsigned char> v(1000);
  for (auto& c : v) c = (unsigned char)rng();
  Hash64 whole;
  whole.update(v.data(), v.size());
  for (int round = 0; round < 50; round++) {  // any split gives the same digest
    Hash64 h;
    size_t off = 0;
    while (off < v.size()) {
      const size_t k = std::min<size_t>(rng() % 70, v.size() - off);
      h.update(v.data() + off, k);
      off += k;
    }
    CHECK(h.digest() == whole.digest());
  }
  for (size_t n : {size_t(0), size_t(1), size_t(7), size_t(31), size_t(32), size_t(33), size_t(64), size_t(1000)}) {
    Hash64 a, b;
    a.update(v.data(), n);
    if (n) b.update(v.data(), n - 1), b.update(v.data() + n - 1, 1);
    CHECK(n == 0 || a.digest() == b.digest());
    std::vector<unsigned char> w(v.begin(), v.begin() + std::ptrdiff_t(n));
    if (n) {
      w[n / 2] ^= 1;
      Hash64 c;
      c.update(w.data(), n);
      CHECK(c.digest() != a.digest());  // (a flipped bit changes it)
    }
  }
  Hash64 e1, e2;
  e2.update("x", 1);
  CHECK(e1.digest() != e2.digest());
}

static void test_names() {
  std::mt19937 rng(2);
  std::vector<int32_t> ids(256 * 3);
  for (auto& x : ids) x = int32_t(rng() % 1000);
  const auto chain = chunk_chain(kFp, 4, ids.data(), 3, 256);
  // the keys are the RAM tier's (chunk_walk), so a RAM chunk and its file share an identity
  ChunkStore<int*> st;
  size_t i = 0;
  chunk_walk(st, chunk_seed(kFp, 4), ids.data(), 3, 256,
             [&](size_t c, ChunkStore<int*>::C*, const Key128& key, uint64_t, const int32_t*) -> ChunkStore<int*>::C* {
               CHECK(key == chain[c].key);
               i++;
               return nullptr;
             });
  CHECK(i == 3);  // (a dry run still computes every key)
  CHECK(chain[0].name.size() == 49 && chain[0].name[32] == '-');
  // the same prefix, the same names (across processes: nothing in-memory goes into them); a different tail, another third chunk
  std::vector<int32_t> ids2 = ids;
  ids2[600]++;
  const auto chain2 = chunk_chain(kFp, 4, ids2.data(), 3, 256);
  CHECK(chain2[0].name == chain[0].name && chain2[1].name == chain[1].name && chain2[2].name != chain[2].name);
  const auto other_fp = chunk_chain(kFp + 1, 4, ids.data(), 3, 256);
  CHECK(other_fp[0].name != chain[0].name);
  std::vector<int32_t> ids3 = ids;
  ids3[0]++;  // an earlier token changes every later name (the parent chain)
  const auto chain3 = chunk_chain(kFp, 4, ids3.data(), 3, 256);
  CHECK(chain3[0].name != chain[0].name && chain3[1].name != chain[1].name && chain3[2].name != chain[2].name);
}

static void test_roundtrip_and_damage() {
  std::mt19937 rng(3);
  const std::string dir = tmpdir("spill_files");
  DiskIndex idx;
  idx.set_dir(dir + "/chunks");
  const Entry e = make_entry(rng, {}, 3, 100, 777);
  Exported ex;
  CHECK(export_entry(e, idx, dir + "/entries/a.q27m", &ex));
  CHECK(ex.written.size() == 3 && idx.files() == 3);
  for (const auto& n : ex.names) CHECK(idx.find(n) && idx.find(n)->refs == 1);
  CHECK(fs::file_size(dir + "/entries/a.q27m") == manifest_file_bytes(e.state.size(), e.tail_pages * e.g.slices * e.g.page_bytes, e.ids.size(), 3));
  CHECK(idx.physical_bytes() == 3 * chunk_file_bytes(e.g, e.chunk_tokens()));

  ManifestHead head;
  CHECK(read_manifest_head(ex.manifest, head));
  CHECK(head.ids == e.ids && head.names == ex.names && head.info.state_bytes == e.state.size() && head.info.g == e.g);

  // read it back into a private linear buffer, segmented (the pinned pool's shape): every byte as the old linear stream had it
  auto read_back = [&](const Entry& want, const std::vector<std::string>& names, const std::string& manifest, int* bad) {
    HostKv dst;
    size_t rem = want.linear.size();
    while (rem) {
      const size_t k = std::min(rem, want.g.page_bytes * (1 + rng() % 5));
      dst.push(KvBuf(new char[k](), KvDel{}), k);
      rem -= k;
    }
    std::vector<char> state;
    DiskRead d;
    plan_read(d, want, idx, manifest, names, &state, &dst);
    const int st = wait_state((d.start(), d));
    if (bad) *bad = d.bad.load();
    return st == 1 && same(want, state, dst);
  };
  int bad = 0;
  CHECK(read_back(e, ex.names, ex.manifest, &bad));

  // the chunk files hold the RAM chunks' bytes as they are: [slice][page in chunk]
  {
    std::FILE* f = std::fopen(idx.path(ex.names[1]).c_str(), "rb");
    CHECK(f);
    std::fseek(f, long(sizeof(ChunkHeader) + e.chunk_tokens() * 4), SEEK_SET);
    HostKv want = chunk_kv(e, 1);
    std::vector<char> got(want.total);
    CHECK(std::fread(got.data(), 1, got.size(), f) == got.size());
    CHECK(std::memcmp(got.data(), want.seg[0].get(), got.size()) == 0);
    std::fclose(f);
  }

  // damage: each case reads as failed, never as OK; a chunk's damage names the chunk
  struct Case { const char* what; std::string file; long off; int cut; int want_bad; };
  const std::vector<Case> cases = {
      {"chunk 0 payload", idx.path(ex.names[0]), long(sizeof(ChunkHeader) + e.chunk_tokens() * 4 + 5), 0, 0},
      {"chunk 1 hash", idx.path(ex.names[1]), -3, 0, 1},
      {"chunk 2 ids", idx.path(ex.names[2]), long(sizeof(ChunkHeader) + 12), 0, 2},
      {"chunk 1 header (key)", idx.path(ex.names[1]), 20, 0, 1},
      {"chunk 2 magic", idx.path(ex.names[2]), 1, 0, 2},
      {"chunk 0 truncated", idx.path(ex.names[0]), -1, 9, 0},
      {"manifest state", ex.manifest, long(sizeof(ManifestHeader) + e.ids.size() * 4 + 3 * kNameField + 10), 0, -1},
      {"manifest tail", ex.manifest, -20, 0, -1},
      {"manifest ids", ex.manifest, long(sizeof(ManifestHeader) + 40), 0, -1},
      {"manifest truncated", ex.manifest, -1, 5, -1},
  };
  for (const Case& c : cases) {
    const std::string keep = c.file + ".keep";
    fs::copy_file(c.file, keep);
    if (c.cut) truncate_by(c.file, size_t(c.cut));
    else flip(c.file, c.off);
    bad = -7;
    const bool ok = read_back(e, ex.names, ex.manifest, &bad);
    if (ok || bad != c.want_bad) std::printf("case %s: ok=%d bad=%d\n", c.what, ok, bad);
    CHECK(!ok);
    CHECK(bad == c.want_bad);
    fs::rename(keep, c.file);
  }
  CHECK(read_back(e, ex.names, ex.manifest, &bad));  // (restored: reads fine again)
  // a manifest whose structure is damaged is not even imported
  {
    const std::string keep = ex.manifest + ".keep";
    fs::copy_file(ex.manifest, keep);
    truncate_by(ex.manifest, 3);
    CHECK(!read_manifest_head(ex.manifest, head));
    fs::rename(keep, ex.manifest);
    fs::copy_file(ex.manifest, keep);
    flip(ex.manifest, 0);
    CHECK(!read_manifest_head(ex.manifest, head));
    fs::rename(keep, ex.manifest);
  }
  CHECK(read_manifest_head(ex.manifest, head));
  // wrong expectations (another model, other tokens, a different chunk list)
  {
    HostKv dst = empty_kv(e.linear.size());
    std::vector<char> state;
    Entry w = e;
    w.fp = kFp + 1;
    DiskRead d1;
    plan_read(d1, w, idx, ex.manifest, ex.names, &state, &dst);
    CHECK(wait_state((d1.start(), d1)) == -1);
    w = e;
    w.ids[10]++;
    DiskRead d2;
    plan_read(d2, w, idx, ex.manifest, ex.names, &state, &dst);
    CHECK(wait_state((d2.start(), d2)) == -1);
    auto names = ex.names;
    std::swap(names[0], names[1]);
    DiskRead d3;
    plan_read(d3, e, idx, ex.manifest, names, &state, &dst);
    CHECK(d3.run_sync() == false);
    DiskRead d4;  // the synchronous run reports the same way
    plan_read(d4, e, idx, ex.manifest, ex.names, &state, &dst);
    CHECK(d4.run_sync() == true && d4.state_now() == 1);
  }
  fs::remove_all(dir);
}

static void test_sharing_and_refcounts() {
  std::mt19937 rng(4);
  const std::string dir = tmpdir("spill_files");
  DiskIndex idx;
  idx.set_dir(dir + "/chunks");
  // three entries share a 2-chunk prefix; the third also shares nothing past it
  const Entry a = make_entry(rng, {}, 3, 100, 500);
  std::vector<int32_t> prefix(a.ids.begin(), a.ids.begin() + std::ptrdiff_t(2 * a.chunk_tokens()));
  Entry b = make_entry(rng, prefix, 3, 30, 500);
  Entry c = make_entry(rng, prefix, 2, 200, 500);
  // (b and c share a's first two chunks' KV bytes: the KV of a shared prefix is the same data, which is what sharing means)
  for (Entry* x : {&b, &c})
    for (uint32_t sl = 0; sl < a.g.slices; sl++)
      std::memcpy(x->linear.data() + size_t(sl) * x->n_pages() * a.g.page_bytes,
                  a.linear.data() + size_t(sl) * a.n_pages() * a.g.page_bytes, size_t(2) * a.g.chunk_pages * a.g.page_bytes);
  Exported ea, eb, ec;
  CHECK(export_entry(a, idx, dir + "/entries/a.q27m", &ea));
  CHECK(ea.written.size() == 3);
  CHECK(export_entry(b, idx, dir + "/entries/b.q27m", &eb));
  CHECK(eb.written.size() == 1);  // only b's own third chunk: the first two exist (verified) and are shared
  CHECK(export_entry(c, idx, dir + "/entries/c.q27m", &ec));
  CHECK(ec.written.empty());  // the 2 chunks it has are both on disk already
  CHECK(idx.files() == 4);
  CHECK(idx.find(ea.names[0])->refs == 3 && idx.find(ea.names[1])->refs == 3);
  CHECK(idx.find(ea.names[2])->refs == 1 && idx.find(eb.names[2])->refs == 1);
  CHECK(ea.names[2] != eb.names[2]);
  const uint64_t cb = chunk_file_bytes(a.g, a.chunk_tokens());
  CHECK(idx.physical_bytes() == 4 * cb);

  // every entry reads back correctly through the shared files
  auto restore = [&](const Entry& e, const Exported& x) {
    HostKv dst = empty_kv(e.linear.size());
    std::vector<char> state;
    DiskRead d;
    plan_read(d, e, idx, x.manifest, x.names, &state, &dst);
    return d.run_sync() && same(e, state, dst);
  };
  CHECK(restore(a, ea) && restore(b, eb) && restore(c, ec));

  // drop a (manifest first, then the refs): its own chunk goes, the shared ones stay
  fs::remove(ea.manifest);
  uint64_t freed = 0;
  for (const auto& n : ea.names) freed += idx.unref(n);
  CHECK(freed == cb);
  CHECK(!fs::exists(idx.path(ea.names[2])) && fs::exists(idx.path(ea.names[0])) && fs::exists(idx.path(ea.names[1])));
  CHECK(idx.find(ea.names[0])->refs == 2);
  CHECK(restore(b, eb) && restore(c, ec));
  // drop b: still c holds the prefix
  fs::remove(eb.manifest);
  for (const auto& n : eb.names) idx.unref(n);
  CHECK(idx.files() == 2 && restore(c, ec));
  // drop c: nothing is left
  fs::remove(ec.manifest);
  for (const auto& n : ec.names) idx.unref(n);
  CHECK(idx.files() == 0 && idx.size() == 0);
  CHECK(fs::is_empty(dir + "/chunks"));
  CHECK(idx.unref("nonexistent") == 0);  // (tolerated)

  // an export that fails after writing chunks removes the ones nobody references, and only those
  CHECK(export_entry(a, idx, dir + "/entries/a.q27m", &ea));
  Entry a2 = a;  // same chunks, but a manifest that cannot be written (its directory is a file)
  {
    std::FILE* f = std::fopen((dir + "/blocker").c_str(), "wb");
    std::fclose(f);
  }
  Exported bad;
  CHECK(!export_entry(b, idx, dir + "/blocker/x.q27m", &bad));  // (its manifest's directory cannot be made)
  (void)a2;
  CHECK(idx.find(ea.names[0])->refs == 1);  // a's chunks untouched
  for (const auto& n : bad.written) CHECK(!idx.find(n));
  fs::remove_all(dir);
}

static void test_crash_leftovers_and_sweep() {
  std::mt19937 rng(5);
  const std::string dir = tmpdir("spill_files");
  DiskIndex idx;
  idx.set_dir(dir + "/chunks");
  const Entry a = make_entry(rng, {}, 2, 64, 100);
  Exported ea;
  CHECK(export_entry(a, idx, dir + "/entries/a.q27m", &ea));
  // leftovers: a half-written chunk and manifest (.tmp), a chunk of an entry whose manifest was never written, an unrelated file
  { std::FILE* f = std::fopen((idx.dir() + "/deadbeef.kvc.tmp").c_str(), "wb"); std::fputs("x", f); std::fclose(f); }
  { std::FILE* f = std::fopen((idx.dir() + "/00-11.kvc").c_str(), "wb"); std::fputs("stray", f); std::fclose(f); }
  { std::FILE* f = std::fopen((idx.dir() + "/readme.txt").c_str(), "wb"); std::fclose(f); }
  // "restart": a fresh index built from the manifests (what spill_import does: names -> ref, files must exist with the right size)
  DiskIndex idx2;
  idx2.set_dir(idx.dir());
  ManifestHead head;
  CHECK(read_manifest_head(ea.manifest, head));
  for (const auto& n : head.names) {
    std::error_code ec;
    CHECK(fs::file_size(idx2.path(n), ec) == chunk_file_bytes(a.g, a.chunk_tokens()));
    if (!idx2.find(n)) idx2.add(n, chunk_file_bytes(a.g, a.chunk_tokens()));
    idx2.ref(n);
  }
  CHECK(idx2.sweep() == 2);  // the .tmp and the stray; the referenced chunks and the foreign file stay
  CHECK(fs::exists(idx.dir() + "/readme.txt") && !fs::exists(idx.dir() + "/00-11.kvc") && !fs::exists(idx.dir() + "/deadbeef.kvc.tmp"));
  CHECK(idx2.files() == 2 && fs::exists(idx2.path(head.names[0])));
  // an entry whose chunk went missing (a crash after the manifest, a disk cleaner): the structural import check fails
  fs::remove(idx2.path(head.names[1]));
  std::error_code ec;
  CHECK(fs::file_size(idx2.path(head.names[1]), ec) != chunk_file_bytes(a.g, a.chunk_tokens()));
  // ... and a restore of it fails cleanly (read error on the chunk, the chunk is named)
  {
    HostKv dst = empty_kv(a.linear.size());
    std::vector<char> state;
    DiskRead d;
    plan_read(d, a, idx2, ea.manifest, head.names, &state, &dst);
    CHECK(d.run_sync() == false && d.bad.load() == 1);
    idx2.forget_file(head.names[1]);  // (what the engine does with a chunk a read failed on: not served, the refs stay)
    CHECK(idx2.find(head.names[1]) && !idx2.find(head.names[1])->present && idx2.files() == 1);
    CHECK(idx2.physical_bytes() == chunk_file_bytes(a.g, a.chunk_tokens()));
  }
  // a damaged chunk that is re-exported (by an entry listing the same chunk) is rewritten, not skipped, and the refs carry on
  {
    Exported again;
    CHECK(idx2.find(head.names[1])->refs == 1);
    CHECK(export_entry(a, idx2, dir + "/entries/a2.q27m", &again));
    CHECK(again.written.size() == 1 && again.written[0] == head.names[1]);
    CHECK(idx2.find(head.names[1])->refs == 2 && idx2.find(head.names[1])->present);
    HostKv dst = empty_kv(a.linear.size());
    std::vector<char> state;
    DiskRead d;
    plan_read(d, a, idx2, ea.manifest, head.names, &state, &dst);
    CHECK(d.run_sync() && same(a, state, dst));
  }
  // an existing file with the right size but other contents is not "the chunk": the export does not skip it
  {
    const std::string name = head.names[0];
    flip(idx2.path(name), long(sizeof(ChunkHeader) + 8));  // a token id
    CHECK(!chunk_file_matches(idx2.path(name), a.fp, chunk_chain(a.fp, 4, a.ids.data(), 2, a.chunk_tokens())[0], "", a.ids.data(), a.chunk_tokens(), a.g));
  }
  fs::remove_all(dir);
}


// export_entry_files / validate_manifest are the functions Model::spill_write / spill_import call: their branches are exercised here.
static void test_export_import_branches() {
  std::mt19937 rng(6);
  const std::string dir = tmpdir("spill_files");
  DiskIndex idx;
  idx.set_dir(dir + "/chunks");
  const Entry a = make_entry(rng, {}, 3, 100, 321);
  const uint64_t cb = chunk_file_bytes(a.g, a.chunk_tokens());
  const auto chain = chunk_chain(a.fp, 4, a.ids.data(), 3, a.chunk_tokens());
  auto names_of = [&] { std::vector<std::string> v; for (auto& c : chain) v.push_back(c.name); return v; };

  // 1. a leftover chunk file from a crash (no manifest, not in the index) that is exactly the chunk is ADOPTED, not rewritten
  {
    fs::create_directories(idx.dir());
    HostKv kv = chunk_kv(a, 0);
    CPieces pieces;
    all_pieces(kv, pieces);
    CHECK(write_chunk_file(idx.path(chain[0].name), a.fp, chain[0], "", a.ids.data(), a.chunk_tokens(), a.g, pieces, nullptr));
    const auto mtime = fs::last_write_time(idx.path(chain[0].name));
    Exported ex;
    CHECK(export_entry(a, idx, dir + "/entries/a.q27m", &ex));
    CHECK(ex.written.size() == 3);  // (new to the index: chunk 0 is adopted from the leftover file, the other two are written)
    CHECK(fs::last_write_time(idx.path(chain[0].name)) == mtime);
    CHECK(idx.find(chain[0].name)->refs == 1 && idx.files() == 3);
  }
  // 2. replacing an entry's manifest (cur_path / cur_names): the new references come first, the old manifest file goes, nothing is lost
  {
    std::string cur = dir + "/entries/a.q27m";
    std::vector<std::string> cur_names = names_of();
    Exported ex;
    CHECK(export_entry(a, idx, dir + "/entries/a2.q27m", &ex, nullptr, &cur, &cur_names));
    CHECK(cur == dir + "/entries/a2.q27m" && !fs::exists(dir + "/entries/a.q27m") && fs::exists(cur));
    CHECK(ex.written.empty());
    for (const auto& n : cur_names) CHECK(idx.find(n)->refs == 1 && fs::exists(idx.path(n)));
    // (rewriting to the same path: the old and the new are one file, refs stay 1)
    Exported ex2;
    CHECK(export_entry(a, idx, cur, &ex2, nullptr, &cur, &cur_names));
    for (const auto& n : cur_names) CHECK(idx.find(n)->refs == 1);
    CHECK(fs::exists(cur));
  }
  // 3. a file that another manifest lists but that is not this chunk (damaged): the export fails, and touches nothing
  {
    const std::string victim = idx.path(chain[1].name);
    fs::copy_file(victim, victim + ".keep");
    flip(victim, long(sizeof(ChunkHeader) + 4 * 5));  // a token id
    Entry b = a;  // the same chunks, a new manifest
    Exported ex;
    std::string err;
    CHECK(!export_entry(b, idx, dir + "/entries/b.q27m", &ex, &err));
    CHECK(err.find("does not match") != std::string::npos);
    CHECK(!fs::exists(dir + "/entries/b.q27m") && !fs::exists(dir + "/entries/b.q27m.tmp"));
    for (const auto& n : names_of()) CHECK(idx.find(n)->refs == 1);
    fs::rename(victim + ".keep", victim);
    // ... and one nobody lists (refs 0, e.g. after a failed read's forget) is rewritten instead
    idx.forget_file(chain[1].name);
    idx.find(chain[1].name)->refs = 0;
    CHECK(export_entry(b, idx, dir + "/entries/b.q27m", &ex, &err));
    CHECK(ex.written == std::vector<std::string>{chain[1].name} && idx.find(chain[1].name)->present && idx.find(chain[1].name)->refs == 1);
    idx.find(chain[1].name)->refs += 1;  // (the reference of a's manifest, zeroed by hand above)
    fs::remove(dir + "/entries/b.q27m");
    for (const auto& n : names_of()) (void)idx.unref(n);  // (b's references go: a's remain)
  }
  // 4. an export that fails on its manifest (after writing chunk files) removes the chunk files it created, and only those
  {
    DiskIndex idx2;
    idx2.set_dir(dir + "/chunks2");
    {
      std::FILE* f = std::fopen((dir + "/blocker").c_str(), "wb");
      std::fclose(f);
    }
    Exported ex;
    CHECK(!export_entry(a, idx2, dir + "/blocker/x.q27m", &ex));
    CHECK(idx2.files() == 0 && fs::is_empty(dir + "/chunks2"));
    // manifest cannot be created (a directory in its place): chunk files were written first, then removed again
    fs::create_directories(dir + "/entries2/m.q27m");
    CHECK(!export_entry(a, idx2, dir + "/entries2/m.q27m", &ex));
    CHECK(idx2.files() == 0 && fs::is_empty(dir + "/chunks2"));
    // the same with chunk files that were already there (referenced by another manifest): they stay
    Exported ok1;
    CHECK(export_entry(a, idx2, dir + "/entries2/ok.q27m", &ok1));
    CHECK(!export_entry(a, idx2, dir + "/entries2/m.q27m", &ex));
    CHECK(idx2.files() == 3 && idx2.find(chain[0].name)->refs == 1);
  }
  // 5. validate_manifest: what spill_import rejects
  {
    DiskIndex fresh;
    fresh.set_dir(idx.dir());
    ManifestHead h;
    std::vector<std::string> unknown;
    const std::string man = dir + "/entries/a2.q27m";
    CHECK(validate_manifest(fresh, man, a.fp, a.g, a.state.size(), 1 << 20, h, unknown) && unknown.size() == 3);
    CHECK(!validate_manifest(fresh, man, a.fp + 1, a.g, a.state.size(), 1 << 20, h, unknown));       // another model
    Geom g2 = a.g;
    g2.page_bytes *= 2;
    CHECK(!validate_manifest(fresh, man, a.fp, g2, a.state.size(), 1 << 20, h, unknown));            // another KV type
    CHECK(!validate_manifest(fresh, man, a.fp, a.g, a.state.size() + 1, 1 << 20, h, unknown));       // another state size
    CHECK(!validate_manifest(fresh, man, a.fp, a.g, a.state.size(), 100, h, unknown));               // longer than max_tokens
    CHECK(!validate_manifest(fresh, dir + "/entries/none.q27m", a.fp, a.g, a.state.size(), 1 << 20, h, unknown));
    // a chunk file that is gone (a manifest referencing a missing chunk: the crash / cleaner case)
    fs::rename(idx.path(chain[2].name), idx.path(chain[2].name) + ".away");
    CHECK(!validate_manifest(fresh, man, a.fp, a.g, a.state.size(), 1 << 20, h, unknown));
    // ... or the wrong size
    fs::rename(idx.path(chain[2].name) + ".away", idx.path(chain[2].name));
    truncate_by(idx.path(chain[2].name), 8);
    CHECK(!validate_manifest(fresh, man, a.fp, a.g, a.state.size(), 1 << 20, h, unknown));
    // (the index trusts a chunk it already has: no stat)
    fresh.add(chain[2].name, cb);
    CHECK(validate_manifest(fresh, man, a.fp, a.g, a.state.size(), 1 << 20, h, unknown) && unknown.size() == 2);
    // a manifest whose names are not the chain of its tokens: write one with swapped names
    {
      ManifestHead good;
      CHECK(read_manifest_head(man, good));
      auto swapped = good.names;
      std::swap(swapped[0], swapped[1]);
      HostKv tail = tail_kv(a);
      CPieces tp;
      all_pieces(tail, tp);
      ManifestInfo mi;
      mi.fingerprint = a.fp, mi.g = a.g, mi.state_bytes = a.state.size(), mi.tail_bytes = tail.total;
      CHECK(write_manifest_file(dir + "/entries/swap.q27m", mi, a.ids, swapped, a.state.data(), tp, nullptr));
      DiskIndex f2;
      f2.set_dir(idx.dir());
      f2.add(chain[0].name, cb), f2.add(chain[1].name, cb), f2.add(chain[2].name, cb);
      CHECK(!validate_manifest(f2, dir + "/entries/swap.q27m", a.fp, a.g, a.state.size(), 1 << 20, h, unknown));
      // a tail whose size does not fit the token count
      mi.tail_bytes = 0;
      CHECK(write_manifest_file(dir + "/entries/notail.q27m", mi, a.ids, good.names, a.state.data(), {}, nullptr));
      CHECK(!validate_manifest(f2, dir + "/entries/notail.q27m", a.fp, a.g, a.state.size(), 1 << 20, h, unknown));
    }
  }
  fs::remove_all(dir);
}

// I/O errors are not verdicts on a file (review of the SSD dedup commits): a read that fails for another reason than "the file is not
// there" / "its contents are wrong" sets io_error and NOT `bad`, so the engine fails that restore only and never deletes the chunk
// file (chunk 0 is in nearly every manifest). A directory in place of the file opens fine on Linux and fails on the first fread
// (EISDIR) -- the same shape as an EIO or a lock that appears after the open. A removal that fails (Windows: a reader holds the file
// without FILE_SHARE_DELETE) goes to the pending-delete set: counted, retried, the file is not served meanwhile.
static void test_io_errors_and_pending_deletes() {
  std::mt19937 rng(11);
  const std::string dir = tmpdir("spill_files_io");
  DiskIndex idx;
  idx.set_dir(dir + "/chunks");
  const Entry a = make_entry(rng, {}, 2, 64, 100);
  Exported ea;
  CHECK(export_entry(a, idx, dir + "/entries/a.q27m", &ea));
  auto read = [&](const std::string& manifest, DiskRead& d, HostKv& dst, std::vector<char>& state) {
    dst = empty_kv(a.linear.size());
    plan_read(d, a, idx, manifest, ea.names, &state, &dst);
    return d.run_sync();
  };
  {  // a chunk file that cannot be read (I/O error): not `bad`, the file is untouched
    const std::string c1 = idx.path(ea.names[1]);
    fs::rename(c1, c1 + ".real");
    fs::create_directory(c1);
    HostKv dst;
    std::vector<char> state;
    DiskRead d;
    CHECK(!read(ea.manifest, d, dst, state));
    CHECK(d.io_error.load() && d.bad.load() == -1);
    fs::remove(c1);
    fs::rename(c1 + ".real", c1);
    DiskRead d2;
    CHECK(read(ea.manifest, d2, dst, state) && !d2.io_error.load() && same(a, state, dst));  // (nothing was lost)
  }
  {  // a manifest that cannot be read: io_error, bad stays -1; a missing / damaged chunk is still `bad` and not an I/O error
    fs::rename(ea.manifest, ea.manifest + ".real");
    fs::create_directory(ea.manifest);
    HostKv dst;
    std::vector<char> state;
    DiskRead d;
    CHECK(!read(ea.manifest, d, dst, state) && d.io_error.load() && d.bad.load() == -1);
    fs::remove(ea.manifest);
    fs::rename(ea.manifest + ".real", ea.manifest);
    const std::string c0 = idx.path(ea.names[0]);
    fs::rename(c0, c0 + ".away");
    DiskRead d2;
    CHECK(!read(ea.manifest, d2, dst, state) && !d2.io_error.load() && d2.bad.load() == 0);
    fs::rename(c0 + ".away", c0);
    truncate_by(c0, 1);
    DiskRead d3;
    CHECK(!read(ea.manifest, d3, dst, state) && !d3.io_error.load() && d3.bad.load() == 0);  // (a short read at EOF is damage)
    fs::copy_file(idx.path(ea.names[1]), c0, fs::copy_options::overwrite_existing);  // (restore something readable)
  }
  {  // a removal that fails: pending, counted, retried
    DiskIndex x;
    x.set_dir(dir + "/chunks2");
    fs::create_directories(x.dir());
    x.add("aa-bb", 1000);
    x.add("cc-dd", 500);
    x.ref("aa-bb");
    x.ref("cc-dd");
    // (a non-empty directory cannot be remove()d: stands for a file another process holds open)
    fs::create_directories(x.path("aa-bb") + "/held");
    x.forget_file("aa-bb");
    CHECK(x.find("aa-bb") && !x.find("aa-bb")->present);  // (not served)
    CHECK(x.pending_files() == 1 && x.physical_bytes() == 1500 && x.files() == 2);  // (still on the disk: still counted)
    CHECK(x.retry_pending() == 0 && x.pending_files() == 1);
    fs::remove_all(x.path("aa-bb"));
    CHECK(x.retry_pending() == 1 && x.pending_files() == 0 && x.physical_bytes() == 500);
    // the last reference goes while the file is held: the entry leaves the index, the file stays counted until it can be removed
    fs::create_directories(x.path("cc-dd") + "/held");
    CHECK(x.unref("cc-dd") == 500 && !x.find("cc-dd") && x.pending_files() == 1 && x.physical_bytes() == 500);
    CHECK(x.sweep() == 0 && x.pending_files() == 1);  // (the sweep retries too: still held)
    fs::remove_all(x.path("cc-dd"));
    CHECK(x.sweep() == 1 && x.pending_files() == 0 && x.physical_bytes() == 0);
    x.add("ee-ff", 10);  // (re-registered: no longer pending)
  }
  {  // a rename that fails is no failure when the file already there is exactly what was to be written
    const std::string target = dir + "/target.kvc";
    fs::create_directories(target + "/held");
    std::string err;
    auto fill = [&](std::FILE* f, Hash64& hh) { return write_hashed(f, "abc", 3, hh); };
    CHECK(!write_file_atomic(target, &err, fill) && !fs::exists(target + ".tmp"));
    CHECK(write_file_atomic(target, &err, fill, [] { return true; }) && !fs::exists(target + ".tmp"));
    CHECK(!write_file_atomic(target, &err, fill, [] { return false; }));
  }
  fs::remove_all(dir);
}

int main() {
  test_hash();
  test_names();
  test_roundtrip_and_damage();
  test_sharing_and_refcounts();
  test_crash_leftovers_and_sweep();
  test_export_import_branches();
  test_io_errors_and_pending_deletes();
  std::printf(fails ? "FAILED (%d)\n" : "ok\n", fails);
  return fails ? 1 : 0;
}
