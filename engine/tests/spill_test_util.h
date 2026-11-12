// Shared helpers of the host tests of the SSD files (test_spill_files.cc, test_spill_read.cc): a synthetic entry (tokens, state,
// a linear slice-major KV stream), and the glue Model::spill_write / spill_load do around the header's functions (chunks
// first, the manifest last; a reader plan over a private linear buffer).
#pragma once
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>

#include "spill_files.h"

namespace q27test {
using namespace q27;
namespace fs = std::filesystem;

inline int fails = 0;
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                 \
      if (++fails > 20) std::exit(1);                                          \
    }                                                                          \
  } while (0)

constexpr uint64_t kFp = 0x1234abcd5678ef01ull;

struct Entry {
  Geom g;
  uint64_t fp = kFp;
  size_t n_chunks = 0, tail_pages = 0, tail_tokens = 0;
  std::vector<int32_t> ids;
  std::vector<char> state, linear;  // linear: [slice][page over n_pages][page_bytes]
  bool ring = false;
  size_t n_pages() const { return n_chunks * g.chunk_pages + tail_pages; }
  size_t chunk_tokens() const { return size_t(g.chunk_pages) * g.page_tokens; }
};

// prefix_ids: the first tokens (so that two entries can share a prefix); the rest random.
inline Entry make_entry(std::mt19937& rng, const std::vector<int32_t>& prefix_ids, size_t n_chunks, size_t tail_tokens,
                        size_t state_bytes, uint32_t slices = 3, uint32_t chunk_pages = 4, uint64_t page_bytes = 4096 * 3) {
  Entry e;
  e.g = {slices, chunk_pages, 64, page_bytes};
  e.n_chunks = n_chunks, e.tail_tokens = tail_tokens;
  e.tail_pages = (tail_tokens + 63) / 64;
  e.ids = prefix_ids;
  const size_t n_ids = n_chunks * e.chunk_tokens() + tail_tokens;
  while (e.ids.size() < n_ids) e.ids.push_back(int32_t(rng() % 100000));
  e.ids.resize(n_ids);
  e.state.resize(state_bytes);
  for (auto& c : e.state) c = char(rng());
  e.linear.resize(size_t(slices) * e.n_pages() * page_bytes);
  for (auto& c : e.linear) c = char(rng());
  return e;
}

inline HostKv wrap(const char* p, size_t n) {
  HostKv kv;
  if (n) {
    KvBuf b(new char[n], KvDel{});
    std::memcpy(b.get(), p, n);
    kv.push(std::move(b), n);
  }
  return kv;
}
inline HostKv empty_kv(size_t n) {
  HostKv kv;
  if (n) kv.push(KvBuf(new char[n](), KvDel{}), n);
  return kv;
}
// The chunk's own HostKv (the RAM tier's layout) and the tail's, cut out of the linear stream.
inline HostKv chunk_kv(const Entry& e, size_t c) {
  std::vector<char> v;
  for (uint32_t sl = 0; sl < e.g.slices; sl++) {
    const char* p = e.linear.data() + (sl * e.n_pages() + c * e.g.chunk_pages) * e.g.page_bytes;
    v.insert(v.end(), p, p + size_t(e.g.chunk_pages) * e.g.page_bytes);
  }
  return wrap(v.data(), v.size());
}
inline HostKv tail_kv(const Entry& e) {
  std::vector<char> v;
  for (uint32_t sl = 0; sl < e.g.slices; sl++) {
    const char* p = e.linear.data() + (sl * e.n_pages() + e.n_chunks * e.g.chunk_pages) * e.g.page_bytes;
    v.insert(v.end(), p, p + e.tail_pages * e.g.page_bytes);
  }
  return wrap(v.data(), v.size());
}

struct Exported {
  std::string manifest;
  std::vector<std::string> names;
  std::vector<std::string> written;  // chunk files this export wrote (the others were already there)
};

// Model::spill_write's file sequence (the real function: spill_files.h export_entry_files). cur_path / cur_names: the entry's existing
// manifest when it is being replaced. false: it threw (err), the index and the files are as the function leaves them.
inline bool export_entry(const Entry& e, DiskIndex& idx, const std::string& manifest, Exported* out, std::string* err = nullptr,
                         std::string* cur_path = nullptr, std::vector<std::string>* cur_names = nullptr, EntryMeta meta = {}) {
  std::string own_path;
  std::vector<std::string> own_names;
  std::string& cp = cur_path ? *cur_path : own_path;
  std::vector<std::string>& cn = cur_names ? *cur_names : own_names;
  const auto chain = chunk_chain(e.fp, int(e.g.chunk_pages), e.ids.data(), e.n_chunks, e.chunk_tokens());
  std::vector<bool> had;
  for (const auto& c : chain) had.push_back(idx.find(c.name) && idx.find(c.name)->present);
  std::vector<HostKv> kvs;
  for (size_t c = 0; c < e.n_chunks; c++) kvs.push_back(chunk_kv(e, c));
  HostKv tail = tail_kv(e);
  out->manifest = manifest;
  out->names.clear(), out->written.clear();
  try {
    export_entry_files(idx, e.fp, e.g, manifest, e.ring, e.ids, e.state.data(), e.state.size(), tail, e.n_chunks,
                       [&](size_t i) -> const HostKv& { return kvs[i]; }, cp, cn, meta);
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return false;
  }
  out->names = cn;
  for (size_t i = 0; i < chain.size(); i++)
    if (!had[i]) out->written.push_back(chain[i].name);
  return true;
}

// Fills a DiskRead for the entry's manifest; the destination is a private linear buffer (dst) of the entry's whole stream.
inline void plan_read(DiskRead& d, const Entry& e, const DiskIndex& idx, const std::string& manifest,
                      const std::vector<std::string>& names, std::vector<char>* state, HostKv* dst,
                      uint32_t manifest_version = kManifestVersion) {
  d.manifest = manifest;
  d.info.version = manifest_version;
  d.info.fingerprint = e.fp, d.info.g = e.g, d.info.ring = e.ring, d.info.state_bytes = e.state.size();
  d.info.tail_bytes = e.tail_pages * e.g.slices * e.g.page_bytes;
  d.ids = e.ids;
  d.names = names;
  state->assign(e.state.size(), 0);
  d.state = state->data();
  run_pieces(*dst, int(e.g.slices), e.g.page_bytes, e.n_pages(), e.n_chunks * e.g.chunk_pages, e.tail_pages, d.tail);
  d.chunk.resize(names.size());
  for (size_t c = 0; c < names.size(); c++) {
    d.chunk_paths.push_back(idx.path(names[c]));
    run_pieces(*dst, int(e.g.slices), e.g.page_bytes, e.n_pages(), c * e.g.chunk_pages, e.g.chunk_pages, d.chunk[c]);
  }
}

inline bool same(const Entry& e, const std::vector<char>& state, const HostKv& dst) {
  if (state != e.state) return false;
  size_t off = 0;
  for (size_t i = 0; i < dst.seg.size(); i++) {
    if (std::memcmp(dst.seg[i].get(), e.linear.data() + off, dst.len[i]) != 0) return false;
    off += dst.len[i];
  }
  return off == e.linear.size();
}

inline int wait_state(DiskRead& d) {
  while (d.state_now() == 0) std::this_thread::yield();  // (the engine thread polls; here it spins)
  d.join();
  return d.state_now();
}

inline void flip(const std::string& path, long off) {
  std::FILE* f = std::fopen(path.c_str(), "r+b");
  CHECK(f);
  if (off < 0) {
    std::fseek(f, 0, SEEK_END);
    off += std::ftell(f);
  }
  std::fseek(f, off, SEEK_SET);
  int c = std::fgetc(f);
  std::fseek(f, off, SEEK_SET);
  std::fputc(c ^ 0x55, f);
  std::fclose(f);
}
inline void truncate_by(const std::string& path, size_t cut) {
  fs::resize_file(path, fs::file_size(path) - cut);
}
inline std::string tmpdir(const char* tag) {
  std::string t = std::string("/tmp/") + tag + "_XXXXXX";
  std::vector<char> b(t.begin(), t.end());
  b.push_back(0);
  CHECK(mkdtemp(b.data()));
  return b.data();
}
}  // namespace q27test
