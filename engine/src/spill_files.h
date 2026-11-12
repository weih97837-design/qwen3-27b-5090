// The SSD tier's files (docs/tiered-cache.md, spill.cc invariants (M)-(Q)). Pure C++ (no CUDA): tests/test_spill_files.cc and
// tests/test_spill_read.cc build it with g++ alone, so the formats, the checksums, the crash behaviour and the disk
// refcounts are tested on the real code.
//
// LAYOUT under SPILL_DIR:
//   chunks/<name>.kvc    one full kChunkPages-page chunk of KV (the RAM tier's dedup unit), shared by every entry whose prompt
//                        starts with the same tokens, written once.
//   entries/<id>.q27m    one manifest per entry: its tokens, recurrent state, the tail (the KV pages past the last full chunk)
//                        and the list of chunk names. Written LAST, after every chunk it lists is durable.
// Names are content-derived, never an in-memory uid (those do not survive a restart): <32 hex of the chunk's chain key>-<16
// hex of Hash64(parent name || the chunk's token ids)>. The chain key already covers the model / KV fingerprint and every
// token up to the chunk's end (chunk_store.h); the second part is an independent hash of the chunk's own tokens and parent
// NAME. A hit is verified, not trusted: a chunk file holds its own ids and its parent's name, so by induction through
// the manifest's list the whole prefix is verified (a 192-bit collision cannot give wrong KV; it fails the export / restore).
//
// FORMATS (little-endian, as the structs lie in memory on x86-64 / ARM64 Windows and Linux; every file ends with Hash64
// of all bytes before it):
//   .kvc   ChunkHeader, n_ids int32 token ids, the payload ([slice][page in chunk][page_bytes], exactly the RAM chunk's HostKv
//          bytes), u64 hash.
//   .q27m  ManifestHeader, n_ids int32, n_chunks x 64-byte nul-padded names, state, tail ([slice][tail page][page_bytes]), u64 hash.
//
// Writes are crash safe: <path>.tmp, fflush + fsync, rename, (fsync of the directory). A leftover .tmp is deleted at start;
// a chunk without a manifest ("stray") is deleted by DiskIndex::sweep once the index was built from the manifests.
//
// DISK REFCOUNTS (DiskIndex): a chunk file's refs = the number of manifests (entries with a path) that list it. The file
// is unlinked by unref() when that reaches 0 and by nothing else. The engine thread owns the index; the reader thread
// never touches it (it gets plain paths).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "chunk_store.h"
#include "host_kv.h"

namespace q27 {

// ---------------------------------------------------------------- Hash64: a streaming 64-bit hash (xxh64-style, own code)

// Word-at-a-time (4 lanes), several GB/s: the SSD restores checksum on the reader thread without becoming the bottleneck.
// Not bit-compatible with XXH64 (and not cryptographic): it detects damage, the stored ids verify identity. Any split of
// the input into update() calls gives the same digest.
class Hash64 {
 public:
  void update(const void* p, size_t n) {
    if (n == 0) return;
    const unsigned char* c = static_cast<const unsigned char*>(p);
    total_ += n;
    if (nbuf_ + n < 32) {
      std::memcpy(buf_ + nbuf_, c, n);
      nbuf_ += n;
      return;
    }
    if (nbuf_) {
      const size_t k = 32 - nbuf_;
      std::memcpy(buf_ + nbuf_, c, k);
      stripe(buf_);
      c += k, n -= k, nbuf_ = 0;
    }
    while (n >= 32) {
      stripe(c);
      c += 32, n -= 32;
    }
    if (n) std::memcpy(buf_, c, n), nbuf_ = n;
  }
  uint64_t digest() const {
    uint64_t h;
    if (total_ >= 32) {
      h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
      for (int i = 0; i < 4; i++) h = merge(h, v_[i]);
    } else {
      h = v_[2] + P5;  // (the seed is 0)
    }
    h += total_;
    const unsigned char* p = buf_;
    size_t r = nbuf_;
    for (; r >= 8; p += 8, r -= 8) {
      h ^= round(0, rd64(p));
      h = rotl(h, 27) * P1 + P4;
    }
    if (r >= 4) {
      uint32_t w;
      std::memcpy(&w, p, 4);
      h ^= uint64_t(w) * P1;
      h = rotl(h, 23) * P2 + P3;
      p += 4, r -= 4;
    }
    for (; r; p++, r--) {
      h ^= uint64_t(*p) * P5;
      h = rotl(h, 11) * P1;
    }
    h ^= h >> 33, h *= P2, h ^= h >> 29, h *= P3, h ^= h >> 32;
    return h;
  }

 private:
  static constexpr uint64_t P1 = 11400714785074694791ull, P2 = 14029467366897019727ull, P3 = 1609587929392839161ull,
                            P4 = 9650029242287828579ull, P5 = 2870177450012600261ull;
  static uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
  static uint64_t rd64(const unsigned char* p) {
    uint64_t w;
    std::memcpy(&w, p, 8);
    return w;
  }
  static uint64_t round(uint64_t acc, uint64_t in) { return rotl(acc + in * P2, 31) * P1; }
  static uint64_t merge(uint64_t acc, uint64_t v) { return (acc ^ round(0, v)) * P1 + P4; }
  void stripe(const unsigned char* p) {
    for (int i = 0; i < 4; i++) v_[i] = round(v_[i], rd64(p + 8 * i));
  }
  uint64_t v_[4] = {P1 + P2, P2, 0, 0 - P1};
  unsigned char buf_[32] = {};
  size_t nbuf_ = 0;
  uint64_t total_ = 0;
};

// ---------------------------------------------------------------- names

inline std::string hex64(uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

struct ChunkId {
  Key128 key;
  std::string name;
};

inline std::string chunk_file_name(const Key128& k, const std::string& parent, const int32_t* ids, size_t n) {
  Hash64 h;
  h.update(parent.data(), parent.size());
  h.update(ids, n * sizeof(int32_t));
  return hex64(k.a) + hex64(k.b) + "-" + hex64(h.digest());
}

// The chain of the n_chunks full chunks of ids (chunk_tokens each): the keys chunk_walk computes, plus the file names.
inline std::vector<ChunkId> chunk_chain(uint64_t fingerprint, int chunk_pages, const int32_t* ids, size_t n_chunks,
                                        size_t chunk_tokens) {
  std::vector<ChunkId> v;
  v.reserve(n_chunks);
  Key128 prev = chunk_seed(fingerprint, chunk_pages);
  for (size_t c = 0; c < n_chunks; c++) {
    const int32_t* p = ids + c * chunk_tokens;
    const Key128 key = chunk_next(prev, p, chunk_tokens);
    v.push_back({key, chunk_file_name(key, c ? v[c - 1].name : std::string(), p, chunk_tokens)});
    prev = key;
  }
  return v;
}

// ---------------------------------------------------------------- files: low level

using Pieces = std::vector<std::pair<char*, size_t>>;          // destinations (reader)
using CPieces = std::vector<std::pair<const char*, size_t>>;   // sources (writer)

// The pieces of every segment of kv, in order (a chunk's bytes, or an entry's tail: both are the file payload as is).
template <class P>
void all_pieces(const HostKv& kv, std::vector<std::pair<P, size_t>>& out) {
  for (size_t i = 0; i < kv.seg.size(); i++) out.emplace_back(kv.seg[i].get(), kv.len[i]);
}
// The pieces of pages [first, first + count) of every slice of a slice-major stream over n_pages pages (a private entry's
// linear kv): the payload of one chunk (first = c * chunk_pages) or of the tail, in file order.
template <class P>
void run_pieces(const HostKv& kv, int slices, size_t page_bytes, size_t n_pages, size_t first, size_t count,
                std::vector<std::pair<P, size_t>>& out) {
  for (int sl = 0; sl < slices; sl++)
    clip_run(kv, (size_t(sl) * n_pages + first) * page_bytes, count * page_bytes,
             [&](size_t, char* h, size_t n) { out.emplace_back(h, n); });
}
template <class P>
size_t pieces_bytes(const std::vector<std::pair<P, size_t>>& v) {
  size_t n = 0;
  for (const auto& p : v) n += p.second;
  return n;
}

inline bool sync_file(std::FILE* f) {
  if (std::fflush(f) != 0) return false;
#ifdef _WIN32
  return _commit(_fileno(f)) == 0;  // (FlushFileBuffers; not compiled in the Linux builds)
#else
  return ::fsync(::fileno(f)) == 0;
#endif
}
// Makes the renames in dir durable where the OS allows it (POSIX); Windows has no directory fsync: the file's own
// flush before the rename is what it offers.
inline void sync_dir(const std::string& dir) {
#ifndef _WIN32
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if (fd >= 0) {
    ::fsync(fd);
    ::close(fd);
  }
#else
  (void)dir;
#endif
}

struct FileCloser {
  std::FILE* f;
  ~FileCloser() {
    if (f) std::fclose(f);
  }
};

inline bool write_hashed(std::FILE* f, const void* p, size_t n, Hash64& h) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    const size_t k = std::min(n, size_t(32) << 20);
    if (std::fwrite(c, 1, k, f) != k) return false;
    h.update(c, k);
    c += k, n -= k;
  }
  return true;
}

// fread + hash, in blocks (the hash runs on the bytes just read: cache-warm); a set `cancel` stops between blocks.
inline bool read_hashed(std::FILE* f, void* p, size_t n, Hash64& h, const std::atomic<bool>* cancel,
                        size_t block = size_t(32) << 20) {
  char* c = static_cast<char*>(p);
  while (n) {
    if (cancel && cancel->load(std::memory_order_relaxed)) return false;
    const size_t k = std::min(n, block);
    if (std::fread(c, 1, k, f) != k) return false;
    h.update(c, k);
    c += k, n -= k;
  }
  return true;
}

// tmp -> rename: the shared tail of both writers. `fill(f, hash)` writes the content; the hash is appended.
template <class Fill>
bool write_file_atomic(const std::string& path, std::string* err, Fill&& fill) {
  const std::string tmp = path + ".tmp";
  bool ok;
  {
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
      if (err) *err = "cannot create " + tmp;
      return false;
    }
    FileCloser guard{f};
    Hash64 h;
    ok = fill(f, h);
    const uint64_t dig = h.digest();
    ok = ok && std::fwrite(&dig, 8, 1, f) == 1 && sync_file(f);
    guard.f = nullptr;
    ok = std::fclose(f) == 0 && ok;
  }
  if (!ok) {
    std::remove(tmp.c_str());
    if (err) *err = "could not write " + tmp;
    return false;
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::remove(tmp.c_str());
    if (err) *err = "could not rename " + tmp + ": " + ec.message();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- chunk files

inline constexpr char kChunkMagic[8] = {'Q', '2', '7', 'C', 'H', 'K', '1', '\0'};
inline constexpr char kManifestMagic[8] = {'Q', '2', '7', 'M', 'A', 'N', '1', '\0'};
inline constexpr uint32_t kFileVersion = 1;
inline constexpr size_t kNameField = 64;  // a name, nul padded (a name is 49 characters)

struct Geom {  // the KV geometry the files were written with (all of it is checked on read)
  uint32_t slices = 0, chunk_pages = 0, page_tokens = 0;
  uint64_t page_bytes = 0;
  uint64_t chunk_payload() const { return uint64_t(slices) * chunk_pages * page_bytes; }
  bool operator==(const Geom& o) const {
    return slices == o.slices && chunk_pages == o.chunk_pages && page_tokens == o.page_tokens && page_bytes == o.page_bytes;
  }
};

struct ChunkHeader {
  char magic[8];
  uint32_t version, slices;
  uint64_t fingerprint, key_a, key_b, page_bytes;
  uint32_t chunk_pages, n_ids, page_tokens, reserved;
  char parent[kNameField];  // the parent chunk's name ("" for the first chunk of a chain)
  uint64_t payload_bytes;
};
static_assert(sizeof(ChunkHeader) == 136, "ChunkHeader layout (no padding: it is compared bytewise)");

inline uint64_t chunk_file_bytes(const Geom& g, size_t n_ids) {
  return sizeof(ChunkHeader) + uint64_t(n_ids) * sizeof(int32_t) + g.chunk_payload() + 8;
}

inline ChunkHeader make_chunk_header(uint64_t fp, const ChunkId& id, const std::string& parent, size_t n_ids, const Geom& g) {
  ChunkHeader h{};
  std::memcpy(h.magic, kChunkMagic, sizeof kChunkMagic);
  h.version = kFileVersion, h.slices = g.slices;
  h.fingerprint = fp, h.key_a = id.key.a, h.key_b = id.key.b, h.page_bytes = g.page_bytes;
  h.chunk_pages = g.chunk_pages, h.n_ids = uint32_t(n_ids), h.page_tokens = g.page_tokens;
  std::memcpy(h.parent, parent.data(), std::min(parent.size(), kNameField - 1));
  h.payload_bytes = g.chunk_payload();
  return h;
}

inline bool chunk_header_ok(const ChunkHeader& h, const ChunkHeader& want) {
  return std::memcmp(&h, &want, sizeof h) == 0;  // (both zero-initialised: padding and the parent field compare too)
}

// payload: the chunk's bytes, exactly g.chunk_payload() in all. Written to <path>.tmp, synced, renamed.
inline bool write_chunk_file(const std::string& path, uint64_t fp, const ChunkId& id, const std::string& parent,
                             const int32_t* ids, size_t n_ids, const Geom& g, const CPieces& payload, std::string* err) {
  if (pieces_bytes(payload) != g.chunk_payload() || parent.size() >= kNameField || n_ids == 0) {
    if (err) *err = "chunk payload does not match its geometry";
    return false;
  }
  const ChunkHeader h = make_chunk_header(fp, id, parent, n_ids, g);
  return write_file_atomic(path, err, [&](std::FILE* f, Hash64& hh) {
    bool ok = write_hashed(f, &h, sizeof h, hh) && write_hashed(f, ids, n_ids * sizeof(int32_t), hh);
    for (const auto& [p, n] : payload) ok = ok && write_hashed(f, p, n, hh);
    return ok;
  });
}

// The cheap check before an export skips an existing chunk file: size, header and token ids (the payload's hash is only
// checked by a restore). false: not this chunk (or damaged).
inline bool chunk_file_matches(const std::string& path, uint64_t fp, const ChunkId& id, const std::string& parent,
                               const int32_t* ids, size_t n_ids, const Geom& g) {
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec || size != chunk_file_bytes(g, n_ids)) return false;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  FileCloser guard{f};
  ChunkHeader h{};
  if (!read_all(f, &h, sizeof h) || !chunk_header_ok(h, make_chunk_header(fp, id, parent, n_ids, g))) return false;
  std::vector<int32_t> got(n_ids);
  return read_all(f, got.data(), n_ids * sizeof(int32_t)) && std::equal(got.begin(), got.end(), ids);
}

// ---------------------------------------------------------------- manifests

struct ManifestHeader {
  char magic[8];
  uint32_t version, slices;
  uint64_t fingerprint, page_bytes;
  uint32_t page_tokens, chunk_pages, ring, n_ids;
  uint32_t n_chunks, reserved;
  uint64_t state_bytes, tail_bytes;
};
static_assert(sizeof(ManifestHeader) == 72, "ManifestHeader layout (no padding: it is compared bytewise)");

struct ManifestInfo {
  uint64_t fingerprint = 0;
  Geom g;
  bool ring = false;
  uint64_t state_bytes = 0, tail_bytes = 0;
};

inline uint64_t manifest_file_bytes(uint64_t state_bytes, uint64_t tail_bytes, size_t n_ids, size_t n_chunks) {
  return sizeof(ManifestHeader) + uint64_t(n_ids) * sizeof(int32_t) + uint64_t(n_chunks) * kNameField + state_bytes +
         tail_bytes + 8;
}

inline ManifestHeader make_manifest_header(const ManifestInfo& m, size_t n_ids, size_t n_chunks) {
  ManifestHeader h{};
  std::memcpy(h.magic, kManifestMagic, sizeof kManifestMagic);
  h.version = kFileVersion, h.slices = m.g.slices, h.fingerprint = m.fingerprint, h.page_bytes = m.g.page_bytes;
  h.page_tokens = m.g.page_tokens, h.chunk_pages = m.g.chunk_pages, h.ring = m.ring ? 1 : 0, h.n_ids = uint32_t(n_ids);
  h.n_chunks = uint32_t(n_chunks), h.state_bytes = m.state_bytes, h.tail_bytes = m.tail_bytes;
  return h;
}

inline void put_name(char (&field)[kNameField], const std::string& name) {
  std::memset(field, 0, kNameField);
  std::memcpy(field, name.data(), std::min(name.size(), kNameField - 1));
}
inline std::string get_name(const char (&field)[kNameField]) {
  size_t n = 0;
  while (n < kNameField && field[n]) n++;
  return std::string(field, n);
}

// Written after every chunk it lists is durable (the caller's order). tail: exactly m.tail_bytes in all.
inline bool write_manifest_file(const std::string& path, const ManifestInfo& m, const std::vector<int32_t>& ids,
                                const std::vector<std::string>& names, const char* state, const CPieces& tail,
                                std::string* err) {
  if (pieces_bytes(tail) != m.tail_bytes || ids.empty()) {
    if (err) *err = "manifest tail does not match its size";
    return false;
  }
  for (const auto& n : names)
    if (n.empty() || n.size() >= kNameField) {
      if (err) *err = "bad chunk name";
      return false;
    }
  const ManifestHeader h = make_manifest_header(m, ids.size(), names.size());
  return write_file_atomic(path, err, [&](std::FILE* f, Hash64& hh) {
    bool ok = write_hashed(f, &h, sizeof h, hh) && write_hashed(f, ids.data(), ids.size() * sizeof(int32_t), hh);
    for (const auto& n : names) {
      char field[kNameField];
      put_name(field, n);
      ok = ok && write_hashed(f, field, sizeof field, hh);
    }
    ok = ok && write_hashed(f, state, m.state_bytes, hh);
    for (const auto& [p, n] : tail) ok = ok && write_hashed(f, p, n, hh);
    return ok;
  });
}

struct ManifestHead {
  ManifestInfo info;
  std::vector<int32_t> ids;
  std::vector<std::string> names;
  uint64_t file_bytes = 0;
};

// The structure of a manifest without its state / tail (what start-up needs): magic, version, and a file size that is
// exactly header + ids + names + state + tail + hash. The hash itself is checked by a restore.
inline bool read_manifest_head(const std::string& path, ManifestHead& out) {
  std::error_code ec;
  const uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) return false;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  FileCloser guard{f};
  ManifestHeader h{};
  if (!read_all(f, &h, sizeof h) || std::memcmp(h.magic, kManifestMagic, sizeof kManifestMagic) != 0 ||
      h.version != kFileVersion || h.n_ids == 0 || h.n_ids > (1u << 28) || h.n_chunks > h.n_ids)
    return false;
  if (size != manifest_file_bytes(h.state_bytes, h.tail_bytes, h.n_ids, h.n_chunks)) return false;
  out.info.fingerprint = h.fingerprint;
  out.info.g = {h.slices, h.chunk_pages, h.page_tokens, h.page_bytes};
  out.info.ring = h.ring != 0;
  out.info.state_bytes = h.state_bytes, out.info.tail_bytes = h.tail_bytes;
  out.file_bytes = size;
  out.ids.resize(h.n_ids);
  if (!read_all(f, out.ids.data(), out.ids.size() * sizeof(int32_t))) return false;
  out.names.clear();
  for (uint32_t i = 0; i < h.n_chunks; i++) {
    char field[kNameField];
    if (!read_all(f, field, sizeof field)) return false;
    out.names.push_back(get_name(field));
    if (out.names.back().empty()) return false;
  }
  return true;
}

// ---------------------------------------------------------------- the background reader

// SPILL_ASYNC_PROMOTE: reads an entry's manifest and chunk files into buffers the engine thread prepared, on a background
// thread, so the engine keeps stepping (run_sync() does the same on the calling thread: spill_load). THREADING CONTRACT: the
// thread touches only this object (a copy of everything it needs: paths, expected ids / names / geometry), the buffers it
// was given (`state`, `tail`, `chunk[i]`: pieces of a HostKv that nobody else reads or frees meanwhile, as the entry is busy)
// and the files. It never touches Model, the spill map, the page pool, the PinnedPool bookkeeping, the DiskIndex or CUDA.
// It ends by storing `st` (release); the engine thread reads it with acquire, and only then joins the thread and uses the
// bytes. The destructor sets `cancel` (the thread stops between blocks) and joins, so a Model can be destroyed while one
// runs. Not movable (the thread holds `this`): keep it in a unique_ptr.
// Every file is verified: magic, version, geometry, fingerprint, the tokens (and for a chunk its key and parent's name), sizes
// and the Hash64 trailer. `bad`: the index of the chunk file that failed (-1: the manifest, or a cancel).
struct DiskRead {
  static constexpr size_t kBlock = size_t(32) << 20;
  // inputs, set before start() and read-only afterwards
  std::string manifest;
  std::vector<std::string> chunk_paths;           // chunk i's file
  ManifestInfo info;                              // the expectation (fingerprint, geometry, ring, state / tail sizes)
  std::vector<int32_t> ids;                       // the tokens the manifest must hold
  std::vector<std::string> names;                 // the chunk names it must list, in order
  char* state = nullptr;                          // info.state_bytes bytes
  Pieces tail;                                    // the tail payload, in file order (sum == info.tail_bytes)
  std::vector<Pieces> chunk;                      // chunk i's payload, in file order (each sum == g.chunk_payload())
  // outputs
  std::atomic<int> st{0};                         // 0 running, 1 ok, -1 failed
  std::atomic<bool> cancel{false};
  std::atomic<int64_t> ns{0};                     // wall time of the read
  std::atomic<int> bad{-1};
  std::thread th;

  DiskRead() = default;
  DiskRead(const DiskRead&) = delete;
  DiskRead& operator=(const DiskRead&) = delete;
  ~DiskRead() {
    cancel.store(true);
    if (th.joinable()) th.join();
  }
  void start() {
    th = std::thread([this] { finish(run_inner()); });
  }
  // On the calling thread (the synchronous restore).
  bool run_sync() {
    const bool ok = finish(run_inner());
    return ok;
  }
  // The engine thread, once st != 0 (acquire): the thread is over.
  void join() {
    if (th.joinable()) th.join();
  }
  int state_now() const { return st.load(std::memory_order_acquire); }

 private:
  bool finish(bool ok) {
    st.store(ok ? 1 : -1, std::memory_order_release);
    return ok;
  }
  bool run_inner() {
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = false;
    try {
      ok = read_manifest() && read_chunks();
    } catch (...) {
      ok = false;
    }
    ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
             std::memory_order_relaxed);
    return ok;
  }
  bool read_trailer(std::FILE* f, const Hash64& h) {
    uint64_t want = 0;
    if (!read_all(f, &want, 8, &cancel, kBlock) || want != h.digest()) return false;
    return std::fgetc(f) == EOF;  // (nothing after the hash)
  }
  bool read_manifest() {
    if (names.size() != chunk_paths.size() || names.size() != chunk.size() || pieces_bytes(tail) != info.tail_bytes) return false;
    std::FILE* f = std::fopen(manifest.c_str(), "rb");
    if (!f) return false;
    FileCloser guard{f};
    Hash64 hh;
    ManifestHeader h{};
    const ManifestHeader want = make_manifest_header(info, ids.size(), names.size());
    if (!read_hashed(f, &h, sizeof h, hh, &cancel, kBlock) || std::memcmp(&h, &want, sizeof h) != 0) return false;
    std::vector<int32_t> got(ids.size());
    if (!read_hashed(f, got.data(), got.size() * sizeof(int32_t), hh, &cancel, kBlock) || got != ids) return false;
    for (const auto& n : names) {
      char field[kNameField], want[kNameField];
      put_name(want, n);
      if (!read_hashed(f, field, sizeof field, hh, &cancel, kBlock) || std::memcmp(field, want, kNameField) != 0) return false;
    }
    if (!read_hashed(f, state, info.state_bytes, hh, &cancel, kBlock)) return false;
    for (const auto& [p, n] : tail)
      if (!read_hashed(f, p, n, hh, &cancel, kBlock)) return false;
    return read_trailer(f, hh);
  }
  bool read_chunks() {
    const size_t ct = size_t(info.g.chunk_pages) * info.g.page_tokens;
    const std::vector<ChunkId> chain = chunk_chain(info.fingerprint, int(info.g.chunk_pages), ids.data(), names.size(), ct);
    for (size_t i = 0; i < names.size(); i++) {
      if (cancel.load(std::memory_order_relaxed)) return false;
      if (chain[i].name != names[i] || pieces_bytes(chunk[i]) != info.g.chunk_payload() || (i + 1) * ct > ids.size()) {
        bad.store(int(i));
        return false;
      }
      if (!read_chunk(i, chain[i], ct)) {
        if (!cancel.load(std::memory_order_relaxed)) bad.store(int(i));
        return false;
      }
    }
    return true;
  }
  bool read_chunk(size_t i, const ChunkId& id, size_t ct) {
    std::FILE* f = std::fopen(chunk_paths[i].c_str(), "rb");
    if (!f) return false;
    FileCloser guard{f};
    Hash64 hh;
    ChunkHeader h{};
    const std::string parent = i ? names[i - 1] : std::string();
    if (!read_hashed(f, &h, sizeof h, hh, &cancel, kBlock) ||
        !chunk_header_ok(h, make_chunk_header(info.fingerprint, id, parent, ct, info.g)))
      return false;
    std::vector<int32_t> got(ct);
    if (!read_hashed(f, got.data(), ct * sizeof(int32_t), hh, &cancel, kBlock) || !std::equal(got.begin(), got.end(), ids.begin() + std::ptrdiff_t(i * ct)))
      return false;
    for (const auto& [p, n] : chunk[i])
      if (!read_hashed(f, p, n, hh, &cancel, kBlock)) return false;
    return read_trailer(f, hh);
  }
};

// ---------------------------------------------------------------- the on-disk chunk index

class DiskIndex {
 public:
  struct Chunk {
    uint64_t bytes = 0;    // the file's size
    int refs = 0;          // manifests (entries with a path) that list it
    bool present = false;  // the file is believed to exist (false after a failed read removed it)
  };

  void set_dir(const std::string& d) { dir_ = d; }
  const std::string& dir() const { return dir_; }
  std::string path(const std::string& name) const { return (std::filesystem::path(dir_) / (name + ".kvc")).string(); }
  Chunk* find(const std::string& name) {
    const auto it = m_.find(name);
    return it == m_.end() ? nullptr : &it->second;
  }
  const Chunk* find(const std::string& name) const {
    const auto it = m_.find(name);
    return it == m_.end() ? nullptr : &it->second;
  }
  void reserve(size_t extra) { m_.reserve(m_.size() + extra); }  // (may throw: before any file is written)
  // Registers a chunk file that exists (refs are the manifests', through ref()).
  Chunk& add(const std::string& name, uint64_t bytes) {
    Chunk& c = m_[name];
    c.bytes = bytes, c.present = true;
    return c;
  }
  void ref(const std::string& name) {
    if (Chunk* c = find(name)) c->refs++;
  }
  // A manifest that listed the chunk is gone: at 0 refs the file is unlinked. Returns the bytes freed. (An unknown name,
  // e.g. one a failed read forgot, is nothing.)
  uint64_t unref(const std::string& name) {
    const auto it = m_.find(name);
    if (it == m_.end()) return 0;
    if (--it->second.refs > 0) return 0;
    return erase(it);
  }
  // A file this process wrote that no manifest took (the export failed): gone unless somebody references it.
  uint64_t drop_unreferenced(const std::string& name) {
    const auto it = m_.find(name);
    return it == m_.end() || it->second.refs > 0 ? 0 : erase(it);
  }
  // A read found the file damaged or missing: it is deleted and not served again; the entries that list it still hold their
  // reference (their restore fails and drops them), a later export writes it afresh.
  void forget_file(const std::string& name) {
    Chunk* c = find(name);
    if (!c || !c->present) return;
    std::error_code ec;
    std::filesystem::remove(path(name), ec);
    c->present = false;
  }
  uint64_t physical_bytes() const {
    uint64_t n = 0;
    for (const auto& kv : m_) n += kv.second.present ? kv.second.bytes : 0;
    return n;
  }
  size_t files() const {
    size_t n = 0;
    for (const auto& kv : m_) n += kv.second.present;
    return n;
  }
  size_t size() const { return m_.size(); }
  // After the index was built from every manifest: removes leftover .tmp files and chunk files nobody lists. Returns how many.
  size_t sweep() {
    size_t n = 0;
    std::error_code ec;
    std::vector<std::filesystem::path> gone;
    for (std::filesystem::directory_iterator it(dir_, ec), end; !ec && it != end; it.increment(ec)) {
      std::error_code e2;
      if (!it->is_regular_file(e2) || e2) continue;
      const std::string ext = it->path().extension().string();
      if (ext == ".tmp") {
        gone.push_back(it->path());
      } else if (ext == ".kvc") {
        const Chunk* c = find(it->path().stem().string());
        if (!c || c->refs <= 0) gone.push_back(it->path());
      }
    }
    for (const auto& p : gone) {
      std::error_code e3;
      if (std::filesystem::remove(p, e3)) n++;
      if (p.extension() == ".kvc") m_.erase(p.stem().string());
    }
    return n;
  }

 private:
  uint64_t erase(std::unordered_map<std::string, Chunk>::iterator it) {
    const uint64_t freed = it->second.present ? it->second.bytes : 0;
    if (it->second.present) {
      std::error_code ec;
      std::filesystem::remove(path(it->first), ec);
    }
    m_.erase(it);
    return freed;
  }
  std::string dir_;
  std::unordered_map<std::string, Chunk> m_;
};

// ---------------------------------------------------------------- export / import of an entry (spill.cc's spill_write / spill_import)

// The whole file sequence of Model::spill_write, in the order of spill.cc (N): every missing chunk file (an existing one is VERIFIED
// before it is skipped; a leftover file nobody lists, e.g. from a crash, is adopted when it is exactly the chunk; a file another manifest
// lists that is NOT the chunk is never overwritten: it throws), fsync of the chunk directory, THEN the manifest, then the commit: the
// new disk references first, then the old manifest's (`cur_path` / `cur_names`: the entry's existing manifest, if any; replaced here).
// kv_of(i) -> const HostKv& (chunk i's bytes). tail: the entry's own KV past its n_chunks full chunks. Throws std::runtime_error with
// every file this call created that nobody lists removed again; nothing is committed then. After the commit nothing throws.
template <class KvOf>
uint64_t export_entry_files(DiskIndex& idx, uint64_t fp, const Geom& g, const std::string& path, bool ring,
                            const std::vector<int32_t>& ids, const char* state, uint64_t state_bytes, const HostKv& tail,
                            size_t n_chunks, KvOf&& kv_of, std::string& cur_path, std::vector<std::string>& cur_names) {
  const size_t ct = size_t(g.chunk_pages) * g.page_tokens;
  if (n_chunks * ct > ids.size()) throw std::runtime_error("more chunks than tokens");
  {
    std::error_code ec;
    std::filesystem::create_directories(idx.dir(), ec);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    if (ec) throw std::runtime_error("could not create the cache directories: " + ec.message());
  }
  const std::vector<ChunkId> chain = chunk_chain(fp, int(g.chunk_pages), ids.data(), n_chunks, ct);
  std::vector<std::string> names, fresh;
  names.reserve(n_chunks), fresh.reserve(n_chunks);
  idx.reserve(n_chunks);          // (before any file is written)
  std::string new_path = path;    // (copied now: the commit below must not throw)
  struct Cleanup {                // a failure removes the chunk files this call wrote that no manifest lists
    DiskIndex& idx;
    std::vector<std::string>& fresh;
    bool armed = true;
    ~Cleanup() {
      if (armed)
        for (const std::string& n : fresh) idx.drop_unreferenced(n);
    }
  } cleanup{idx, fresh};
  for (size_t i = 0; i < n_chunks; i++) {
    const std::string& name = chain[i].name;
    const std::string parent = i ? chain[i - 1].name : std::string();
    const int32_t* cid = ids.data() + i * ct;
    names.push_back(name);
    const DiskIndex::Chunk* have = idx.find(name);
    if ((!have || have->present) && chunk_file_matches(idx.path(name), fp, chain[i], parent, cid, ct, g)) {
      if (!have) {  // (a file a crash left, with no manifest: adopted)
        idx.add(name, chunk_file_bytes(g, ct));
        fresh.push_back(name);
      }
      continue;
    }
    if (have && have->present && have->refs > 0)  // another entry's manifest lists a file that is not this chunk: never overwritten
      throw std::runtime_error("a chunk file on the SSD does not match its name: " + name);
    CPieces pieces;
    all_pieces(kv_of(i), pieces);
    std::string err;
    if (!write_chunk_file(idx.path(name), fp, chain[i], parent, cid, ct, g, pieces, &err)) throw std::runtime_error(err);
    idx.add(name, chunk_file_bytes(g, ct));
    fresh.push_back(name);
  }
  if (!fresh.empty()) sync_dir(idx.dir());
  CPieces tp;
  all_pieces(tail, tp);
  ManifestInfo mi;
  mi.fingerprint = fp, mi.g = g, mi.ring = ring, mi.state_bytes = state_bytes, mi.tail_bytes = tail.total;
  std::string err;
  if (!write_manifest_file(path, mi, ids, names, state, tp, &err)) throw std::runtime_error(err);
  sync_dir(std::filesystem::path(path).parent_path().string());
  std::error_code ec;
  uint64_t mbytes = std::filesystem::file_size(path, ec);
  if (ec) mbytes = manifest_file_bytes(state_bytes, tail.total, ids.size(), n_chunks);
  cleanup.armed = false;
  // Commit: nothing below throws.
  for (const std::string& n : names) idx.ref(n);
  const std::string old_path = std::move(cur_path);
  const std::vector<std::string> old_names = std::move(cur_names);
  for (const std::string& n : old_names) idx.unref(n);
  if (!old_path.empty() && old_path != new_path) std::filesystem::remove(old_path, ec);
  cur_path = std::move(new_path);
  cur_names = std::move(names);
  return mbytes;
}

// spill_import's checks: the manifest at `path` is structurally sound, was written for this model / KV type (fingerprint, geometry, state
// size), its tail size fits its token count, its chunk names are the chain of its OWN tokens, and every chunk file it lists is
// there with exactly the size it must have. `unknown`: the listed files that are not in the index yet (the caller adds them, then
// refs every name). false: not importable (the caller deletes the manifest).
inline bool validate_manifest(DiskIndex& idx, const std::string& path, uint64_t fp, const Geom& g, uint64_t state_bytes,
                              size_t max_ids, ManifestHead& h, std::vector<std::string>& unknown) {
  if (!read_manifest_head(path, h)) return false;
  const size_t n_ids = h.ids.size(), nch = h.names.size(), ct = size_t(g.chunk_pages) * g.page_tokens;
  if (h.info.fingerprint != fp || !(h.info.g == g) || h.info.state_bytes != state_bytes || n_ids > max_ids || nch * ct > n_ids ||
      g.page_tokens == 0)
    return false;
  const size_t npages = (n_ids + g.page_tokens - 1) / g.page_tokens;
  if (h.info.tail_bytes != uint64_t(npages - nch * g.chunk_pages) * g.slices * g.page_bytes) return false;
  const std::vector<ChunkId> chain = chunk_chain(fp, int(g.chunk_pages), h.ids.data(), nch, ct);
  const uint64_t cb = chunk_file_bytes(g, ct);
  idx.reserve(nch);
  unknown.clear();
  for (size_t i = 0; i < nch; i++) {
    if (chain[i].name != h.names[i]) return false;
    const DiskIndex::Chunk* c = idx.find(h.names[i]);
    if (c && c->present) continue;
    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(idx.path(h.names[i]), ec);
    if (ec || size != cb) return false;
    unknown.push_back(h.names[i]);
  }
  return true;
}

}  // namespace q27
