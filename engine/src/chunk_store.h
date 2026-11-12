// Content-addressed, reference-counted KV chunks for the RAM tier of the tiered cache (docs/tiered-cache.md,
// spill.cc "RAM dedup"). Pure C++ (no CUDA): tests/test_chunk_store.cc builds it with g++ alone.
//
// A spilled entry is the KV of tokens [0, pos). Entries of agents with one long system prompt, or several checkpoints
// of one conversation, share their first tokens, so storing each entry whole would keep those bytes several times.
// The full 4096-token (kChunkPages pages) blocks of an entry are therefore stored once, in a ChunkStore, and entries
// reference them; what is left (fewer than kChunkPages pages, up to and including the partial last page) is the
// entry's own "tail".
//
//  * Chunk layout: a chunk is its own HostKv, slice-major INSIDE the chunk: page j (of kChunkPages) of slice sl is at
//    (sl * kChunkPages + j) * page_bytes. (An entry's whole-stream layout is slice-major over ALL its pages, so a chunk
//    is not a contiguous slice of it; copy_runs (host_kv.h) is called once per chunk with that chunk's pages.)
//  * Key chain: key[0] = seed(fingerprint, chunk_pages); key[c] = H128(key[c - 1] || the token ids of chunk c). A
//    chunk's KV depends on every token before it, and so does its key. 128 bits come from two independent 64-bit
//    lanes; a hit is VERIFIED against the stored ids of that chunk, and the chain continues from the STORAGE key
//    (which differs from the nominal one after a collision), so a collision can only cost a miss, never wrong KV.
//  * Collisions: find() probes nominal, nominal + 1 (in b), ... until a key is absent (miss; that free key is where
//    the new chunk goes) or holds the same ids (hit).
//  * Refcounts are explicit: ref() / unref(), taken by the entries that list the chunk. A chunk with refs == 0 is an
//    "orphan": it stays resident (so a deeper checkpoint that replaces its shallower predecessor re-finds it) until
//    the owner reclaims it (extract). The store knows nothing of events or copies: `done` / `inflight` are the
//    owner's (Model::spill_wait, spill_chunk_unref).
//  * Everything here is on the engine thread (no locks).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "host_kv.h"

namespace q27 {

inline constexpr int kChunkPages = 64;  // pages per chunk (64 tokens each): 4096 tokens

struct Key128 {
  uint64_t a = 0, b = 0;
  bool operator==(const Key128& o) const { return a == o.a && b == o.b; }
  bool operator<(const Key128& o) const { return a != o.a ? a < o.a : b < o.b; }
};

// Two independent 64-bit multiply-xorshift lanes (portable: no __int128, so MSVC builds it too). Not cryptographic:
// the stored ids verify every hit.
class Hash128 {
 public:
  void u64(uint64_t w) {
    a_ = mix(a_ ^ w, 0x9E3779B97F4A7C15ull, 0xBF58476D1CE4E5B9ull, 29);
    b_ = mix(b_ + (w ^ 0xD6E8FEB86659FD93ull), 0xC2B2AE3D27D4EB4Full, 0x94D049BB133111EBull, 31) ;
    n_++;
  }
  void bytes(const void* p, size_t n) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; i++) u64(c[i] | (uint64_t(n - i) << 8));
  }
  // Token ids as little-endian 32-bit values, two per word (endianness-independent).
  void ids(const int32_t* p, size_t n) {
    size_t i = 0;
    for (; i + 1 < n; i += 2) u64(uint64_t(uint32_t(p[i])) | (uint64_t(uint32_t(p[i + 1])) << 32));
    if (i < n) u64(uint64_t(uint32_t(p[i])));
    u64(n);
  }
  Key128 finish() const {
    Key128 k;
    k.a = mix(a_ ^ n_, 0xFF51AFD7ED558CCDull, 0xC4CEB9FE1A85EC53ull, 33);
    k.b = mix(b_ + n_, 0x9FB21C651E98DF25ull, 0xD6E8FEB86659FD93ull, 32);
    return k;
  }

 private:
  static uint64_t mix(uint64_t x, uint64_t m1, uint64_t m2, int s) {
    x *= m1;
    x ^= x >> s;
    x *= m2;
    x ^= x >> (s + 3);
    return x;
  }
  uint64_t a_ = 0x243F6A8885A308D3ull, b_ = 0x13198A2E03707344ull, n_ = 0;
};

inline Key128 chunk_seed(uint64_t fingerprint, int chunk_pages) {
  Hash128 h;
  h.u64(fingerprint);
  h.bytes("q27chunk1", 9);
  h.u64(uint64_t(chunk_pages));
  return h.finish();
}

inline Key128 chunk_next(const Key128& prev, const int32_t* ids, size_t n) {
  Hash128 h;
  h.u64(prev.a);
  h.u64(prev.b);
  h.ids(ids, n);
  return h.finish();
}

template <class Ev>
struct Chunk {
  Key128 skey;                // the key it is stored under (find's probing may have moved it off the nominal one)
  HostKv kv;                  // [kChunkPages-page layout]: slice-major within the chunk
  std::vector<int32_t> ids;   // the chunk's tokens (verification of a hit)
  int refs = 0;               // entries that list it (+ an entry under construction)
  Ev done{};                  // owner's: the copy that last wrote / read kv (null: none)
  bool inflight = false;      // owner's: `done` may not have completed yet
  size_t bytes() const { return kv.total; }
};

template <class Ev>
class ChunkStore {
 public:
  using C = Chunk<Ev>;

  // The chunk of nominal key `nominal` whose ids equal ids[0, n) (a hit), else nullptr. *skey: where it is / where a
  // new one for this chain position goes.
  C* find(Key128 nominal, const int32_t* ids, size_t n, Key128* skey) {
    Key128 k = nominal;
    for (;;) {
      auto it = m_.find(k);
      if (it == m_.end()) {
        *skey = k;
        return nullptr;
      }
      if (it->second.ids.size() == n && std::equal(ids, ids + n, it->second.ids.begin())) {
        *skey = k;
        return &it->second;
      }
      k.b++;  // (a collision: the next probe)
    }
  }

  // Registers a new chunk (refs 0: an orphan until ref()) holding kv. Throws bad_alloc with nothing changed (kv stays
  // with the caller's HostKv only if the throw happens before the move: it is moved last).
  C* add(const Key128& skey, const int32_t* ids, size_t n, HostKv&& kv) {
    C c;
    c.skey = skey;
    c.ids.assign(ids, ids + n);
    auto it = m_.emplace(skey, std::move(c)).first;  // (a node allocation: the only other throw)
    it->second.kv = std::move(kv);
    bytes_ += it->second.kv.total;
    orphans_++;
    return &it->second;
  }

  void ref(C* c) {
    if (c->refs++ == 0) orphans_--;
  }
  // True when the count reached 0 (the chunk is an orphan now; the owner decides when it goes).
  bool unref(C* c) {
    if (--c->refs > 0) return false;
    orphans_++;
    return true;
  }

  // Removes the chunk (whatever its refs) and returns it: its HostKv is the caller's to free or park.
  C extract(C* c) {
    if (c->refs <= 0) orphans_--;
    bytes_ -= c->kv.total;
    C out = std::move(*c);
    m_.erase(out.skey);
    return out;
  }

  std::vector<C*> orphans() {
    std::vector<C*> v;
    if (orphans_ > 0)
      for (auto& kv : m_)
        if (kv.second.refs <= 0) v.push_back(&kv.second);
    return v;
  }
  template <class F>
  void for_each(F&& fn) {
    for (auto& kv : m_) fn(kv.second);
  }
  void clear() {
    m_.clear();
    bytes_ = orphans_ = 0;
  }
  size_t size() const { return m_.size(); }
  size_t bytes() const { return bytes_; }  // physical bytes of the resident chunks (orphans included)
  size_t orphan_count() const { return orphans_; }

 private:
  std::map<Key128, C> m_;  // (node-based: a chunk never moves while it is registered)
  size_t bytes_ = 0, orphans_ = 0;
};

// The key chain of the n_full chunks of ids (chunk_tokens each): fn(c, hit, skey, ids_of_chunk) per chunk, hit being the
// verified resident chunk or nullptr; false from fn stops. A miss continues from the key it would be stored under.
template <class Store, class Fn>
void chunk_walk(Store& st, Key128 seed, const int32_t* ids, size_t n_full, size_t chunk_tokens, Fn&& fn) {
  Key128 prev = seed;
  for (size_t c = 0; c < n_full; c++) {
    const int32_t* p = ids + c * chunk_tokens;
    Key128 skey;
    auto* hit = st.find(chunk_next(prev, p, chunk_tokens), p, chunk_tokens, &skey);
    if (!fn(c, hit, skey, p)) return;
    prev = skey;
  }
}

// Bytes of the chunks of ids that are not resident (a dry run: nothing is registered).
template <class Store>
size_t chunk_missing_bytes(Store& st, Key128 seed, const int32_t* ids, size_t n_full, size_t chunk_tokens,
                           size_t chunk_bytes) {
  size_t miss = 0;
  chunk_walk(st, seed, ids, n_full, chunk_tokens, [&](size_t, auto* hit, const Key128&, const int32_t*) {
    if (!hit) miss += chunk_bytes;
    return true;
  });
  return miss;
}

// Takes a reference on each of the n_full chunks of ids: resident ones as they are, the others registered with a
// fresh HostKv from alloc(chunk_bytes) (their bytes are NOT filled: fresh lists (index, chunk) for the caller's
// copies). `refd` receives every chunk referenced so far, so after a throw the caller unrefs exactly those. Capacity
// is reserved first: the only throws are alloc() and the store's node allocation, each before its chunk is listed.
template <class Store, class Alloc>
void chunk_acquire(Store& st, Key128 seed, const int32_t* ids, size_t n_full, size_t chunk_tokens, size_t chunk_bytes,
                   Alloc&& alloc, std::vector<typename Store::C*>& refd,
                   std::vector<std::pair<size_t, typename Store::C*>>& fresh) {
  refd.reserve(refd.size() + n_full);
  fresh.reserve(fresh.size() + n_full);
  chunk_walk(st, seed, ids, n_full, chunk_tokens, [&](size_t c, auto* hit, const Key128& skey, const int32_t* p) {
    if (!hit) {
      HostKv kv = alloc(chunk_bytes);
      hit = st.add(skey, p, chunk_tokens, std::move(kv));
      fresh.emplace_back(c, hit);  // (reserved: no throw from here)
    }
    st.ref(hit);
    refd.push_back(hit);
    return true;
  });
}

// The entry's linear slice-major stream (the .q27kv payload: page i of slice sl at (sl * n + i) * page_bytes, n =
// chunks * chunk_pages + tail_pages) from its chunks (each [slice][chunk_pages] inside) and tail ([slice][tail_pages]
// inside): fn(const char*, bytes) in file order. With no chunks this is the tail's stream unchanged.
template <class C, class F>
void linear_pieces(const std::vector<C*>& chunks, const HostKv& tail, int slices, size_t page_bytes, size_t chunk_pages,
                   size_t tail_pages, F&& fn) {
  for (int sl = 0; sl < slices; sl++) {
    for (const C* c : chunks)
      clip_run(c->kv, size_t(sl) * chunk_pages * page_bytes, chunk_pages * page_bytes,
               [&](size_t, char* h, size_t n) { fn(static_cast<const char*>(h), n); });
    if (tail_pages)
      clip_run(tail, size_t(sl) * tail_pages * page_bytes, tail_pages * page_bytes,
               [&](size_t, char* h, size_t n) { fn(static_cast<const char*>(h), n); });
  }
}

// Which entry spill_enforce_ram evicts: the one that frees the most host memory, the least recently used among those
// within a factor 2 of the best (so an entry whose chunks are shared with others, which frees almost nothing, is kept
// in favour of one that gives real memory back, without always sacrificing the newest big entry).
struct VictimCand {
  uint64_t last_use = 0;
  size_t freed = 0;
};
inline int pick_victim(const std::vector<VictimCand>& v) {
  size_t best = 0;
  for (const VictimCand& c : v) best = std::max(best, c.freed);
  int pick = -1;
  for (size_t i = 0; i < v.size(); i++)
    if (v[i].freed * 2 >= best && (pick < 0 || v[i].last_use < v[size_t(pick)].last_use)) pick = int(i);
  return pick;
}

}  // namespace q27
