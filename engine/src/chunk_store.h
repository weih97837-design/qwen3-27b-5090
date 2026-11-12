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
//    lanes. A hit is VERIFIED, not trusted: the chunk's stored ids must equal the requested ones AND its parent (the
//    previous chunk of the chain, by the store's unique serial `uid`, 0 for the first chunk) must be the chunk the walk
//    just hit / registered. By induction a hit has the same ids in every chunk up to it, i.e. the same whole prefix, so
//    a 128-bit collision can never give another prefix's KV (exact, not just "a miss").
//  * Collisions: the store is a multimap keyed by the nominal key; chunks that collide simply coexist under one key and
//    find() checks every candidate of that key (ids + parent uid). There is no probing, so removing a chunk never
//    leaves a hole that hides another one. A chunk whose parent was freed and re-registered has a new uid and is a
//    miss (never stale KV).
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
#include <new>
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
  Key128 key;                 // the nominal key it is registered under
  uint64_t uid = 0;           // unique among all chunks the store ever registered (> 0)
  uint64_t parent = 0;        // uid of the previous chunk of its chain (0: the first chunk)
  HostKv kv;                  // [kChunkPages-page layout]: slice-major within the chunk
  std::vector<int32_t> ids;   // the chunk's tokens (verification of a hit, with `parent`)
  int refs = 0;               // entries that list it (+ an entry under construction)
  Ev done{};                  // owner's: the copy that last wrote / read kv (null: none)
  bool inflight = false;      // owner's: `done` may not have completed yet
  size_t bytes() const { return kv.total; }
};

template <class Ev>
class ChunkStore {
 public:
  using C = Chunk<Ev>;

  // The chunk of nominal key `nominal` whose parent is `parent` (uid; 0 for the first chunk of a chain) and whose ids
  // equal ids[0, n) (a hit), else nullptr. Every chunk registered under the key is a candidate (a collision).
  C* find(const Key128& nominal, uint64_t parent, const int32_t* ids, size_t n) {
    auto r = m_.equal_range(nominal);
    for (auto it = r.first; it != r.second; ++it)
      if (it->second.parent == parent && it->second.ids.size() == n && std::equal(ids, ids + n, it->second.ids.begin()))
        return &it->second;
    return nullptr;
  }

  // Registers a new chunk (refs 0: an orphan until ref()) holding kv. Throws bad_alloc with nothing changed (kv stays
  // with the caller's HostKv only if the throw happens before the move: it is moved last).
  C* add(const Key128& key, uint64_t parent, const int32_t* ids, size_t n, HostKv&& kv) {
    C c;
    c.key = key;
    c.parent = parent;
    c.ids.assign(ids, ids + n);
    auto it = m_.emplace(key, std::move(c));  // (a node allocation: the only other throw)
    it->second.uid = ++next_uid_;
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
    auto r = m_.equal_range(c->key);
    auto it = r.first;
    while (it != r.second && &it->second != c) ++it;
    if (it == r.second) return C{};  // (not registered: nothing to do)
    if (c->refs <= 0) orphans_--;
    bytes_ -= c->kv.total;
    C out = std::move(it->second);
    m_.erase(it);
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
  std::multimap<Key128, C> m_;  // (node-based: a chunk never moves while it is registered)
  size_t bytes_ = 0, orphans_ = 0;
  uint64_t next_uid_ = 0;  // (never reused, not even after clear())
};

// Marks "the chain has no resident chunk here": a miss's children cannot hit (their parent does not exist).
inline constexpr uint64_t kNoParent = ~uint64_t(0);

// The key chain of the n_full chunks of ids (chunk_tokens each): fn(c, hit, key, parent_uid, ids_of_chunk) per chunk, hit
// being the verified resident chunk or nullptr. fn returns the chunk now at that position (the hit, or one it
// registered) or nullptr (a dry run): the next position's parent is that chunk, and after nullptr no later chunk can hit.
template <class Store, class Fn>
void chunk_walk(Store& st, Key128 seed, const int32_t* ids, size_t n_full, size_t chunk_tokens, Fn&& fn) {
  Key128 prev = seed;
  uint64_t parent = 0;
  for (size_t c = 0; c < n_full; c++) {
    const int32_t* p = ids + c * chunk_tokens;
    const Key128 key = chunk_next(prev, p, chunk_tokens);
    auto* hit = parent == kNoParent ? nullptr : st.find(key, parent, p, chunk_tokens);
    auto* cur = fn(c, hit, key, parent, p);
    parent = cur ? cur->uid : kNoParent;
    prev = key;
  }
}

// Bytes of the chunks of ids that are not resident (a dry run: nothing is registered).
template <class Store>
size_t chunk_missing_bytes(Store& st, Key128 seed, const int32_t* ids, size_t n_full, size_t chunk_tokens,
                           size_t chunk_bytes) {
  size_t miss = 0;
  chunk_walk(st, seed, ids, n_full, chunk_tokens,
             [&](size_t, typename Store::C* hit, const Key128&, uint64_t, const int32_t*) -> typename Store::C* {
               if (!hit) miss += chunk_bytes;
               return hit;
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
  chunk_walk(st, seed, ids, n_full, chunk_tokens,
             [&](size_t c, typename Store::C* hit, const Key128& key, uint64_t parent,
                 const int32_t* p) -> typename Store::C* {
               if (!hit) {
                 HostKv kv = alloc(chunk_bytes);
                 hit = st.add(key, parent, p, chunk_tokens, std::move(kv));
                 fresh.emplace_back(c, hit);  // (reserved: no throw from here)
               }
               st.ref(hit);
               refd.push_back(hit);
               return hit;
             });
}

// Frees a chunk that reached 0 refs, WITHOUT waiting for its copy (spill.cc invariant (I)). An inflight chunk (a copy that
// fills or reads it may still run) is parked in `retired` (a vector of {HostKv kv; Ev done; bool async}) with its own
// event, async = true, and the owner's reap frees it once the event completed. The capacity is reserved before the chunk
// leaves the store; if that fails (bad_alloc) the only way left is wait(chunk), the old synchronous fallback. A chunk
// that is not inflight is extracted and its memory goes back at once; put_event(done) recycles its (complete) event.
template <class Store, class Retired, class Wait, class PutEvent>
void chunk_free(Store& st, typename Store::C* c, std::vector<Retired>& retired, Wait&& wait, PutEvent&& put_event) {
  if (c->inflight) {
    bool reserved = true;
    try {
      retired.reserve(retired.size() + 1);
    } catch (const std::bad_alloc&) {
      reserved = false;
    }
    if (reserved) {
      typename Store::C out = st.extract(c);
      retired.push_back(Retired{std::move(out.kv), out.done, true});  // (reserved: no throw)
      return;
    }
    wait(c);
    c->inflight = false;
  }
  typename Store::C out = st.extract(c);
  put_event(out.done);  // (complete: not inflight)
}

// The entry's linear slice-major stream (the old linear file payload; now the layout of a private entry's kv: page i of slice sl at (sl * n + i) * page_bytes, n =
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
// in favour of one that gives real memory back, without always sacrificing the newest big entry). `state`: the part of
// `freed` that is the recurrent state alone (an entry whose KV is shared or pinned frees only that).
struct VictimCand {
  uint64_t last_use = 0;
  size_t freed = 0;
  size_t state = 0;
};
inline int pick_victim(const std::vector<VictimCand>& v) {
  size_t best = 0;
  for (const VictimCand& c : v) best = std::max(best, c.freed);
  int pick = -1;
  for (size_t i = 0; i < v.size(); i++)
    if (v[i].freed * 2 >= best && (pick < 0 || v[i].last_use < v[size_t(pick)].last_use)) pick = int(i);
  return pick;
}

// The two classes of victims of spill_enforce_ram: pass 0 the idle backups (their checkpoint is still in the pool, so
// dropping one loses nothing and writes nothing), pass 1 the sole copies (written to the SSD, or dropped). Pass 0 goes
// first (pick_victim inside it: most bytes freed first), EXCEPT when the best backup frees nothing beyond its state
// while the best sole copy frees at least 4x as much: then the sole copy goes (it lands on the SSD), instead of
// sacrificing backups one by one for the state bytes only. Every call removes one candidate from memory, so the loop
// that calls it ends.
struct VictimChoice {
  int pass = -1;  // 0 / 1; -1: no candidate
  int idx = -1;
};
inline VictimChoice choose_victim(const std::vector<VictimCand>& backups, const std::vector<VictimCand>& sole) {
  const int k0 = pick_victim(backups), k1 = pick_victim(sole);
  if (k0 >= 0) {
    const VictimCand& b = backups[size_t(k0)];
    size_t best_sole = 0;
    for (const VictimCand& c : sole) best_sole = std::max(best_sole, c.freed);
    if (k1 >= 0 && b.freed <= b.state && best_sole > 0 && best_sole >= 4 * b.freed) return {1, k1};
    return {0, k0};
  }
  return k1 >= 0 ? VictimChoice{1, k1} : VictimChoice{};
}

}  // namespace q27
