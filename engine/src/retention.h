// Value-based retention of the tiered cache (docs/tiered-cache.md "Retention"): what is dropped first when RAM or the SSD is full.
// Pure C++ (no CUDA): tests/test_retention.cc builds it with g++ alone. frontend/tiercache.py implements the same order for the SSD
// (it only has the numbers the engine reports: spill_meta / spill_agents); frontend/tests/test_retention.py checks that the two orders
// agree on random candidates.
//
// ONE order for a tier (the victim is the minimum):
//   1. the agent that uses the MOST bytes in this tier (fractional attribution below; SPILL_FAIR_AGENTS=0 skips this step), then
//      within that agent
//   2. superseded entries first (an intermediate checkpoint of a conversation whose deeper checkpoint exists),
//   3. never-hit entries before entries that were restored at least once (hits),
//   4. the least recently used.
// (chunk_store.h pick_victim adds the old "frees the most host memory" refinement inside step 4.)
//
// Superseded (superseded_flags): the entry's tokens are a strict prefix of another entry's of the same chain, that other entry is
// usable in its place (covers(): its draft ring is valid where the shallow one's is), the entry is longer than kKeepShortTokens
// (shorter ones are the shared system prompts new conversations start from), and it is neither a BRANCH POINT (every entry that
// extends it lies on the one chain of the deepest of them) nor a SHARED prefix (the entry and its extenders belong to >= 2
// distinct agents). It is the logic of Model::ckpt_covered / spill_subsumed, run over the spilled entries (RAM and SSD) and the
// checkpoints in the pool (a deeper one that is still in VRAM supersedes the spilled shallow one too).
//
// Attribution (agent_use): an entry's own bytes (tail, recurrent state, manifest) belong to its agent; a chunk that entries of
// several agents reference is split between the DISTINCT agents referencing it (bytes / n), not between the entries. Agent 0 is
// "unknown" (its own bucket).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace q27 {

inline constexpr int kKeepShortTokens = 16384;

// Prefix tests without comparing whole token vectors: every node carries chained hashes of its full kBlockTokens-token blocks
// (block_hashes: h[k] = H(h[k-1], block k), computed once per entry / checkpoint, cached by the owner). Two nodes agree on their first
// k blocks iff their first k hashes are equal (64 bits: a collision would only mis-rank an entry, never lose data), and the tokens
// past the last full block (< kBlockTokens) are compared directly.
inline constexpr size_t kBlockTokens = 256;

inline std::vector<uint64_t> block_hashes(const std::vector<int32_t>& ids) {
  std::vector<uint64_t> h(ids.size() / kBlockTokens);
  uint64_t prev = 0x9E3779B97F4A7C15ull;
  for (size_t k = 0; k < h.size(); k++) {
    uint64_t x = prev;
    for (size_t i = k * kBlockTokens; i < (k + 1) * kBlockTokens; i++) {
      x ^= uint64_t(uint32_t(ids[i])) + 0x9E3779B97F4A7C15ull + (x << 6) + (x >> 2);
      x *= 0xFF51AFD7ED558CCDull;
    }
    prev = h[k] = x ^ (x >> 29);
  }
  return h;
}

struct RetNode {
  const std::vector<int32_t>* ids = nullptr;
  const std::vector<uint64_t>* bh = nullptr;  // block_hashes(*ids)
  bool ring = false;
  uint64_t owner = 0;
  bool durable = false;  // a manifest on the SSD exists (the node survives a restart); pool checkpoints and RAM-only entries are not
};

namespace ret_detail {
// a's tokens are a prefix of b's (a no longer than b).
inline bool prefix_of(const RetNode& a, const RetNode& b) {
  const size_t na = a.ids->size();
  if (na > b.ids->size()) return false;
  const size_t m = na / kBlockTokens;
  if (!std::equal(a.bh->begin(), a.bh->begin() + std::ptrdiff_t(m), b.bh->begin())) return false;
  return std::equal(a.ids->begin() + std::ptrdiff_t(m * kBlockTokens), a.ids->end(), b.ids->begin() + std::ptrdiff_t(m * kBlockTokens));
}
}  // namespace ret_detail

// flags[i] for i < n_check: whether nodes[i] is superseded by another node (see the top). Nodes past n_check only count as
// extenders. durable_only: the standing-in (deepest) extender must be durable (on the SSD); the branch / shared tests still look at
// every extender (conservative). A flag set this way means the conversation's prefix survives a restart without this entry. Cost: a sort by the block hashes (short arrays), then for each node the nodes that share its full blocks (adjacent in
// that order), each checked on the < kBlockTokens tokens past them.
inline void superseded_flags(const std::vector<RetNode>& nodes, size_t n_check, std::vector<char>& flags, bool durable_only = false) {
  flags.assign(n_check, 0);
  const size_t n = nodes.size();
  if (n < 2) return;
  std::vector<size_t> ord(n);
  for (size_t i = 0; i < n; i++) ord[i] = i;
  auto less = [&](size_t a, size_t b) {
    const auto& x = *nodes[a].bh;
    const auto& y = *nodes[b].bh;
    if (x == y) return a < b;
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  };
  std::sort(ord.begin(), ord.end(), less);
  std::vector<size_t> pos(n);  // position in `ord`
  for (size_t p = 0; p < n; p++) pos[ord[p]] = p;
  for (size_t i = 0; i < n_check; i++) {
    const RetNode& c = nodes[i];
    if (int(c.ids->size()) <= kKeepShortTokens) continue;
    const size_t m = c.ids->size() / kBlockTokens;
    // the group of nodes with c's exact block-hash array starts somewhere at or before c; the strict extenders are in it (a longer
    // tail) or after it (more blocks) while the first m hashes stay equal to c's
    size_t g = pos[i];
    while (g > 0 && *nodes[ord[g - 1]].bh == *c.bh) g--;
    const RetNode* deepest = nullptr;  // the longest extender that can stand in for c (its ring covers c's)
    std::vector<const RetNode*> ext;
    for (size_t q = g; q < n; q++) {
      const RetNode& o = nodes[ord[q]];
      if (o.bh->size() < m || !std::equal(c.bh->begin(), c.bh->begin() + std::ptrdiff_t(m), o.bh->begin())) break;
      if (&o == &c || o.ids->size() <= c.ids->size() || !ret_detail::prefix_of(c, o)) continue;  // (an equal copy, e.g. a live backup next to its pool checkpoint, is no extender)
      ext.push_back(&o);
      if (o.ring >= c.ring && (!durable_only || o.durable) && (!deepest || o.ids->size() > deepest->ids->size())) deepest = &o;
    }
    if (!deepest) continue;
    bool ok = true;
    bool multi = false;
    for (const RetNode* o : ext) {
      if (!ret_detail::prefix_of(*o, *deepest)) {  // a path that leaves the deepest one: c is a branch point
        ok = false;
        break;
      }
      if (o->owner != c.owner) multi = true;
    }
    flags[i] = ok && !multi;
  }
}

// The reuse count a sequence inherits from the checkpoint it resumed from (Model::prefix_begin): the checkpoints it saves later (the
// deeper ones that replace the matched one, which is then subsumed when it is evicted) carry it, so the tip of a conversation that
// is being reused keeps its hits. A match of at most kKeepShortTokens (a system prompt every new conversation starts from) does not
// propagate: a popular prefix must not make every conversation that merely began with it look reused.
inline uint32_t inherited_hits(int matched_tokens, uint32_t hits) { return matched_tokens > kKeepShortTokens ? hits : 0; }

// One entry's share of a tier: its own bytes, and the shared pieces (a chunk's identity and size) it references.
struct UseEntry {
  uint64_t owner = 0;
  double own = 0;
  std::vector<std::pair<const void*, double>> shared;
};

// Bytes per agent in the tier: own bytes whole, a shared piece / the number of DISTINCT agents referencing it.
inline std::map<uint64_t, double> agent_use(const std::vector<UseEntry>& entries) {
  struct Piece {
    double bytes = 0;
    std::vector<uint64_t> owners;
  };
  std::map<const void*, Piece> pieces;
  std::map<uint64_t, double> out;
  for (const UseEntry& e : entries) {
    out[e.owner] += e.own;
    for (const auto& [id, bytes] : e.shared) {
      Piece& p = pieces[id];
      p.bytes = bytes;
      if (std::find(p.owners.begin(), p.owners.end(), e.owner) == p.owners.end()) p.owners.push_back(e.owner);
    }
  }
  for (const auto& kv : pieces)
    for (uint64_t o : kv.second.owners) out[o] += kv.second.bytes / double(kv.second.owners.size());
  return out;
}

}  // namespace q27
