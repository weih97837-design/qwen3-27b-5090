// Value-based retention of the tiered cache (docs/tiered-cache.md "Retention"): what is dropped first when RAM or the SSD is full.
// Pure C++ (no CUDA): tests/test_retention.cc builds it with g++ alone. frontend/tiercache.py implements the same order for the SSD
// (it only has the numbers the engine reports: spill_meta / spill_agents); frontend/tests/test_retention.py checks that the two orders
// agree on random candidates.
//
// ONE order for a tier (the victim is the minimum):
//   0. entries of COLD agents first (SPILL_COLD_FIRST=0 skips this step; cold_agents() below). Among them step 1 picks the agent
//      using the most bytes, so "the cold agent with the most bytes" goes first, then steps 2-4 inside it,
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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <set>
#include <string>
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

// Wall-clock seconds for the age of an entry (a monotonic clock: only differences are used).
inline double ret_now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// COLD agents of a tier (SPILL_COLD_AFTER_S, SPILL_COLD_FIRST): an agent none of whose entries in the tier has hits > 0 and whose NEWEST
// entry there is at least `after` seconds old (born: when the entry came to the tier). A just-created conversation is not cold yet.
// Agent 0 is the "unknown" bucket, not an agent: never cold. `entries` is the whole tier (also entries that cannot be victims now:
// busy ones count for the agent's hits and age). frontend/tiercache.py _cold_owners is the same rule, from the times it keeps.
struct ColdEntry {
  uint64_t owner = 0;
  uint32_t hits = 0;
  double born = 0;
};
inline std::set<uint64_t> cold_agents(const std::vector<ColdEntry>& entries, double now, double after) {
  std::map<uint64_t, std::pair<bool, double>> a;  // owner -> (any hit, newest born)
  for (const ColdEntry& e : entries) {
    auto it = a.emplace(e.owner, std::make_pair(false, e.born)).first;
    it->second.first = it->second.first || e.hits > 0;
    it->second.second = std::max(it->second.second, e.born);
  }
  std::set<uint64_t> cold;
  for (const auto& [o, v] : a)
    if (o != 0 && !v.first && now - v.second >= after) cold.insert(o);
  return cold;
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

// ======================================================================================================================================
// CACHE_POLICY=smart (docs/tiered-cache.md "CACHE_POLICY=smart"). A GreedyDual / Marconi / SGLang-StateCost hybrid that replaces the
// legacy order above for the RAM tier (engine: spill_enforce_ram), the SSD tier (frontend/tiercache.py, from the per-entry numbers
// spill_meta reports) and the VRAM checkpoint choice (Model::evict_checkpoint). CACHE_POLICY=legacy (the default) leaves every
// legacy path as it was: nothing below is called then. frontend/cache_policy.py is the same algorithm in Python (the SSD tier and
// tools/cache_sim.py); frontend/tests/test_cache_policy.py runs both on random candidates (engine/tests/smart_policy_cli.cc).
//
//   gain(e)     = prefill seconds restoring e saves over restoring its parent:
//                 T(dp, de) = t0 * (de - dp) + (k / 2000) * (de^2 - dp^2) us   [t0: us per token; k: us per token per 1K of depth]
//   value(e)    = gain(e) / bytes_GiB(e) * (1 + hits(e)) * (junction(e) ? junction_factor : 1) * (superseded(e) ? superseded_factor : 1)
//   priority(e) = stamp(e) + value(e)        stamp(e): the tier's watermark L at the entry's last use (GdTier::stamp_at)
//   L           = max(L, priority of the victim) after each eviction (GreedyDual ageing; max: L never goes down)
//   victim      = the minimum priority among the eligible LEAVES of the tier (no retained strict extender in the same tier); among
//                 all eligible entries only when there is no leaf; an entry younger than min_resident_s in the tier is not
//                 eligible unless every candidate is young (then the oldest one goes). Ties: the older (age_key), then the lower index.
// parent(e): the deepest retained entry that is a strict prefix of e (dp = its tokens, 0 for none). junction(e): e is a BRANCH
// POINT (two of its extenders diverge) or its extenders belong to an owner other than e's (shared by >= 2 agents).
// Priorities are recomputed every time a victim is chosen (dropping a shallow entry raises its extenders' gain).

struct SmartParams {
  // Prefill time per token at depth d: t0 + k * d / 1000 microseconds (k: microseconds per token per 1,000 tokens of depth).
  // Defaults: the owner's RTX 5090 (DFlash2, NVFP4 KV, log extract of 2026-11-27): ~17,000 tok/s for a 30K prompt (mean depth 15K:
  // 58.7 us/token) and ~13,600 tok/s for a 125K prompt (mean depth 62.5K: 73.4 us/token) => k = 14.7 us / 47.5K = 0.31 us per 1K
  // tokens of depth, t0 = 58.7 - 0.31 * 15 = 54 us. CACHE_PREFILL_T0_US / CACHE_PREFILL_K_US (the latter per 1,000 tokens of depth).
  // TODO(auto-fit): worker.py measure_prefill already measures the prefill speed per prompt size; fit (t0, k) from it and pass them
  // here (the only hook needed: these two fields). Not wired yet on purpose (the owner validates the fixed defaults first).
  double t0_us = 54.0;
  double k_us = 0.31;
  double superseded_factor = 0.25;  // CACHE_SUPERSEDED_FACTOR: value of an entry a deeper one of its chain stands in for
  double junction_factor = 2.0;     // CACHE_JUNCTION_FACTOR: value of a branch point / a prefix shared by >= 2 agents
  double min_resident_s = 300;      // CACHE_MIN_RESIDENT_S: RAM / SSD entries younger than this are protected (VRAM: not used)
};
inline constexpr double kSmartMinBytes = 1 << 20;  // bytes floor of value() (an entry that frees nothing must not divide by 0)

// CACHE_POLICY: 0 legacy (unset, empty or "legacy"), 1 smart, -1 anything else (the caller reports it). Case and blanks ignored.
inline int cache_policy_of(const char* v) {
  if (!v) return 0;
  std::string s;
  for (const char* p = v; *p; p++)
    if (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') s.push_back(char(*p >= 'A' && *p <= 'Z' ? *p - 'A' + 'a' : *p));
  return s.empty() || s == "legacy" ? 0 : s == "smart" ? 1 : -1;
}

// The constants from the environment (get(name) -> const char* or null, e.g. getenv). An unset, empty, non-numeric, negative or
// non-finite value keeps the default.
template <class Get>
inline SmartParams smart_params_from_env(Get&& get) {
  SmartParams p;
  auto rd = [&](const char* name, double& out) {
    const char* e = get(name);
    if (!e || !*e) return;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end != e && std::isfinite(v) && v >= 0) out = v;
  };
  rd("CACHE_PREFILL_T0_US", p.t0_us);
  rd("CACHE_PREFILL_K_US", p.k_us);
  rd("CACHE_SUPERSEDED_FACTOR", p.superseded_factor);
  rd("CACHE_JUNCTION_FACTOR", p.junction_factor);
  rd("CACHE_MIN_RESIDENT_S", p.min_resident_s);
  return p;
}

// Seconds of prefill from depth `from` to depth `to` (0 when to <= from).
inline double prefill_seconds(const SmartParams& p, double from, double to) {
  if (!(to > from)) return 0;
  return 1e-6 * (p.t0_us * (to - from) + 0.5 * (p.k_us * 1e-3) * (to * to - from * from));
}

// Tiers an entry is in (bit mask).
inline constexpr uint8_t kTierVram = 1, kTierRam = 2, kTierSsd = 4;

struct ChainInfo {
  int parent = -1;          // index of the deepest node that is a strict prefix (-1: none)
  int parent_depth = 0;     // its tokens (0: none)
  uint8_t desc_tiers = 0;   // OR of the tiers of every strict extender (leaf in tier T: !(desc_tiers & T))
  bool branch = false;      // two extenders diverge (not all on the chain of the deepest one)
  bool shared = false;      // an extender belongs to another owner
  bool junction() const { return branch || shared; }
};

// Chain relations of nodes (any mix of pool checkpoints and spilled entries; tiers[i]: node i's tier bits). Equal copies (the same
// tokens: a live backup next to its checkpoint) are neither each other's parent nor extender. Cost: the sort of superseded_flags,
// then for each node the nodes that share its full blocks (adjacent in that order).
inline std::vector<ChainInfo> chain_info(const std::vector<RetNode>& nodes, const std::vector<uint8_t>& tiers) {
  const size_t n = nodes.size();
  std::vector<ChainInfo> out(n);
  if (n < 2) return out;
  std::vector<size_t> ord(n);
  for (size_t i = 0; i < n; i++) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
    const auto& x = *nodes[a].bh;
    const auto& y = *nodes[b].bh;
    if (x == y) return a < b;
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  });
  std::vector<size_t> pos(n);
  for (size_t p = 0; p < n; p++) pos[ord[p]] = p;
  std::vector<size_t> ext;
  for (size_t i = 0; i < n; i++) {
    const RetNode& c = nodes[i];
    const size_t m = c.ids->size() / kBlockTokens;
    size_t g = pos[i];
    while (g > 0 && *nodes[ord[g - 1]].bh == *c.bh) g--;
    ext.clear();
    size_t deepest = n;
    for (size_t q = g; q < n; q++) {
      const size_t j = ord[q];
      const RetNode& o = nodes[j];
      if (o.bh->size() < m || !std::equal(c.bh->begin(), c.bh->begin() + std::ptrdiff_t(m), o.bh->begin())) break;
      if (j == i || o.ids->size() <= c.ids->size() || !ret_detail::prefix_of(c, o)) continue;
      ext.push_back(j);
      if (deepest == n || o.ids->size() > nodes[deepest].ids->size()) deepest = j;
    }
    ChainInfo& ci = out[i];
    for (size_t j : ext) {
      ci.desc_tiers = uint8_t(ci.desc_tiers | (j < tiers.size() ? tiers[j] : 0));
      if (nodes[j].owner != c.owner) ci.shared = true;
      if (!ret_detail::prefix_of(nodes[j], nodes[deepest])) ci.branch = true;
      ChainInfo& cj = out[j];  // c is a strict prefix of j: its parent unless a deeper prefix was found
      if (int(c.ids->size()) > cj.parent_depth || cj.parent < 0) cj.parent = int(i), cj.parent_depth = int(c.ids->size());
    }
  }
  return out;
}

// One candidate of a smart pick (the caller fills it from the entry and its ChainInfo).
struct SmartCand {
  double depth = 0, parent_depth = 0;  // tokens
  double bytes = 0;                    // physical bytes evicting it frees in this tier
  uint32_t hits = 0;
  bool junction = false, superseded = false;
  bool leaf = true;                    // no retained strict extender in this tier
  bool young = false;                  // in the tier for less than min_resident_s (protected)
  double stamp = 0;                    // the tier's L at the entry's last use
  double age_key = 0;                  // smaller: older (the young fallback, ties)
  double state = 0;                    // the part of `bytes` that is the recurrent state alone (choose_victim's backup rule)
};

inline double smart_value(const SmartParams& p, const SmartCand& c) {
  const double gib = std::max(c.bytes, kSmartMinBytes) / double(1ull << 30);
  double v = prefill_seconds(p, c.parent_depth, c.depth) / gib * (1.0 + double(c.hits));
  if (c.junction) v *= p.junction_factor;
  if (c.superseded) v *= p.superseded_factor;
  return v;
}
inline double smart_priority(const SmartParams& p, const SmartCand& c) { return c.stamp + smart_value(p, c); }

enum SmartReason : int { kSmartLeaf = 0, kSmartNonLeaf = 1, kSmartYoung = 2 };
struct SmartPick {
  int idx = -1;
  double priority = 0;
  int reason = kSmartLeaf;
  bool young_skipped = false;  // a young candidate was passed over (the protection acted)
};

inline SmartPick smart_pick(const std::vector<SmartCand>& v, const SmartParams& p) {
  SmartPick r;
  if (v.empty()) return r;
  bool any_old = false, any_young = false, old_leaf = false;
  for (const SmartCand& c : v) {
    any_old = any_old || !c.young;
    any_young = any_young || c.young;
    old_leaf = old_leaf || (!c.young && c.leaf);
  }
  if (!any_old) {  // all young: the oldest goes
    size_t b = 0;
    for (size_t i = 1; i < v.size(); i++)
      if (v[i].age_key < v[b].age_key) b = i;
    r.idx = int(b), r.priority = smart_priority(p, v[b]), r.reason = kSmartYoung;
    return r;
  }
  r.young_skipped = any_young;
  r.reason = old_leaf ? kSmartLeaf : kSmartNonLeaf;
  for (size_t i = 0; i < v.size(); i++) {
    const SmartCand& c = v[i];
    if (c.young || (old_leaf && !c.leaf)) continue;
    const double pr = smart_priority(p, c);
    if (r.idx < 0 || pr < r.priority || (pr == r.priority && c.age_key < v[size_t(r.idx)].age_key))
      r.idx = int(i), r.priority = pr;
  }
  return r;
}

// The watermark of one tier, the stamps of its entries and its counters. Uses are identified by a monotonic clock (the engine's
// use_clock_: Checkpoint / Spilled last_use). stamp_at(c) is L as it was when the use at clock c happened: the value set by the
// last eviction recorded at a clock < c (0 before the first). Only the entry that was used carries its clock (its ancestors keep
// theirs). The record keeps the last kMaxSteps raises (older uses get the oldest kept L: a lower stamp, never a higher one).
class GdTier {
 public:
  static constexpr size_t kMaxSteps = 4096;
  double L() const { return L_; }
  double stamp_at(uint64_t clock) const {
    auto it = std::lower_bound(steps_.begin(), steps_.end(), clock, [](const std::pair<uint64_t, double>& s, uint64_t c) { return s.first < c; });
    if (it == steps_.begin()) return floor_;
    return std::prev(it)->second;
  }
  // A victim was evicted at clock `now` (the latest use clock so far).
  void evicted(const SmartPick& pk, uint64_t now) {
    if (pk.idx < 0) return;
    if (pk.reason >= 0 && pk.reason < 3) ev_[pk.reason]++;
    young_skipped_ += pk.young_skipped ? 1 : 0;
    prio_sum_ += pk.priority;
    L_ = std::max(L_, pk.priority);
    if (!steps_.empty() && steps_.back().first >= now) steps_.back().second = L_;  // (same clock: the later L)
    else steps_.emplace_back(now, L_);
    if (steps_.size() > kMaxSteps) {
      const size_t cut = steps_.size() / 2;
      floor_ = steps_[cut - 1].second;
      steps_.erase(steps_.begin(), steps_.begin() + std::ptrdiff_t(cut));
    }
  }
  // out[6]: evictions of a leaf, of a non-leaf (no leaf was eligible), of a young entry (all were young), picks that passed over a
  // young entry, the mean victim priority and L (both x 1000, rounded).
  void stats(int64_t* out) const {
    const int64_t n = ev_[0] + ev_[1] + ev_[2];
    out[0] = ev_[0], out[1] = ev_[1], out[2] = ev_[2], out[3] = young_skipped_;
    out[4] = n ? int64_t(std::llround(prio_sum_ / double(n) * 1000.0)) : 0;
    out[5] = int64_t(std::llround(L_ * 1000.0));
  }

 private:
  double L_ = 0, floor_ = 0, prio_sum_ = 0;
  int64_t ev_[3] = {0, 0, 0}, young_skipped_ = 0;
  std::vector<std::pair<uint64_t, double>> steps_;
};

// Pages of `pages` that nothing else references (pool.refs(p) == 1): what evicting a pool checkpoint gives back to the KV pool.
template <class Pool>
inline size_t sole_pages(const std::vector<int32_t>& pages, const Pool& pool) {
  size_t n = 0;
  for (int32_t p : pages) n += pool.refs(p) == 1 ? 1 : 0;
  return n;
}

namespace ret_detail {
template <class V>
inline const std::vector<uint64_t>* cached_bh(const std::vector<int32_t>& ids, V& h) {
  if (h.size() != ids.size() / kBlockTokens) h = block_hashes(ids);  // (once per entry / checkpoint: its tokens never change)
  return &h;
}
}  // namespace ret_detail

// Smart victim of the RAM tier (Model::spill_enforce_ram). spill: map id -> Spilled (fields ids, bh, ring, owner, path, last_use,
// born, hits, state_bytes, live, busy, in_ram()); ckpts: the pool's checkpoints (ids, bh, ring, owner, loading). Candidates: entries
// in RAM and not busy; idle backups (live) are class 0 and go first as in choose_victim (they lose nothing: their checkpoint is
// still in the pool), unless the best backup frees only its state while the best sole copy frees >= 4x that. parent / junction come
// from every visible node (pool + RAM + SSD), leaf from the RAM tier. freed(s): host bytes erasing s gives back (spill_freed);
// superseded(id): Model::spill_superseded. Returns spill.end() when there is no candidate; *pick (if not null) gets the pick.
template <class Map, class Ckpts, class Freed, class Sup>
inline typename Map::iterator smart_ram_victim(Map& spill, const Ckpts& ckpts, const SmartParams& p, const GdTier& gd, double now_s,
                                               Freed&& freed, Sup&& superseded, SmartPick* pick = nullptr) {
  std::vector<RetNode> nodes;
  std::vector<uint8_t> tiers;
  std::vector<typename Map::iterator> its;
  nodes.reserve(spill.size() + ckpts.size()), tiers.reserve(spill.size() + ckpts.size()), its.reserve(spill.size());
  for (auto it = spill.begin(); it != spill.end(); ++it) {
    const auto& s = it->second;
    nodes.push_back({&s.ids, ret_detail::cached_bh(s.ids, s.bh), s.ring, s.owner, !s.path.empty()});
    tiers.push_back(uint8_t((s.in_ram() ? kTierRam : 0) | (!s.path.empty() ? kTierSsd : 0)));
    its.push_back(it);
  }
  for (const auto& c : ckpts)
    if (!c.loading) {
      nodes.push_back({&c.ids, ret_detail::cached_bh(c.ids, c.bh), c.ring, c.owner, false});
      tiers.push_back(kTierVram);
    }
  const std::vector<ChainInfo> ci = chain_info(nodes, tiers);
  std::vector<SmartCand> cand[2];
  std::vector<size_t> at[2];
  for (size_t i = 0; i < its.size(); i++) {
    const auto& s = its[i]->second;
    if (!s.in_ram() || s.busy) continue;
    SmartCand c;
    c.depth = double(s.ids.size()), c.parent_depth = double(ci[i].parent_depth);
    c.bytes = double(freed(s)), c.state = double(s.state_bytes), c.hits = s.hits;
    c.junction = ci[i].junction(), c.superseded = superseded(its[i]->first);
    c.leaf = !(ci[i].desc_tiers & kTierRam);
    c.young = now_s - s.born < p.min_resident_s;
    c.stamp = gd.stamp_at(s.last_use), c.age_key = s.born;
    const int cls = s.live ? 0 : 1;
    cand[cls].push_back(c), at[cls].push_back(i);
  }
  const SmartPick k0 = smart_pick(cand[0], p), k1 = smart_pick(cand[1], p);
  int cls = -1;
  if (k0.idx >= 0) {
    const SmartCand& b = cand[0][size_t(k0.idx)];
    double best_sole = 0;
    for (const SmartCand& c : cand[1]) best_sole = std::max(best_sole, c.bytes);
    cls = (k1.idx >= 0 && b.bytes <= b.state && best_sole > 0 && best_sole >= 4 * b.bytes) ? 1 : 0;
  } else if (k1.idx >= 0) {
    cls = 1;
  }
  if (cls < 0) return spill.end();
  const SmartPick& k = cls ? k1 : k0;
  if (pick) *pick = k;
  return its[at[cls][size_t(k.idx)]];
}

// Smart victim among the pool's checkpoints (Model::evict_checkpoint): those neither loading nor held (the caller keeps the legacy
// fallbacks for held / loading ones). parent / leaf / junction within the pool. bytes(c): what evicting it frees (pages only it
// holds, or the state slot when a slot is wanted); protect(c): not a victim unless every candidate is protected (a sibling-share
// checkpoint planned for a waiting request). No age protection in VRAM (its pages turn over in seconds): age_key is last_use.
template <class Ckpts, class Bytes, class Protect>
inline typename Ckpts::iterator smart_vram_victim(Ckpts& ckpts, const SmartParams& p, const GdTier& gd, Bytes&& bytes, Protect&& protect,
                                                  SmartPick* pick = nullptr) {
  std::vector<RetNode> nodes;
  std::vector<uint8_t> tiers;
  std::vector<typename Ckpts::iterator> its;
  for (auto it = ckpts.begin(); it != ckpts.end(); ++it)
    if (!it->loading) {
      nodes.push_back({&it->ids, ret_detail::cached_bh(it->ids, it->bh), it->ring, it->owner, false});
      tiers.push_back(kTierVram);
      its.push_back(it);
    }
  const std::vector<ChainInfo> ci = chain_info(nodes, tiers);
  std::vector<SmartCand> cand;
  std::vector<size_t> at;
  for (size_t i = 0; i < its.size(); i++) {
    const auto& c = *its[i];
    if (c.held) continue;
    SmartCand s;
    s.depth = double(c.ids.size()), s.parent_depth = double(ci[i].parent_depth);
    s.bytes = double(bytes(c)), s.hits = c.hits, s.junction = ci[i].junction();
    s.leaf = !(ci[i].desc_tiers & kTierVram);
    s.young = bool(protect(c));
    s.stamp = gd.stamp_at(c.last_use), s.age_key = double(c.last_use);
    cand.push_back(s), at.push_back(i);
  }
  const SmartPick k = smart_pick(cand, p);
  if (k.idx < 0) return ckpts.end();
  if (pick) *pick = k;
  return its[at[size_t(k.idx)]];
}

// The per-entry numbers frontend/tiercache.py needs for the SSD tier (appended to spill_meta under smart): for each entry of spill, in
// map order, its ChainInfo over every visible node (pool + RAM + SSD).
template <class Map, class Ckpts>
inline std::vector<ChainInfo> smart_spill_chains(const Map& spill, const Ckpts& ckpts) {
  std::vector<RetNode> nodes;
  std::vector<uint8_t> tiers;
  for (const auto& kv : spill) {
    const auto& s = kv.second;
    nodes.push_back({&s.ids, ret_detail::cached_bh(s.ids, s.bh), s.ring, s.owner, !s.path.empty()});
    tiers.push_back(uint8_t((s.in_ram() ? kTierRam : 0) | (!s.path.empty() ? kTierSsd : 0)));
  }
  const size_t n = nodes.size();
  for (const auto& c : ckpts)
    if (!c.loading) {
      nodes.push_back({&c.ids, ret_detail::cached_bh(c.ids, c.bh), c.ring, c.owner, false});
      tiers.push_back(kTierVram);
    }
  std::vector<ChainInfo> ci = chain_info(nodes, tiers);
  ci.resize(n);
  return ci;
}

}  // namespace q27
