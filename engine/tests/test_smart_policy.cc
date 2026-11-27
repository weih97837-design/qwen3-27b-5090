// Host-only test of CACHE_POLICY=smart (src/retention.h: chain_info, smart_pick, GdTier, smart_ram_victim, smart_vram_victim,
// smart_spill_chains, the environment). No CUDA, no GPU. The engine's structs are replaced by fakes with the same member names (the
// templates are instantiated with them exactly as spill.cc / model.cc instantiate them with Spilled / Checkpoint).
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Isrc tests/test_smart_policy.cc -o /tmp/test_smart_policy && /tmp/test_smart_policy
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <string>

#include "retention.h"

using namespace q27;

static int fails = 0;
#define CHECK(c)                                               \
  do {                                                         \
    if (!(c)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      if (++fails > 30) std::exit(1);                          \
    }                                                          \
  } while (0)

static std::vector<int32_t> toks(int32_t base, size_t n) {
  std::vector<int32_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = int32_t((uint32_t(base) * 1000003u + uint32_t(i)) & 0x7fffffffu);
  return v;
}
static std::vector<int32_t> extend(const std::vector<int32_t>& a, size_t n, size_t extra, int32_t salt) {
  std::vector<int32_t> v(a.begin(), a.begin() + std::ptrdiff_t(n));
  for (size_t i = 0; i < extra; i++) v.push_back(salt * 7919 + int32_t(i));
  return v;
}

// Fakes of Spilled / Checkpoint / PagePool with the members the templates use.
struct FakeSpilled {
  std::vector<int32_t> ids;
  mutable std::vector<uint64_t> bh;
  bool ring = false;
  uint64_t owner = 0;
  std::string path;
  uint64_t last_use = 0;
  double born = 0;
  uint32_t hits = 0;
  size_t state_bytes = 190u << 20;
  bool live = false, busy = false, ram = true;
  size_t freed = 0;
  bool sup = false;
  bool in_ram() const { return ram; }
};
struct FakeCkpt {
  std::vector<int32_t> ids;
  mutable std::vector<uint64_t> bh;
  bool ring = false;
  uint64_t owner = 0;
  bool loading = false, held = false;
  uint64_t last_use = 0;
  uint32_t hits = 0;
  std::vector<int32_t> pages;
  uint64_t share_until = 0;
};
struct FakePool {
  std::vector<int> r;
  int refs(int p) const { return r[size_t(p)]; }
};

static std::vector<ChainInfo> chains(const std::vector<std::vector<int32_t>>& ids, const std::vector<uint64_t>& owner,
                                     const std::vector<uint8_t>& tiers) {
  static std::list<std::vector<uint64_t>> keep;  // (the nodes point into these)
  std::vector<RetNode> nodes;
  for (size_t i = 0; i < ids.size(); i++) {
    keep.push_back(block_hashes(ids[i]));
    nodes.push_back({&ids[i], &keep.back(), false, owner[i]});
  }
  return chain_info(nodes, tiers);
}

static void test_env_and_formula() {
  CHECK(cache_policy_of(nullptr) == 0 && cache_policy_of("") == 0 && cache_policy_of(" Legacy ") == 0);
  CHECK(cache_policy_of("smart") == 1 && cache_policy_of("SMART\n") == 1 && cache_policy_of("lru") == -1);
  std::map<std::string, std::string> env = {{"CACHE_PREFILL_T0_US", "60"}, {"CACHE_PREFILL_K_US", "x"},
                                            {"CACHE_SUPERSEDED_FACTOR", "-1"}, {"CACHE_MIN_RESIDENT_S", "0"},
                                            {"CACHE_JUNCTION_FACTOR", "3.5"}, {"CACHE_HIT_WEIGHT", "0"}};
  const SmartParams p = smart_params_from_env([&](const char* n) -> const char* {
    auto it = env.find(n);
    return it == env.end() ? nullptr : it->second.c_str();
  });
  CHECK(p.t0_us == 60 && p.k_us == 0.31 && p.superseded_factor == 0.25 && p.min_resident_s == 0 && p.junction_factor == 3.5);
  CHECK(p.hit_weight == 0 && SmartParams().hit_weight == 0.0);
  const SmartParams d;
  // The defaults give the measured speeds: 30K prompt ~17,000 tok/s, 125K ~13,600 tok/s.
  const double tps30 = 30000 / prefill_seconds(d, 0, 30000), tps125 = 125000 / prefill_seconds(d, 0, 125000);
  CHECK(tps30 > 16800 && tps30 < 17300);
  CHECK(tps125 > 13400 && tps125 < 13800);
  CHECK(prefill_seconds(d, 5000, 5000) == 0 && prefill_seconds(d, 6000, 5000) == 0);
  // T(a, c) = T(a, b) + T(b, c)
  CHECK(std::abs(prefill_seconds(d, 0, 90000) - prefill_seconds(d, 0, 30000) - prefill_seconds(d, 30000, 90000)) < 1e-9);
}

static void test_chain_info() {
  const auto sys = toks(1, 9000);                 // a shared system prompt
  const auto a1 = extend(sys, 9000, 21000, 11);   // agent 1's conversation (30K)
  const auto a2 = extend(a1, 30000, 30000, 12);   // its continuation (60K)
  const auto a3 = extend(a2, 60000, 30000, 13);   // and again (90K)
  const auto b1 = extend(a1, 20000, 20000, 21);   // diverges from a1 at 20K (40K)
  const auto c1 = extend(sys, 9000, 30000, 31);   // agent 2 from the same system prompt
  //                      0    1   2   3   4   5
  const std::vector<std::vector<int32_t>> ids = {sys, a1, a2, a3, b1, c1};
  const std::vector<uint64_t> owner = {0, 1, 1, 1, 1, 2};
  const std::vector<uint8_t> tier = {kTierRam, kTierRam, kTierSsd, kTierVram, kTierRam, kTierSsd};
  const auto ci = chains(ids, owner, tier);
  CHECK(ci[0].parent == -1 && ci[0].parent_depth == 0);
  CHECK(ci[1].parent == 0 && ci[1].parent_depth == 9000);
  CHECK(ci[2].parent == 1 && ci[2].parent_depth == 30000);
  CHECK(ci[3].parent == 2 && ci[3].parent_depth == 60000);
  CHECK(ci[4].parent == 0 && ci[4].parent_depth == 9000);  // (b1 leaves a1 at 20K: a1 is no prefix of it)
  CHECK(ci[5].parent == 0);
  CHECK(ci[0].branch && ci[0].shared && ci[0].junction());          // the system prompt: a1 / b1 / c1 diverge, two owners
  CHECK(!ci[1].branch && !ci[1].shared && !ci[1].junction());       // a1: only its own continuations extend it
  CHECK(ci[1].desc_tiers == (kTierSsd | kTierVram));                // a2 on the SSD, a3 in VRAM
  CHECK(!(ci[1].desc_tiers & kTierRam));                            // => a1 is a leaf of the RAM tier
  CHECK(ci[3].desc_tiers == 0 && ci[4].desc_tiers == 0 && ci[5].desc_tiers == 0);
  // an equal copy (a live backup next to its checkpoint) is neither parent nor extender
  const auto ci2 = chains({a1, a1, a2}, {1, 1, 1}, {kTierRam, kTierVram, kTierRam});
  CHECK(ci2[0].desc_tiers == kTierRam && ci2[1].desc_tiers == kTierRam && ci2[2].parent_depth == 30000);
  CHECK(!ci2[0].branch && !ci2[0].junction());
  // a branch point with ONE owner is a junction too
  const auto ci3 = chains({a1, a2, extend(a1, 30000, 500, 99)}, {1, 1, 1}, {kTierRam, kTierRam, kTierRam});
  CHECK(ci3[0].branch && !ci3[0].shared && ci3[0].junction());
}

static SmartCand cand(double depth, double parent, double gib, uint32_t hits = 0, double stamp = 0, double age = 0) {
  SmartCand c;
  c.depth = depth, c.parent_depth = parent, c.bytes = gib * double(1ull << 30), c.hits = hits, c.stamp = stamp, c.age_key = age;
  return c;
}

static void test_value_and_pick() {
  SmartParams p;
  p.hit_weight = 1.0;  // (the default is 0: hits ignored)
  // value: gain / GiB * (1 + hits) * junction * superseded
  SmartCand c = cand(60000, 30000, 1.0);
  const double base = smart_value(p, c);
  CHECK(std::abs(base - prefill_seconds(p, 30000, 60000)) < 1e-12);
  c.hits = 3;
  CHECK(std::abs(smart_value(p, c) - 4 * base) < 1e-9);
  c.hits = 0, c.junction = true;
  CHECK(std::abs(smart_value(p, c) - 2 * base) < 1e-9);
  c.junction = false, c.superseded = true;
  CHECK(std::abs(smart_value(p, c) - 0.25 * base) < 1e-9);
  c.superseded = false, c.bytes = 0;  // (floor: no division by zero)
  CHECK(std::isfinite(smart_value(p, c)) && smart_value(p, c) > base);

  // leaf-first: the non-leaf with the lowest value is not taken while a leaf exists
  std::vector<SmartCand> v = {cand(30000, 0, 1.0), cand(60000, 30000, 1.0), cand(90000, 60000, 1.0)};
  v[0].leaf = false, v[1].leaf = false;  // a chain of three in one tier: only the tip is a leaf
  SmartPick k = smart_pick(v, p);
  CHECK(k.idx == 2 && k.reason == kSmartLeaf && !k.young_skipped);
  CHECK(std::abs(k.priority - smart_priority(p, v[2])) < 1e-12);
  // no leaf eligible: the minimum among the rest (forced non-leaf)
  v[2].young = true;
  k = smart_pick(v, p);
  CHECK(k.reason == kSmartNonLeaf && k.young_skipped);
  CHECK(k.idx == (smart_value(p, v[0]) < smart_value(p, v[1]) ? 0 : 1));
  // all young: the oldest goes
  for (auto& x : v) x.young = true;
  v[0].age_key = 5, v[1].age_key = 2, v[2].age_key = 9;
  k = smart_pick(v, p);
  CHECK(k.idx == 1 && k.reason == kSmartYoung);
  // the recency stamp dominates value
  std::vector<SmartCand> w = {cand(90000, 0, 2.0, 0, 50.0), cand(20000, 0, 0.5, 0, 0.0)};
  CHECK(smart_pick(w, p).idx == 1);
  w[1].stamp = 100;
  CHECK(smart_pick(w, p).idx == 0);
  // ties: the older first
  std::vector<SmartCand> t = {cand(30000, 0, 1, 0, 0, 7), cand(30000, 0, 1, 0, 0, 3)};
  CHECK(smart_pick(t, p).idx == 1);
  CHECK(smart_pick({}, p).idx == -1);
}

static void test_gd_ageing() {
  const SmartParams p;
  GdTier gd;
  CHECK(gd.L() == 0 && gd.stamp_at(1) == 0 && gd.stamp_at(1000) == 0);
  SmartPick k;
  k.idx = 0, k.priority = 3.0, k.reason = kSmartLeaf;
  gd.evicted(k, 10);  // L = 3 from clock 10 on
  CHECK(gd.L() == 3.0);
  CHECK(gd.stamp_at(10) == 0 && gd.stamp_at(11) == 3.0);  // a use at clock 10 happened before that eviction
  k.priority = 1.0, k.reason = kSmartNonLeaf;
  gd.evicted(k, 20);  // L never goes down
  CHECK(gd.L() == 3.0 && gd.stamp_at(25) == 3.0);
  k.priority = 7.5, k.reason = kSmartYoung, k.young_skipped = true;
  gd.evicted(k, 30);
  CHECK(gd.stamp_at(30) == 3.0 && gd.stamp_at(31) == 7.5);
  int64_t st[6];
  gd.stats(st);
  CHECK(st[0] == 1 && st[1] == 1 && st[2] == 1 && st[3] == 1 && st[4] == 3833 && st[5] == 7500);
  // an entry used after the last eviction outranks an equally valuable one used before it (GreedyDual recency)
  SmartCand old = cand(40000, 0, 1, 0, gd.stamp_at(5)), fresh = cand(40000, 0, 1, 0, gd.stamp_at(40));
  CHECK(smart_pick({fresh, old}, p).idx == 1);
  // bounded record: old uses get the oldest kept L (never a higher one)
  GdTier g2;
  for (uint64_t i = 1; i <= GdTier::kMaxSteps + 10; i++) {
    SmartPick q;
    q.idx = 0, q.priority = double(i);
    g2.evicted(q, i * 2);
  }
  CHECK(g2.stamp_at(1) <= g2.stamp_at(GdTier::kMaxSteps) && g2.stamp_at(1) > 0);
  CHECK(g2.stamp_at((GdTier::kMaxSteps + 10) * 2 + 1) == double(GdTier::kMaxSteps + 10));
}

// The RAM tier with fake entries: diverging chains, junction protection, young protection, superseded kept, recompute at pop.
static void test_ram_victims() {
  const SmartParams p;  // min_resident_s 300
  const auto sys = toks(1, 9000);
  const auto a1 = extend(sys, 9000, 21000, 11);   // 30K
  const auto a2 = extend(a1, 30000, 30000, 12);   // 60K
  const auto b1 = extend(a1, 20000, 30000, 21);   // diverges from a1 at 20K: 50K
  const auto c1 = toks(5, 40000);                 // unrelated, agent 3
  std::map<uint64_t, FakeSpilled> spill;
  std::vector<FakeCkpt> ckpts;
  const double now = 10000;
  auto add = [&](uint64_t id, const std::vector<int32_t>& ids, uint64_t owner, double gib, double born, uint64_t last_use) {
    FakeSpilled s;
    s.ids = ids, s.owner = owner, s.freed = size_t(gib * double(1ull << 30)), s.born = born, s.last_use = last_use;
    spill[id] = std::move(s);
  };
  add(1, sys, 1, 0.35, 100, 1);
  add(2, a1, 1, 0.6, 200, 2);
  add(3, a2, 1, 0.7, 300, 3);
  add(4, b1, 2, 0.75, 400, 4);
  add(5, c1, 3, 0.9, 500, 5);
  GdTier gd;
  auto freed = [](const FakeSpilled& s) { return s.freed; };
  auto sup = [&](uint64_t id) { return spill.at(id).sup; };
  std::vector<uint64_t> order;
  SmartPick pk;
  auto pop = [&]() {
    auto it = smart_ram_victim(spill, ckpts, p, gd, now, 100, freed, sup, &pk);  // (records the eviction at clock 100)
    if (it == spill.end()) return uint64_t(0);
    const uint64_t id = it->first;
    spill.erase(it);
    return id;
  };
  // leaves: a2 (3), b1 (4), c1 (5); a1 and the system prompt are not (a2 / b1 extend them). Values vs their parents: a2
  // T(30K, 60K) / 0.7 GiB, b1 T(9K, 50K) / 0.75 (b1 leaves a1 at 20K: its parent is the system prompt), c1 T(0, 40K) / 0.9.
  const double va2 = prefill_seconds(p, 30000, 60000) / 0.7, vb1 = prefill_seconds(p, 9000, 50000) / 0.75,
               vc1 = prefill_seconds(p, 0, 40000) / 0.9;
  CHECK(vc1 < va2 && va2 < vb1);
  uint64_t v1 = pop();
  CHECK(v1 == 5 && pk.reason == kSmartLeaf);
  CHECK(std::abs(pk.priority - vc1) < 1e-6);  // (stamps 0: nothing was evicted before)
  // next: a2 (its value is the lowest left). The stamps of the rest are still 0 (their last uses are before clock 100).
  CHECK(pop() == 3);
  // recompute at pop: a1 is a leaf of RAM now (a2 is gone), with T(9K, 30K) / 0.6; b1 still T(9K, 50K) / 0.75
  const double va1 = prefill_seconds(p, 9000, 30000) / 0.6;
  CHECK(pop() == (va1 < vb1 ? 2u : 4u));
  // the system prompt (a junction: a1, b1 diverge and two owners) is never taken while a leaf exists
  while (spill.size() > 1) CHECK(pop() != 1);
  CHECK(spill.begin()->first == 1);
  CHECK(pop() == 1 && pk.reason == kSmartLeaf);  // (the last one: a leaf now)

  // young protection: a just-spilled big low-value entry is not the victim while an older entry exists
  spill.clear();
  add(10, toks(7, 20000), 1, 4.0, now - 10, 1);   // 10 s in RAM, very low value per byte
  add(11, toks(8, 60000), 2, 1.0, now - 5000, 2);  // old, high value
  uint64_t v = pop();
  CHECK(v == 11 && pk.young_skipped);
  CHECK(pop() == 10 && pk.reason == kSmartYoung);  // alone: the young one goes (the oldest of the young)

  // superseded: kept and ranked with the reduced value, not dropped first regardless of value
  spill.clear();
  add(20, a1, 1, 0.6, 0, 1);  // a deeper checkpoint (a2) of its chain is in the pool: superseded
  spill[20].sup = true;
  add(21, toks(9, 18000), 2, 0.4, 0, 1);
  FakeCkpt deep;
  deep.ids = a2;
  ckpts.push_back(deep);
  const double v20 = 0.25 * prefill_seconds(p, 0, 30000) / 0.6, v21 = prefill_seconds(p, 0, 18000) / 0.4;
  CHECK(pop() == (v20 < v21 ? 20u : 21u));
  ckpts.clear();

  // backups (live) go first, as in choose_victim; busy / not-in-RAM entries are no candidates
  spill.clear();
  add(30, toks(10, 30000), 1, 1.0, 0, 1);
  add(31, toks(11, 30000), 1, 1.0, 0, 9);
  spill[31].live = true;
  add(32, toks(12, 30000), 1, 0.1, 0, 1);
  spill[32].busy = true;
  add(33, toks(13, 30000), 1, 0.1, 0, 1);
  spill[33].ram = false, spill[33].path = "x.q27m";
  gd = GdTier();
  const double L0 = gd.L();
  CHECK(pop() == 31);
  CHECK(gd.L() == L0);  // dropping an idle backup loses nothing: the watermark does not move
  CHECK(pop() == 30);
  CHECK(gd.L() > L0);
  CHECK(pop() == 0);  // (only busy / on-disk entries left)
}

static void test_vram_victims() {
  const SmartParams p;
  GdTier gd;
  const auto a1 = toks(1, 30000), a2 = extend(a1, 30000, 30000, 2), b1 = toks(3, 20000);
  std::vector<FakeCkpt> ck(4);
  ck[0].ids = a1, ck[0].last_use = 1;
  ck[1].ids = a2, ck[1].last_use = 2;
  ck[2].ids = b1, ck[2].last_use = 3;
  ck[3].ids = toks(4, 50000), ck[3].last_use = 4, ck[3].held = true;  // (held: the legacy fallback's, never a smart candidate)
  FakePool pool{std::vector<int>(4000, 1)};
  // pages: a1 [0, 469), a2 shares them and owns [469, 938); b1 [1000, 1313) of which half are a running sequence's too
  for (int i = 0; i < 469; i++) ck[0].pages.push_back(i), ck[1].pages.push_back(i), pool.r[size_t(i)] = 2;
  for (int i = 469; i < 938; i++) ck[1].pages.push_back(i);
  for (int i = 1000; i < 1313; i++) ck[2].pages.push_back(i), pool.r[size_t(i)] = i < 1156 ? 2 : 1;
  CHECK(sole_pages(ck[0].pages, pool) == 0 && sole_pages(ck[1].pages, pool) == 469 && sole_pages(ck[2].pages, pool) == 157);
  const double page = 18432.0 * 64;
  auto bytes = [&](const FakeCkpt& c) { return double(sole_pages(c.pages, pool)) * page; };
  auto none = [](const FakeCkpt&) { return false; };
  SmartPick pk;
  auto it = smart_vram_victim(ck, p, gd, 7, bytes, none, &pk);
  CHECK(gd.L() == pk.priority && gd.stamp_at(8) == pk.priority);  // (recorded)
  // leaves: a2 and b1 (a1 is a2's prefix). a1 frees no page: never chosen while a leaf exists.
  const double va2 = prefill_seconds(p, 30000, 60000) / (469 * page / double(1ull << 30));
  const double vb1 = prefill_seconds(p, 0, 20000) / (157 * page / double(1ull << 30));
  CHECK(it != ck.end() && (it - ck.begin()) == (va2 < vb1 ? 1 : 2) && pk.reason == kSmartLeaf);
  // a sibling-share checkpoint is protected
  auto protect_b1 = [&](const FakeCkpt& c) { return &c == &ck[2]; };
  it = smart_vram_victim(ck, p, gd, 7, bytes, protect_b1, &pk);
  CHECK((it - ck.begin()) == 1 && pk.young_skipped);
  // loading ones are invisible (not even an extender: a1 becomes a leaf when a2 loads)
  ck[1].loading = true;
  ck[2].held = true;
  it = smart_vram_victim(ck, p, gd, 7, bytes, none, &pk);
  CHECK((it - ck.begin()) == 0 && pk.reason == kSmartLeaf);
  for (auto& c : ck) c.held = true;
  CHECK(smart_vram_victim(ck, p, gd, 7, bytes, none, &pk) == ck.end());
}

static void test_spill_chains() {
  const auto a1 = toks(1, 30000), a2 = extend(a1, 30000, 30000, 2);
  std::map<uint64_t, FakeSpilled> spill;
  spill[1].ids = a1, spill[1].ram = false, spill[1].path = "a1.q27m";
  spill[2].ids = a2, spill[2].ram = true;
  std::vector<FakeCkpt> ck(1);
  ck[0].ids = extend(a2, 60000, 1000, 3);
  const auto ci = smart_spill_chains(spill, ck);
  CHECK(ci.size() == 2);
  CHECK(ci[0].parent_depth == 0 && ci[0].desc_tiers == (kTierRam | kTierVram));  // a1 is a leaf of the SSD
  CHECK(ci[1].parent_depth == 30000 && ci[1].desc_tiers == kTierVram);
}

// Random chains: chain_info agrees with a brute-force prefix scan.
static void test_chain_info_random() {
  uint64_t seed = 12345;
  auto rnd = [&](uint64_t n) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return (seed >> 33) % n;
  };
  for (int trial = 0; trial < 200; trial++) {
    std::vector<std::vector<int32_t>> ids;
    std::vector<uint64_t> owner;
    std::vector<uint8_t> tier;
    const size_t n = 1 + rnd(12);
    for (size_t i = 0; i < n; i++) {
      if (i > 0 && rnd(3) != 0) {
        const auto& base = ids[rnd(i)];
        const size_t keep = rnd(base.size() + 1);
        ids.push_back(extend(base, keep, rnd(3) * 300 + rnd(2) * 77, int32_t(trial * 100 + int(i))));
      } else {
        ids.push_back(toks(int32_t(trial * 100 + int(i)), 1 + rnd(2000)));
      }
      owner.push_back(rnd(3));
      tier.push_back(uint8_t(1u << rnd(3)));
    }
    const auto ci = chains(ids, owner, tier);
    for (size_t i = 0; i < n; i++) {
      int pd = 0;
      uint8_t dt = 0;
      bool shared = false;
      std::vector<size_t> ext;
      for (size_t j = 0; j < n; j++) {
        if (j == i) continue;
        const bool j_pre_i = ids[j].size() < ids[i].size() && std::equal(ids[j].begin(), ids[j].end(), ids[i].begin());
        const bool i_pre_j = ids[i].size() < ids[j].size() && std::equal(ids[i].begin(), ids[i].end(), ids[j].begin());
        if (j_pre_i) pd = std::max(pd, int(ids[j].size()));
        if (i_pre_j) dt = uint8_t(dt | tier[j]), shared = shared || owner[j] != owner[i], ext.push_back(j);
      }
      bool branch = false;
      if (!ext.empty()) {
        size_t d = ext[0];
        for (size_t j : ext)
          if (ids[j].size() > ids[d].size()) d = j;
        for (size_t j : ext)
          if (!std::equal(ids[j].begin(), ids[j].end(), ids[d].begin())) branch = true;
      }
      CHECK(ci[i].parent_depth == pd);
      CHECK(ci[i].desc_tiers == dt);
      CHECK(ci[i].shared == shared);
      CHECK(ci[i].branch == branch);
    }
  }
}

int main() {
  test_env_and_formula();
  test_chain_info();
  test_value_and_pick();
  test_gd_ageing();
  test_ram_victims();
  test_vram_victims();
  test_spill_chains();
  test_chain_info_random();
  if (fails) {
    std::printf("%d failure(s)\n", fails);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
