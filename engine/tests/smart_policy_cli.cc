// A small driver of the CACHE_POLICY=smart functions of src/retention.h, for frontend/tests/test_cache_policy.py (the Python and C++
// victim orders must agree). Built by that test with g++ (skipped without one). Input on stdin, one query per block:
//   pick <t0_us> <k_us> <superseded_factor> <junction_factor> <hit_weight> <n>
//     n lines: depth parent_depth bytes hits junction superseded leaf young stamp age_key
//   -> "pick <idx> <priority %.17g> <reason> <young_skipped>"
//   chain <n>
//     n lines: owner tiers len id_0 .. id_{len-1}
//   -> n lines "<parent> <parent_depth> <desc_tiers> <branch> <shared>"
//   ram <now_s> <min_resident_s> <n>          (smart_ram_victim with fake entries; t0 / k / factors: the defaults)
//     n lines: live freed state hits superseded born last_use owner len id_0 ..   (all in RAM, none busy)
//   -> "ram <index of the victim or -1>"
#include <cstdio>
#include <iostream>
#include <map>
#include <string>

#include "retention.h"

using namespace q27;

struct Ent {
  std::vector<int32_t> ids;
  mutable std::vector<uint64_t> bh;
  bool ring = false;
  uint64_t owner = 0;
  std::string path;
  uint64_t last_use = 0;
  double born = 0;
  uint32_t hits = 0;
  size_t state_bytes = 0;
  bool live = false, busy = false;
  size_t freed = 0;
  bool sup = false;
  bool in_ram() const { return true; }
};
struct Ck {
  std::vector<int32_t> ids;
  mutable std::vector<uint64_t> bh;
  bool ring = false;
  uint64_t owner = 0;
  bool loading = false;
};

int main() {
  std::string cmd;
  while (std::cin >> cmd) {
    if (cmd == "pick") {
      SmartParams p;
      size_t n = 0;
      std::cin >> p.t0_us >> p.k_us >> p.superseded_factor >> p.junction_factor >> p.hit_weight >> n;
      std::vector<SmartCand> v(n);
      for (auto& c : v) {
        int j, s, l, y;
        std::cin >> c.depth >> c.parent_depth >> c.bytes >> c.hits >> j >> s >> l >> y >> c.stamp >> c.age_key;
        c.junction = j, c.superseded = s, c.leaf = l, c.young = y;
      }
      const SmartPick k = smart_pick(v, p);
      std::printf("pick %d %.17g %d %d\n", k.idx, k.priority, k.reason, k.young_skipped ? 1 : 0);
    } else if (cmd == "chain") {
      size_t n = 0;
      std::cin >> n;
      std::vector<std::vector<int32_t>> ids(n);
      std::vector<std::vector<uint64_t>> bh(n);
      std::vector<RetNode> nodes(n);
      std::vector<uint8_t> tiers(n);
      for (size_t i = 0; i < n; i++) {
        uint64_t owner;
        int t;
        size_t len;
        std::cin >> owner >> t >> len;
        ids[i].resize(len);
        for (auto& x : ids[i]) std::cin >> x;
        tiers[i] = uint8_t(t);
        bh[i] = block_hashes(ids[i]);
        nodes[i] = {&ids[i], &bh[i], false, owner};
      }
      for (const ChainInfo& c : chain_info(nodes, tiers))
        std::printf("%d %d %d %d %d\n", c.parent, c.parent_depth, int(c.desc_tiers), c.branch ? 1 : 0, c.shared ? 1 : 0);
    } else if (cmd == "ram") {
      SmartParams p;
      double now = 0;
      size_t n = 0;
      std::cin >> now >> p.min_resident_s >> n;
      std::map<uint64_t, Ent> spill;
      for (size_t i = 0; i < n; i++) {
        Ent e;
        int live, sup;
        size_t len;
        std::cin >> live >> e.freed >> e.state_bytes >> e.hits >> sup >> e.born >> e.last_use >> e.owner >> len;
        e.live = live, e.sup = sup;
        e.ids.resize(len);
        for (auto& x : e.ids) std::cin >> x;
        spill[i] = std::move(e);
      }
      std::vector<Ck> ck;
      GdTier gd;
      auto it = smart_ram_victim(spill, ck, p, gd, now, 0, [](const Ent& e) { return e.freed; },
                                 [&](uint64_t id) { return spill.at(id).sup; });
      std::printf("ram %lld\n", it == spill.end() ? -1LL : (long long)it->first);
    }
    std::fflush(stdout);
  }
  return 0;
}
