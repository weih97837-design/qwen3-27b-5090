// Host-only test of the SSD gate (src/ssd_gate.h: the admission the RAM cap applies before it writes a victim to the SSD) and of
// smart_ram_victim's victim ChainInfo (retention.h). No CUDA, no GPU.
//   g++ -std=c++20 -O1 -Wall -Wextra -fsanitize=address,undefined -Isrc tests/test_ssd_gate.cc -o /tmp/test_ssd_gate && /tmp/test_ssd_gate
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "ssd_gate.h"

using namespace q27;

static int fails = 0;
#define CHECK(c)                                               \
  do {                                                         \
    if (!(c)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      if (++fails > 30) std::exit(1);                          \
    }                                                          \
  } while (0)

static SsdGateIn entry(double bytes, double parent, uint32_t hits, uint64_t last_use, bool junction = false, bool sup = false) {
  SsdGateIn e;
  e.bytes = bytes, e.parent_depth = parent, e.hits = hits, e.last_use = last_use, e.junction = junction, e.superseded = sup;
  return e;
}

static double prio(const SmartParams& p, int64_t tokens, const SsdGateIn& e, double stamp) {
  SmartCand c;
  c.depth = double(tokens), c.parent_depth = e.parent_depth, c.bytes = e.bytes, c.hits = e.hits;
  c.junction = e.junction, c.superseded = e.superseded, c.stamp = stamp;
  return smart_priority(p, c);
}

static void test_off_by_default() {
  SsdGate g;
  SmartParams p;
  int calls = 0;
  auto in = [&] { calls++; return entry(1e9, 0, 0, 1); };
  CHECK(!g.on());
  CHECK(g.admit(10, true, p, in));          // (never pushed: every victim is written, as before)
  CHECK(g.admit(1 << 20, false, p, in));
  CHECK(calls == 0 && g.checked() == 0 && g.rejected() == 0);
  // pushed, then turned off again: off, the counters keep what they had
  g.set(true, 32768, 0, 0, nullptr, nullptr, 0);
  CHECK(!g.admit(100, false, p, in) && g.rejected_short() == 1);
  g.set(false, 32768, 5.0, 0, nullptr, nullptr, 0);
  CHECK(!g.on() && g.admit(100, true, p, in) && g.rejected() == 1 && g.checked() == 1);
}

static void test_min_tokens() {
  SsdGate g;
  SmartParams p;
  int calls = 0;
  auto in = [&] { calls++; return entry(1e9, 0, 0, 1); };
  g.set(true, 32768, 0, 0, nullptr, nullptr, 0);
  CHECK(g.verdict(32767, false, p, in) == kSsdShort);
  CHECK(g.verdict(32768, false, p, in) == kSsdAdmit);  // (tiercache._short: pos < min_tokens)
  CHECK(g.verdict(10, true, p, in) == kSsdShort);      // (also under smart, before the watermark)
  g.set(true, 0, 0, 0, nullptr, nullptr, 0);
  CHECK(g.verdict(1, false, p, in) == kSsdAdmit);      // 0: off
  g.set(true, -5, 0, 0, nullptr, nullptr, 0);
  CHECK(g.min_tokens() == 0 && g.verdict(1, false, p, in) == kSsdAdmit);
  CHECK(calls == 0);  // (no watermark test ran: its inputs were never asked for)
}

static void test_watermark() {
  SmartParams p;
  SsdGate g;
  int calls = 0;
  SsdGateIn e = entry(2.0 * (1 << 30), 0, 0, 50);
  auto in = [&] { calls++; return e; };
  // L <= 0 (nothing evicted from the SSD yet): no test
  g.set(true, 0, 0.0, 0, nullptr, nullptr, 0);
  CHECK(g.verdict(40000, true, p, in) == kSsdAdmit && calls == 0);
  // legacy (smart = false): no watermark test even with an L
  g.set(true, 0, 1e9, 0, nullptr, nullptr, 0);
  CHECK(g.verdict(40000, false, p, in) == kSsdAdmit && calls == 0);
  // smart, L > 0: priority = stamp + value against L
  const uint64_t clocks[] = {10, 40, 60};
  const double values[] = {1.0, 2.0, 3.0};
  const double v = prio(p, 40000, e, 0);  // the value alone
  CHECK(v > 0);
  g.set(true, 0, 3.0, 0.5, clocks, values, 3);
  // last_use 50: the last step with a clock < 50 is (40, 2.0)
  CHECK(g.stamp_at(50) == 2.0);
  const int want = prio(p, 40000, e, 2.0) > 3.0 ? kSsdAdmit : kSsdBelowWatermark;
  CHECK(g.verdict(40000, true, p, in) == want && calls == 1);
  // a big value is admitted, a small one is not
  e = entry(64.0 * (1 << 20), 0, 0, 50);   // (small bytes: high value)
  CHECK(g.verdict(100000, true, p, in) == kSsdAdmit);
  e = entry(8.0 * (1 << 30), 39000, 0, 5); // (old use: stamp = floor 0.5; a tiny gain over its parent, big bytes)
  CHECK(g.stamp_at(5) == 0.5 && g.verdict(40000, true, p, in) == kSsdBelowWatermark);
  // equal priority is rejected (tiercache._smart_admit: priority > L)
  {
    SsdGateIn z = entry(1 << 30, 40000, 0, 1);  // (no gain: value 0, priority = stamp = floor)
    SsdGate h;
    h.set(true, 0, 0.75, 0.75, nullptr, nullptr, 0);
    CHECK(h.verdict(40000, true, p, [&] { return z; }) == kSsdBelowWatermark);
    h.set(true, 0, 0.75, 0.5, nullptr, nullptr, 0);
    CHECK(h.verdict(40000, true, p, [&] { return z; }) == kSsdBelowWatermark);
  }
  // hits / junction / superseded enter the value as in smart_value
  p.hit_weight = 1.0;
  SsdGateIn a = entry(4.0 * (1 << 30), 0, 3, 1, true, false), b = entry(4.0 * (1 << 30), 0, 3, 1, false, true);
  SsdGate k;
  const double pa = prio(p, 60000, a, 0), pb = prio(p, 60000, b, 0);
  k.set(true, 0, (pa + pb) / 2, 0, nullptr, nullptr, 0);
  CHECK(pa > pb);
  CHECK(k.verdict(60000, true, p, [&] { return a; }) == kSsdAdmit);
  CHECK(k.verdict(60000, true, p, [&] { return b; }) == kSsdBelowWatermark);
}

static void test_stamp_at_matches_gd_tier() {
  // the pushed record reproduces GdTier::stamp_at (the frontend's GdTier is the same class in Python)
  GdTier gd;
  SmartPick pk;
  pk.idx = 0;
  uint64_t clock = 0;
  std::vector<uint64_t> cl;
  std::vector<double> va;
  for (int i = 0; i < 9000; i++) {  // (> kMaxSteps: the record is halved, floor moves)
    clock += uint64_t(1 + (i * 7) % 3);
    pk.priority = 0.001 * double(i) + double((i * 13) % 5);
    gd.evicted(pk, clock);
  }
  // the record as the frontend sends its own (clocks, values, floor): GdTier keeps it private, so it is rebuilt from stamp_at (a step at
  // clock c with value v makes stamp_at(c + 1) = v != stamp_at(c))
  SsdGate g;
  for (uint64_t c = 0; c <= clock; c++)
    if (gd.stamp_at(c + 1) != gd.stamp_at(c)) cl.push_back(c), va.push_back(gd.stamp_at(c + 1));
  g.set(true, 0, gd.L(), gd.stamp_at(0), cl.data(), va.data(), cl.size());
  for (uint64_t c = 0; c <= clock + 3; c++) CHECK(g.stamp_at(c) == gd.stamp_at(c));
}

static void test_counters_add_up() {
  SsdGate g;
  SmartParams p;
  g.set(true, 32768, 3.0, 0, nullptr, nullptr, 0);
  int written = 0, dropped = 0;
  for (int i = 0; i < 500; i++) {
    const int64_t tokens = 1000 + (i * 977) % 120000;
    const SsdGateIn e = entry(double(1 + i % 13) * double(256 << 20), double((i * 31) % 20000), uint32_t(i % 3), uint64_t(i));
    if (g.admit(tokens, i % 4 != 0, p, [&] { return e; })) written++;
    else dropped++;
  }
  CHECK(written > 0 && dropped > 0);
  CHECK(g.checked() == written + dropped);
  CHECK(g.rejected() == dropped && g.rejected_short() + g.rejected_watermark() == dropped);
  CHECK(g.rejected_short() > 0 && g.rejected_watermark() > 0);
}

// smart_ram_victim's victim ChainInfo is the victim's entry of chain_info over every visible node.
struct FakeSpilled {
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
  bool in_ram() const { return true; }
};
struct FakeCk {
  std::vector<int32_t> ids;
  mutable std::vector<uint64_t> bh;
  bool ring = false;
  uint64_t owner = 0;
  bool loading = false;
};

static void test_victim_chain_info() {
  std::vector<int32_t> base(40000);
  for (size_t i = 0; i < base.size(); i++) base[i] = int32_t(i * 7 + 3);
  std::map<uint64_t, FakeSpilled> spill;
  FakeSpilled a, b, c;
  a.ids.assign(base.begin(), base.begin() + 20000), a.owner = 1, a.last_use = 1;
  b.ids = base, b.owner = 2, b.last_use = 2;  // (extends a: a's extender of another owner -> a is a junction)
  c.ids.assign(base.begin(), base.begin() + 30000), c.owner = 1, c.last_use = 3;
  spill[1] = a, spill[2] = b, spill[3] = c;
  std::vector<FakeCk> ck;
  SmartParams p;
  p.min_resident_s = 0;
  GdTier gd;
  ChainInfo vci;
  vci.parent_depth = -7;
  auto it = smart_ram_victim(spill, ck, p, gd, 1000.0, 10, [](const FakeSpilled&) { return size_t(1) << 30; },
                             [](uint64_t) { return false; }, nullptr, &vci);
  CHECK(it != spill.end());
  // the same chain_info as smart_spill_chains, at the victim's map index
  const std::vector<ChainInfo> ci = smart_spill_chains(spill, ck);
  size_t idx = 0;
  for (auto j = spill.begin(); j != it; ++j) idx++;
  CHECK(vci.parent_depth == ci[idx].parent_depth && vci.parent == ci[idx].parent);
  CHECK(vci.junction() == ci[idx].junction() && vci.desc_tiers == ci[idx].desc_tiers);
  CHECK(vci.parent_depth != -7);
  // the leaf b (deepest) is the victim, parent c (30000 tokens)
  CHECK(it->first == 2 && vci.parent_depth == 30000);
}

int main() {
  test_off_by_default();
  test_min_tokens();
  test_watermark();
  test_stamp_at_matches_gd_tier();
  test_counters_add_up();
  test_victim_chain_info();
  if (fails) {
    std::printf("%d failure(s)\n", fails);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
