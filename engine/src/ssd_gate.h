// SSD admission BEFORE the engine writes (docs/tiered-cache.md "SSD admission in the engine"). Pure C++ (no CUDA):
// tests/test_ssd_gate.cc builds it with g++ alone.
//
// The engine's RAM cap (Model::spill_enforce_ram) writes a sole-copy victim to the SSD (spill_write). frontend/tiercache.py used to judge
// such an entry only afterwards (the first tick that saw it on the SSD: too short, or under CACHE_POLICY=smart below the SSD's
// watermark -> deleted again), so most of those writes were wasted I/O and SSD wear. The frontend now pushes its admission state to the
// engine (q27_spill_set_ssd_gate, from TierPolicy.start / every tick) and the engine runs the same test before spill_write; a rejected
// victim is dropped (as an entry without an SSD tier) and counted here. The frontend keeps its after-the-fact checks as a safety net.
//
// The rule (ssd_verdict), the same as tiercache.py's for the tick path:
//   1. tokens < min_tokens (SPILL_SSD_MIN_TOKENS; 0: off)               -> rejected ("short"; tiercache._short)
//   2. smart (the engine runs CACHE_POLICY=smart) and the SSD's watermark L > 0:
//        priority = stamp_at(last_use) + smart_value(entry)              (retention.h smart_value; stamp from the pushed SSD record)
//        priority <= L                                                   -> rejected ("watermark"; tiercache._smart_admit)
//      L <= 0 (nothing was evicted from the SSD yet): no watermark test.
//   anything else                                                        -> admitted (written as before)
// The smart inputs are those tiercache._cand takes for a RAM entry: depth = tokens, parent_depth / junction from the chain over every
// visible node (spill_meta's parent_pos / junction), hits, superseded (spill_meta's plain flag), bytes = what writing it costs now
// (spill_new_disk: spill_list_ex's new_disk_bytes), stamp = the SSD tier's L at the entry's last use. leaf / young do not enter a
// priority. SPILL_SSD_WARM_ONLY / the probation quota (legacy) stay in the frontend only (they need the agents' hits across tiers and
// the `auto:` owners the frontend tracks).
//
// Never pushed (an older frontend, or the SSD tier is off): the gate is OFF and the engine writes exactly as before.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "retention.h"

namespace q27 {

enum SsdVerdict : int { kSsdAdmit = 0, kSsdShort = 1, kSsdBelowWatermark = 2 };

// The value / priority inputs of one entry (filled only when the watermark test runs: see SsdGate::verdict).
struct SsdGateIn {
  double bytes = 0;          // SSD bytes writing it costs now (manifest + chunk files not on the SSD yet)
  double parent_depth = 0;   // tokens of the deepest retained strict prefix (0: none)
  uint32_t hits = 0;
  bool junction = false, superseded = false;
  uint64_t last_use = 0;     // the engine's use clock at the entry's last use
};

class SsdGate {
 public:
  // enabled = false: off (today's behaviour). min_tokens < 0 counts as 0. watermark: the SSD tier's L (frontend GdTier.L); floor,
  // clocks[0..n), values[0..n): its stamp record (GdTier.floor / clocks / values, clocks ascending).
  void set(bool enabled, int64_t min_tokens, double watermark, double floor, const uint64_t* clocks, const double* values, size_t n) {
    on_ = enabled;
    min_tokens_ = std::max<int64_t>(0, min_tokens);
    L_ = watermark, floor_ = floor;
    steps_.clear();
    if (enabled && clocks && values) {
      steps_.reserve(n);
      for (size_t i = 0; i < n; i++) steps_.emplace_back(clocks[i], values[i]);
    }
  }
  bool on() const { return on_; }
  int64_t min_tokens() const { return min_tokens_; }
  double watermark() const { return L_; }

  // The SSD tier's L at use clock c: as GdTier::stamp_at and frontend cache_policy.GdTier.stamp_at (the value of the last step with
  // a clock < c; floor before the first).
  double stamp_at(uint64_t c) const {
    auto it = std::lower_bound(steps_.begin(), steps_.end(), c,
                               [](const std::pair<uint64_t, double>& s, uint64_t x) { return s.first < x; });
    if (it == steps_.begin()) return floor_;
    return std::prev(it)->second;
  }

  // The verdict for an entry of `tokens` tokens (no counting). smart: the engine runs CACHE_POLICY=smart. in(): returns SsdGateIn,
  // called only when the watermark test runs (its inputs cost a chain walk / chunk names).
  template <class In>
  int verdict(int64_t tokens, bool smart, const SmartParams& p, In&& in) const {
    if (!on_) return kSsdAdmit;
    if (min_tokens_ > 0 && tokens < min_tokens_) return kSsdShort;
    if (!smart || !(L_ > 0)) return kSsdAdmit;
    const SsdGateIn e = in();
    SmartCand c;
    c.depth = double(tokens), c.parent_depth = e.parent_depth, c.bytes = e.bytes, c.hits = e.hits;
    c.junction = e.junction, c.superseded = e.superseded;
    c.stamp = stamp_at(e.last_use);
    return smart_priority(p, c) > L_ ? kSsdAdmit : kSsdBelowWatermark;
  }

  // verdict() + the counters. True: write it.
  template <class In>
  bool admit(int64_t tokens, bool smart, const SmartParams& p, In&& in) {
    if (!on_) return true;
    const int v = verdict(tokens, smart, p, in);
    checked_++;
    if (v == kSsdShort) rejected_short_++;
    if (v == kSsdBelowWatermark) rejected_watermark_++;
    return v == kSsdAdmit;
  }

  int64_t checked() const { return checked_; }  // victims the gate judged (admitted + rejected)
  int64_t rejected() const { return rejected_short_ + rejected_watermark_; }
  int64_t rejected_short() const { return rejected_short_; }
  int64_t rejected_watermark() const { return rejected_watermark_; }

 private:
  bool on_ = false;
  int64_t min_tokens_ = 0;
  double L_ = 0, floor_ = 0;
  std::vector<std::pair<uint64_t, double>> steps_;
  int64_t checked_ = 0, rejected_short_ = 0, rejected_watermark_ = 0;
};

}  // namespace q27
