"""CACHE_POLICY=smart: the value-based retention policy, in Python (docs/tiered-cache.md "CACHE_POLICY=smart").

The same algorithm as engine/src/retention.h (SmartParams, prefill_seconds, smart_value, smart_pick, GdTier, chain_info); the engine
uses it for the RAM tier and the VRAM checkpoints, frontend/tiercache.py for the SSD tier and tools/cache_sim.py for the simulator.
frontend/tests/test_cache_policy.py runs this module and the C++ one (engine/tests/smart_policy_cli.cc) on the same random inputs.
No dependencies (the simulator imports it from tools/).

    gain(e)     = T(parent depth, depth): T(a, b) = 1e-6 * (t0 * (b - a) + (k / 2000) * (b^2 - a^2)) seconds
    value(e)    = gain / GiB(bytes) * (1 + hit_weight * hits) * (junction_factor if junction) * (superseded_factor if superseded)
    priority(e) = stamp(e) + value(e)     stamp: the tier's watermark L at the entry's last use
    victim      = min priority among the non-young leaves; no leaf: among all non-young; all young: the oldest
    L           = max(L, victim priority)
"""

import bisect
import math
import os

LEGACY, SMART = "legacy", "smart"
POLICIES = (LEGACY, SMART)
GIB = float(1 << 30)
MIN_BYTES = float(1 << 20)       # retention.h kSmartMinBytes
TIER_VRAM, TIER_RAM, TIER_SSD = 1, 2, 4
LEAF, NON_LEAF, YOUNG = 0, 1, 2  # retention.h SmartReason
REASONS = ("leaf", "non_leaf", "young")
MAX_STEPS = 4096                 # GdTier::kMaxSteps

# The constants: (variable, attribute, default). docs/tiered-cache.md and docs/configuration.md list them.
PARAMS = (("CACHE_PREFILL_T0_US", "t0_us", 54.0),          # us per token at depth 0
          ("CACHE_PREFILL_K_US", "k_us", 0.31),            # us per token per 1,000 tokens of depth
          ("CACHE_SUPERSEDED_FACTOR", "superseded_factor", 0.25),
          ("CACHE_JUNCTION_FACTOR", "junction_factor", 2.0),
          ("CACHE_HIT_WEIGHT", "hit_weight", 0.0),          # value x (1 + hit_weight * hits)
          ("CACHE_MIN_RESIDENT_S", "min_resident_s", 300.0))
ENV_NAMES = ("CACHE_POLICY",) + tuple(p[0] for p in PARAMS)


def policy_of(text):
    """CACHE_POLICY: "legacy" (unset, empty or legacy) or "smart"; ValueError for anything else (retention.h cache_policy_of: -1)."""
    s = "".join((text or "").split()).lower()
    if s in ("", LEGACY):
        return LEGACY
    if s == SMART:
        return SMART
    raise ValueError(f"CACHE_POLICY must be legacy or smart, got {text!r}")


class SmartParams:
    def __init__(self, **kw):
        for _, attr, default in PARAMS:
            setattr(self, attr, float(kw.pop(attr, default)))
        if kw:
            raise TypeError(f"unknown parameters: {sorted(kw)}")

    @classmethod
    def from_env(cls, environ=None):
        """An unset, empty, non-numeric, negative or non-finite value keeps the default (as retention.h smart_params_from_env)."""
        env = os.environ if environ is None else environ
        kw = {}
        for name, attr, _ in PARAMS:
            text = env.get(name)
            if not text or not text.strip():
                continue
            try:
                v = float(text)
            except ValueError:
                continue
            if math.isfinite(v) and v >= 0:
                kw[attr] = v
        return cls(**kw)

    def as_dict(self):
        return {attr: getattr(self, attr) for _, attr, _ in PARAMS}


def prefill_seconds(p, a, b):
    if not b > a:
        return 0.0
    return 1e-6 * (p.t0_us * (b - a) + 0.5 * (p.k_us * 1e-3) * (b * b - a * a))


class Cand:
    """One candidate (retention.h SmartCand)."""
    __slots__ = ("depth", "parent_depth", "bytes", "hits", "junction", "superseded", "leaf", "young", "stamp", "age_key", "state", "ref")

    def __init__(self, depth, parent_depth=0, bytes=0, hits=0, junction=False, superseded=False, leaf=True, young=False, stamp=0.0,
                 age_key=0.0, state=0, ref=None):
        self.depth, self.parent_depth, self.bytes, self.hits = float(depth), float(parent_depth), float(bytes), int(hits)
        self.junction, self.superseded, self.leaf, self.young = bool(junction), bool(superseded), bool(leaf), bool(young)
        self.stamp, self.age_key, self.state, self.ref = float(stamp), float(age_key), float(state), ref


def smart_value(p, c):
    gib = max(c.bytes, MIN_BYTES) / GIB
    v = prefill_seconds(p, c.parent_depth, c.depth) / gib * (1.0 + p.hit_weight * float(c.hits))
    if c.junction:
        v *= p.junction_factor
    if c.superseded:
        v *= p.superseded_factor
    return v


def smart_priority(p, c):
    return c.stamp + smart_value(p, c)


class Pick:
    __slots__ = ("idx", "priority", "reason", "young_skipped")

    def __init__(self, idx=-1, priority=0.0, reason=LEAF, young_skipped=False):
        self.idx, self.priority, self.reason, self.young_skipped = idx, priority, reason, young_skipped

    def __repr__(self):
        return f"Pick({self.idx}, {self.priority:.6g}, {REASONS[self.reason]}, young_skipped={self.young_skipped})"


def smart_pick(cands, p):
    """retention.h smart_pick: the victim's index (Pick.idx, -1 when there is none)."""
    if not cands:
        return Pick()
    any_old = any(not c.young for c in cands)
    if not any_old:
        b = 0
        for i in range(1, len(cands)):
            if cands[i].age_key < cands[b].age_key:
                b = i
        return Pick(b, smart_priority(p, cands[b]), YOUNG, False)
    old_leaf = any(not c.young and c.leaf for c in cands)
    r = Pick(-1, 0.0, LEAF if old_leaf else NON_LEAF, any(c.young for c in cands))
    for i, c in enumerate(cands):
        if c.young or (old_leaf and not c.leaf):
            continue
        pr = smart_priority(p, c)
        if r.idx < 0 or pr < r.priority or (pr == r.priority and c.age_key < cands[r.idx].age_key):
            r.idx, r.priority = i, pr
    return r


def smart_choose(backups, sole, p):
    """retention.h smart_ram_victim's class rule: (class, Pick) with class 0 (an idle backup) first unless the best backup frees only
    its state while the best sole copy frees >= 4x that; (-1, Pick()) with no candidate."""
    k0, k1 = smart_pick(backups, p), smart_pick(sole, p)
    if k0.idx >= 0:
        b = backups[k0.idx]
        best_sole = max((c.bytes for c in sole), default=0.0)
        if k1.idx >= 0 and b.bytes <= b.state and best_sole > 0 and best_sole >= 4 * b.bytes:
            return 1, k1
        return 0, k0
    return (1, k1) if k1.idx >= 0 else (-1, Pick())


class GdTier:
    """retention.h GdTier: the watermark L, the stamps by use clock (stamp_at) and the counters."""

    def __init__(self):
        self.L, self.floor, self.prio_sum = 0.0, 0.0, 0.0
        self.ev = [0, 0, 0]
        self.young_skipped = 0
        self.clocks, self.values = [], []

    def stamp_at(self, clock):
        i = bisect.bisect_left(self.clocks, clock)  # (the raises at clocks < clock)
        return self.values[i - 1] if i > 0 else self.floor

    def evicted(self, pick, now):
        if pick.idx < 0:
            return
        self.ev[pick.reason] += 1
        self.young_skipped += 1 if pick.young_skipped else 0
        self.prio_sum += pick.priority
        self.L = max(self.L, pick.priority)
        if self.clocks and self.clocks[-1] >= now:
            self.values[-1] = self.L
        else:
            self.clocks.append(now)
            self.values.append(self.L)
        if len(self.clocks) > MAX_STEPS:
            cut = len(self.clocks) // 2
            self.floor = self.values[cut - 1]
            del self.clocks[:cut], self.values[:cut]

    def stats(self):
        n = sum(self.ev)
        return {"evictions_leaf": self.ev[0], "evictions_non_leaf": self.ev[1], "evictions_young": self.ev[2],
                "young_skipped": self.young_skipped, "victim_priority_mean": self.prio_sum / n if n else None, "watermark": self.L}


def engine_counters(sp):
    """The engine's smart counters (q27lib spill_stats: cache_smart, ram_smart_*, vram_smart_*) as {tier: GdTier.stats()-like dict}; {}
    from an engine without them."""
    out = {}
    for tier in ("ram", "vram"):
        k = f"{tier}_smart_evictions_leaf"
        if k not in sp:
            continue
        n = sum(sp.get(f"{tier}_smart_evictions_{r}", 0) for r in REASONS)
        out[tier] = {"evictions_leaf": sp[k], "evictions_non_leaf": sp.get(f"{tier}_smart_evictions_non_leaf", 0),
                     "evictions_young": sp.get(f"{tier}_smart_evictions_young", 0), "young_skipped": sp.get(f"{tier}_smart_young_skipped", 0),
                     "victim_priority_mean": sp.get(f"{tier}_smart_victim_priority_mean_milli", 0) / 1000 if n else None,
                     "watermark": sp.get(f"{tier}_smart_watermark_milli", 0) / 1000}
    return out


def status_block(sp, ts):
    """For the stats UI (stats.tiers_block): the policy name and, per tier (ram / vram from the engine, ssd from tiercache), the
    evictions by reason, the mean victim priority and the watermark. sp: engine.spill_stats(), ts: TierPolicy.status()."""
    policy = ts.get("cache_policy") or (SMART if sp.get("cache_smart") == 1 else LEGACY)
    tiers = engine_counters(sp)
    if "ssd_smart_evictions_leaf" in ts:
        tiers["ssd"] = {k[len("ssd_smart_"):]: v for k, v in ts.items() if k.startswith("ssd_smart_")}
    return {"cache_policy": policy, "cache_smart": tiers if policy == SMART else None,
            "ssd_not_admitted": ts.get("ssd_not_admitted", 0)}


def metrics(sp, ts):
    """/metrics lines of the policy: q27:cache_policy{policy=...} 1, and under smart q27:cache_evictions_total{tier,reason},
    q27:cache_young_skipped_total{tier}, q27:cache_victim_priority_mean{tier}, q27:cache_watermark{tier}, q27:cache_ssd_not_admitted_total."""
    st = status_block(sp, ts)
    out = {f'q27:cache_policy{{policy="{st["cache_policy"]}"}}': 1}
    for tier, c in (st["cache_smart"] or {}).items():
        for r in REASONS:
            out[f'q27:cache_evictions_total{{tier="{tier}",reason="{r}"}}'] = c.get(f"evictions_{r}", 0)
        out[f'q27:cache_young_skipped_total{{tier="{tier}"}}'] = c.get("young_skipped", 0)
        if c.get("victim_priority_mean") is not None:
            out[f'q27:cache_victim_priority_mean{{tier="{tier}"}}'] = c["victim_priority_mean"]
        out[f'q27:cache_watermark{{tier="{tier}"}}'] = c.get("watermark") or 0
    if st["cache_policy"] == SMART:
        out["q27:cache_ssd_not_admitted_total"] = st["ssd_not_admitted"]
    return out


def chain_info(nodes):
    """retention.h chain_info on (ids, owner, tiers) triples (ids: a sequence of token ids or block hashes): per node a dict with
    parent (index or -1), parent_depth, desc_tiers, branch, shared. O(n^2): for tests and small sets (the simulator keeps a trie)."""
    ids = [tuple(n[0]) for n in nodes]
    out = [{"parent": -1, "parent_depth": 0, "desc_tiers": 0, "branch": False, "shared": False} for _ in nodes]

    def pre(a, b):  # a strict prefix of b
        return len(a) < len(b) and b[:len(a)] == a

    for i, c in enumerate(ids):
        ext = [j for j in range(len(ids)) if j != i and pre(c, ids[j])]
        if not ext:
            continue
        deepest = max(ext, key=lambda j: (len(ids[j]), -j))
        for j in ext:
            out[i]["desc_tiers"] |= nodes[j][2]
            out[i]["shared"] = out[i]["shared"] or nodes[j][1] != nodes[i][1]
            if ids[deepest][:len(ids[j])] != ids[j]:
                out[i]["branch"] = True
            if len(c) > out[j]["parent_depth"] or out[j]["parent"] < 0:
                out[j]["parent"], out[j]["parent_depth"] = i, len(c)
    return out
