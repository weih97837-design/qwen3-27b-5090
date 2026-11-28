"""The SSD gate (engine/src/ssd_gate.h; docs/tiered-cache.md "SSD 寫入前的准入"): TierPolicy pushes its SSD admission to the engine at
start() and every tick, so the engine's RAM cap rejects a victim BEFORE writing it, with the same rule the policy applies after the fact
(SPILL_SSD_MIN_TOKENS; under smart the SSD's watermark). The fake engine applies the pushed gate the way spill_enforce_ram does (with
cache_policy.ssd_verdict, the Python twin of SsdGate::verdict; test_cache_policy.py checks the twin against the C++). No GPU."""

import bisect
import pathlib
import random
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cache_policy as cp  # noqa: E402
import stats  # noqa: E402
import tiercache  # noqa: E402
from test_retention import RetEngine  # noqa: E402
from test_tiercache import B, Clock  # noqa: E402
from tiercache import TierPolicy  # noqa: E402


class GateEngine(RetEngine):
    """RetEngine with spill_set_ssd_gate (the calls are recorded) and a RAM-pressure demotion that applies the pushed gate as
    Model::spill_enforce_ram does: admitted -> written to a file, rejected -> dropped and counted (ssd_gate_rejected)."""

    def __init__(self, smart=False):
        super().__init__()
        self.smart = smart  # (the engine's own CACHE_POLICY: the watermark test runs only under smart)
        self.gate_calls = []
        self.gate = None
        self.attempts = self.written_by_cap = 0
        self.rejected = {"short": 0, "watermark": 0}

    def spill_set_ssd_gate(self, on, min_tokens=0, watermark=0.0, floor=0.0, clocks=(), values=()):
        self.gate_calls.append((on, min_tokens, watermark, floor, list(clocks), list(values)))
        self.gate = self.gate_calls[-1] if on else None
        return True

    def stamp_at(self, c):
        _, _, _, floor, clocks, values = self.gate
        i = bisect.bisect_left(clocks, c)
        return values[i - 1] if i > 0 else floor

    def ram_pressure(self, entry_id, directory, params):
        """The RAM cap made entry_id its victim (a sole copy): the gate decides before the write."""
        e = self.entries[entry_id]
        self.attempts += 1
        if self.gate is not None:
            _, min_tokens, L, _, _, _ = self.gate
            v = cp.ssd_verdict(params, min_tokens, L, self.stamp_at, e["pos"], self.smart,
                               lambda: (e["ram_bytes"], e.get("parent_pos", 0), e.get("hits", 0), e.get("junction", False),
                                        e.get("superseded", False), e.get("last_use", 0)))
            if v != cp.SSD_ADMIT:
                self.rejected["short" if v == cp.SSD_SHORT else "watermark"] += 1
                self.entries.pop(entry_id)
                return False
        path = pathlib.Path(directory) / f"cap-{entry_id}{tiercache.SUFFIX}"
        path.write_bytes(b"x" * e["ram_bytes"])
        e.update(ram_bytes=0, on_disk=True, path=str(path))
        self.written_by_cap += 1
        return True

    def spill_stats(self):
        st = super().spill_stats()
        st.update(ssd_gate_on=int(self.gate is not None), ssd_gate_rejected=sum(self.rejected.values()),
                  ssd_gate_rejected_short=self.rejected["short"], ssd_gate_rejected_watermark=self.rejected["watermark"],
                  ssd_gate_checked=self.attempts if self.gate is not None else 0)
        return st


def make(tmp_path, smart=False, ssd_gib=10 ** 9, eng=None, **kw):
    eng = eng if eng is not None else GateEngine(smart=smart)
    clock = Clock()
    d = tmp_path / "spill"
    tier = TierPolicy(eng, str(d), ram_retention_s=10 ** 6, ssd_gib=ssd_gib * B, ssd_min_free_gib=0, ttl_days=7, tick_s=0, clock=clock,
                      disk_free=lambda _d: 10 ** 12, log=lambda _m: None, ssd_min_tokens=8192,
                      cache_policy="smart" if smart else "legacy", smart_params=cp.SmartParams(min_resident_s=0), **kw)
    tier.start()
    return types.SimpleNamespace(eng=eng, tier=tier, dir=d, clock=clock)


def test_legacy_pushes_min_tokens_at_start_and_every_tick(tmp_path):
    env = make(tmp_path)
    assert env.eng.gate_calls == [(True, 8192, 0.0, 0.0, [], [])]  # (start: before any victim is written)
    for t in (1, 2, 3):
        env.clock.t = t
        env.tier.tick(idle=True)
    assert len(env.eng.gate_calls) == 4 and all(c == (True, 8192, 0.0, 0.0, [], []) for c in env.eng.gate_calls)
    assert env.tier.status()["ssd_gate"] == {"min_tokens": 8192, "watermark": 0.0, "steps": 0, "smart": False}


def test_smart_pushes_the_watermark_and_its_stamp_record(tmp_path):
    env = make(tmp_path, smart=True)
    assert env.eng.gate_calls[-1] == (True, 8192, 0.0, 0.0, [], [])  # (nothing evicted from the SSD yet: L 0, no watermark test)
    env.tier.gd.evicted(cp.Pick(0, 3.5, cp.LEAF), 40)
    env.tier.gd.evicted(cp.Pick(0, 7.25, cp.LEAF), 90)
    env.clock.t = 1
    env.tier.tick(idle=True)
    on, mt, L, floor, clocks, values = env.eng.gate_calls[-1]
    assert (on, mt, L, floor, clocks, values) == (True, 8192, 7.25, 0.0, [40, 90], [3.5, 7.25])
    # the push after make_room carries the watermark a tick's own SSD eviction raised
    a = env.eng.add_disk(env.dir, 300, pos=40_000)
    b = env.eng.add_disk(env.dir, 300, pos=60_000)
    env.eng.entries[a].update(last_use=100)
    env.eng.entries[b].update(last_use=100)
    env.tier.written.update({a: 0.0, b: 0.0})
    env.tier.budget = 400
    env.clock.t = 2
    env.tier.tick(idle=True)
    assert env.tier.gd.L > 7.25 and env.eng.gate_calls[-1][2] == env.tier.gd.L
    assert env.eng.gate_calls[-1][4] == env.tier.gd.clocks and env.eng.gate_calls[-1][5] == env.tier.gd.values


def test_nothing_is_pushed_without_an_ssd_tier(tmp_path):
    env = make(tmp_path, ssd_gib=0)
    env.clock.t = 1
    env.tier.tick(idle=True)
    env.tier.flush(5)
    assert env.eng.gate_calls == [] and env.tier.status()["ssd_gate"] is None


def test_an_engine_without_the_call_or_a_failing_one(tmp_path):
    env = make(tmp_path, eng=RetEngine())  # (no spill_set_ssd_gate: the engine writes as before, the policy judges after)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.tier.gate is None and env.tier.last_error is None

    class Old(GateEngine):
        def spill_set_ssd_gate(self, *a, **k):  # (q27lib with an older library: nothing done)
            return False

    env = make(tmp_path / "old", eng=Old())
    assert env.tier.gate is None

    class Boom(GateEngine):
        def spill_set_ssd_gate(self, *a, **k):
            raise RuntimeError("q27_spill_set_ssd_gate: bad")

    env = make(tmp_path / "boom", eng=Boom())
    env.clock.t = 1
    env.tier.tick(idle=True)  # (never raises into the worker)
    assert env.tier.gate is None and "spill_set_ssd_gate" in env.tier.last_error


def _ram(eng, rng, i):
    pos = rng.choice([2_000, 6_000, 8_191, 8_192, 12_000, 20_000, 40_000, 90_000, 130_000])
    eid = eng.add_ram(rng.choice([64, 300, 1200, 5000]), pos=pos, hits=rng.choice([0, 0, 1, 3]),
                      superseded=rng.random() < 0.2)
    eng.entries[eid].update(parent_pos=rng.choice([0, 0, pos // 2, max(0, pos - 500)]), junction=rng.random() < 0.2,
                            last_use=rng.randrange(0, 200))
    return eid


def test_written_plus_rejected_is_every_demotion_and_the_safety_net_finds_nothing(tmp_path):
    for smart in (False, True):
        env = make(tmp_path / str(smart), smart=smart)
        if smart:  # (an SSD that already evicted: a watermark, with a record of raises at use clocks 50 / 120)
            env.tier.gd.evicted(cp.Pick(0, 400.0, cp.LEAF), 50)  # (the values here are large: a few KiB entries, floored to 1 MiB)
            env.tier.gd.evicted(cp.Pick(0, 1500.0, cp.LEAF), 120)
            env.clock.t = 1
            env.tier.tick(idle=True)
        rng = random.Random(7 + smart)
        for i in range(300):
            env.eng.ram_pressure(_ram(env.eng, rng, i), env.tier.files_dir, env.tier.smart_params)
        sp = env.eng.spill_stats()
        assert env.eng.written_by_cap + sp["ssd_gate_rejected"] == env.eng.attempts == 300
        assert sp["ssd_gate_checked"] == 300 and sp["ssd_gate_rejected"] == sp["ssd_gate_rejected_short"] + sp["ssd_gate_rejected_watermark"]
        assert sp["ssd_gate_rejected_short"] > 0 and env.eng.written_by_cap > 0
        assert (sp["ssd_gate_rejected_watermark"] > 0) == smart
        # every entry the cap wrote passes the policy's own after-the-fact checks: the safety net deletes none of them
        before = (env.tier.not_admitted, env.tier.skipped_short)
        drops = len(env.eng.drops)
        env.clock.t = 2
        env.tier.tick(idle=True)
        assert (env.tier.not_admitted, env.tier.skipped_short) == before and len(env.eng.drops) == drops
        blk = stats.tiers_block(env.eng, env.tier, environ={})
        assert blk["ssd_gate_rejected"] == sp["ssd_gate_rejected"] and blk["ssd_gate_on"] == 1
        assert blk["ssd_gate"]["smart"] == smart


def test_without_a_push_the_engine_writes_everything_and_the_policy_judges_after(tmp_path):
    """The old behaviour (an older frontend, or an engine without the gate): every victim is written, the tick deletes the ones below
    the watermark afterwards (ssd_not_admitted). Shows what the gate saves."""
    env = make(tmp_path, smart=True)
    env.eng.gate = None
    env.eng.spill_set_ssd_gate = None  # (the policy cannot push: written first, judged after)
    env.tier.gd.evicted(cp.Pick(0, 1500.0, cp.LEAF), 120)
    rng = random.Random(3)
    ids = [_ram(env.eng, rng, i) for i in range(100)]
    for eid in ids:
        env.eng.ram_pressure(eid, env.tier.files_dir, env.tier.smart_params)
    assert env.eng.written_by_cap == 100
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.tier.not_admitted > 0  # (the wasted writes the gate now avoids)


def _stamp_of(gate):
    _, _, _, floor, clocks, values = gate
    return lambda c: values[bisect.bisect_left(clocks, c) - 1] if bisect.bisect_left(clocks, c) > 0 else floor


def test_the_pushed_gate_decides_as_the_policy_does(tmp_path):
    """On random RAM entries: the verdict from the pushed values (what the engine computes) equals TierPolicy's own tick-path decision
    (_short, then under smart _smart_admit)."""
    rng = random.Random(11)
    for smart in (False, True):
        env = make(tmp_path / f"a{smart}", smart=smart, eng=GateEngine(smart=smart))
        for k in range(rng.randrange(1, 40)):
            env.tier.gd.evicted(cp.Pick(0, rng.uniform(0, 3000), cp.LEAF), rng.randrange(0, 300))
        env.clock.t = 1
        env.tier.tick(idle=True)
        gate = env.eng.gate_calls[-1]
        stamp = _stamp_of(gate)
        agree, seen = 0, set()
        for i in range(400):
            e = env.eng.entries[_ram(env.eng, rng, i)]
            policy = not env.tier._short(e) and (not smart or env.tier._smart_admit(e, env.clock.t))
            v = cp.ssd_verdict(env.tier.smart_params, gate[1], gate[2], stamp, e["pos"], smart,
                               lambda: (e["ram_bytes"], e["parent_pos"], e["hits"], e["junction"], e["superseded"], e["last_use"]))
            assert (v == cp.SSD_ADMIT) == policy, (e, v)
            agree += 1
            seen.add(v)
        assert agree == 400
        assert seen == ({cp.SSD_ADMIT, cp.SSD_SHORT, cp.SSD_BELOW_WATERMARK} if smart else {cp.SSD_ADMIT, cp.SSD_SHORT})
