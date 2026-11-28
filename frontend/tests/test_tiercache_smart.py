"""CACHE_POLICY=smart on the SSD tier (frontend/tiercache.py with frontend/cache_policy.py; docs/tiered-cache.md "CACHE_POLICY=smart").
The legacy order is tests/test_retention.py's (unchanged: the default). No GPU."""

import pathlib
import random
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cache_policy as cp  # noqa: E402
import stats  # noqa: E402
import tiercache  # noqa: E402
from test_retention import RetEngine  # noqa: E402
from test_tiercache import B, Clock, FakeEngine  # noqa: E402
from tiercache import TierPolicy  # noqa: E402

GiB = 1 << 30


def make(tmp_path, budget=10_000, min_resident_s=0.0, **kw):
    eng = RetEngine()
    clock = Clock()
    d = tmp_path / "spill"
    tier = TierPolicy(eng, str(d), ram_retention_s=0, ssd_gib=budget * B, ssd_min_free_gib=0, ttl_days=7, tick_s=0, clock=clock,
                      disk_free=lambda _d: 10 ** 12, log=lambda _m: None, ssd_min_tokens=8192, cache_policy="smart",
                      smart_params=cp.SmartParams(min_resident_s=min_resident_s), **kw)
    tier.start()
    return types.SimpleNamespace(eng=eng, tier=tier, dir=d, clock=clock)


def disk(env, size, pos, parent=0, desc=0, junction=False, hits=0, superseded=False, last_use=0, owner=1):
    i = env.eng.add_disk(env.dir, size, owner=owner, hits=hits, superseded=superseded, pos=pos)
    env.eng.entries[i].update(parent_pos=parent, desc_tiers=desc, junction=junction, last_use=last_use)
    return i


def test_from_env_reads_the_policy_and_its_constants(monkeypatch):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    for v in ("CACHE_POLICY", "CACHE_MIN_RESIDENT_S", "CACHE_HIT_WEIGHT"):
        monkeypatch.delenv(v, raising=False)
    t = TierPolicy.from_env(FakeEngine(), log=lambda _m: None)
    assert t.cache_policy == "smart" and t.smart  # (the default)
    monkeypatch.setenv("CACHE_POLICY", "legacy")
    t = TierPolicy.from_env(FakeEngine(), log=lambda _m: None)
    assert t.cache_policy == "legacy" and not t.smart  # (explicit legacy wins)
    monkeypatch.setenv("CACHE_POLICY", "Smart")
    monkeypatch.setenv("CACHE_MIN_RESIDENT_S", "60")
    monkeypatch.setenv("CACHE_HIT_WEIGHT", "0")
    t = TierPolicy.from_env(FakeEngine(), log=lambda _m: None)
    assert t.smart and t.smart_params.min_resident_s == 60 and t.smart_params.hit_weight == 0
    monkeypatch.setenv("CACHE_POLICY", "lfu")
    with pytest.raises(ValueError, match="CACHE_POLICY"):
        TierPolicy.from_env(FakeEngine(), log=lambda _m: None)


def test_victim_is_the_lowest_value_leaf_and_the_watermark_rises(tmp_path):
    env = make(tmp_path)
    # a chain on the SSD: 30K -> 60K (the 30K one is not a leaf of the SSD); an unrelated 40K entry; a junction 20K one
    shallow = disk(env, 300, 30_000, desc=cp.TIER_SSD)
    deep = disk(env, 300, 60_000, parent=30_000)
    other = disk(env, 300, 40_000)
    junc = disk(env, 300, 20_000, junction=True)
    env.tier.written.update({shallow: 1.0, deep: 2.0, other: 3.0, junc: 4.0})
    env.clock.t = 100.0
    p = env.tier.smart_params
    # values (same bytes): deep T(30K, 60K), other T(0, 40K), junc 2 * T(0, 20K); the shallow one is no candidate while leaves exist
    vals = {deep: cp.prefill_seconds(p, 30_000, 60_000), other: cp.prefill_seconds(p, 0, 40_000), junc: 2 * cp.prefill_seconds(p, 0, 20_000)}
    first = min(vals, key=vals.get)
    env.tier.budget = 900  # one of four must go
    assert env.tier.make_room(0)
    assert env.eng.drops == [first] and shallow in env.eng.entries
    assert env.tier.gd.L > 0 and env.tier.gd.stats()["evictions_leaf"] == 1
    st = env.tier.status()
    assert st["cache_policy"] == "smart" and st["ssd_smart_evictions_leaf"] == 1 and st["ssd_smart_watermark"] == env.tier.gd.L


def test_young_entries_are_protected_unless_all_are_young(tmp_path):
    env = make(tmp_path, min_resident_s=300)
    env.clock.t = 10_000
    old = disk(env, 300, 60_000)       # high value, old
    young = disk(env, 300, 18_000)     # low value, written 10 s ago
    env.tier.written.update({old: 1_000.0, young: 9_990.0})
    env.tier.budget = 400
    assert env.tier.make_room(0)
    assert env.eng.drops == [old] and env.tier.gd.stats()["young_skipped"] == 1
    env2 = make(tmp_path / "2", min_resident_s=300)
    env2.clock.t = 10_000
    a, b = disk(env2, 300, 60_000), disk(env2, 300, 18_000)
    env2.tier.written.update({a: 9_950.0, b: 9_990.0})  # both young: the oldest goes
    env2.tier.budget = 400
    assert env2.tier.make_room(0)
    assert env2.eng.drops == [a] and env2.tier.gd.stats()["evictions_young"] == 1


def test_superseded_entries_are_written_not_dropped(tmp_path):
    env = make(tmp_path)
    sup = env.eng.add_ram(100, owner=1, superseded=True, pos=40_000)
    env.eng.entries[sup]["superseded_durable"] = True  # (legacy would drop it: tests/test_retention.py)
    env.clock.t = 0
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.eng.exports == [sup] and not env.eng.drops and env.tier.status()["ssd_skipped_superseded"] == 0
    # and flush writes it too
    env2 = make(tmp_path / "f")
    s2 = env2.eng.add_ram(100, pos=40_000, superseded=True)
    env2.eng.entries[s2]["superseded_durable"] = True
    assert env2.tier.flush(10)["written"] == 1 and env2.eng.exports == [s2]


def test_admission_below_the_watermark_is_refused(tmp_path):
    env = make(tmp_path)
    env.tier.gd.evicted(cp.Pick(0, 5000.0, cp.LEAF), 100)  # (the SSD evicted an entry of priority 5000 at use clock 100)
    low = env.eng.add_ram(200, owner=1, pos=20_000)
    env.eng.entries[low].update(last_use=50)  # used before that eviction: stamp 0 + value (~1,200: 200 B, floored to 1 MiB) < L
    high = env.eng.add_ram(200, owner=1, pos=40_000)
    env.eng.entries[high].update(last_use=150)  # used after it: stamp 5000 + value > L
    env.clock.t = 0
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.eng.exports == [high] and low in env.eng.drops and env.tier.not_admitted == 1
    assert env.tier.status()["ssd_not_admitted"] == 1


def test_an_entry_the_engine_wrote_is_checked_at_first_sight(tmp_path):
    env = make(tmp_path)
    env.tier.tick(idle=True)
    env.tier.gd.evicted(cp.Pick(0, 50.0, cp.LEAF), 100)
    weak = disk(env, 100, 20_000, last_use=10)    # appeared on the SSD (the engine's RAM cap wrote it), old last use
    strong = disk(env, 100, 40_000, last_use=200)
    for i in (weak, strong):
        env.eng.entries[i]["unique_disk_bytes"] = 1 << 30
    env.clock.t = 5
    env.tier.tick(idle=True)
    assert weak in env.eng.drops and strong in env.eng.entries and env.tier.not_admitted == 1


def test_legacy_ignores_the_smart_fields(tmp_path):
    """The default policy never looks at parent_pos / desc_tiers / junction: the legacy order of test_retention.py."""
    env = make(tmp_path)
    env.tier.smart = False
    env.tier.cache_policy = "legacy"
    env.tier.fair_agents = env.tier.cold_first = False
    a = disk(env, 300, 60_000, desc=cp.TIER_SSD)   # (a smart non-leaf)
    b = disk(env, 300, 18_000, junction=True)
    env.tier.written.update({a: 1.0, b: 2.0})
    env.tier.budget = 400
    assert env.tier.make_room(0)
    assert env.eng.drops == [a]  # (legacy: the oldest written, whatever the chain)
    assert env.tier.gd.stats()["evictions_leaf"] == 0


def test_status_block_and_metrics():
    sp = {"cache_smart": 1, "ram_smart_evictions_leaf": 4, "ram_smart_evictions_non_leaf": 1, "ram_smart_evictions_young": 0,
          "ram_smart_young_skipped": 2, "ram_smart_victim_priority_mean_milli": 2500, "ram_smart_watermark_milli": 4000,
          "vram_smart_evictions_leaf": 0, "vram_smart_evictions_non_leaf": 0, "vram_smart_evictions_young": 0,
          "vram_smart_young_skipped": 0, "vram_smart_victim_priority_mean_milli": 0, "vram_smart_watermark_milli": 0}
    ts = {"cache_policy": "smart", "ssd_not_admitted": 3, **{f"ssd_smart_{k}": v for k, v in cp.GdTier().stats().items()}}
    b = cp.status_block(sp, ts)
    assert b["cache_policy"] == "smart" and b["cache_smart"]["ram"]["victim_priority_mean"] == 2.5
    assert b["cache_smart"]["vram"]["victim_priority_mean"] is None and "ssd" in b["cache_smart"]
    m = cp.metrics(sp, ts)
    assert m['q27:cache_policy{policy="smart"}'] == 1 and m['q27:cache_evictions_total{tier="ram",reason="leaf"}'] == 4
    assert m['q27:cache_young_skipped_total{tier="ram"}'] == 2 and m["q27:cache_ssd_not_admitted_total"] == 3
    leg = cp.metrics({}, {"cache_policy": "legacy"})
    assert leg == {'q27:cache_policy{policy="legacy"}': 1}
    assert cp.status_block({}, {})["cache_policy"] == "legacy"  # (an older engine and policy)

    class Eng:
        def spill_stats(self):
            return {"entries": 0, "in_ram": 0, "on_disk": 0, "ram_bytes": 0, "spilled": 0, "restored": 0, "written": 0,
                    "dropped": 0, "subsumed": 0, "failed": 0, **sp}

    class Tiers:
        directory = None
        ram_retention_s = 1
        ttl_s = 86400.0

        def status(self):
            return dict(ts, ssd_bytes=0)
    out = stats.tiers_block(Eng(), Tiers(), environ={})
    assert out["cache_policy"] == "smart" and out["cache_smart"]["ram"]["evictions_leaf"] == 4


def test_q27lib_spill_meta_parses_the_smart_columns(monkeypatch):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "engine" / "python"))
    import q27lib

    class E(q27lib.Engine):
        has_retention = True

        def __init__(self):
            pass

        def _text(self, fn, what):
            return [["7", "3", "2", "1", "0"], ["8", "3", "0", "0", "0", "30720", "6", "1"]]
    monkeypatch.setattr(q27lib, "lib", lambda: types.SimpleNamespace(q27_spill_meta=None))
    m = E().spill_meta()
    assert m[7] == {"owner": 3, "hits": 2, "superseded": True, "superseded_durable": False}
    assert m[8]["parent_pos"] == 30720 and m[8]["desc_tiers"] == 6 and m[8]["junction"] is True


# ---------------------------------------------------------------- the SSD order under smart agrees with the engine's smart_pick

@pytest.fixture(scope="module")
def engine_pick(tmp_path_factory):
    import shutil
    import subprocess
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++")
    root = pathlib.Path(__file__).resolve().parents[2]
    exe = tmp_path_factory.mktemp("cli") / "smart_policy_cli"
    subprocess.run([gxx, "-std=c++20", "-O1", f"-I{root / 'engine' / 'src'}", str(root / "engine" / "tests" / "smart_policy_cli.cc"),
                    "-o", str(exe)], check=True)
    return lambda text: subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True).stdout.split()


def test_ssd_smart_victims_agree_with_the_engine_on_random_candidates(tmp_path, engine_pick):
    rng = random.Random(9)
    env = make(tmp_path, min_resident_s=300)
    p = env.tier.smart_params
    for trial in range(300):
        env.tier.gd = cp.GdTier()
        for k in range(rng.randint(0, 3)):
            env.tier.gd.evicted(cp.Pick(0, rng.random() * 5, cp.LEAF), 10 * (k + 1))
        now = 10_000.0
        env.clock.t = now
        entries, lines = [], []
        for i in range(rng.randint(1, 8)):
            e = {"id": i + 1, "pos": rng.choice([20_000, 30_000, 60_000, 90_000]), "parent_pos": rng.choice([0, 0, 9_000, 18_000]),
                 "desc_tiers": rng.choice([0, 0, cp.TIER_SSD, cp.TIER_RAM]), "junction": rng.random() < 0.2,
                 "superseded": rng.random() < 0.3, "hits": rng.choice([0, 0, 2]), "last_use": rng.choice([0, 15, 25, 40]),
                 "unique_disk_bytes": rng.choice([200 << 20, 1 << 30, 3 << 30]), "on_disk": True, "owner": 1}
            env.tier.written[e["id"]] = now - rng.choice([10, 200, 400, 5000])
            entries.append(e)
            c = env.tier._cand(e, now)
            lines.append(f"{c.depth!r} {c.parent_depth!r} {c.bytes!r} {c.hits} {int(c.junction)} {int(c.superseded)} {int(c.leaf)} "
                         f"{int(c.young)} {c.stamp!r} {c.age_key!r}")
        text = f"pick {p.t0_us!r} {p.k_us!r} {p.superseded_factor!r} {p.junction_factor!r} {p.hit_weight!r} {len(entries)}\n" + "\n".join(lines) + "\n"
        out = engine_pick(text)
        assert env.tier._pick_victim(entries, 0)["id"] == entries[int(out[1])]["id"], (trial, out)
