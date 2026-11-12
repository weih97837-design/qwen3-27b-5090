"""Value-based retention (docs/tiered-cache.md "Retention"; engine/src/retention.h): the frontend's side. The victim order of the SSD
(agent using the most bytes, then superseded, never-hit before hit, then the old rule), RAM -> SSD (superseded are dropped, not
written), the per-agent bytes and reuse figures in the stats, SPILL_FAIR_AGENTS, and the agent id the engine is told. No GPU."""

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import agent as agents  # noqa: E402
import stats  # noqa: E402
import tiercache  # noqa: E402
from test_tiercache import B, Clock, FakeEngine  # noqa: E402
from tiercache import TierPolicy  # noqa: E402

GiB = 1 << 30


class RetEngine(FakeEngine):
    """FakeEngine whose entries carry the engine's retention fields, and that reports per-agent bytes like spill_agents."""

    def __init__(self):
        super().__init__()
        self.agent_calls = 0

    def add_disk(self, tmp, size, owner=0, hits=0, superseded=False, pos=40_000):
        entry_id = self.add_ram(size, pos=pos)
        e = self.entries[entry_id]
        path = pathlib.Path(tmp) / f"{entry_id}{tiercache.SUFFIX}"
        path.write_bytes(b"x" * size)
        e.update(ram_bytes=0, on_disk=True, path=str(path), owner=owner, hits=hits, superseded=superseded)
        return entry_id

    def add_ram(self, size, pos=10_000, owner=0, hits=0, superseded=False):
        entry_id = super().add_ram(size, pos=pos)
        self.entries[entry_id].update(owner=owner, hits=hits, superseded=superseded)
        return entry_id

    def spill_agents(self):
        self.agent_calls += 1
        rows = {}
        for e in self.entries.values():
            r = rows.setdefault(e.get("owner", 0), {"owner": e.get("owner", 0), "ram_bytes": 0, "ssd_bytes": 0,
                                                    "ram_entries": 0, "ssd_entries": 0})
            if e["ram_bytes"] > 0:
                r["ram_bytes"] += e["ram_bytes"]
                r["ram_entries"] += 1
            if e["on_disk"]:
                r["ssd_bytes"] += len(open(e["path"], "rb").read())
                r["ssd_entries"] += 1
        return list(rows.values())


def make(tmp_path, budget=1000, fair=True, retention=0, **kw):
    eng = RetEngine()
    clock = Clock()
    d = tmp_path / "spill"
    tier = TierPolicy(eng, str(d), ram_retention_s=retention, ssd_gib=budget * B, ssd_min_free_gib=0, ttl_days=7, tick_s=0, clock=clock,
                      disk_free=lambda _d: 10 ** 12, log=lambda _m: None, ssd_min_tokens=8192, fair_agents=fair, **kw)
    tier.start()
    return types.SimpleNamespace(eng=eng, tier=tier, dir=d, clock=clock)


def test_ssd_victim_is_the_biggest_agents_never_hit_entry_before_an_older_hit_one(tmp_path):
    env = make(tmp_path)
    hit = env.eng.add_disk(env.dir, 400, owner=1, hits=3)      # oldest
    never = env.eng.add_disk(env.dir, 400, owner=1)
    small = env.eng.add_disk(env.dir, 100, owner=2)            # older than `never`, but agent 2 uses 100 B against 800
    env.tier.written.update({hit: 1.0, small: 2.0, never: 3.0})
    assert env.tier.make_room(300)  # used 900 + 300 > 1000: one victim
    assert env.eng.drops == [never]
    assert small in env.eng.entries and hit in env.eng.entries


def test_ssd_superseded_goes_first_then_never_hit_then_hit_then_old_rule(tmp_path):
    env = make(tmp_path, budget=10_000)
    hit_old = env.eng.add_disk(env.dir, 100, owner=1, hits=2)
    never_mid = env.eng.add_disk(env.dir, 100, owner=1)
    sup_new_hit = env.eng.add_disk(env.dir, 100, owner=1, hits=1, superseded=True)
    sup_newer = env.eng.add_disk(env.dir, 100, owner=1, superseded=True)
    never_new = env.eng.add_disk(env.dir, 100, owner=1)
    env.tier.written.update({hit_old: 1.0, never_mid: 2.0, sup_new_hit: 3.0, sup_newer: 4.0, never_new: 5.0})
    env.tier.budget = 100  # everything but one entry must go
    assert env.tier.make_room(0)
    # superseded first (never-hit one before the hit one), then never-hit by age, the hit one stays
    assert env.eng.drops == [sup_newer, sup_new_hit, never_mid, never_new]
    assert list(env.eng.entries) == [hit_old]


def test_ssd_fairness_is_elastic_and_can_be_switched_off(tmp_path):
    for fair in (True, False):
        env = make(tmp_path / str(fair), budget=1000, fair=fair)
        big_hit = env.eng.add_disk(env.dir, 450, owner=1, hits=1)
        big2 = env.eng.add_disk(env.dir, 450, owner=1, hits=1)
        small_sup = env.eng.add_disk(env.dir, 100, owner=2, superseded=True)
        env.tier.written.update({big_hit: 1.0, big2: 2.0, small_sup: 3.0})
        assert env.tier.make_room(100)
        # fair: the biggest agent pays (its older, equally valued entry); unfair: the superseded entry goes, whoever owns it
        assert env.eng.drops == ([big_hit] if fair else [small_sup])


def test_ssd_the_agent_with_the_most_bytes_changes_as_entries_go(tmp_path):
    env = make(tmp_path, budget=10_000)
    a = [env.eng.add_disk(env.dir, 300, owner=1) for _ in range(3)]   # 900
    b = [env.eng.add_disk(env.dir, 250, owner=2) for _ in range(3)]   # 750
    env.tier.written.update({e: float(i) for i, e in enumerate(a + b)})
    env.tier.budget = 1000  # used 1650: 650 must go
    assert env.tier.make_room(0)
    # 1: a[0] (900 > 750) -> agent 1 now 600 < 750 -> b[0] -> 500 < 600 -> a[1] -> ... until used <= 1000
    assert env.eng.drops == [a[0], b[0], a[1]]


def test_defaults_without_retention_fields_are_the_old_oldest_first(tmp_path):
    env = make(tmp_path, budget=10_000)
    ids = [env.eng.add_disk(env.dir, 100) for _ in range(4)]
    env.tier.written.update({ids[0]: 4.0, ids[1]: 1.0, ids[2]: 3.0, ids[3]: 2.0})
    env.tier.budget = 200
    assert env.tier.make_room(0)
    assert env.eng.drops == [ids[1], ids[3]]
    # an engine whose entries carry no retention keys at all behaves the same
    env2 = make(tmp_path / "plain", budget=10_000)
    plain = [{"id": i, "on_disk": True, "logical_bytes": 100, "unique_disk_bytes": 100} for i in (1, 2, 3)]
    env2.tier.written.update({1: 3.0, 2: 1.0, 3: 2.0})
    assert env2.tier._pick_victim(plain, 0)["id"] == 2


def test_ram_to_ssd_superseded_entries_are_dropped_and_reused_ones_are_written_first(tmp_path):
    env = make(tmp_path, budget=10_000)
    old_never = env.eng.add_ram(200, owner=1)
    sup = env.eng.add_ram(200, owner=1, superseded=True)
    newer_hit = env.eng.add_ram(200, owner=1, hits=2)
    env.clock.t = 0
    env.tier.tick(idle=True)          # (first sight: retention counts from here; ram_retention_s is 0)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert sup not in env.eng.entries and env.eng.drops[0] == sup
    assert env.eng.exports == [newer_hit, old_never]        # (reused first, then by age)
    st = env.tier.status()
    assert st["ssd_skipped_superseded"] == 1


def test_flush_does_not_write_superseded_entries(tmp_path):
    env = make(tmp_path, budget=10_000)
    sup = env.eng.add_ram(100, pos=9000, superseded=True)
    keep = env.eng.add_ram(100, pos=9000)
    assert env.tier.flush(10)["written"] == 1
    assert env.eng.exports == [keep] and sup in env.eng.entries


def test_status_reuse_split_and_agent_rows(tmp_path):
    env = make(tmp_path, budget=10_000, retention=3600)
    env.eng.add_disk(env.dir, 300, owner=1, hits=2)
    env.eng.add_disk(env.dir, 100, owner=1)
    env.eng.add_disk(env.dir, 50, owner=2, superseded=True)
    env.eng.add_ram(40, owner=2, hits=1)
    env.eng.add_ram(60, owner=3)
    env.clock.t = 5
    env.tier.tick(idle=False)
    st = env.tier.status()
    assert (st["disk_reused_entries"], st["disk_reused_bytes"]) == (1, 300)
    assert (st["disk_never_entries"], st["disk_never_bytes"]) == (2, 150)
    assert (st["disk_superseded_entries"], st["disk_superseded_bytes"]) == (1, 50)
    assert (st["ram_reused_entries"], st["ram_never_entries"]) == (1, 1)
    assert {r["owner"] for r in st["agent_rows"]} == {1, 2, 3} and st["fair_agents"] is True


def test_from_env_reads_spill_fair_agents(monkeypatch):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.delenv("SPILL_FAIR_AGENTS", raising=False)
    assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None).fair_agents is True
    for text, want in (("1", True), ("0", False), ("false", False), ("", True), ("yes", True)):
        monkeypatch.setenv("SPILL_FAIR_AGENTS", text)
        assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None).fair_agents is want, text


def test_owner_id_is_stable_nonzero_and_distinct():
    a, b = agents.owner_id("manager"), agents.owner_id("worker")
    assert a == agents.owner_id("manager") and a != b and 0 < a < (1 << 63) and 0 < b < (1 << 63)
    assert agents.owner_id(None) == 0 and agents.owner_id("") == 0
    assert agents.owner_id("manager") == int.from_bytes(__import__("hashlib").blake2b(b"manager", digest_size=8).digest(), "little") & ((1 << 63) - 1)  # (stable across processes: blake2b, not hash())


def test_agents_block_fills_ram_and_ssd_bytes_per_agent():
    totals = {"alpha": {"requests": 3, "hit_requests": 1, "prompt_tokens": 100, "cached_tokens": 50},
              "beta": {"requests": 1, "hit_requests": 0, "prompt_tokens": 10, "cached_tokens": 0}}
    use = [{"owner": agents.owner_id("alpha"), "ram_bytes": 3 * GiB, "ssd_bytes": 20 * GiB, "ram_entries": 4, "ssd_entries": 30},
           {"owner": agents.owner_id("beta"), "ram_bytes": GiB, "ssd_bytes": 0, "ram_entries": 1, "ssd_entries": 0},
           {"owner": 0, "ram_bytes": 5, "ssd_bytes": 7, "ram_entries": 1, "ssd_entries": 1},
           {"owner": 12345, "ram_bytes": 10, "ssd_bytes": 10, "ram_entries": 1, "ssd_entries": 2}]  # (an agent of an earlier run)
    rows = {r["agent"]: r for r in stats.agents_block([], totals, use)}
    assert rows["alpha"]["ram_bytes"] == 3 * GiB and rows["alpha"]["ssd_bytes"] == 20 * GiB and rows["alpha"]["ssd_entries"] == 30
    assert rows["beta"]["ram_bytes"] == GiB and rows["beta"]["ssd_bytes"] == 0
    assert rows["unknown"]["ram_bytes"] == 15 and rows["unknown"]["ssd_bytes"] == 17 and rows["unknown"]["ssd_entries"] == 3
    # without the engine's figures (an older engine) they stay None
    plain = {r["agent"]: r for r in stats.agents_block([], totals, None)}
    assert plain["alpha"]["ram_bytes"] is None and plain["alpha"]["ssd_bytes"] is None
    json_ok = __import__("json").dumps(list(rows.values()))
    assert json_ok


def test_tiers_block_reuse_rates():
    class Eng:
        def spill_stats(self):
            return {"entries": 3, "in_ram": 2, "on_disk": 1, "ram_bytes": 1, "spilled": 40, "restored": 7, "written": 20,
                    "dropped": 1, "subsumed": 0, "failed": 0, "restored_ram": 4, "restored_disk": 3}

    class Tiers:
        directory = None
        ram_retention_s = 1
        ttl_s = 86400.0

        def status(self):
            return {"ssd_bytes": 0, "fair_agents": False, "ram_reused_entries": 2, "ssd_skipped_superseded": 5}
    out = stats.tiers_block(Eng(), Tiers(), environ={})
    assert out["ram_reuse_rate"] == 0.1 and out["ssd_reuse_rate"] == 0.15
    assert out["ram_restored"] == 4 and out["ssd_restored"] == 3 and out["fair_agents"] is False
    assert out["ram_reused_entries"] == 2 and out["ram_never_entries"] == 0 and out["ssd_skipped_superseded"] == 5
    old = Eng()
    old.spill_stats = lambda: {"entries": 0, "in_ram": 0, "on_disk": 0, "ram_bytes": 0, "spilled": 0, "restored": 0,
                               "written": 0, "dropped": 0, "subsumed": 0, "failed": 0}
    out = stats.tiers_block(old, Tiers(), environ={})
    assert out["ram_reuse_rate"] is None and out["ssd_reuse_rate"] is None


def test_begin_tells_the_engine_which_agent_owns_the_sequence():
    import worker
    calls = []

    class Eng:
        def set_owner(self, owner):
            calls.append(("owner", owner))

        def prefix_begin(self, ids, at=(), spec=False):
            calls.append(("begin",))
            return 0

        def spill_sync_lookup(self, on):
            pass

    sched = worker.Scheduler.__new__(worker.Scheduler)
    sched.engine, sched.spec, sched.cur = Eng(), False, None
    s = types.SimpleNamespace(no_sync=False, eng_ids=[1, 2], at=(), agent="alpha", cached=1, started=False, fed=0)
    sched.release_restored = lambda s: None
    try:
        sched.begin(s)
    except AttributeError:
        pass  # (the rest of begin needs a full scheduler; only the order of the first two calls matters here)
    assert calls[:2] == [("owner", agents.owner_id("alpha")), ("begin",)]


# ---------------------------------------------------------------- the dedup layout (what the real engine takes) and the order of the engine

def test_dedup_layout_make_room_applies_the_retention_order(tmp_path):
    """_make_room_dedup with an engine that reports owner / hits / superseded (spill_list) and per-agent bytes (spill_agents)."""
    from test_tiercache import FakeDedupEngine

    class Eng(FakeDedupEngine):
        def add_disk(self, owner, hits=0, superseded=False, name=None, size=200):
            entry_id = self.add_ram([(name or f"U{self.next_id}", size)], tail=50, pos=40_000)
            e = self.entries[entry_id]
            path = pathlib.Path(self.dir) / f"{entry_id}.q27m"
            path.write_text(json.dumps({"chunks": e["chunks"], "tail": e["tail"]}))
            self.spill_export(entry_id, str(path))
            self.entries[entry_id].update(owner=owner, hits=hits, superseded=superseded)
            return entry_id

        def spill_list(self):
            return [dict(e) for _, e in sorted(self.entries.items())]

        def spill_agents(self):
            rows = {}
            for e in self.entries.values():
                if e["on_disk"]:
                    r = rows.setdefault(e["owner"], {"owner": e["owner"], "ram_bytes": 0, "ssd_bytes": 0, "ram_entries": 0, "ssd_entries": 0})
                    r["ssd_bytes"] += e["logical_bytes"]
                    r["ssd_entries"] += 1
            return list(rows.values())

    import json
    eng = Eng()
    eng.dir = tmp_path / "spill" / "entries"
    clock = Clock()
    tier = TierPolicy(eng, str(tmp_path / "spill"), ram_retention_s=0, ssd_gib=10_000 * B, ssd_min_free_gib=0, ttl_days=7, tick_s=0,
                      clock=clock, disk_free=lambda _d: 10 ** 12, log=lambda _m: None)
    tier.start()
    eng.dir.mkdir(parents=True, exist_ok=True)
    big_hit = eng.add_disk(1, hits=4)
    big_never = eng.add_disk(1)
    big_sup = eng.add_disk(1, superseded=True)
    small_never = eng.add_disk(2)
    tier.written.update({big_hit: 1.0, small_never: 2.0, big_never: 3.0, big_sup: 4.0})
    tier.budget = 2 * 260  # two of four entries may stay
    assert tier.make_room(0)
    # agent 1 uses 780 B against 260: superseded first, then its never-hit one; the hit entry and the small agent's stay
    assert eng.drops == [big_sup, big_never]
    assert sorted(eng.entries) == sorted([big_hit, small_never])


def _engine_pick(cands, fair):
    """A port of chunk_store.h pick_victim for candidates of equal `freed` (what the SSD's order compares): agent with the most bytes
    (ties: smaller id), then superseded, then never hit, then least recently used."""
    pool = list(cands)
    if fair:
        top = max(pool, key=lambda c: (c["use"], -c["owner"]))["owner"]
        pool = [c for c in pool if c["owner"] == top]
    best = min((0 if c["superseded"] else 2) + (1 if c["hits"] > 0 else 0) for c in pool)
    pool = [c for c in pool if (0 if c["superseded"] else 2) + (1 if c["hits"] > 0 else 0) == best]
    return min(pool, key=lambda c: c["age"])["id"]


def test_ssd_order_agrees_with_the_engines_on_random_candidates(tmp_path):
    import random
    rng = random.Random(5)
    for fair in (True, False):
        env = make(tmp_path / str(fair), budget=10 ** 9, fair=fair)
        for trial in range(300):
            n = rng.randint(1, 9)
            owners = {o: float(rng.choice([10, 100, 500, 500])) for o in range(rng.randint(1, 4))}
            cands = [{"id": i + 1, "owner": rng.choice(list(owners)), "hits": rng.choice([0, 0, 1, 3]),
                      "superseded": rng.random() < 0.3, "age": i + 1 + rng.random()} for i in range(n)]
            rng.shuffle(cands)
            for c in cands:
                c["use"] = owners[c["owner"]]
            env.tier.ssd_use = dict(owners)
            env.tier.written = {c["id"]: c["age"] for c in cands}
            entries = [{"id": c["id"], "owner": c["owner"], "hits": c["hits"], "superseded": c["superseded"],
                        "unique_disk_bytes": 100, "logical_bytes": 100, "on_disk": True} for c in cands]
            assert env.tier._pick_victim(entries, 0)["id"] == _engine_pick(cands, fair), (trial, fair)
