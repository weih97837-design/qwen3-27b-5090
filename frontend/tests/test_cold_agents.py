"""Never-hit entries, cold agents, warm-only SSD writes and the probation quota (docs/tiered-cache.md "Retention"; engine/src/retention.h
cold_agents, chunk_store.h pick_victim step 0). No GPU."""

import pathlib
import random
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import tiercache  # noqa: E402
from test_retention import B, make, _engine_pick  # noqa: E402

BIG = 40_000  # tokens: above KEEP_SHORT_TOKENS and SPILL_SSD_MIN_TOKENS


def test_keep_short_tokens_is_the_engines():
    text = (pathlib.Path(__file__).resolve().parents[2] / "engine/src/retention.h").read_text()
    m = re.search(r"kKeepShortTokens\s*=\s*(\d+)", text)
    assert m and int(m.group(1)) == tiercache.KEEP_SHORT_TOKENS


# ------------------------------------------------------------------ A: never-hit time-to-live

def test_unused_ttl_drops_never_hit_entries_from_ram_and_ssd(tmp_path):
    env = make(tmp_path, budget=10_000, retention=10 ** 9, unused_ttl_s=1800)
    ram_never = env.eng.add_ram(100, pos=BIG, owner=1, hits=0)
    ram_hit = env.eng.add_ram(100, pos=BIG, owner=1, hits=2)
    ram_short = env.eng.add_ram(100, pos=10_000, owner=1, hits=0)      # a shared system-prompt prefix: kept
    disk_never = env.eng.add_disk(env.dir, 100, owner=2, hits=0)
    disk_hit = env.eng.add_disk(env.dir, 100, owner=2, hits=1)
    env.clock.t = 100
    env.tier.tick(idle=False)
    assert env.eng.drops == []
    env.tier.written[disk_never] = 100
    env.tier.written[disk_hit] = 100
    env.clock.t = 100 + 1799
    env.tier.tick(idle=False)
    assert env.eng.drops == []
    env.clock.t = 100 + 1800
    env.tier.tick(idle=False)
    assert sorted(env.eng.drops) == sorted([ram_never, disk_never])
    assert {ram_hit, ram_short, disk_hit} <= set(env.eng.entries)
    assert env.tier.status()["expired"] == 2


def test_unused_ttl_off_and_inert_without_hits_field(tmp_path):
    env = make(tmp_path, budget=10_000, retention=10 ** 9, unused_ttl_s=0)
    env.eng.add_ram(100, pos=BIG, owner=1)
    env.clock.t = 10 ** 6
    env.tier.tick(idle=False)
    assert env.eng.drops == []
    env = make(tmp_path / "x", budget=10_000, retention=10 ** 9, unused_ttl_s=60)
    rid = env.eng.add_ram(100, pos=BIG)
    del env.eng.entries[rid]["hits"]  # an engine without the retention fields
    env.tier.tick(idle=False)
    env.clock.t = 10 ** 6
    env.tier.tick(idle=False)
    assert env.eng.drops == [] and env.tier.status()["expired"] == 0


def test_unused_ttl_skips_backups_and_busy_entries_and_works_without_ssd(tmp_path):
    env = make(tmp_path, budget=10_000, retention=10 ** 9, unused_ttl_s=60)
    backup = env.eng.add_ram(100, pos=BIG, owner=1)
    env.eng.entries[backup]["backup"] = True
    busy = env.eng.add_ram(100, pos=BIG, owner=1)
    env.eng.entries[busy]["busy"] = True
    plain = env.eng.add_ram(100, pos=BIG, owner=1)
    env.tier.tick(idle=False)
    env.clock.t = 1000
    env.tier.tick(idle=False)
    assert env.eng.drops == [plain]
    env.tier.directory = None  # no SSD tier: RAM entries still expire
    other = env.eng.add_ram(100, pos=BIG, owner=1)
    env.clock.t = 1001
    env.tier.tick(idle=False)
    env.clock.t = 2000
    env.tier.tick(idle=False)
    assert other in env.eng.drops


# ------------------------------------------------------------------ B: cold agents first

def _tier(tmp_path, **kw):
    env = make(tmp_path, budget=10_000, cold_first=True, cold_after_s=600, **kw)
    env.clock.t = 1000.0
    return env


def test_cold_agent_goes_before_the_biggest_agent_and_the_biggest_cold_one_first(tmp_path):
    env = _tier(tmp_path)
    big_hot = [env.eng.add_disk(env.dir, 500, owner=1, hits=1), env.eng.add_disk(env.dir, 500, owner=1)]
    cold_small = env.eng.add_disk(env.dir, 100, owner=2)
    cold_mid = [env.eng.add_disk(env.dir, 200, owner=3), env.eng.add_disk(env.dir, 200, owner=3)]
    fresh = env.eng.add_disk(env.dir, 100, owner=4)          # no hit but just created: not cold
    env.tier.written.update({big_hot[0]: 900, big_hot[1]: 900, cold_small: 100, cold_mid[0]: 100, cold_mid[1]: 150, fresh: 900})
    entries = env.eng.spill_list()
    env.tier.ssd_use = {1: 1000.0, 2: 100.0, 3: 400.0, 4: 100.0}
    assert env.tier._pick_victim(entries, 0)["id"] == cold_mid[0]   # cold agents 2 (100 B) and 3 (400 B): the one with the most bytes
    env.tier.cold_first = False
    assert env.tier._pick_victim(entries, 0)["id"] == big_hot[1]    # old order: the agent with the most bytes, its never-hit entry


def test_cold_needs_no_hit_anywhere_in_the_tier_and_an_old_newest_entry(tmp_path):
    env = _tier(tmp_path)
    a_old = env.eng.add_disk(env.dir, 100, owner=1)
    a_hit = env.eng.add_disk(env.dir, 100, owner=1, hits=1)           # one hit makes the whole agent warm
    b_old = env.eng.add_disk(env.dir, 100, owner=2)
    b_new = env.eng.add_disk(env.dir, 100, owner=2)                   # a newer entry keeps the agent not cold
    c = env.eng.add_disk(env.dir, 100, owner=0)                       # the unknown bucket is never cold
    env.tier.written.update({a_old: 0, a_hit: 0, b_old: 0, b_new: 999, c: 0})
    entries = env.eng.spill_list()
    cold = tiercache._cold_owners(entries, env.tier.written, 1000.0, 600)
    assert cold == set()
    env.tier.written[b_new] = 400  # 600 s old: cold (the boundary counts)
    assert tiercache._cold_owners(entries, env.tier.written, 1000.0, 600) == {2}


def test_cold_victim_is_only_chosen_among_candidates_but_judged_over_the_whole_tier(tmp_path):
    env = _tier(tmp_path)
    busy_hit = env.eng.add_disk(env.dir, 100, owner=1, hits=3)
    env.eng.entries[busy_hit]["busy"] = True                          # not a candidate, but its hit makes agent 1 warm
    cold_cand = env.eng.add_disk(env.dir, 100, owner=1)
    other = env.eng.add_disk(env.dir, 100, owner=2)
    env.tier.written.update({busy_hit: 0, cold_cand: 0, other: 0})
    entries = env.eng.spill_list()
    cands = [e for e in entries if not e.get("busy")]
    assert env.tier._pick_victim(cands, 0, entries)["id"] in (cold_cand, other)
    env.tier.ssd_use = {1: 900.0, 2: 100.0}
    assert env.tier._pick_victim(cands, 0, entries)["id"] == other   # agent 1 is warm (the busy entry's hit): only agent 2 is cold


@pytest.mark.parametrize("fair", [True, False])
def test_random_agreement_with_the_engines_order_including_cold_agents(tmp_path, fair):
    rng = random.Random(11)
    env = make(tmp_path, budget=10 ** 9, fair=fair, cold_first=True, cold_after_s=600)
    now = 10_000.0
    env.clock.t = now
    for trial in range(400):
        n = rng.randint(1, 10)
        owners = {o: float(rng.choice([10, 100, 500, 500])) for o in range(rng.randint(0, 4), rng.randint(5, 7))}
        cands = [{"id": i + 1, "owner": rng.choice(list(owners)), "hits": rng.choice([0, 0, 0, 1, 3]),
                  "superseded": rng.random() < 0.3, "age": now - rng.choice([10, 300, 599, 600, 601, 5000]) + rng.random() * 0.01}
                 for i in range(n)]
        for c in cands:
            c["use"] = owners[c["owner"]]
        # the engine's cold_agents (retention.h) as the reference: no hit in any entry, newest entry >= 600 s old, agent 0 never
        cold = set()
        for o in {c["owner"] for c in cands}:
            mine = [c for c in cands if c["owner"] == o]
            if o != 0 and not any(c["hits"] > 0 for c in mine) and now - max(c["age"] for c in mine) >= 600:
                cold.add(o)
        for c in cands:
            c["cold"] = c["owner"] in cold
        # ("age" is when the entry was written: the oldest of the group goes first, as the engine's last_use; ids are shuffled)
        rng.shuffle(cands)
        env.tier.ssd_use = dict(owners)
        env.tier.written = {c["id"]: c["age"] for c in cands}
        entries = [{"id": c["id"], "owner": c["owner"], "hits": c["hits"], "superseded": c["superseded"],
                    "unique_disk_bytes": 100, "logical_bytes": 100, "on_disk": True} for c in cands]
        want = _engine_pick([dict(c, age=c["age"]) for c in cands], fair, cold_first=True)
        assert env.tier._pick_victim(entries, 0)["id"] == want, (trial, fair)


# ------------------------------------------------------------------ C: warm only + probation

def _warm_env(tmp_path, probation_gib=1000 * B, **kw):
    env = make(tmp_path, budget=10_000, retention=0, ssd_warm_only=True, probation_gib=probation_gib, **kw)
    return env


def test_auto_agents_are_not_written_until_warm_and_agents_with_ids_use_the_probation(tmp_path):
    env = _warm_env(tmp_path)
    env.tier.note_agent("auto:12345678", 7)
    auto = env.eng.add_ram(100, pos=BIG, owner=7)
    named = env.eng.add_ram(100, pos=BIG, owner=8)
    hit = env.eng.add_ram(100, pos=BIG, owner=9, hits=1)
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert auto in env.eng.drops and auto not in env.eng.exports
    assert set(env.eng.exports) == {named, hit}
    assert env.tier.status()["not_written"] == 1


def test_warm_agent_entries_are_not_counted_against_the_quota(tmp_path):
    env = _warm_env(tmp_path, probation_gib=150 * B)
    warm_hit = env.eng.add_disk(env.dir, 100, owner=5, hits=2)
    warm_never = env.eng.add_disk(env.dir, 100, owner=5)               # its agent has a hit: warm, not unproven
    warm_new = env.eng.add_ram(100, pos=BIG, owner=5)
    unproven = env.eng.add_ram(100, pos=BIG, owner=6)
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert set(env.eng.exports) == {warm_new, unproven}
    assert env.eng.drops == [] and warm_hit in env.eng.entries and warm_never in env.eng.entries


def test_probation_quota_drops_the_oldest_unproven_entry_first(tmp_path):
    env = _warm_env(tmp_path, probation_gib=250 * B)
    first = env.eng.add_ram(100, pos=BIG, owner=11)
    second = env.eng.add_ram(100, pos=BIG, owner=12)
    third = env.eng.add_ram(100, pos=BIG, owner=13)
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    # all three are unproven: 300 B > 250 B, so the oldest written one goes when the third is written
    assert set(env.eng.exports) == {first, second, third}
    assert len(env.eng.drops) == 1
    assert env.eng.drops[0] in (first, second, third)
    survivors = set(env.eng.entries)
    assert len(survivors) == 2
    dropped = env.eng.drops[0]
    # the oldest by write time is the first one the loop wrote, which is the one dropped
    assert dropped == env.eng.exports[0]
    assert env.tier.status()["not_written"] == 1


def test_probation_removes_the_oldest_unproven_on_the_ssd_and_an_oversized_entry_is_refused(tmp_path):
    env = _warm_env(tmp_path, probation_gib=250 * B)
    old = env.eng.add_disk(env.dir, 100, owner=21)
    mid = env.eng.add_disk(env.dir, 100, owner=22)
    new = env.eng.add_disk(env.dir, 100, owner=23)
    env.tier.written.update({old: 1.0, mid: 2.0, new: 3.0})
    huge = env.eng.add_ram(300, pos=BIG, owner=24)                     # larger than the whole quota: dropped, never written
    env.clock.t = 5
    env.tier.tick(idle=True)
    env.clock.t = 6
    env.tier.tick(idle=True)
    assert env.eng.drops[0] == old and huge in env.eng.drops and huge not in env.eng.exports
    assert mid in env.eng.entries and new in env.eng.entries


def test_an_agent_that_gets_a_hit_leaves_the_probation(tmp_path):
    env = _warm_env(tmp_path, probation_gib=100 * B)
    a = env.eng.add_disk(env.dir, 100, owner=31)
    b = env.eng.add_disk(env.dir, 100, owner=32)
    env.tier.written.update({a: 1.0, b: 2.0})
    env.eng.entries[a]["hits"] = 1                                      # agent 31 proved itself
    env.clock.t = 5
    env.tier.tick(idle=True)
    assert env.eng.drops == [] and b in env.eng.entries                 # 100 B of unproven (b) fits the 100 B quota


def test_warm_only_off_writes_everything_and_short_prefixes_are_exempt(tmp_path):
    env = make(tmp_path, budget=10_000, retention=0, ssd_warm_only=False, probation_gib=0)
    env.tier.note_agent("auto:aaaaaaaa", 7)
    e1 = env.eng.add_ram(100, pos=BIG, owner=7)
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.eng.exports == [e1]
    env = _warm_env(tmp_path / "s", probation_gib=0)
    env.tier.note_agent("auto:bbbbbbbb", 7)
    short = env.eng.add_ram(100, pos=BIG, owner=7)
    env.eng.entries[short]["pos"] = 9000
    env.tier.ssd_min_tokens = 0
    env.tier.tick(idle=True)
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.eng.exports == [short]


def test_flush_applies_the_warm_only_gate(tmp_path):
    env = _warm_env(tmp_path)
    env.tier.note_agent("auto:cccccccc", 7)
    auto = env.eng.add_ram(100, pos=BIG, owner=7)
    named = env.eng.add_ram(100, pos=BIG, owner=8)
    assert env.tier.flush(10)["written"] == 1
    assert env.eng.exports == [named] and auto in env.eng.drops


def test_from_env_reads_the_new_settings(monkeypatch):
    from test_retention import FakeEngine
    monkeypatch.setenv("SPILL_RAM_GIB", "8")
    for name in ("SPILL_UNUSED_TTL_S", "SPILL_COLD_AFTER_S", "SPILL_COLD_FIRST", "SPILL_SSD_WARM_ONLY", "SPILL_PROBATION_GIB"):
        monkeypatch.delenv(name, raising=False)
    t = tiercache.TierPolicy.from_env(FakeEngine(), log=lambda _m: None)
    assert (t.unused_ttl_s, t.cold_after_s, t.cold_first, t.ssd_warm_only, t.probation_bytes) == (0, 600, True, False, 4 * tiercache.GiB)
    monkeypatch.setenv("SPILL_UNUSED_TTL_S", "1800")
    monkeypatch.setenv("SPILL_COLD_FIRST", "0")
    monkeypatch.setenv("SPILL_SSD_WARM_ONLY", "on")
    monkeypatch.setenv("SPILL_PROBATION_GIB", "0.5")
    t = tiercache.TierPolicy.from_env(FakeEngine(), log=lambda _m: None)
    assert (t.unused_ttl_s, t.cold_first, t.ssd_warm_only, t.probation_bytes) == (1800, False, True, 0.5 * tiercache.GiB)
