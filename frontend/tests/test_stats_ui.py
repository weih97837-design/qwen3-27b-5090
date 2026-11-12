"""The UI's lanes, tiered-cache block, host and deploy info (stats.py, worker.Scheduler.stats) against a stub engine:
no GPU. The data reaches /ui/state through Hub.tick()."""

import collections
import json
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import stats  # noqa: E402
import worker  # noqa: E402

GiB = 1 << 30


def row(rid, out=0, decode_wall=0.0, phase="decode", prompt=100, cached=40, age=1.0):
    return {"rid": rid, "phase": phase, "prompt": prompt, "cached": cached, "out": out,
            "decode_wall": decode_wall, "age": age}


# ------------------------------------------------------------ slot_rows

def test_slot_rows_parked_keep_their_lane_and_the_rest_fill_free_lanes():
    seqs = [("running", None, row("a", out=5, decode_wall=2.0)),
            ("parked", 1, row("b")),          # parking area 1 -> lane 2
            ("admitted", None, row("c"))]
    out = stats.slot_rows(seqs, 3)
    assert [r["lane"] for r in out] == [1, 2, 3]
    assert [r["rid"] for r in out] == ["a", "b", "c"]
    assert [r["state"] for r in out] == ["running", "parked", "admitted"]
    assert out[0]["tok_s"] == pytest.approx(4 / 2.0)
    assert out[1]["tok_s"] is None and out[2]["tok_s"] is None  # out <= 1 or no decode time


def test_slot_rows_pads_with_idle_and_truncates_to_lanes():
    out = stats.slot_rows([("admitted", None, row("x"))], 4)
    assert len(out) == 4
    assert out[0]["state"] == "admitted" and out[0]["rid"] == "x"
    for r in out[1:]:
        assert r == {"lane": r["lane"], "state": "idle", "open": True, "rid": None, "phase": None, "prompt": 0,
                     "cached": 0, "out": 0, "tok_s": None, "age": 0.0}
    assert [r["lane"] for r in out] == [1, 2, 3, 4]
    assert len(stats.slot_rows([("admitted", None, row(str(i))) for i in range(5)], 3)) == 3


def test_slot_rows_parked_area_out_of_range_falls_back_to_a_free_lane():
    out = stats.slot_rows([("parked", 7, row("p"))], 3)
    assert out[0]["rid"] == "p" and out[0]["lane"] == 1


def test_slot_rows_empty_is_all_idle():
    out = stats.slot_rows([], 2)
    assert [r["state"] for r in out] == ["idle", "idle"]


def test_slot_rows_target_closes_idle_lanes_above_it():
    out = stats.slot_rows([("admitted", None, row("x"))], 4, target=2)
    assert [r["state"] for r in out] == ["admitted", "idle", "closed", "closed"]
    assert [r["open"] for r in out] == [True, True, False, False]
    assert all(r["tok_s"] is None for r in out[2:])


def test_slot_rows_busy_lane_above_target_keeps_its_state_but_is_not_open():
    memory = {"x": 3}
    out = stats.slot_rows([("running", None, row("x", out=3, decode_wall=1.0))], 4, target=2, last_lanes=memory)
    assert out[2]["rid"] == "x" and out[2]["state"] == "running" and out[2]["open"] is False
    assert out[2]["tok_s"] == pytest.approx(2.0)
    assert out[3]["state"] == "closed" and out[3]["open"] is False


def test_slot_rows_default_target_opens_every_lane():
    out = stats.slot_rows([], 3)
    assert all(r["open"] for r in out)


def test_slot_rows_lanes_stay_put_when_another_request_finishes():
    memory = {}
    snap1 = stats.slot_rows([("running", None, row("a", out=4, decode_wall=1.0)),
                             ("admitted", None, row("b")), ("admitted", None, row("c"))], 4, last_lanes=memory)
    assert {r["rid"]: r["lane"] for r in snap1 if r["rid"]} == {"a": 1, "b": 2, "c": 3}
    assert memory == {"a": 1, "b": 2, "c": 3}
    # a finishes: b and c keep lanes 2 and 3 (without the memory, b would move to lane 1)
    snap2 = stats.slot_rows([("admitted", None, row("b")), ("admitted", None, row("c"))], 4, last_lanes=memory)
    assert {r["rid"]: r["lane"] for r in snap2 if r["rid"]} == {"b": 2, "c": 3}
    assert snap2[0]["state"] == "idle"
    assert memory == {"b": 2, "c": 3}  # finished rids dropped
    # a new request takes the freed lane 1
    snap3 = stats.slot_rows([("admitted", None, row("d")), ("admitted", None, row("b")),
                             ("admitted", None, row("c"))], 4, last_lanes=memory)
    assert {r["rid"]: r["lane"] for r in snap3 if r["rid"]} == {"d": 1, "b": 2, "c": 3}


# ------------------------------------------------------------ tiers_block

class FakeEngine:
    def __init__(self, fail=False):
        self.fail = fail

    def spill_stats(self):
        if self.fail:
            raise RuntimeError("q27_spill_stats: boom")
        return {"entries": 3, "in_ram": 2, "on_disk": 1, "ram_bytes": 5 * GiB, "spilled": 4, "restored": 2,
                "written": 1, "dropped": 0, "subsumed": 6, "failed": 0}


class FakeTiers:
    def __init__(self, directory, status_error=None):
        self.directory = directory
        self.ram_retention_s = 3600.0
        self.ttl_s = 7 * 86400.0
        self.status_error = status_error

    def status(self):
        if self.status_error:
            raise self.status_error
        return {"ram_entries": 2, "ram_bytes": 5 * GiB, "disk_entries": 1, "ssd_bytes": 1234,
                "ssd_budget_bytes": 100 * GiB, "last_error": None}


def test_tiers_block_none_without_a_policy():
    assert stats.tiers_block(FakeEngine(), None) is None


def test_tiers_block_fields(tmp_path):
    env = {"SPILL_RAM_GIB": "8"}
    out = stats.tiers_block(FakeEngine(), FakeTiers(tmp_path), environ=env)
    assert out["ram_bytes"] == 5 * GiB and out["ram_cap_bytes"] == 8 * GiB
    assert out["in_ram"] == 2 and out["on_disk"] == 1 and out["ssd_bytes"] == 1234
    assert out["ssd_cap_bytes"] == 100 * GiB
    assert out["disk_free_bytes"] > 0
    assert out["spilled"] == 4 and out["restored"] == 2 and out["written"] == 1
    assert out["dropped"] == 0 and out["subsumed"] == 6 and out["failed"] == 0
    assert out["last_error"] is None and out["retention_s"] == 3600.0 and out["ttl_days"] == 7.0
    assert out["dir"] == str(tmp_path)
    json.dumps(out)


def test_tiers_block_no_directory_and_unset_ram():
    out = stats.tiers_block(FakeEngine(), FakeTiers(None), environ={})
    assert out["dir"] is None and out["disk_free_bytes"] is None and out["ram_cap_bytes"] is None


def test_tiers_block_error_path_never_raises():
    assert stats.tiers_block(FakeEngine(fail=True), FakeTiers(None), environ={}) == {
        "error": "RuntimeError: q27_spill_stats: boom"}
    out = stats.tiers_block(FakeEngine(), FakeTiers(None, status_error=OSError("gone")), environ={})
    assert out == {"error": "OSError: gone"}


# ------------------------------------------------------------ host / deploy

def test_host_info_reads_meminfo(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal:       65536000 kB\nMemFree:  1 kB\nMemAvailable:   40960000 kB\n", encoding="utf-8")
    out = stats.host_info(time.time() - 12.5, meminfo=str(mi))
    assert out["ram_total_bytes"] == 65536000 * 1024 and out["ram_avail_bytes"] == 40960000 * 1024
    assert out["cpu_count"] is not None and out["uptime_s"] == pytest.approx(12.5, abs=0.5)
    assert set(out) == {"cpu_count", "load1", "load5", "ram_total_bytes", "ram_avail_bytes", "uptime_s"}


def test_host_info_without_meminfo_or_loadavg(tmp_path, monkeypatch):
    monkeypatch.setattr(stats.os, "getloadavg", lambda: (_ for _ in ()).throw(OSError("no")))
    out = stats.host_info(0.0, now=5.0, meminfo=str(tmp_path / "missing"))
    assert out["ram_total_bytes"] is None and out["ram_avail_bytes"] is None
    assert out["load1"] is None and out["load5"] is None and out["uptime_s"] == 5.0


def test_deploy_info_settings_from_environment():
    env = {"MAX_SEQS": "4", "SPILL_RAM_GIB": "8"}
    out = stats.deploy_info(100.0, "ready", environ=env)
    assert out["started_at"] == 100.0 and out["engine_state"] == "ready"
    assert set(out["settings"]) == set(stats.DEPLOY_SETTINGS)
    assert out["settings"]["MAX_SEQS"] == "4" and out["settings"]["SPILL_RAM_GIB"] == "8"
    assert out["settings"]["KV_DTYPE"] is None
    json.dumps(out)


# ------------------------------------------------------------ Scheduler.stats + Hub.tick

class Dev(dict):
    pass


class StubEngine(FakeEngine):
    def cache_stats(self):
        return {"free_pages": 10}

    def device_memory(self):
        return {"free": 1, "total": 2, "free_at_start": 1, "budget": 2}


def seq(rid, now, out=0, first_token=None, area=None, phase="decode", started=True):
    s = worker.Seq(rid, {"ids": [1, 2, 3, 4], "max_new": 16, "temperature": 0.0, "top_k": 1, "top_p": 1.0}, now)
    s.out = [0] * out
    s.first_token = first_token
    s.area = area
    s.phase = phase
    s.started = started
    s.fed = 4
    return s


def make_sched(engine, tiers, alive, waiting, cur, engine_seqs=3):
    sched = worker.Scheduler.__new__(worker.Scheduler)
    sched.engine, sched.tiers = engine, tiers
    sched.alive, sched.waiting, sched.cur = alive, collections.deque(waiting), cur
    sched.engine_seqs, sched.max_seqs, sched.blocked = engine_seqs, engine_seqs, None
    sched.slot_lanes = {}
    sched.m = collections.Counter()
    sched.last_stats = 0.0
    return sched


def test_scheduler_stats_lanes_tiers_and_hub_tick():
    now = time.time()
    a = seq("a", now, out=5, first_token=now - 2.0)  # running
    b = seq("b", now, area=1)                        # parked on area 1 -> lane 2
    c = seq("c", now)                                # admitted, no area
    w = seq("w", now, started=False)                 # waiting: no lane
    sched = make_sched(StubEngine(), FakeTiers(None), [a, b, c], [w], cur=a)
    snap = sched.stats()

    assert [r["lane"] for r in snap["slots"]] == [1, 2, 3]
    assert [r["rid"] for r in snap["slots"]] == ["a", "b", "c"]
    assert [r["state"] for r in snap["slots"]] == ["running", "parked", "admitted"]
    assert snap["slots"][0]["tok_s"] == pytest.approx(4 / 2.0, rel=0.05)
    assert snap["tiers"]["ram_bytes"] == 5 * GiB and snap["tiers"]["ssd_cap_bytes"] == 100 * GiB
    assert len(snap["seqs"]) == 4  # the waiting request is still listed in seqs

    hub = stats.Hub()
    hub.set_snapshot(snap)
    tick = hub.tick()
    assert tick["slots"] == snap["slots"]
    assert tick["tiers"]["in_ram"] == 2
    assert set(tick["host"]) == {"cpu_count", "load1", "load5", "ram_total_bytes", "ram_avail_bytes", "uptime_s"}
    assert tick["deploy"]["engine_state"] == "loading"
    assert set(tick["deploy"]["settings"]) == set(stats.DEPLOY_SETTINGS)
    json.dumps(tick)


def test_scheduler_stats_target_below_engine_seqs_closes_tiles_and_keeps_lanes():
    now = time.time()
    a = seq("a", now, out=5, first_token=now - 2.0)
    b = seq("b", now, area=1)  # parked on lane 2
    c = seq("c", now)
    sched = make_sched(StubEngine(), FakeTiers(None), [a, b, c], [], cur=a, engine_seqs=8)
    sched.max_seqs = 4
    snap = sched.stats()
    assert [r["state"] for r in snap["slots"]] == ["running", "parked", "admitted", "idle"] + ["closed"] * 4
    assert [r["open"] for r in snap["slots"]] == [True] * 4 + [False] * 4
    assert snap["target_seqs"] == 4
    # a finishes; c stays on lane 3, b on lane 2
    sched.alive = [b, c]
    sched.cur = None
    snap2 = sched.stats()
    assert [(r["lane"], r["rid"]) for r in snap2["slots"] if r["rid"]] == [(2, "b"), (3, "c")]
    json.dumps(snap2)


def test_scheduler_stats_survives_a_tiers_error():
    sched = make_sched(StubEngine(), FakeTiers(None, status_error=RuntimeError("bad")), [], [], cur=None)
    snap = sched.stats()
    assert snap["tiers"] == {"error": "RuntimeError: bad"}
    assert [r["state"] for r in snap["slots"]] == ["idle"] * 3


def test_hub_tick_without_snapshot():
    tick = stats.Hub().tick()
    assert tick["slots"] == [] and tick["tiers"] is None
    json.dumps(tick)
