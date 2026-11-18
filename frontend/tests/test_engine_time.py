"""The engine time split (引擎時間分配): worker.EngineClock around the turns, Hub.tick()'s window shares, /metrics, and the page.
No GPU: a Scheduler built without __init__ and a stub engine, as the other tests do."""

import collections
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import stats  # noqa: E402
import worker  # noqa: E402

KINDS = ("prefill", "decode", "restore", "spill", "media", "idle", "other")
HTML = (pathlib.Path(__file__).resolve().parents[1] / "ui/index.html").read_text()


class Eng:
    """prefix_begin 'restores' (bumps the restored counter) and takes a little time."""

    def __init__(self):
        self.restored, self.begin_s = 0, 0.0

    def spill_stats(self):
        return {"restored": self.restored}

    def prefix_begin(self, ids, at=None, spec=False):
        time.sleep(self.begin_s)
        if self.begin_s:
            self.restored += 1
        return 0

    def cache_stats(self):
        return {"free_pages": 1, "checkpoint_only_pages": 0, "checkpoints": 0, "evicted": 0}

    def device_memory(self):
        return {}

    def spill_sync_lookup(self, on):
        pass


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, *a):
        self.sent.append(a)


def sched():
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.tiers = Eng(), Conn(), object()
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.cur, s.last_active, s.backup_done, s.spec = None, 0.0, True, False
    s.clock = worker.EngineClock()
    return s


def shares(snap):
    return {k: v / snap["wall"] for k, v in snap["seconds"].items()}


def test_the_shares_sum_to_one_and_every_kind_is_reported():
    s = sched()
    time.sleep(0.01)
    snap = s.clock.snapshot()
    assert set(snap["seconds"]) == set(KINDS)
    assert abs(sum(shares(snap).values()) - 1.0) < 1e-6
    assert snap["seconds"]["other"] > 0 and snap["seconds"]["prefill"] == 0


def test_a_prefill_and_a_decode_turn_add_to_their_own_kind():
    s = sched()
    s.prefill_turn = lambda q: time.sleep(0.03)
    s.decode_turn = lambda q: time.sleep(0.02)
    s.batch_turn = lambda q: time.sleep(0.02)
    s.guarded("prefill", object())
    a = s.clock.snapshot()["seconds"]
    assert a["prefill"] >= 0.03 and a["decode"] == 0
    s.guarded("decode", object())
    seqs = [worker.Seq(r, {"ids": [1], "max_new": 5, "temperature": 1.0, "top_k": 20, "top_p": 0.95}, 0.0) for r in "ab"]
    for q in seqs:
        q.phase = worker.DECODE
    s.guarded("batch", seqs)
    b = s.clock.snapshot()["seconds"]
    assert b["decode"] >= 0.04 and b["prefill"] >= a["prefill"]
    assert abs(sum(shares(s.clock.snapshot()).values()) - 1.0) < 1e-6


def test_work_inside_a_turn_is_taken_out_of_the_turn():
    s = sched()

    def turn(q):
        time.sleep(0.02)  # the turn's own work
        s.clock.add("media", 0.05)  # (as if 0.05 s of encode ran inside; added, not slept: the sum is checked below)
        s.clock.add("restore", 0.04)

    s.prefill_turn = turn
    s.guarded("prefill", object())
    sec = s.clock.snapshot()["seconds"]
    assert sec["media"] == 0.05 and sec["restore"] == 0.04
    assert sec["prefill"] < 0.02 + 0.03  # the turn's 0.02 s, not 0.02 + 0.09 (the inner kinds were taken out; clamped at 0)


def test_a_turn_in_progress_shows_in_a_snapshot():
    s = sched()
    seen = {}

    def turn(q):
        time.sleep(0.03)
        seen.update(s.clock.snapshot()["seconds"])  # (Scheduler.stats() from a long decode turn's report())

    s.decode_turn = turn
    s.guarded("decode", object())
    assert seen["decode"] >= 0.03


def test_a_synchronous_tier_restore_in_prefix_begin_counts_as_restore():
    s = sched()
    seq = worker.Seq("a", {"ids": list(range(10)), "max_new": 5, "temperature": 1.0, "top_k": 20, "top_p": 0.95}, 0.0)
    s.engine.begin_s = 0.03
    s.clock.enter("prefill")
    try:
        s.begin(seq)
    except Exception:  # noqa: BLE001 - the rest of begin needs the full scheduler; the restore was timed before it
        pass
    s.clock.leave()
    sec = s.clock.snapshot()["seconds"]
    assert sec["restore"] >= 0.03 and sec["prefill"] < 0.03


def test_a_prefix_begin_without_a_restore_is_not_restore_time():
    s = sched()
    seq = worker.Seq("a", {"ids": list(range(10)), "max_new": 5, "temperature": 1.0, "top_k": 20, "top_p": 0.95}, 0.0)
    s.engine.begin_s = 0.0
    try:
        s.begin(seq)
    except Exception:  # noqa: BLE001
        pass
    assert s.clock.snapshot()["seconds"]["restore"] == 0


def test_idle_and_spill_spans_are_added_outside_turns():
    s = sched()
    s.clock.add("idle", 0.5)
    s.clock.add("spill", 0.25)
    sec = s.clock.snapshot()["seconds"]
    assert sec["idle"] == 0.5 and sec["spill"] == 0.25 and s.clock.inner == 0.0


def test_metrics_carry_the_cumulative_seconds_per_kind():
    s = sched()
    s.tiers = None
    s.max_seqs, s.engine_seqs, s.blocked = 2, 2, None
    s.clock.add("idle", 0.5)
    m = s.metrics()
    assert {k for k in m if k.startswith("q27:engine_seconds_total")} == {f'q27:engine_seconds_total{{kind="{k}"}}' for k in KINDS}
    assert m['q27:engine_seconds_total{kind="idle"}'] == 0.5


def test_a_scheduler_without_a_clock_reports_none():
    s = object.__new__(worker.Scheduler)
    assert s.clock.snapshot() is None
    s.clock.enter("decode")
    s.clock.leave()
    s.clock.add("idle", 1.0)


# ------------------------------------------------------------ Hub

def et(wall, **sec):
    d = dict.fromkeys(KINDS[:-1], 0.0)
    d.update(sec)
    d["other"] = max(0.0, wall - sum(d.values()))
    return {"wall": wall, "seconds": d}


def snap(wall, prefilled=0, generated=0, **sec):
    return {"t": time.time(), "seqs": [], "cache": None, "device": {"free_at_start": 1, "total": 2}, "running": 0, "waiting": 0,
            "blocked": None, "target_seqs": 1, "engine_time": et(wall, **sec), "prefilled": prefilled, "generated": generated}


def test_tick_has_recent_and_total_shares_that_sum_to_one():
    hub = stats.Hub()
    hub.set_snapshot(snap(100, prefill=10, decode=50, idle=40, prefilled=1000, generated=200))
    hub.set_snapshot(snap(130, prefill=25, decode=65, idle=40, prefilled=4000, generated=500))
    blk = hub.tick()["engine_time"]
    r, t = blk["recent"], blk["total"]
    assert abs(sum(r["share"].values()) - 1.0) < 1e-9 and abs(sum(t["share"].values()) - 1.0) < 1e-9
    assert r["wall"] == 30 and abs(r["share"]["prefill"] - 0.5) < 1e-9 and abs(r["share"]["decode"] - 0.5) < 1e-9
    assert abs(r["seconds"]["prefill"] - 15) < 1e-9 and abs(r["seconds"]["decode"] - 15) < 1e-9
    assert r["prefill_tokens"] == 3000 and r["decode_tokens"] == 300
    assert t["wall"] == 130 and abs(t["share"]["idle"] - 40 / 130) < 1e-9


def test_the_recent_window_is_the_last_60_seconds():
    hub = stats.Hub()
    for w, d in ((10, 10), (40, 40), (70, 70), (100, 70)):  # busy decoding to 70 s, then 30 s idle
        hub.set_snapshot(snap(w, decode=d, idle=w - d))
    r = hub.tick()["engine_time"]["recent"]
    assert r["wall"] == 60 and abs(r["share"]["decode"] - 30 / 60) < 1e-9 and abs(r["share"]["idle"] - 30 / 60) < 1e-9


def test_one_snapshot_has_only_the_total_and_an_old_engine_has_no_block():
    hub = stats.Hub()
    hub.set_snapshot(snap(5, decode=5))
    blk = hub.tick()["engine_time"]
    assert blk["recent"] is None and blk["total"]["share"]["decode"] == 1.0
    old = stats.Hub()
    s = snap(5)
    del s["engine_time"]
    old.set_snapshot(s)
    assert old.tick()["engine_time"] is None


def test_a_restarted_engine_starts_the_window_again():
    hub = stats.Hub()
    hub.set_snapshot(snap(500, decode=500))
    hub.set_snapshot(snap(3, prefill=3))  # the new engine's clock starts at 0
    blk = hub.tick()["engine_time"]
    assert blk["recent"] is None and blk["total"]["share"]["prefill"] == 1.0


# ------------------------------------------------------------ the page

def test_the_page_has_the_engine_time_card():
    assert 'id="et-sec"' in HTML and "引擎時間分配" in HTML and "drawEngineTime();" in HTML
    assert "預填與解碼是輪流使用 GPU，不是同時。" in HTML
    for label in ("預填（prefill）", "解碼（decode）", "載回（restore）", "寫入（spill）", "媒體", "閒置", "其他", "累計"):
        assert label in HTML
    assert 'block("engine_time")' in HTML and "sec.hidden = true" in HTML  # (absent: the card hides)
