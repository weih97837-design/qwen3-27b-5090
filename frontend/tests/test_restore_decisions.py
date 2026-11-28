"""Restore-decision counters (worker.Scheduler.note_restore; q27:restore_decisions_total{tier,outcome}; the stats snapshot's
"restore_decisions"; the trace's req.restore / restore_tier): every outcome of maybe_restore / poll_restores / begin, with the fake engine of
test_restore.py. The counters only count: the decisions themselves are test_restore.py's. No GPU."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import worker  # noqa: E402
from test_restore import GB, Clock, Eng, ram_match, sched, seq  # noqa: E402


def key(tier, outcome):
    return f'q27:restore_decisions_total{{tier="{tier}",outcome="{outcome}"}}'


def counts(s):
    return {k: v for k, v in s.m.items() if k.startswith("q27:restore_decisions_total")}


class RefEng(Eng):
    """Eng whose spill_stats has the engine's spill_restore_begin refusal counters; `refuse` names the counter a begin that returns 0
    moves (as Model::spill_restore_begin does)."""

    def __init__(self, refuse=None, **kw):
        super().__init__(**kw)
        self.refuse = refuse
        self.ctr = {"restore_refused_busy": 0, "restore_refused_room": 0, "restore_dropped_missing": 0, "restore_failed": 0,
                    "restored": 0, "restored_ram": 0, "restored_disk": 0}

    def spill_restore_begin(self, entry_id):
        r = super().spill_restore_begin(entry_id)
        if not r and self.refuse:
            for k in self.refuse if isinstance(self.refuse, tuple) else (self.refuse,):
                self.ctr[k] += 1
        return r

    def spill_stats(self):
        return dict(self.ctr, restoring=0)


def run(eng, **kw):
    s = sched(eng, **kw)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    return s, a


def test_no_match_and_started_per_tier():
    s, a = run(Eng(match=None))
    assert counts(s) == {key("none", "no_match"): 1}
    assert (a.restore_outcome, a.restore_tier) == ("no_match", None)
    s, a = run(Eng(match=ram_match(), begin=1))
    assert counts(s) == {key("ram", "matched"): 1, key("ram", "started"): 1}
    assert (a.restore_outcome, a.restore_tier) == ("started", "ram")
    s, a = run(Eng(match=ram_match(on_disk=True, nbytes=4 * GB, pos=200_000), begin=2))
    assert counts(s) == {key("ssd", "matched"): 1, key("ssd", "started"): 1}
    # counted once per stay in the queue: further passes do not count again
    s.admit_waiting()
    s.admit_waiting()
    assert s.m[key("ssd", "matched")] == 1


def test_joined_and_skipped_busy_and_the_ends(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a, b = seq("a"), seq("b")
    s.waiting.extend([a, b])
    s.admit_waiting()
    assert s.m[key("ram", "started")] == 1 and s.m[key("ram", "joined")] == 1 and s.m[key("ram", "matched")] == 2
    assert b.restore_outcome == "joined"
    eng.finished = [(7, True)]
    s.admit_waiting()
    assert s.m[key("ram", "restored")] == 1 and a.restore_outcome == b.restore_outcome == "restored"
    # a failed restore
    s2, c = run(Eng(match=ram_match(eid=8), begin=1))
    s2.engine.finished = [(8, False)]
    s2.admit_waiting()
    assert s2.m[key("ram", "failed")] == 1 and c.restore_outcome == "failed"
    # timed out, then the late result: counted once (timed_out), and a newcomer for the late entry is skipped_busy
    eng3 = Eng(match=ram_match(eid=9), begin=2)
    s3 = sched(eng3)
    d = seq("d")
    s3.waiting.append(d)
    s3.admit_waiting()
    clk.t += worker.RESTORE_TIMEOUT_S + 1
    s3.admit_waiting()
    assert s3.m[key("ram", "timed_out")] == 1 and d.restore_outcome == "timed_out"
    e = seq("e")
    s3.waiting.append(e)
    s3.admit_waiting()
    assert s3.m[key("ram", "skipped_busy")] == 1 and e.restore_outcome == "skipped_busy"
    eng3.finished = [(9, True)]
    s3.poll_restores()
    assert s3.m[key("ram", "restored")] == 0 and s3.m[key("ram", "timed_out")] == 1
    # matched == the sum of the per-check decisions; started == the sum of the ends
    for sc in (s, s2, s3):
        for tier in ("ram", "ssd"):
            assert sc.m[key(tier, "matched")] == sum(sc.m[key(tier, o)] for o in worker.RESTORE_DECISIONS)
            assert sc.m[key(tier, "started")] == sum(sc.m[key(tier, o)] for o in worker.RESTORE_ENDS)


def test_not_worth_loading_keeps_its_numbers():
    s, a = run(Eng(match=ram_match(pos=1_000, nbytes=3 * GB, on_disk=True), begin=1))
    assert counts(s) == {key("ssd", "matched"): 1, key("ssd", "not_worth_loading"): 1}
    assert a.no_sync and a.restore_outcome == "not_worth_loading" and a.restore_tier == "ssd"
    w = s.restore_decisions()["not_worth"]["ssd"]
    load = 3 * GB / worker.RESTORE_BW_SSD + worker.RESTORE_OVERHEAD_S
    assert w["n"] == 1 and w["bytes"] == 3 * GB and abs(w["load_s"] - load) < 1e-9
    assert abs(w["recompute_s"] - 1_000 / worker.PREFILL_TPS_DEFAULT) < 1e-9
    assert w["last"] == {"tokens": 1_000, "bytes": 3 * GB, "load_s": w["load_s"], "recompute_s": w["recompute_s"],
                         "bandwidth": worker.RESTORE_BW_SSD, "prefill_tps": worker.PREFILL_TPS_DEFAULT}
    assert s.m['q27:restore_not_worth_bytes_total{tier="ssd"}'] == 3 * GB
    assert abs(s.m['q27:restore_not_worth_load_seconds_total{tier="ssd"}'] - load) < 1e-9
    # a second one sums
    b = seq("b")
    s.waiting.append(b)
    s.admit_waiting()
    assert s.restore_decisions()["not_worth"]["ssd"]["n"] == 2 and s.restore_decisions()["not_worth"]["ssd"]["bytes"] == 6 * GB


def test_no_kv_room():
    eng = Eng(match=ram_match(), begin=1)
    eng.cache_stats = lambda: {"free_pages": 0, "checkpoint_only_pages": 0, "park_pages": 171, "checkpoints": 0, "evicted": 0}
    s, a = run(eng)
    assert counts(s) == {key("ram", "matched"): 1, key("ram", "no_kv_room"): 1} and eng.begin_calls == []
    assert a.restore_outcome == "no_kv_room"


def test_engine_refusals_are_told_apart():
    for refuse, outcome in (("restore_refused_room", "spill_room_refused"), ("restore_refused_busy", "skipped_busy"),
                            (("restore_dropped_missing", "restore_failed"), "chunk_missing"), ("restore_failed", "damaged"),
                            (None, "begin_refused")):
        s, a = run(RefEng(refuse=refuse, match=ram_match(on_disk=True, nbytes=GB, pos=100_000), begin=0))
        assert counts(s) == {key("ssd", "matched"): 1, key("ssd", outcome): 1}, refuse
        assert a.restore_outcome == outcome and a.restoring is None and s.alive == [a]  # (admitted as before)
    # an engine without the counters (test_restore's Eng): begin_refused
    s, a = run(Eng(match=ram_match(), begin=0))
    assert counts(s) == {key("ram", "matched"): 1, key("ram", "begin_refused"): 1}


def test_lookup_failures_with_and_without_a_match():
    eng = Eng()

    def boom(ids, spec=False):
        raise RuntimeError("engine said no")

    eng.spill_match = boom
    s, a = run(eng)
    assert counts(s) == {key("none", "lookup_failed"): 1} and s.alive == [a]
    eng = Eng(match=ram_match(), begin=1)
    eng.cache_stats = lambda: (_ for _ in ()).throw(RuntimeError("cache_stats failed"))
    s, a = run(eng)
    assert counts(s) == {key("ram", "matched"): 1, key("ram", "lookup_failed"): 1}


def test_counting_never_changes_a_decision():
    """A failure inside the counters is swallowed: the request is restored exactly as without them."""
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    s.restore_counts = s.restore_not_worth = object()  # (any counting failure)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert a.restoring is not None and eng.begin_calls == [7] and s.m["q27:restores_started_total"] == 1
    s2, b = run(Eng(match=ram_match(pos=1_000, nbytes=3 * GB), begin=1))
    s2.restore_not_worth = object()
    c = seq("c")
    s2.waiting.append(c)
    s2.admit_waiting()
    assert c.no_sync and c.restoring is None and c in s2.alive


def test_not_checked_and_sync_restored_at_begin():
    class SyncEng(RefEng):
        def prefix_begin(self, ids, at=(), spec=False):  # (prefix_begin promoted an SSD entry itself)
            self.ctr["restored"] += 1
            self.ctr["restored_disk"] += 1
            return 0

    eng = SyncEng(match=None)
    s = sched(eng, on=False)  # (delay hits off: maybe_restore never looks)
    s.tiers = object()
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    s.begin(a)
    assert counts(s) == {key("none", "not_checked"): 1, key("ssd", "sync_restored"): 1}
    assert a.restore_outcome == "not_checked"
    # with delay hits on, a checked request is not "not_checked"
    s2 = sched(RefEng(match=None))
    s2.tiers = object()
    b = seq("b")
    s2.waiting.append(b)
    s2.admit_waiting()
    s2.begin(b)
    assert counts(s2) == {key("none", "no_match"): 1}
    # without a tier policy nothing is counted at begin
    s3 = sched(Eng(match=None), on=False)
    c = seq("c")
    s3.waiting.append(c)
    s3.admit_waiting()
    s3.begin(c)
    assert counts(s3) == {}


def test_snapshot_block():
    s, a = run(Eng(match=ram_match(), begin=1))
    rd = s.restore_decisions()
    assert rd["counts"] == {"ram": {"matched": 1, "started": 1}} and rd["not_worth"] == {}
    assert rd["outcomes"] == list(worker.RESTORE_OUTCOMES) and "no_match" in rd["none_outcomes"]
    st = s.stats()
    assert st["restore_decisions"]["counts"]["ram"]["started"] == 1
    # a Scheduler with nothing counted yet
    assert sched(Eng()).restore_decisions()["counts"] == {}


def test_the_tiers_block_carries_the_decisions_for_the_tier_panel():
    s, a = run(Eng(match=ram_match(), begin=1))
    blk = s.tiers_with_decisions({"ram_bytes": 1})
    assert blk["restore_decisions"]["counts"]["ram"]["started"] == 1
    assert s.tiers_with_decisions(None) is None and "restore_decisions" not in s.tiers_with_decisions({"error": "x"})


def test_metrics_have_the_gate_and_refusal_series():
    class Tiers:
        def status(self):
            return {"ssd_bytes": 0, "ssd_budget_bytes": 0, "cache_policy": "smart"}

    eng = RefEng(match=ram_match(), begin=1)
    base = dict(ram_bytes=0, in_ram=0, on_disk=0, spilled=0, written=0, dropped=0, subsumed=0, failed=0,
                ssd_gate_rejected=5, ssd_gate_rejected_short=2, ssd_gate_rejected_watermark=3, ssd_gate_checked=9)
    eng.spill_stats = lambda: dict(eng.ctr, **base)
    s = sched(eng)
    s.tiers, s.clock, s.share, s.engine_seqs = Tiers(), worker.EngineClock(), 1.0, 4
    s.client_metrics = lambda: {}
    s.waiting.append(seq("a"))
    s.admit_waiting()
    m = s.metrics()
    assert m["q27:tier_ssd_gate_rejected"] == 5 and m["q27:tier_ssd_gate_rejected_watermark"] == 3 and m["q27:tier_ssd_gate_checked"] == 9
    assert m["q27:tier_restore_refused_room"] == 0 and m[key("ram", "started")] == 1


def test_metrics_have_the_speed_counters_only_from_an_engine_that_reports_them():
    class Tiers:
        def status(self):
            return {"ssd_bytes": 0, "ssd_budget_bytes": 0, "cache_policy": "smart"}

    eng = RefEng(match=ram_match(), begin=1)
    base = dict(ram_bytes=0, in_ram=0, on_disk=0, spilled=0, written=0, dropped=0, subsumed=0, failed=0,
                ssd_read_bytes=4 * 10**9, ssd_read_ns=2 * 10**9, h2d_bytes=3 * 10**9, h2d_ns=5 * 10**8)
    eng.spill_stats = lambda: dict(eng.ctr, **base)
    s = sched(eng)
    s.tiers, s.clock, s.share, s.engine_seqs = Tiers(), worker.EngineClock(), 1.0, 4
    s.client_metrics = lambda: {}
    s.waiting.append(seq("a"))
    s.admit_waiting()
    m = s.metrics()
    assert m["q27:ssd_read_bytes_total"] == 4 * 10**9 and m["q27:ssd_read_seconds_total"] == 2.0
    assert m["q27:h2d_bytes_total"] == 3 * 10**9 and m["q27:h2d_seconds_total"] == 0.5
    del base["ssd_read_bytes"], base["ssd_read_ns"], base["h2d_bytes"], base["h2d_ns"]  # an older engine
    s2 = sched(eng)
    s2.tiers, s2.clock, s2.share, s2.engine_seqs = Tiers(), worker.EngineClock(), 1.0, 4
    s2.client_metrics = lambda: {}
    m2 = s2.metrics()
    assert "q27:ssd_read_bytes_total" not in m2 and "q27:h2d_seconds_total" not in m2
