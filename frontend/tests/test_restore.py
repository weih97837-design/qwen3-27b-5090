"""Delay-hit scheduling (SPILL_ASYNC_PROMOTE, worker.Scheduler.admit_waiting / maybe_restore / poll_restores) against a
fake engine: the "restoring" state machine, the load-vs-recompute decision, timeouts and failures, and that with the flag
off the admission is the old head-of-line loop. No GPU."""

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402

GB = 10**9


class Eng:
    """spill_match / spill_restore_begin / spill_restore_poll as the engine's API behaves (queued: begin returns 1 for an
    entry in RAM and 2 for one on disk; the id comes out of the poll once the test says the load ended)."""

    def __init__(self, match=None, begin=1):
        self.match, self.begin_result = match, begin
        self.match_calls, self.begin_calls, self.poll_calls = [], [], 0
        self.finished, self.failed, self.sync = [], 0, []
        self.released = []
        self.pos = 0
        self.resets = []

    def spill_match(self, ids, spec=False):
        self.match_calls.append((len(ids), spec))
        return dict(self.match) if self.match else None

    def reset(self):
        self.resets.append(len(self.begin_calls))  # (how many begins came before this reset)

    def spill_restore_begin(self, entry_id):
        self.begin_calls.append(entry_id)
        return self.begin_result

    def spill_restore_poll(self, cap=64):
        self.poll_calls += 1
        out, self.finished = self.finished, []
        return out

    def spill_release(self, entry_id):
        self.released.append(entry_id)
        return True

    def spill_stats(self):
        return {"restore_failed": self.failed, "restoring": 0}

    def spill_sync_lookup(self, on):
        self.sync.append(bool(on))

    def cache_stats(self):
        return {"free_pages": 10**6, "checkpoint_only_pages": 0, "park_pages": 171, "checkpoints": 0, "evicted": 0}

    def device_memory(self):
        return [0, 0, 0, 0]

    def prefix_begin(self, ids, at=(), spec=False):
        self.sync.append("prefix_begin")
        return 0


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def sched(engine, seqs=4, on=True, pinned=True):
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.tiers = engine, Conn(), None
    s.engine_seqs = s.max_seqs = seqs
    s.admit_kv, s.kv_reserve, s.min_tps = False, 4096, 0.0
    s.rates, s.blocked, s.held_until = {}, None, (None, 0.0)
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.slot_lanes, s.last_stats, s.spec, s.cur = {}, 0.0, False, None
    s.async_promote, s.restores, s.restore_bw, s.restore_pinned = on, {}, {}, pinned
    return s


def seq(rid, n=20_000):
    return worker.Seq(rid, {"ids": [hash(rid) % 1000 * 1000 + i for i in range(n)], "max_new": 10, "temperature": 1.0,
                            "top_k": 20, "top_p": 0.95}, 0.0)


def ram_match(pos=50_000, nbytes=3 * GB, on_disk=False, eid=7):
    return {"id": eid, "pos": pos, "bytes": nbytes, "on_disk": on_disk}


def test_flag_off_is_the_old_head_of_line_admission():
    eng = Eng(match=ram_match())
    s = sched(eng, seqs=2, on=False)
    a, b, c = seq("a"), seq("b"), seq("c")
    s.waiting.extend([a, b, c])
    s.admit_waiting()
    assert s.alive == [a, b] and list(s.waiting) == [c]
    assert eng.match_calls == [] and eng.begin_calls == [] and eng.poll_calls == 0
    assert all(q.restoring is None for q in (a, b, c))


def test_ram_hit_delays_the_request_and_admits_the_next_ones():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a, b = seq("a"), seq("b")
    s.waiting.extend([a, b])
    eng.match = ram_match()  # only a's prompt matches; b's lookup finds nothing
    calls = []

    def match(ids, spec=False):
        calls.append(len(ids))
        return ram_match() if ids is a.eng_ids else None

    eng.spill_match = match
    s.admit_waiting()
    assert a.restoring is not None and s.alive == [b] and list(s.waiting) == [a]
    assert eng.begin_calls == [7]
    assert s.restores[7]["seqs"] == [a]
    # still loading: the next pass keeps it waiting and does not ask again
    s.admit_waiting()
    assert s.alive == [b] and a.restoring is not None and eng.begin_calls == [7] and len(calls) == 2
    # the load ends: the request is admissible again, in its own place
    eng.finished = [(7, True)]
    s.admit_waiting()
    assert a.restoring is None and s.alive == [b, a] and not s.waiting and 7 not in s.restores
    assert s.m["q27:restores_started_total"] == 1


def test_a_restoring_request_does_not_take_an_engine_slot():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng, seqs=1)
    a, b = seq("a"), seq("b")
    s.waiting.extend([a, b])
    eng.spill_match = lambda ids, spec=False: ram_match() if ids is a.eng_ids else None
    s.admit_waiting()
    assert s.alive == [b] and list(s.waiting) == [a]  # (a waits for its load, not for b's slot)


def test_disk_entry_begin_2_and_bandwidth_is_measured_per_tier(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(nbytes=4 * GB, on_disk=True, eid=9), begin=2)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert a.restoring is not None and eng.begin_calls == [9]
    clk.t += 4.0
    eng.finished = [(9, True)]
    s.poll_restores()
    assert a.restoring is None
    assert s.restore_bw["ssd"] == 4 * GB / 4.0 and "ram" not in s.restore_bw
    assert s.bandwidth(True) == 1 * GB  # measured, inside the clamp
    s.restore_bw["ssd"] = 1  # absurdly slow: clamped so restores do not lock themselves out
    assert s.bandwidth(True) == worker.RESTORE_BW_SSD * worker.RESTORE_BW_CLAMP[0]
    assert s.bandwidth(False) == worker.RESTORE_BW_RAM_PINNED
    s.restore_pinned = False
    assert s.bandwidth(False) == worker.RESTORE_BW_RAM_PAGEABLE


def test_timeout_releases_the_request_and_ignores_the_late_result(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=2)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert a.restoring is not None
    clk.t += worker.RESTORE_TIMEOUT_S - 1
    s.admit_waiting()
    assert a.restoring is not None and not s.alive
    clk.t += 2
    s.admit_waiting()
    assert a.restoring is None and s.alive == [a]  # admitted: prefix_begin falls back
    assert s.m["q27:restore_timeouts_total"] == 1
    # a new request for the same entry does not wait for the late load, nor starts another
    b = seq("b")
    s.waiting.append(b)
    s.admit_waiting()
    assert b.restoring is None and s.alive == [a, b] and eng.begin_calls == [7]
    eng.finished = [(7, True)]
    s.poll_restores()
    assert s.restore_bw == {} and 7 not in s.restores  # (a late load teaches nothing about the speed)


def test_failed_restore_releases_the_request_without_a_speed_sample():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    eng.finished = [(7, False)]
    s.admit_waiting()
    assert a.restoring is None and s.alive == [a] and s.restore_bw == {}


def test_begin_returning_0_admits_as_before():
    eng = Eng(match=ram_match(), begin=0)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert s.alive == [a] and a.restoring is None and not a.no_sync and not s.restores
    s.waiting.append(b := seq("b"))
    s.admit_waiting()  # (asked once per request)
    assert len(eng.begin_calls) == 2 and b in s.alive


def test_two_requests_share_one_restore():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a, b = seq("a"), seq("b")
    s.waiting.extend([a, b])
    s.admit_waiting()
    assert eng.begin_calls == [7] and s.restores[7]["seqs"] == [a, b]
    assert a.restoring is b.restoring and not s.alive
    eng.finished = [(7, True)]
    s.admit_waiting()
    assert s.alive == [a, b]


def test_load_versus_recompute_decision():
    s = sched(Eng())
    m = ram_match(pos=50_000, nbytes=3 * GB)  # 3 GB / 12 GB/s = 0.25 s (+ 0.05) against 50k / 8000 = 6.25 s
    assert s.worth_loading(m)
    assert not s.worth_loading(ram_match(pos=1_000, nbytes=3 * GB))  # 1000 tokens take 0.125 s to compute
    # the measured prefill speed counts
    s.measure_prefill(8192, 0.5)  # 16,384 tok/s
    assert s.prefill_tps == 16384
    assert s.est_recompute_s(16384) == 1.0
    # exactly at the margin: not worth it
    s.restore_pinned, s.prefill_tps = False, 1000.0
    load = 6 * GB / worker.RESTORE_BW_RAM_PAGEABLE + worker.RESTORE_OVERHEAD_S
    pos = int(load / worker.RESTORE_MARGIN * 1000.0)
    assert not s.worth_loading(ram_match(pos=pos, nbytes=6 * GB))
    assert s.worth_loading(ram_match(pos=pos + 5, nbytes=6 * GB))
    # an SSD entry: 4.5 GB at 2 GB/s = 2.3 s
    assert s.worth_loading(ram_match(pos=200_000, nbytes=int(4.5 * GB), on_disk=True))  # 200 s to compute at 1000 tok/s
    s.prefill_tps = 200_000.0
    assert not s.worth_loading(ram_match(pos=200_000, nbytes=int(4.5 * GB), on_disk=True))  # 1 s to compute


def test_not_worth_loading_recomputes_and_keeps_prefix_begin_from_loading_it():
    eng = Eng(match=ram_match(pos=500, nbytes=3 * GB), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert s.alive == [a] and a.restoring is None and a.no_sync and eng.begin_calls == []
    s.begin(a)
    assert eng.sync == [False, "prefix_begin", True]  # off around prefix_begin, then on again


def test_engine_error_in_match_falls_back_to_the_normal_path():
    eng = Eng()

    def boom(ids, spec=False):
        raise RuntimeError("q27_spill_match: boom")

    eng.spill_match = boom
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert s.alive == [a] and a.restoring is None


def test_poll_error_keeps_the_restores_and_polls_again(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    good = eng.spill_restore_poll
    calls = []

    def boom(cap=64):
        calls.append(1)
        raise RuntimeError("poll")

    eng.spill_restore_poll = boom
    s.admit_waiting()
    assert a.restoring is not None and s.restores and s.async_promote and len(calls) == 1
    eng.spill_restore_poll = good  # the next iteration polls again and the queued restore completes
    eng.finished = [(7, True)]
    s.admit_waiting()
    assert a.restoring is None and s.alive == [a] and not s.restores


def test_poll_error_still_times_the_waiters_out(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()

    def boom(cap=64):
        raise RuntimeError("poll")

    eng.spill_restore_poll = boom
    clk.t += worker.RESTORE_TIMEOUT_S + 1
    s.admit_waiting()
    assert a.restoring is None and s.alive == [a] and eng.released == [7]


def test_an_admission_that_says_no_still_stops_the_pass():
    eng = Eng()
    s = sched(eng)
    s.admit_kv = True
    run = seq("r")
    run.started, run.fed = True, len(run.eng_ids)
    s.alive = [run]
    s.engine.cache_stats = lambda: {"free_pages": 0, "checkpoint_only_pages": 0, "park_pages": 171, "checkpoints": 0,
                                    "evicted": 0}
    a, b = seq("a", 64_000), seq("b", 64)
    s.waiting.extend([a, b])
    s.admit_waiting()
    assert s.alive == [run] and list(s.waiting) == [a, b] and s.blocked == "kv"


def test_decode_turns_end_when_a_restore_releases_a_request(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    s.last_restore_poll = 0.0
    assert not s.restore_due()  # nothing pending
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    mono = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: mono[0])
    assert not s.restore_due()  # polled, nothing ended
    polls = eng.poll_calls
    assert not s.restore_due() and eng.poll_calls == polls  # (throttled: the same instant)
    mono[0] += 1
    eng.finished = [(7, True)]
    assert s.restore_due() and a.restoring is None


def test_stats_show_the_restoring_state_and_count():
    eng = Eng(match=ram_match(on_disk=True), begin=2)
    s = sched(eng)
    a, b = seq("a"), seq("b")
    s.waiting.extend([a, b])
    eng.spill_match = lambda ids, spec=False: ram_match(on_disk=True) if ids is a.eng_ids else None
    s.admit_waiting()
    snap = s.stats()
    states = {r["rid"]: r["state"] for r in snap["seqs"]}
    assert states == {"a": "restoring", "b": "admitted"}
    row = next(r for r in snap["seqs"] if r["rid"] == "a")
    assert row["restore"]["on_disk"] and row["restore"]["tokens"] == 50_000
    assert snap["restoring"] == 1 and snap["waiting"] == 1
    assert [r["rid"] for r in snap["slots"] if r["rid"]] == ["b"]  # (a restoring request has no lane)
    json.dumps(snap)
    eng.finished = [(7, True)]
    s.admit_waiting()
    assert s.stats()["restoring"] == 0


# ---- the tier policy leaves an entry that is being restored alone; the tiers block reports the restores

def test_tier_policy_skips_busy_entries(tmp_path):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import test_vram_backup as tvb

    eng = tvb.Base()
    idle, busy = eng.add_ram(size=100), eng.add_ram(size=100)
    eng.entries[busy]["busy"] = True
    t = tvb.tiers_for(eng, directory=tmp_path)
    assert t.flush(10.0) == {"written": 1, "left": 0}
    assert eng.exports == [idle]


def test_tiers_block_reports_async_restores():
    import stats

    class E:
        def spill_stats(self):
            return {"entries": 1, "in_ram": 1, "on_disk": 0, "ram_bytes": 1, "spilled": 1, "restored": 1, "written": 0,
                    "dropped": 0, "subsumed": 0, "failed": 0, "restoring": 2, "restored_async": 5, "restore_failed": 1,
                    "restore_ms_total": 2000, "restore_bytes_total": 24 * GB}

    class T:
        directory = None
        ram_retention_s = ttl_s = 1.0

        def status(self):
            return {"ssd_bytes": 0, "ssd_budget_bytes": 1, "last_error": None}

    out = stats.tiers_block(E(), T(), environ={})
    assert out["restoring"] == 2 and out["restored_async"] == 5 and out["restore_failed"] == 1
    assert out["restore_gbps"] == 12.0


def test_a_failure_does_not_poison_later_speed_samples(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    for _ in range(3):  # (polls that end nothing must not reset the failure baseline)
        s.poll_restores()
    clk.t += 1.0
    eng.finished = [(7, False)]
    s.poll_restores()
    assert a.restoring is None and s.restore_bw == {}
    # a later restore that succeeds teaches the speed, though the engine's failure counter stays at 1
    eng.match = ram_match(eid=8)
    b = seq("b")
    s.waiting.append(b)
    s.admit_waiting()
    s.poll_restores()
    clk.t += 0.5
    eng.finished = [(8, True)]
    s.poll_restores()
    assert s.restore_bw == {"ram": 3 * GB / 0.5}


# ---- review fixes: sync lookup restore, fit check, KvFull retry, release, per-id attribution, restart, no blocking

class KvFull(Exception):
    pass


def test_sync_lookup_is_restored_when_prefix_begin_raises():
    eng = Eng(match=ram_match(pos=500, nbytes=3 * GB), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert a.no_sync
    calls = {"n": 0}

    def prefix_begin(ids, at=(), spec=False):
        eng.sync.append("prefix_begin")
        calls["n"] += 1
        if calls["n"] == 1:
            raise KvFull("pool")
        return 0

    eng.prefix_begin = prefix_begin
    try:
        s.begin(a)
        assert False, "KvFull expected"
    except KvFull:
        pass
    assert eng.sync == [False, "prefix_begin", True] and s.cur is None
    # the retry (after a preemption or a restore) is still told not to promote, and ends on True again
    eng.sync.clear()
    s.begin(a)
    assert a.no_sync and eng.sync == [False, "prefix_begin", True]


def test_kvfull_in_a_turn_retries_with_no_sync_kept():
    eng = Eng(match=ram_match(pos=500, nbytes=3 * GB), begin=1)
    s = sched(eng)
    s.KvFull = KvFull
    s.backup_done, s.last_active = False, 0.0
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    n = {"k": 0}

    def prefill_turn(q):
        s.begin(q)
        n["k"] += 1
        if n["k"] == 1:
            raise KvFull("pool")

    s.prefill_turn = prefill_turn
    # a lone request that runs out finishes with "length" (nobody to preempt) - what matters: sync is restored each time
    s.finish = lambda q, reason, **kw: setattr(q, "dead", True)
    s.guarded("prefill", a)
    assert eng.sync == [False, "prefix_begin", True]
    assert a.no_sync


def test_restore_that_does_not_fit_is_not_started():
    eng = Eng(match=ram_match(pos=50_000, nbytes=3 * GB), begin=1)
    s = sched(eng)
    # 50k tokens = 196 pages; the pool has 100 free
    eng.cache_stats = lambda: {"free_pages": 100, "checkpoint_only_pages": 0, "park_pages": 171, "checkpoints": 0,
                               "evicted": 0}
    a = seq("a", n=60_000)
    s.waiting.append(a)
    s.admit_waiting()
    assert eng.begin_calls == [] and a.restoring is None and s.alive == [a] and not a.no_sync and not s.restores
    # with room it starts
    eng.cache_stats = lambda: {"free_pages": 10_000, "checkpoint_only_pages": 0, "park_pages": 171, "checkpoints": 0,
                               "evicted": 0}
    b = seq("b", n=60_000)
    s.waiting.append(b)
    s.admit_waiting()
    assert eng.begin_calls == [7] and b.restoring is not None


def test_restore_fit_counts_what_running_requests_reserve():
    eng = Eng(match=ram_match(pos=20_000, nbytes=GB), begin=1)
    s = sched(eng)
    run = seq("r", n=100_000)
    run.started, run.fed = True, 10_000  # still needs 90k tokens of prompt = 352 pages (+ output)
    s.alive = [run]
    # the restore alone (79 pages + the rest of the new prompt) would fit in 600 pages, but not next to the runner
    eng.cache_stats = lambda: {"free_pages": 600, "checkpoint_only_pages": 0, "park_pages": 0, "checkpoints": 0,
                               "evicted": 0}
    a = seq("a", n=30_000)
    s.waiting.append(a)
    s.admit_waiting()
    assert eng.begin_calls == [] and a.restoring is None


def test_kvfull_polls_restores_once_and_retries_without_preempting():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    s.KvFull = KvFull
    s.backup_done, s.last_active = False, 0.0
    a, b = seq("a"), seq("b")
    s.waiting.append(a)
    s.admit_waiting()
    assert a.restoring is not None
    run = b
    run.started, run.fed = True, 10
    s.alive.append(run)
    n = {"k": 0}

    def decode_turn(q):
        n["k"] += 1
        if n["k"] == 1:
            eng.finished = [(7, True)]  # the restore ends while the turn runs
            raise KvFull("pool")
        return "ok"

    s.decode_turn = decode_turn
    s.suspend = lambda v: (_ for _ in ()).throw(AssertionError("must not preempt"))
    assert s.guarded("decode", run) == "ok"
    assert n["k"] == 2 and a.restoring is None
    # nothing completes: falls through to the preemption as before
    s.restores = {}
    n["k"] = 0
    s.suspend = lambda v: s.alive.remove(v)
    s.alive = [run]
    s.finish = lambda q, reason, **kw: setattr(q, "dead", True)
    s.guarded("decode", run)
    assert run.dead  # (lone request that cannot be held: finished "length")


def test_release_on_timeout_cancel_and_finish(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    s.by_id = {}
    s.history, s.decoder = collections.deque(), None
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    clk.t += worker.RESTORE_TIMEOUT_S + 1
    s.admit_waiting()
    assert eng.released == [7]  # timeout
    # cancel while pending: the last waiter leaving releases it
    eng2 = Eng(match=ram_match(eid=9), begin=1)
    s2 = sched(eng2)
    s2.by_id, s2.history, s2.decoder = {}, collections.deque(), None
    b, c = seq("b"), seq("c")
    s2.by_id.update({"b": b, "c": c})
    s2.waiting.extend([b, c])
    s2.admit_waiting()
    assert s2.restores[9]["seqs"] == [b, c]
    s2.cancel("b")
    assert eng2.released == [] and s2.restores[9]["seqs"] == [c]
    s2.cancel("c")
    assert eng2.released == [9] and s2.restores[9]["late"]
    # a request cancelled after the restore ended but before it began releases the held checkpoint
    eng3 = Eng(match=ram_match(eid=11), begin=1)
    s3 = sched(eng3)
    s3.by_id, s3.history, s3.decoder = {}, collections.deque(), None
    d = seq("d")
    s3.by_id["d"] = d
    s3.waiting.append(d)
    s3.admit_waiting()
    eng3.finished = [(11, True)]
    s3.poll_restores()
    assert d.restoring is None and d.restore_id == 11 and eng3.released == []
    s3.cancel("d")
    assert eng3.released == [11]


def test_release_after_begin_and_without_engine_support():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    eng.finished = [(7, True)]
    s.poll_restores()
    s.begin(a)
    assert eng.released == [7] and a.restore_id is None
    del Eng.spill_release  # an older engine build: tolerated
    try:
        b = seq("b")
        b.restore_id = 3
        s.begin(b)
        assert b.restore_id is None
    finally:
        Eng.spill_release = lambda self, entry_id: (self.released.append(entry_id), True)[1]


def test_failure_is_attributed_to_its_own_id(monkeypatch):
    clk = Clock()
    monkeypatch.setattr(worker.time, "time", clk)
    eng = Eng(match=ram_match(eid=1), begin=1)
    s = sched(eng)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    eng.match = ram_match(eid=2, nbytes=6 * GB)
    b = seq("b")
    s.waiting.append(b)
    s.admit_waiting()
    clk.t += 1.0
    eng.finished = [(1, False), (2, True)]  # one poll: 1 failed, 2 worked
    s.poll_restores()
    assert a.restoring is None and b.restoring is None
    assert s.restore_bw == {"ram": 6 * GB / 1.0}  # only the success is a sample


def test_restart_resets_the_delay_hit_state():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    v = seq("v")
    v.restore_checked, v.no_sync, v.restore_id = True, True, 5
    v.started, v.fed, v.phase = True, 100, worker.DECODE
    s.restart(v)
    assert not v.restore_checked and not v.no_sync and v.restore_id is None and v.restoring is None


def test_poll_never_blocks_while_restores_are_pending():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    waits = []

    class C(Conn):
        def poll(self, t=0.0):
            waits.append(t)
            return False

    s.conn = C()
    s.cache_full = s.pin_full = True
    s.poll(block=True, wait=5.0)
    assert waits[0] == 5.0  # (nothing pending: the long idle wait)
    waits.clear()
    s.restores = {7: {"late": True, "seqs": []}}
    s.poll(block=True, wait=5.0)
    assert waits[0] == worker.RESTORE_POLL_S


def test_head_ignores_restoring_requests():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    a, b = seq("a"), seq("b")
    s.waiting.extend([a, b])
    eng.spill_match = lambda ids, spec=False: ram_match() if ids is a.eng_ids else None
    s.admit_waiting()
    assert a.restoring is not None and s.head() is None  # b was admitted
    c = seq("c")
    s.waiting.append(c)
    assert s.head() is c


def test_finished_restore_is_admitted_without_charging_its_prefix_twice():
    """A request whose restore has finished still holds the restored prefix (restore_id set, restore_pos = its tokens):
    admit() counts those pages as covered, not as need on top of the room the restore took. With cached=0 the same
    request is charged for them twice and is not admitted."""
    eng = Eng(match=None)
    eng.cache_stats = lambda: {"free_pages": 116, "checkpoint_only_pages": 0, "park_pages": 0, "checkpoints": 0,
                               "evicted": 0}
    s = sched(eng)
    s.admit_kv = True
    a = seq("a", n=64)  # running: 1 prompt page + 1 output page
    s.alive.append(a)
    r = seq("r", n=20_000)  # 313 prompt pages; 200 of them (12800 tokens) come from the restored checkpoint
    r.restore_id, r.restore_pos = 5, 12_800
    # room 116 = need of a (2) + need of r without the restored prefix (1 + 113 = 114, the output page included)
    assert s.kv_room(r, cached=0) is False
    assert s.kv_room(r, cached=r.restore_pos) is True
    assert s.admit(r) is True
    assert s.blocked is None

    # once the restore is released (no restore_id), the same request is charged for its whole prompt again
    r.restore_id, r.restore_pos = None, 0
    assert s.admit(r) is False
    assert s.blocked == "kv"


def test_finished_cur_is_reset_before_restore_begin():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    done = seq("done")
    done.dead = True
    s.cur = done
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert eng.resets == [0] and eng.begin_calls == [7]  # reset first, then the restore
    assert s.cur is None and a.restoring is not None


def test_running_cur_is_not_reset_for_a_restore():
    eng = Eng(match=ram_match(), begin=1)
    s = sched(eng)
    run = seq("run")
    run.started = True
    s.cur = run
    s.alive.append(run)
    a = seq("a")
    s.waiting.append(a)
    s.admit_waiting()
    assert eng.resets == [] and s.cur is run and eng.begin_calls == [7]
