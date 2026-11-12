"""Requests that share a prefix with one still computing it (worker.Scheduler.submit / can_prefill /
plan_sibling_checkpoint), and the end checkpoint of a cancelled request, against a fake engine: no GPU."""

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402

IM = 99999


class Eng:
    """Records add_checkpoint_at calls; `accept` is what the engine answers."""

    def __init__(self, accept=True):
        self.accept, self.calls, self.checkpoints, self.suspended, self.dropped = accept, [], 0, [], []

    def add_checkpoint_at(self, area, pos):
        self.calls.append((area, pos))
        return self.accept

    def checkpoint(self):
        self.checkpoints += 1

    def seq_suspend(self, a):
        self.suspended.append(a)

    def seq_drop(self, a):
        self.dropped.append(a)

    def metrics_stub(self):
        return {}


class Conn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def sched(engine):
    s = object.__new__(worker.Scheduler)
    s.engine, s.conn, s.tiers = engine, Conn(), None
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.history, s.by_id, s.cur, s.im_start = collections.deque(maxlen=64), {}, None, IM
    s.last_active, s.restores, s.restore_bw = 0.0, {}, {}
    s.touch = lambda: None
    s.metrics = lambda: {}
    s.send = lambda *a, **k: None
    return s


def req(ids, max_new=10):
    return {"ids": ids, "max_new": max_new, "temperature": 1.0, "top_k": 20, "top_p": 0.95}


def prompt(tail, shared=1000):
    # a system message (the shared prefix) then a per-request tail
    return [IM] + list(range(shared)) + [IM] + tail


def started_sibling(s, ids, fed, area=None, make_cur=False):
    o = worker.Seq("o", req(ids), 0.0)
    o.started, o.fed, o.area = True, fed, area
    s.alive.append(o)
    if make_cur:
        s.cur = o
    return o


def test_sibling_not_started_gets_the_checkpoint_in_its_plan_and_newcomer_waits():
    eng = Eng()
    s = sched(eng)
    o = worker.Seq("o", req(prompt([5] * 300)), 0.0)
    s.alive.append(o)
    s.submit("n", req(prompt([6] * 300)))
    n = s.by_id["n"]
    assert n.share_with is o and n.share_len == 1002 // 64 * 64 and n.reusable == n.share_len
    assert n.share_len in o.at and eng.calls == []
    assert not s.can_prefill(n)
    o.started, o.fed = True, n.share_len  # (o has run past it)
    assert s.can_prefill(n)


def test_sibling_started_before_share_gets_an_engine_checkpoint():
    eng = Eng()
    s = sched(eng)
    o = started_sibling(s, prompt([5] * 300), fed=128, make_cur=True)
    s.submit("n", req(prompt([6] * 300)))
    n = s.by_id["n"]
    assert eng.calls == [(-1, n.share_len)] and n.share_with is o and n.reusable == n.share_len
    assert not s.can_prefill(n)
    o.fed = n.share_len
    assert s.can_prefill(n)


def test_parked_sibling_is_addressed_by_its_area():
    eng = Eng()
    s = sched(eng)
    started_sibling(s, prompt([5] * 300), fed=128, area=2)
    s.submit("n", req(prompt([6] * 300)))
    assert eng.calls == [(2, s.by_id["n"].share_len)]


def test_engine_refusal_means_no_waiting_and_no_phantom_reusable():
    eng = Eng(accept=False)
    s = sched(eng)
    started_sibling(s, prompt([5] * 300), fed=128, make_cur=True)
    s.submit("n", req(prompt([6] * 300)))
    n = s.by_id["n"]
    assert n.share_with is None and n.reusable == 0 and s.can_prefill(n)


def test_sibling_past_the_share_is_not_waited_for():
    eng = Eng()
    s = sched(eng)
    started_sibling(s, prompt([5] * 300), fed=1100, make_cur=True)
    s.submit("n", req(prompt([6] * 300)))
    n = s.by_id["n"]
    assert n.share_with is None and eng.calls == [] and s.can_prefill(n)


def test_sibling_too_close_to_its_prompt_end_is_not_asked():
    eng = Eng()
    s = sched(eng)
    # o's prompt ends 62 tokens after the rounded share position: a final chunk that short would take the decode GEMMs
    started_sibling(s, prompt([5] * 20), fed=128, make_cur=True)
    s.submit("n", req(prompt([6] * 300)))
    assert eng.calls == [] and s.by_id["n"].share_with is None


def test_small_overlap_below_the_threshold_is_ignored():
    eng = Eng()
    s = sched(eng)
    s.alive.append(worker.Seq("o", req(prompt([5] * 300, shared=130)), 0.0))
    s.submit("n", req(prompt([6] * 300, shared=130)))  # shares 131 -> 128 tokens: not beyond SHARE_MIN
    assert s.by_id["n"].share_with is None
    s.alive.clear()
    s.alive.append(worker.Seq("o2", req(prompt([5] * 300, shared=200)), 0.0))
    s.submit("m", req(prompt([6] * 300, shared=200)))  # 201 -> 192 tokens
    assert s.by_id["m"].share_with is not None


def finish_cancelled(phase, started=True, parked=False):
    eng = Eng()
    s = sched(eng)
    o = worker.Seq("o", req(prompt([5] * 300)), 0.0)
    o.started, o.phase, o.out = started, phase, [1, 2, 3]
    s.alive.append(o)
    s.by_id["o"] = o
    if parked:
        o.area = 1
    else:
        s.cur = o
    s.cancel("o")
    return eng


def test_cancel_after_the_prompt_saves_the_end_checkpoint():
    assert finish_cancelled(worker.DECODE).checkpoints == 1
    eng = finish_cancelled(worker.DECODE, parked=True)
    assert eng.suspended == [1] and eng.dropped == []


def test_cancel_during_the_prefill_saves_nothing():
    assert finish_cancelled(worker.PREFILL).checkpoints == 0
    eng = finish_cancelled(worker.PREFILL, parked=True)
    assert eng.suspended == [] and eng.dropped == [1]


def test_error_still_saves_nothing():
    eng = Eng()
    s = sched(eng)
    o = worker.Seq("o", req(prompt([5] * 300)), 0.0)
    o.started, o.phase = True, worker.DECODE
    s.alive.append(o)
    s.cur = o
    s.finish(o, "error", save=False, notify=False)
    assert eng.checkpoints == 0


def test_unstarted_sibling_share_is_marked_in_the_engine():
    eng = Eng()
    marks = []
    eng.share_mark = marks.append
    s = sched(eng)
    s.alive.append(worker.Seq("o", req(prompt([5] * 300)), 0.0))
    s.submit("n", req(prompt([6] * 300)))
    assert marks == [s.by_id["n"].share_len]


def test_waiter_of_a_preempted_sibling_does_not_stall():
    eng = Eng()
    s = sched(eng)
    o = worker.Seq("o", req(prompt([5] * 300)), 0.0)
    s.alive.append(o)
    s.submit("n", req(prompt([6] * 300)))
    n = s.by_id["n"]
    s.alive.append(n)
    assert not s.can_prefill(n)
    s.alive.remove(o)  # (o was preempted back into the waiting queue, where the full slots keep it)
    s.waiting.appendleft(o)
    assert s.can_prefill(n) and n.share_with is None and n.share_len == 0


def test_next_candidate_is_tried_when_the_engine_refuses_the_longest():
    class E(Eng):
        def add_checkpoint_at(self, area, pos):
            self.calls.append((area, pos))
            return area == 3

    eng = E()
    s = sched(eng)
    started_sibling(s, prompt([5] * 300, shared=1500), fed=128, area=2)  # (shares 1472 tokens: the longer one)
    b = started_sibling(s, prompt([5] * 300), fed=128, area=3)  # (shares 960)
    s.submit("n", req(prompt([6] * 300, shared=1500)))
    n = s.by_id["n"]
    assert eng.calls == [(2, 1472), (3, 960)]
    assert n.share_with is b and n.share_len == 960 and n.reusable == 960


def test_old_library_without_optional_symbols_still_loads(monkeypatch):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "engine" / "python"))
    import q27lib

    class Old:
        @staticmethod
        def q27_last_error():  # (the only entry point this "older build" has)
            return b""

    monkeypatch.setattr(q27lib, "_lib", None)
    monkeypatch.setattr(q27lib.ctypes, "CDLL", lambda path: Old())
    assert q27lib.lib() is not None
    assert "q27_share_mark" in q27lib.SIGNATURES and not hasattr(q27lib.lib(), "q27_share_mark")
    monkeypatch.setattr(q27lib, "_lib", None)
