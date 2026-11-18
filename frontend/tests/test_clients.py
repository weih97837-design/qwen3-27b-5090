"""Client counting (clients.py, Scheduler.client_count, stats.clients_block, the q27:client_* metrics): counting only."""

import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import clients  # noqa: E402
import stats  # noqa: E402
import worker  # noqa: E402
from test_agent_id import IM, Eng, req  # noqa: E402

NAMES = "192.168.1.11=PC-A, 192.168.1.12=PC-B,10.0.0.0/8=Lab,10.1.0.0/16=Lab1,bad,x=y,=z"


def test_header_wins_and_is_sanitized():
    assert clients.resolve({"X-Client-Id": "PC-A"}, "192.168.1.12", NAMES) == "PC-A"
    assert clients.resolve({"x-client-id": "a b$c"}, "1.2.3.4") == "a_b_c"
    assert clients.resolve({"X-Client-Id": "x" * 100}, "1.2.3.4") == "x" * 64
    assert clients.resolve({"X-Client-Id": "  "}, "192.168.1.11", NAMES) == "PC-A"  # empty header = absent
    assert clients.resolve({"X-Client-Id": "me"}, "127.0.0.1") == "me"


def test_ip_mapping_cidr_and_fallback():
    r = lambda ip, h=None: clients.resolve(h or {}, ip, NAMES)  # noqa: E731
    assert r("192.168.1.11") == "PC-A" and r("192.168.1.12") == "PC-B"
    assert r("192.168.1.13") == "192.168.1.13"
    assert r("10.9.9.9") == "Lab" and r("10.1.2.3") == "Lab1"  # (the longest prefix)
    assert r("::ffff:192.168.1.11") == "PC-A"
    assert clients.resolve({}, "8.8.8.8", "") == "8.8.8.8"
    assert clients.resolve({}, None) == "unknown" and clients.resolve({}, "garbage$") == "garbage_"


def test_loopback_is_local():
    assert clients.resolve({}, "127.0.0.1", NAMES) == "local" and clients.resolve({}, "::1") == "local"


def test_forwarded_for_only_from_a_loopback_or_private_peer():
    xff = {"X-Forwarded-For": "192.168.1.12, 10.0.0.5"}
    assert clients.resolve(xff, "127.0.0.1", NAMES) == "PC-B"       # a proxy on this machine
    assert clients.resolve(xff, "192.168.1.1", NAMES) == "PC-B"     # a proxy on the LAN
    assert clients.resolve(xff, "8.8.8.8", NAMES) == "8.8.8.8"      # public peer: the header is ignored
    assert clients.resolve({"X-Forwarded-For": "junk"}, "127.0.0.1") == "local"
    assert clients.resolve({"X-Forwarded-For": "127.0.0.1"}, "192.168.1.5") == "local"
    assert clients.resolve({**xff, "X-Client-Id": "own"}, "127.0.0.1", NAMES) == "own"


def test_request_meta_carries_client_name_and_keeps_the_user_agent():
    m = stats.request_meta("/v1/chat/completions", {"User-Agent": "ua", "X-Client-Id": "PC-A"}, {}, "1.2.3.4")
    assert m["client_name"] == "PC-A" and m["client"] == "ua"
    m = stats.request_meta("/v1/chat/completions", {}, {}, "192.168.1.11", NAMES)
    assert m["client_name"] == "PC-A"


def sched():
    s = object.__new__(worker.Scheduler)
    s.engine, s.tiers = Eng(), None
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.history, s.by_id, s.cur, s.im_start = collections.deque(maxlen=64), {}, None, IM
    s.last_active, s.restores, s.restore_bw = 0.0, {}, {}
    s.touch = lambda: None
    s.send = lambda *a, **k: None
    s.metrics = lambda: {}
    return s


P = [IM] + list(range(100, 300)) + [IM] + list(range(5)) + [IM, 1]


def test_counters_across_submit_begin_finish():
    s = sched()
    s.submit("r1", req(P, client="PC-A"))
    s.submit("r2", req(P, client="PC-A"))
    s.submit("r3", req(P))
    assert {x.client for x in s.waiting} == {"PC-A", "unknown"}
    t = s.client_totals["PC-A"]
    assert t["requests"] == 2 and t["prompt_tokens"] == 0
    r1 = s.by_id["r1"]
    r1.arrival -= 2.0
    s.waiting.remove(r1)
    s.alive.append(r1)
    r1.begun = time.time()
    s.client_count(r1, "begin", prompt=len(P), cached=64, wait=2.0, media=0)
    snap_rows = [{"rid": "r1", "state": "running", "client": "PC-A", "prompt": len(P)},
                 {"rid": "r2", "state": "waiting", "client": "PC-A", "prompt": 10},
                 {"rid": "r3", "state": "restoring", "client": "unknown", "prompt": 5}]
    rows = {r["client"]: r for r in stats.clients_block(snap_rows, s.client_totals)}
    a = rows["PC-A"]
    assert (a["running"], a["waiting"], a["requests"], a["waiting_tokens"]) == (1, 1, 2, 10)
    assert a["prompt_tokens"] == len(P) and a["cached_tokens"] == 64 and a["hit"] == 64 / len(P)
    assert a["avg_wait_s"] == 2.0 and a["last_seen"] < 5 and a["errors"] == 0
    assert rows["unknown"]["waiting"] == 1 and rows["unknown"]["avg_wait_s"] is None and rows["unknown"]["hit"] is None
    assert [r["client"] for r in stats.clients_block(snap_rows, s.client_totals)][0] == "PC-A"
    r1.out = [1, 2, 3]
    s.finish(r1, "stop", save=False)
    r2 = s.by_id["r2"]
    s.finish(r2, "cancelled", save=False)
    r3 = s.by_id["r3"]
    s.finish(r3, "error", save=False)
    a = s.client_totals["PC-A"]
    assert a["output_tokens"] == 3 and a["errors"] == 0 and a["cancelled"] == 1
    assert s.client_totals["unknown"]["errors"] == 1


def test_finished_log_info_has_the_client():
    s = sched()
    sent = []
    s.send = lambda *a, **k: sent.append(a)
    s.metrics = lambda: {}
    s.submit("r1", req(P, client="PC-B"))
    s.finish(s.by_id["r1"], "stop", save=False)
    assert sent[-1][1] == "r1" and sent[-1][3]["client"] == "PC-B"


def test_cap_evicts_the_least_recently_seen(monkeypatch):
    monkeypatch.setattr(worker, "CLIENT_STATS", 3)
    s = sched()
    for i, name in enumerate(["a", "b", "c"]):
        s.submit(f"r{i}", req(P, client=name))
    s.submit("r9", req(P, client="a"))  # (a is the most recent now)
    s.submit("r10", req(P, client="d"))
    assert list(s.client_totals) == ["c", "a", "d"]
    assert len(s.client_metrics()) == 3 * 11


def test_metrics_escape_labels_and_snapshot_is_json():
    s = sched()
    s.submit("r1", req(P, client='we"ird\\na\nme'))
    m = s.client_metrics()
    key = 'q27:client_requests_total{client="we\\"ird\\\\na\\nme"}'
    assert m[key] == 1 and m['q27:client_waiting{client="we\\"ird\\\\na\\nme"}'] == 1
    assert "\n" not in "".join(m)
    assert clients.label('a"b') == 'a\\"b'
    json.dumps(stats.clients_block([], s.client_totals))
    h = stats.Hub()
    assert h.tick()["clients"] == []
