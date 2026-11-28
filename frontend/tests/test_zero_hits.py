"""The live zero-hit view (tools/zero_hit.py, fed by stats.HUB.finished; GET /admin/zero_hits; /metrics; the status page). The worker's
finish fields are checked on a real Scheduler with a fake engine (test_request_trace.sched). No GPU, no engine process."""

import array
import http.client
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import types

import pytest

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "tools"))

import request_trace as rt  # noqa: E402
import server  # noqa: E402
import stats  # noqa: E402
import zero_hit  # noqa: E402
from test_agent_id import IM, req  # noqa: E402
from test_request_trace import sched  # noqa: E402

LONG = [IM] + list(range(100, 4200))  # 4201 tokens: 16 full blocks, so h15 exists
LONG2 = LONG[:4096] + [IM] + list(range(9000, 9100))  # shares the first 4096 tokens with LONG
SHORT = [IM] + list(range(100, 300))


def zh_of(prompt, agent, begun, restore=None, tier=None):
    """The worker's "zh" info for a request with this prompt (what finish() builds from head_hashes)."""
    h = rt.head_hashes(array.array("i", prompt).tobytes())
    return {"agent": agent, "begun": begun, "h0": h[0] if h else None, "h15": h[15] if len(h) >= 16 else None,
            "restore": restore, "restore_tier": tier}


@pytest.fixture
def live(monkeypatch):
    """A fresh live view, on (ZERO_HIT_STATS and REQUEST_TRACE_HTTP unset); REQUEST_TRACE stays 0 as in conftest."""
    monkeypatch.delenv("ZERO_HIT_STATS", raising=False)
    monkeypatch.delenv("REQUEST_TRACE_HTTP", raising=False)
    view = zero_hit.Live(time.time())
    monkeypatch.setattr(stats, "ZERO", view)
    return view


@pytest.fixture
def base(live):
    srv = server.HTTPServer(("127.0.0.1", 0), server.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def engine_stub(monkeypatch):
    """server.Handler.srv for /metrics: only the engine's metrics dict."""
    srv = types.SimpleNamespace(engine=types.SimpleNamespace(metrics={"q27:requests_total": 3}))
    monkeypatch.setattr(server.Handler, "srv", srv)
    return srv


def get(base, path):
    c = http.client.HTTPConnection(base[len("http://"):], timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def observe(view, prompt, agent, begun, cached=0):
    return view.observe(zero_hit.from_head(zh_of(prompt, agent, begun), len(prompt), cached))


# ------------------------------------------------------------------ the endpoint

def test_endpoint_is_on_by_default_and_has_every_reason(base):
    code, data = get(base, "/admin/zero_hits")
    import json
    j = json.loads(data)
    assert code == 200
    assert set(j) >= {"window_s", "since_ts", "threshold_tokens", "totals", "recent", "agents_top"}
    assert set(j["totals"]) == set(zero_hit.REASONS) and j["threshold_tokens"] == 4096
    assert all(v == {"count": 0, "tokens": 0} for v in j["totals"].values())


def test_endpoint_works_with_the_request_trace_off(base, monkeypatch):
    monkeypatch.setenv("REQUEST_TRACE", "0")  # (conftest's setting; the view does not depend on the trace files)
    assert get(base, "/admin/zero_hits")[0] == 200


def test_endpoint_follows_the_two_switches(base, monkeypatch):
    for v in ("0", "off", "FALSE", "no", "none", ""):
        monkeypatch.setenv("ZERO_HIT_STATS", v)
        code, data = get(base, "/admin/zero_hits")
        assert code == 404 and b"ZERO_HIT_STATS=0" in data, v
    monkeypatch.setenv("ZERO_HIT_STATS", "1")
    assert get(base, "/admin/zero_hits")[0] == 200
    monkeypatch.delenv("ZERO_HIT_STATS")
    monkeypatch.setenv("REQUEST_TRACE_HTTP", "0")
    code, data = get(base, "/admin/zero_hits")
    assert code == 404 and b"REQUEST_TRACE_HTTP=0" in data


def test_since_keeps_only_the_recent_entries_after_it(base, live):
    for i, ts in enumerate((1000.0, 2000.0, 3000.0)):
        observe(live, LONG2 if i else LONG, f"a{i}", ts)
    import json
    j = json.loads(get(base, "/admin/zero_hits")[1])
    assert [e["ts"] for e in j["recent"]] == [1000.0, 2000.0, 3000.0]
    j = json.loads(get(base, "/admin/zero_hits?since=1500")[1])
    assert [e["ts"] for e in j["recent"]] == [2000.0, 3000.0]
    assert get(base, "/admin/zero_hits?since=nope")[0] == 400


def test_recent_entries_have_the_documented_fields(base, live):
    observe(live, LONG, "a", 1000.0)
    import json
    e = json.loads(get(base, "/admin/zero_hits")[1])["recent"][0]
    assert set(e) == {"ts", "agent", "prompt_tokens", "reason", "gap_s", "earlier_age_s", "restore", "restore_tier"}
    assert e["prompt_tokens"] == len(LONG) and e["agent"] == "a"


def test_a_broken_record_never_raises_into_the_reader(monkeypatch, live):
    monkeypatch.setattr(stats, "ZERO", live)
    stats.zero_hit_observe({}, 0, 0)  # (a broken record: the reader thread must keep going)
    assert live.snapshot(now=1.0)["totals"]["first_seen"]["count"] == 0


# ------------------------------------------------------------------ the feed: HUB.finished

def test_a_finished_request_with_zh_is_counted_and_the_field_stays_out_of_the_log(live):
    info = {"prompt_tokens": len(LONG), "completion_tokens": 3, "cached_tokens": 0, "total": 1.0,
            "zh": zh_of(LONG, "a/s1", 1000.0)}
    stats.HUB.finished("r1", "stop", info)
    assert "zh" not in info
    snap = live.snapshot(now=1000.0)
    assert snap["totals"]["window_start"]["count"] == 1  # the view started at 'now' here, so the first request is the window's
    assert snap["recent"][0]["agent"] == "a/s1" and snap["recent"][0]["prompt_tokens"] == len(LONG)


def test_a_finished_request_without_zh_or_with_a_hit_is_not_zero_hit(live):
    stats.HUB.finished("r1", "stop", {"prompt_tokens": len(LONG), "cached_tokens": 0, "total": 1.0})
    stats.HUB.finished("r2", "stop", {"prompt_tokens": len(LONG), "cached_tokens": 4096, "total": 1.0,
                                      "zh": zh_of(LONG, "a", 1000.0)})
    assert sum(v["count"] for v in live.snapshot(now=1.0)["totals"].values()) == 0


def test_the_view_off_means_no_counting(live, monkeypatch):
    monkeypatch.setenv("ZERO_HIT_STATS", "0")
    stats.HUB.finished("r1", "stop", {"prompt_tokens": len(LONG), "cached_tokens": 0, "total": 1.0,
                                      "zh": zh_of(LONG, "a", 1000.0)})
    assert sum(v["count"] for v in live.snapshot(now=1.0)["totals"].values()) == 0


def test_the_view_shows_a_returned_agent_and_an_evicted_one(live):
    observe(live, LONG, "a", 1000.0)
    assert observe(live, LONG2, "a", 1100.0)["reason"] == "evicted_or_missed"
    r = observe(live, LONG, "a", 1100.0 + 601)
    assert r["reason"] == "agent_returned_after_gap" and r["gap_s"] == 601.0


# ------------------------------------------------------------------ /metrics

def test_metrics_lists_every_reason_with_zeros(base, engine_stub, live):
    _, data = get(base, "/metrics")
    text = data.decode()
    for r in zero_hit.REASONS:
        assert f'q27:zero_hit_requests_total{{reason="{r}"}} 0' in text
        assert f'q27:zero_hit_tokens_total{{reason="{r}"}} 0' in text
    assert "q27:requests_total 3" in text  # the engine's own series stay


def test_metrics_counts_follow_the_view(base, engine_stub, live):
    observe(live, LONG, "a", 1000.0)
    text = get(base, "/metrics")[1].decode()
    assert f'q27:zero_hit_requests_total{{reason="window_start"}} 1' in text
    assert f'q27:zero_hit_tokens_total{{reason="window_start"}} {len(LONG)}' in text


def test_metrics_have_no_zero_hit_series_when_the_view_is_off(base, engine_stub, live, monkeypatch):
    monkeypatch.setenv("ZERO_HIT_STATS", "0")
    assert "zero_hit" not in get(base, "/metrics")[1].decode()


# ------------------------------------------------------------------ the engine side: the zh info of finish()

class Sent(list):
    """The messages a worker sends (send(*msg) records each as a tuple)."""


def test_finish_sends_zh_for_a_request_that_began(tmp_path):
    s = sched(tmp_path, env=False)
    sent = Sent()
    s.send = lambda *a, **k: sent.append(a)
    s.submit("r1", req(LONG, agent_id="manager/task-9"))
    r = s.by_id["r1"]
    s.begin(r)
    s.finish(r, "stop", save=False)
    done = [a for a in sent if a[0] == "done"]
    assert done, sent
    zh = done[0][3]["zh"]
    assert zh["agent"] == "manager/task-9" and zh["begun"] == r.begun
    assert zh["h0"] == rt.block_chain(LONG)[0][0] and zh["h15"] == rt.block_chain(LONG)[0][15]
    assert zh["restore"] == r.restore_outcome and zh["restore_tier"] == r.restore_tier


def test_finish_sends_no_zh_for_a_request_that_never_began(tmp_path):
    s = sched(tmp_path, env=False)
    sent = Sent()
    s.send = lambda *a, **k: sent.append(a)
    s.submit("r1", req(LONG, agent_id="manager/task-9"))
    s.finish(s.by_id["r1"], "cancelled", save=False)  # cancelled while queued: no req event in the trace either
    done = [a for a in sent if a[0] == "done"]
    assert done and "zh" not in done[0][3]


# ------------------------------------------------------------------ the status page

def test_the_status_page_has_the_card_and_its_script_parses(tmp_path):
    html = (HERE.parents[1] / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="zh-sec"' in html and 'id="zh-body"' in html and "loadZeroHits" in html
    assert "/admin/zero_hits" in html and "setInterval(loadZeroHits" in html
    for name in zero_hit.REASONS:
        assert f'["{name}"' in html, name  # (the card names every reason, in its ZH list)
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    import re
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert scripts
    f = tmp_path / "zh_check.js"
    f.write_text(max(scripts, key=len), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
