"""Agent identity: X-Agent-Id / "user" (agent.py, stats.request_meta), the worker's auto identity, the per-agent
stats block, the checkpoint notes and the Log column."""

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import agent  # noqa: E402
import stats  # noqa: E402
import worker  # noqa: E402

IM = 7


def meta(headers=None, body=None):
    return stats.request_meta("/v1/chat/completions", headers or {}, body or {})


def test_header_wins_over_body_user():
    assert meta({"X-Agent-Id": "manager/t1"}, {"user": "other"})["agent_id"] == "manager/t1"
    assert meta({}, {"user": "other"})["agent_id"] == "other"
    assert meta({"x-agent-id": "low"}, {})["agent_id"] == "low"  # (a plain dict: any case)
    assert meta({"X-Agent-Id": "  "}, {"user": "body"})["agent_id"] == "body"  # empty header = absent
    assert meta()["agent_id"] is None
    assert meta({}, {"user": 5})["agent_id"] is None and meta({}, {"user": ""})["agent_id"] is None


def test_sanitize():
    assert agent.sanitize("a b$c") == "a_b_c"
    assert agent.sanitize("ok._/-9") == "ok._/-9"
    assert agent.sanitize("é") == "_"
    assert agent.sanitize("x" * 300) == "x" * 128
    assert agent.sanitize("   ") is None and agent.sanitize(None) is None


def test_agent_is_the_part_before_the_first_slash():
    assert agent.agent_of("manager/task-123") == agent.agent_of("manager/probe") == "manager"
    assert agent.agent_of("solo") == "solo"
    assert agent.agent_of("/x") == "_" and agent.agent_of(None) is None
    assert agent.agent_of("auto:3f2a9c1b") == "auto:3f2a9c1b"


def test_auto_id_stable_for_the_same_opening_and_different_for_another_system_prompt():
    a = [IM, 1, 2, 3, IM, 4, 5, IM, 6]            # system, user, generation prompt
    a2 = [IM, 1, 2, 3, IM, 4, 5, IM, 8, 9, IM, 6, IM, 6]  # a later turn of the same conversation
    b = [IM, 9, 9, 9, IM, 4, 5, IM, 6]
    ia, ia2, ib = (agent.auto_id(p, IM) for p in (a, a2, b))
    assert ia == ia2 and ia != ib
    assert ia.startswith("auto:") and len(ia) == len("auto:") + 8


def req(ids, **kw):
    return dict({"ids": ids, "max_new": 4, "temperature": 1.0, "top_k": 20, "top_p": 0.95}, **kw)


class Eng:
    def memory_report(self):
        return {}


def sched():
    s = object.__new__(worker.Scheduler)
    s.engine, s.tiers = Eng(), None
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.history, s.by_id, s.cur, s.im_start = collections.deque(maxlen=64), {}, None, IM
    s.last_active, s.restores, s.restore_bw = 0.0, {}, {}
    s.touch = lambda: None
    s.send = lambda *a, **k: None
    return s


def test_submit_sets_agent_fields_and_auto_identity():
    s = sched()
    p = [IM] + list(range(100, 300)) + [IM] + list(range(5)) + [IM, 1]
    s.submit("r1", req(p, agent_id="manager/t1"))
    s.submit("r2", req(p))
    s.submit("r3", req([IM] + list(range(500, 700)) + [IM] + list(range(5)) + [IM, 1]))
    r1, r2, r3 = (s.by_id[k] for k in ("r1", "r2", "r3"))
    assert (r1.agent, r1.agent_id) == ("manager", "manager/t1")
    assert r2.agent == r2.agent_id == agent.auto_id(p, IM) and r2.agent.startswith("auto:")
    assert r3.agent != r2.agent


def test_agent_of_ids_attributes_planned_checkpoints_and_is_bounded(monkeypatch):
    s = sched()
    p = [IM] + list(range(100, 300)) + [IM] + list(range(5)) + [IM, 1]
    s.submit("r1", req(p, agent_id="alpha/x"))
    r1 = s.by_id["r1"]
    assert r1.at
    for pos in r1.at:
        assert s.agent_of_ids(p, pos) == "alpha"
    assert s.agent_of_ids(p[:50] + [0], 50) is None
    monkeypatch.setattr(worker, "AGENT_NOTES", 3)
    s.note_checkpoints(r1, p, [10, 11, 12, 13, 14])
    assert len(s.agent_notes) == 3 and s.agent_of_ids(p, 14) == "alpha" and s.agent_of_ids(p, 10) is None


def test_hub_tick_agents_and_log_record_carry_the_agent():
    h = stats.Hub()
    h.local.meta = meta({"X-Agent-Id": "mgr/a"}, {"messages": [{"role": "user", "content": "hi"}]})
    h.submitted("req-1")
    assert h.log[-1]["meta"]["agent_id"] == "mgr/a"
    h.finished("req-1", "stop", {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 5, "ttft": 0.1, "total": 1.0,
                                 "reusable_tokens": 5, "computed_tokens": 5, "prefill_s": 0.1, "decode_s": 0.5, "steps": 2,
                                 "queued": 0.0, "preempted": 0, "agent": "mgr", "agent_id": "mgr/a"})
    rec = h.log[-1]["rec"]
    assert rec["agent"] == "mgr" and rec["agent_id"] == "mgr/a"
    assert h.tick()["agents"] == []


def test_ui_has_agents_table_and_log_column():
    html = (pathlib.Path(__file__).resolve().parents[1] / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="agents-sec"' in html and "function drawAgents" in html and 'el("th", {}, "Agent")' in html
    assert "r.agent" in html
