"""Request trace recorder (request_trace.py, docs/request-trace.md) and the AGENT_STATS knob."""

import array
import collections
import importlib
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import request_trace as rt  # noqa: E402
import stats  # noqa: E402
import worker  # noqa: E402
from test_agent_id import IM, req  # noqa: E402

SECRET_TEXT = "ZEBRA-SECRET-PROMPT-TEXT"
P = [IM] + list(range(100, 100 + 700)) + [IM] + list(range(5)) + [IM, 1]  # 706 tokens: 2 full blocks


def ids(n, seed=0):
    return [(i * 7919 + seed) % 150000 for i in range(n)]


# ---------------------------------------------------------------- hashing

def test_chain_is_deterministic_and_has_the_prefix_property():
    a = ids(256 * 5 + 100)
    h1, d1 = rt.block_chain(a)
    h2, d2 = rt.block_chain(list(a))
    assert h1 == h2 and d1 == d2 and len(h1) == 5 and all(len(h) == 16 for h in h1)
    longer = a[:256 * 3] + ids(900, seed=5)
    hl, _ = rt.block_chain(longer)
    assert hl[:3] == h1[:3] and hl[3] != h1[3]
    assert rt.block_chain(a[:255])[0] == []                      # (full blocks only)
    assert rt.block_chain(a[:256 * 2 + 255])[0] == h1[:2]
    # position matters: the same block after a different prefix hashes differently
    assert rt.block_chain(ids(256, 1) + a[:256])[0][1] != h1[0]
    # the bytes are the int32 ids: the same chain from bytes
    assert rt.block_chain(None, data=array.array("i", a).tobytes())[0] == h1
    # chained by the digest: continuing from the last digest gives the same tail
    _, d = rt.block_chain(a[:256 * 2])
    assert rt.block_chain(a[256 * 2:256 * 5], prev=d)[0] == h1[2:5]


def test_hash_time_125k():
    a = ids(125_000)
    t = time.perf_counter()
    h, _ = rt.block_chain(a)
    dt = time.perf_counter() - t
    print(f"125K prompt: {len(h)} blocks in {dt * 1000:.2f} ms")
    assert len(h) == 488 and dt < 0.25


def test_hash_agent_is_stable_and_keeps_the_slash_structure():
    assert rt.hash_agent("manager/t1") == rt.hash_agent("manager/t1")
    a, b = rt.hash_agent("manager/t1").split("/"), rt.hash_agent("manager/t2").split("/")
    assert a[0] == b[0] and a[1] != b[1] and "manager" not in rt.hash_agent("manager/t1")


# ---------------------------------------------------------------- recorder

def read(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines()]


class FakeEngine:
    def __init__(self, cached=256):
        self.cached = cached
        self.begun = []

    def cache_stats(self):
        return {"free_pages": 10, "checkpoint_only_pages": 3, "pool_pages": 100, "checkpoints": 4, "evicted": 7}

    def prefix_begin(self, ids_, at=(), spec=False):
        self.begun.append(list(ids_))
        return self.cached

    def spill_stats(self):
        return {"spilled": 1, "restored": 2, "dropped": 3, "written": 4}


def sched(tmp_path, env=True, **kw):
    s = object.__new__(worker.Scheduler)
    s.engine, s.tiers = FakeEngine(), None
    s.alive, s.waiting, s.m = [], collections.deque(), collections.Counter()
    s.history, s.by_id, s.cur, s.im_start = collections.deque(maxlen=64), {}, None, IM
    s.last_active, s.restores, s.restore_bw = 0.0, {}, {}
    s.last_metrics = s.last_stats = 0.0
    s.touch = lambda: None
    s.send = lambda *a, **k: None
    s.metrics = lambda: {}
    s.spec, s.kv_reserve = False, 4096
    s.settings = s.stats = lambda: {}
    s.clock = worker.EngineClock()
    if env:
        s.trace = rt.from_env({"REQUEST_TRACE": str(tmp_path), **kw}, {"PREFILL_CHUNK": 2048}, lambda m: None)
    return s


def files(tmp_path):
    return sorted(pathlib.Path(tmp_path).glob("trace-*.jsonl"))


def test_off_by_default(tmp_path):
    assert rt.from_env({}, {}) is None and rt.from_env({"REQUEST_TRACE": ""}, {}) is None
    s = sched(tmp_path, env=False)
    assert s.trace is None
    s.submit("r1", req(P, agent_id="m/t"))
    r = s.by_id["r1"]
    s.begin(r)
    s.finish(r, "stop", save=False)
    s.report()
    assert files(tmp_path) == []


def test_events_fields_and_no_content(tmp_path):
    s = sched(tmp_path)
    prompt = P + [IM + 1000 * 0 + 150000 + i for i in range(3)]  # (media placeholders are ids >= the vocabulary)
    s.submit("r1", req(prompt, agent_id="manager/task-9", client="PC-A", secret_text=SECRET_TEXT))
    r = s.by_id["r1"]
    s.waiting.remove(r)
    s.alive.append(r)
    s.begin(r)
    r.out = [31337, 31338]
    r.first_token = r.arrival + 0.5
    s.report(force=True)   # (the snapshot cadence of the stats page also drives the trace's snap)
    s.report(force=True)   # (not due again: no second snap)
    s.finish(r, "stop", save=False)
    ev = read(files(tmp_path)[0])
    assert [e["t"] for e in ev] == ["hdr", "req", "snap", "done"]
    hdr, rq, sn, dn = ev
    assert hdr["version"] == 1 and hdr["block_tokens"] == 256 and hdr["engine"] and hdr["settings"]["PREFILL_CHUNK"] == 2048
    assert {"ts", "rid", "agent", "client", "prompt_tokens", "media_tokens", "max_tokens", "cached", "reusable", "diverged_at",
            "restored_from", "queued_s", "blocks"} <= rq.keys()
    assert rq["agent"] == "manager/task-9" and rq["client"] == "PC-A" and rq["prompt_tokens"] == len(prompt)
    assert rq["cached"] == 256 and rq["restored_from"] is None and rq["diverged_at"] is None and rq["max_tokens"] == 4
    assert rq["blocks"] == rt.block_chain(prompt)[0] and len(rq["blocks"]) == 2   # (the ids the engine's prefix_begin got)
    assert s.engine.begun[0] == prompt
    assert {"running", "waiting", "kv_free_pages", "kv_checkpoint_only_pages", "kv_pool_pages", "checkpoints",
            "q27:prefix_cache_evictions_total", "q27:tier_spilled"} <= sn.keys() or "q27:tier_spilled" not in sn
    assert sn["running"] == 1 and sn["kv_pool_pages"] == 100 and sn["q27:prefix_cache_evictions_total"] == 7
    assert dn["reason"] == "stop" and dn["output_tokens"] == 2 and dn["total_tokens"] == len(prompt) + 2
    assert dn["ttft_s"] == 0.5 and dn["preempted"] == 0 and dn["computed_tokens"] == 0 and dn["end_blocks"] == []
    raw = files(tmp_path)[0].read_text()
    assert SECRET_TEXT not in raw and "31337" not in raw and "manager/task-9" in raw
    for tok in (str(prompt[10]), str(prompt[300]), '"ids"', '"prompt"', '"text"'):
        assert f",{tok}," not in raw.replace(" ", "") and f"[{tok}," not in raw


def test_end_blocks_cover_what_lies_beyond_the_prompts_last_full_block(tmp_path):
    s = sched(tmp_path)
    prompt = ids(256 * 2 + 40)
    s.submit("r1", req(prompt))
    r = s.by_id["r1"]
    s.waiting.remove(r)
    s.alive.append(r)
    s.begin(r)
    r.out = ids(600, seed=3)
    s.finish(r, "length", save=False)
    dn = read(files(tmp_path)[0])[-1]
    allh = rt.block_chain(prompt + r.out)[0]
    assert len(allh) == (552 + 600) // 256 == 4 and dn["end_blocks"] == allh[2:] and dn["total_tokens"] == 1152
    # a request that never began hashes nothing
    s.submit("r2", req(prompt))
    s.finish(s.by_id["r2"], "cancelled", save=False)
    assert read(files(tmp_path)[0])[-1]["end_blocks"] == []


def test_hashed_agents(tmp_path):
    s = sched(tmp_path, REQUEST_TRACE_HASH_AGENTS="1")
    s.submit("r1", req(P, agent_id="manager/task-9"))
    r = s.by_id["r1"]
    s.waiting.remove(r)
    s.alive.append(r)
    s.begin(r)
    raw = files(tmp_path)[0].read_text()
    assert "manager" not in raw and rt.hash_agent("manager/task-9") in raw


def test_rotation_and_size_cap(tmp_path):
    rec = rt.Recorder(str(tmp_path), {"engine": "x", "settings": {}}, max_mib=0.002, file_mib=0.0005, log=lambda m: None)
    first = rec.path
    for i in range(400):
        rec.emit({"t": "snap", "ts": i, "pad": "x" * 100})
    fs = files(tmp_path)
    assert rec.path != first and len(fs) >= 2
    assert sum(f.stat().st_size for f in fs) <= 0.002 * 1048576 + 600 + 120   # (the cap, plus the file being written)
    assert not pathlib.Path(first).exists()                                    # (the oldest went first)
    assert all(read(f)[0]["t"] == "hdr" for f in fs)
    assert all(f.name.startswith("trace-") and f.name.endswith(".jsonl") for f in fs)


def test_prefix_name_and_directory(tmp_path):
    rec = rt.Recorder(str(tmp_path / "sub" / "agentload"), {}, log=lambda m: None)
    assert pathlib.Path(rec.path).parent.name == "sub" and pathlib.Path(rec.path).name.startswith("agentload-")
    rec2 = rt.Recorder(str(tmp_path / "d") + "/", {}, log=lambda m: None)
    assert pathlib.Path(rec2.path).parent.name == "d" and pathlib.Path(rec2.path).name.startswith("trace-")


def test_self_disables_on_a_write_error(tmp_path):
    logs = []
    rec = rt.Recorder(str(tmp_path), {}, log=logs.append)
    rec.emit({"t": "snap", "ts": 1})

    class Broken:
        def write(self, *_):
            raise OSError("disk full")

        def flush(self):
            pass

        def close(self):
            pass

    rec.f = Broken()
    rec.emit({"t": "snap", "ts": 2})          # (must not raise)
    rec.emit({"t": "snap", "ts": 3})
    assert rec.dead and len(logs) == 1 and "disk full" in logs[0]
    s = sched(tmp_path)
    s.trace = rec
    s.submit("r1", req(P))
    r = s.by_id["r1"]
    s.begin(r)                                  # (the scheduler goes on)
    s.finish(r, "stop", save=False)
    assert len(read(files(tmp_path)[0])) == 2   # (hdr + the first snap)


def test_unwritable_target_starts_without_trace(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    logs = []
    assert rt.from_env({"REQUEST_TRACE": str(f / "sub" / "t")}, {}, logs.append) is None and logs


# ---------------------------------------------------------------- AGENT_STATS and the unknown split

def test_agent_stats_env(monkeypatch):
    try:
        monkeypatch.delenv("AGENT_STATS", raising=False)
        assert importlib.reload(worker).AGENT_STATS == 1024
        monkeypatch.setenv("AGENT_STATS", "300")
        assert importlib.reload(worker).AGENT_STATS == 300
        monkeypatch.setenv("AGENT_STATS", "")
        assert importlib.reload(worker).AGENT_STATS == 1024
    finally:
        monkeypatch.delenv("AGENT_STATS", raising=False)
        importlib.reload(worker)


def test_unknown_row_is_split_into_owner_zero_and_overflow_owners():
    import agent
    totals = {"alpha": {"requests": 3, "hit_requests": 1, "prompt_tokens": 100, "cached_tokens": 50}}
    use = [{"owner": agent.owner_id("alpha"), "ram_bytes": 4, "ssd_bytes": 40, "ram_entries": 1, "ssd_entries": 2},
           {"owner": 0, "ram_bytes": 5, "ssd_bytes": 70, "ram_entries": 1, "ssd_entries": 1}]
    use += [{"owner": 1000 + i, "ram_bytes": 1, "ssd_bytes": 10, "ram_entries": 1, "ssd_entries": 1} for i in range(5)]
    rows = {r["agent"]: r for r in stats.agents_block([], totals, use)}
    z, o = rows[stats.UNKNOWN_AGENT], rows[stats.OVERFLOW_AGENT]
    assert z["kind"] == "owner0" and z["ram_bytes"] == 5 and z["ssd_bytes"] == 70 and "count" not in z
    assert o["kind"] == "overflow" and o["count"] == 5 and o["ram_bytes"] == 5 and o["ssd_bytes"] == 50 and o["ssd_entries"] == 5
    assert z["ssd_bytes"] + o["ssd_bytes"] == 120 and z["ram_bytes"] + o["ram_bytes"] == 10  # (what the old lump held)
    assert len(rows) == 3
    # no overflow owners, no overflow row; no owner 0, no unknown row
    rows = {r["agent"]: r for r in stats.agents_block([], totals, use[:1])}
    assert set(rows) == {"alpha"}


def test_header_names_the_build(tmp_path, monkeypatch):
    """The header carries the package's app\\VERSION (None in a checkout) so two versions' traces can be told apart."""
    import request_trace
    (tmp_path / "frontend").mkdir()
    (tmp_path / "VERSION").write_text("v9.9.9-test", encoding="ascii")
    fake = tmp_path / "frontend" / "request_trace.py"
    fake.write_text("")
    monkeypatch.setattr(request_trace, "__file__", str(fake))
    assert request_trace.package_version() == "v9.9.9-test"
    (tmp_path / "VERSION").unlink()
    assert request_trace.package_version() is None
