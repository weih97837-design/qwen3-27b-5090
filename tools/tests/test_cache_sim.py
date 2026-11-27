"""tools/cache_sim.py and tools/cache_sim_gen.py on small synthetic workloads (CPU only).

The scaled-down capacities (6 GiB RAM, 16 GiB SSD, 3000 VRAM pages) put every tier under pressure with ~700 requests; at that scale
the default CACHE_MIN_RESIDENT_S (300 s) would make nearly every RAM entry "young" (the oldest-goes fallback, i.e. FIFO), so the tests
pass explicit SmartParams(min_resident_s=0): they test the value order, not the residency guard (tests/test_cache_policy.py does)."""

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))

import cache_policy as cp  # noqa: E402
import cache_sim as sim  # noqa: E402
import cache_sim_gen as gen  # noqa: E402
import trace_summary  # noqa: E402

SMALL = {"ram_gib": 6, "ssd_gib": 16, "vram_pages": 3000}
HUGE = {"ram_gib": 1e4, "ssd_gib": 1e5, "vram_pages": 10 ** 7, "checkpoint_slots": 10 ** 6}
PARAMS = cp.SmartParams(min_resident_s=0)


def trace(tmp_path, workload, seed=1, n=700, extra=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"{workload}{seed}.jsonl"
    gen.main([str(p), "--workload", workload, "--requests", str(n), "--agents", "24", "--gap-s", "120", "--seed", str(seed), *extra])
    return p


def test_generator_writes_the_trace_format(tmp_path):
    p = trace(tmp_path, "mixed", n=200)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows[0]["t"] == "hdr" and rows[0]["block_tokens"] == 256
    reqs = [r for r in rows if r["t"] == "req"]
    dones = {r["rid"]: r for r in rows if r["t"] == "done"}
    assert len(reqs) == 200 and set(dones) == {r["rid"] for r in reqs}
    for r in reqs:
        assert len(r["blocks"]) == r["prompt_tokens"] // 256 and 0 <= r["reusable"] <= len(r["blocks"]) * 256
        d = dones[r["rid"]]
        assert d["ts"] >= r["ts"] and len(r["blocks"]) + len(d["end_blocks"]) == d["total_tokens"] // 256
    s = trace_summary.summarize(*trace_summary.load([str(p)]))
    assert s["requests"] == 200 and s["diverged_share"] > 0
    assert trace(tmp_path / "again", "mixed", n=200).read_text().splitlines()[1:] == p.read_text().splitlines()[1:]  # (seeded)


def test_continuations_extend_their_conversation(tmp_path):
    reqs, dones, _ = trace_summary.load([str(trace(tmp_path, "continuation", n=150))])
    by = {d["rid"]: d for d in dones}
    last = {}
    for r in reqs:
        prev = last.get(r["agent"])
        if prev is not None and len(r["blocks"]) > len(prev):
            assert r["blocks"][:len(prev)] == prev  # (a pure continuation: the previous prompt + output is a prefix)
        last[r["agent"]] = r["blocks"] + by[r["rid"]]["end_blocks"]
    assert all(r.get("diverged_at") is None for r in reqs)


@pytest.mark.parametrize("seed", [1, 2])
def test_smart_beats_legacy_on_a_diverging_heavy_agent_workload(tmp_path, seed):
    ev = sim.load([str(trace(tmp_path, "diverging", seed=seed, extra=("--churn", "0.08")))])
    legacy = sim.simulate(ev, "legacy", params=PARAMS, **SMALL)  # (today's defaults: SPILL_FAIR_AGENTS=1, SPILL_COLD_FIRST=1)
    smart = sim.simulate(ev, "smart", params=PARAMS, **SMALL)
    assert smart["missed_share"] < 0.8 * legacy["missed_share"], (smart["missed_share"], legacy["missed_share"])
    assert smart["wasted_prefill_s"] < legacy["wasted_prefill_s"]
    assert smart["recomputed_tokens"] < legacy["recomputed_tokens"]
    assert smart["subsumed"] == 0 and legacy["subsumed"] > 0  # (smart spills what legacy drops)
    ideal = sim.simulate(ev, "ideal")
    assert ideal["missed_share"] <= smart["missed_share"] and ideal["recomputed_tokens"] <= smart["recomputed_tokens"]


def test_pure_continuations_without_pressure_give_the_same_hits(tmp_path):
    """Sanity: with nothing evicted the policies cannot differ (the same checkpoints, the same lookups)."""
    ev = sim.load([str(trace(tmp_path, "continuation", n=400))])
    a = sim.simulate(ev, "legacy", params=PARAMS, **HUGE)
    b = sim.simulate(ev, "smart", params=PARAMS, **HUGE)
    c = sim.simulate(ev, "ideal")
    for k in ("cached_tokens", "recomputed_tokens", "missed_share", "vram_hits"):
        assert a[k] == b[k] == c[k], k
    assert a["vram_evictions_per_request"] == b["vram_evictions_per_request"] == 0
    assert a["missed_share"] < 0.01


def test_every_policy_runs_and_reports(tmp_path):
    ev = sim.load([str(trace(tmp_path, "mixed", n=300))])
    rows = [sim.simulate(ev, pol, params=PARAMS, **SMALL) for pol in sim.POLICIES]
    for r in rows:
        assert r["requests"] == 300 and 0 <= r["missed_share"] <= 1 and r["recomputed_tokens"] > 0
        assert r["policy"] in sim.POLICIES
    ideal = rows[sim.POLICIES.index("ideal")]
    assert all(ideal["recomputed_tokens"] <= r["recomputed_tokens"] for r in rows)
    text = sim.render(rows)
    assert "sglang_statecost" in text and "marconi_utility" in text
    assert sim.main([str(tmp_path / "mixed1.jsonl"), "--policy", "legacy,legacy-nofair,smart", "--ram-gib", "6", "--ssd-gib", "16",
                     "--vram-pages", "3000", "--json"]) == 0


def test_incremental_junction_matches_a_subtree_walk(tmp_path):
    """The simulator keeps branch / shared over all tiers incrementally (forks, own); a walk of the subtree must agree."""
    ev = sim.load([str(trace(tmp_path, "diverging", n=300, extra=("--churn", "0.1")))])
    s = sim.Sim("smart", params=PARAMS, **SMALL)

    def walk(n):
        branch = shared = False
        st = [n]
        while st:
            x = st.pop()
            if len(x.children) >= 2:
                branch = True
            for k in x.children:
                shared = shared or k.owner != n.owner
                st.append(k)
        return branch, shared
    rng = random.Random(1)
    checked = 0
    for i, (ts, kind, rec) in enumerate(ev):
        s.now = ts
        (s.begin if kind == 0 else s.done)(rec)
        if i % 40 == 0:
            for n in rng.sample(list(s.nodes.values()), min(60, len(s.nodes))):
                assert s.junction(n) == walk(n)
                assert n.desc == [sum(1 for d in _desc(n) if d.tiers & t) for t in (sim.V, sim.R, sim.S)]
                checked += 1
    assert checked > 100


def _desc(n):
    st, out = list(n.children), []
    while st:
        x = st.pop()
        out.append(x)
        st.extend(x.children)
    return out
