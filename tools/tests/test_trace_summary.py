"""tools/trace_summary.py on a small synthetic trace."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trace_summary as ts  # noqa: E402


def req(ts_, agent, prompt, cached, reusable, div=None):
    return {"t": "req", "ts": ts_, "rid": f"r{ts_}", "agent": agent, "prompt_tokens": prompt, "cached": cached,
            "reusable": reusable, "diverged_at": div, "blocks": []}


def test_summary(tmp_path):
    rows = [{"t": "hdr", "version": 1}, req(0, "a", 20000, 0, 0), req(10, "a", 21000, 0, 20000, div=19000),
            req(25, "a", 22000, 21000, 21500), req(5, "b", 1000, 512, 512), {"t": "snap", "ts": 1}, {"t": "done", "ts": 2}]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n{torn")
    reqs, dones, snaps = ts.load([str(p)])
    s = ts.summarize(reqs, dones, snaps)
    assert s["requests"] == 4 and s["done"] == 1 and s["snaps"] == 1
    assert s["per_agent"] == {"a": 3, "b": 1}
    assert s["long_prompts"] == 3 and abs(s["long_zero_hit_share"] - 2 / 3) < 1e-9
    assert s["diverged_share"] == 0.25 and s["divergence_histogram"] == {16384: 1}
    assert s["gap_s"]["a"]["median"] == 12.5 and s["gap_s"]["a"]["n"] == 2 and "b" not in s["gap_s"]
    assert s["reusable_not_cached_tokens"] == 20000 + 500
    assert abs(s["mean_cached"] - 21512 / 4) < 1e-9
    assert "requests 4" in ts.render(s)
    assert ts.main([str(p)]) == 0
