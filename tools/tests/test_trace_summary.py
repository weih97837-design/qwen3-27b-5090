"""tools/trace_summary.py on a small synthetic trace."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trace_summary as ts  # noqa: E402


def req(ts_, agent, prompt, cached, reusable, div=None, src=None):
    r = {"t": "req", "ts": ts_, "rid": f"r{ts_}", "agent": agent, "prompt_tokens": prompt, "cached": cached,
         "reusable": reusable, "diverged_at": div, "blocks": []}
    if src is not None:
        r["reusable_from"] = src
    return r


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
    assert "reusable_from" not in s  # (a trace written before the field)


def test_reusable_from_share():
    reqs = [req(0, "a", 20000, 0, 0, src="none"), req(1, "a", 21000, 0, 20000, src="same"), req(2, "b", 9000, 0, 9000, src="other"),
            req(3, "c", 30000, 0, 11000, src="other"), req(4, "d", 5000, 0, 0)]
    s = ts.summarize(reqs)
    assert s["reusable_from"] == {"none": {"requests": 1, "reusable_tokens": 0}, "same": {"requests": 1, "reusable_tokens": 20000},
                                  "other": {"requests": 2, "reusable_tokens": 20000}}
    assert s["reusable_from_other_share"] == 0.5
    assert "other agents: 50.0%" in ts.render(s)


def test_anchor_share_synthetic():
    import array
    import random
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend"))
    import request_trace as rt

    def rnd(n, seed):
        r = random.Random(seed)
        return [r.randrange(1000, 150000) for _ in range(n)]

    def anchors(x):
        return rt.anchors(array.array("i", x).tobytes(), 0)

    shared = rnd(5000, 1)
    a_ids = rnd(1000, 2) + shared + rnd(500, 3)
    b_ids = rnd(2222, 4) + shared                       # B repeats A's 5K block behind a different header
    c_ids = rnd(6000, 5)                                # all new
    in_shared = set(anchors(shared))
    b_anchors = anchors(b_ids)
    b_in_block = [h for h in b_anchors if h in in_shared]
    assert len(b_in_block) > 20

    def row(t, agent, ids_, cached=0):
        return {**req(t, agent, len(ids_), cached, 0), "new_tokens": len(ids_), "new_anchors": anchors(ids_)}

    rows = [row(0, "x", a_ids), row(1, "y", b_ids), row(2, "x", c_ids)]
    a = ts.anchor_stats(rows)
    exp_b = len(b_in_block) / len(b_anchors)
    assert 0.55 < exp_b < 0.8                                 # (the rest of B's anchors lie in its own new header)
    assert a["requests"] == 3 and abs(a["share_p90"] - exp_b) < 1e-9
    assert a["share_histogram"].get("0%") == 2           # A (first) and C (all new content): nothing seen before
    assert abs(a["shareable_est"]["any"] - exp_b * len(b_ids)) < 1e-6
    assert a["shareable_est"]["other"] == a["shareable_est"]["any"] and a["shareable_est"]["same"] == 0
    # B's anchors inside the shared block: all seen (about 100%), whatever the header
    only_block = row(1, "y", shared)
    assert ts.anchor_stats([row(0, "x", a_ids), only_block])["share_median"] > 0.95
    # same agent
    a2 = ts.anchor_stats([row(0, "x", a_ids), row(1, "x", b_ids)])
    assert a2["shareable_est"]["same"] == a2["shareable_est"]["any"] > 0 and a2["shareable_est"]["other"] == 0
    out = ts.render(ts.summarize(rows))
    assert "caveat" in out and "cached 0" in out


def test_anchor_section_skipped_for_older_traces():
    s = ts.summarize([req(0, "a", 1000, 0, 0)])
    assert s["anchors"] is None
    assert "no new_anchors" in ts.render(s)
