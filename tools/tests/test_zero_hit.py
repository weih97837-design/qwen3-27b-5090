"""tools/zero_hit.py: the zero-hit classification (offline summary and the live view) on synthetic prompts with the real block chains
(frontend/request_trace.block_chain, so the prefix semantics are the engine's). No engine, no server."""

import array
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))                # tools
sys.path.insert(0, str(HERE.parents[2] / "frontend"))   # request_trace (block_chain, head_hashes)

import request_trace as rt  # noqa: E402
import zero_hit as zh  # noqa: E402


def ids(seed, n):
    r = random.Random(seed)
    return [r.randrange(1, 150000) for _ in range(n)]


A = ids(1, 6000)                       # the base prompt
B = A[:4096] + ids(2, 1904)            # shares exactly 4096 tokens (16 blocks) with A
C = A[:4095] + ids(3, 1905)            # shares 4095 tokens: the 16th block differs
D = A[:256] + ids(4, 5744)             # shares only the first block with A
E = ids(5, 5000)                       # nothing in common


def rec(ts, agent, prompt, cached=0, restore=None, tier=None):
    """A normalized record as from_req makes it, from the prompt's real chain."""
    b = rt.block_chain(prompt)[0]
    return {"ts": float(ts), "agent": agent, "prompt_tokens": len(prompt), "cached": cached,
            "h0": b[0] if b else None, "h15": b[zh.PREFIX_BLOCKS - 1] if len(b) >= zh.PREFIX_BLOCKS else None,
            "restore": restore, "restore_tier": tier}


def reasons(records, **kw):
    c = zh.Classifier(**kw)
    return [(r["reason"] if (r := c.feed(x)) else None) for x in records]


# ------------------------------------------------------------------ the reasons, in order

def test_first_seen_for_a_brand_new_agent_and_prefix():
    c = zh.Classifier(window_start=0)
    assert c.feed(rec(1000, "a", A)) == {"reason": "first_seen", "gap_s": None, "earlier_age_s": None}


def test_window_start_covers_the_first_300_seconds_only():
    c = zh.Classifier(window_start=0)
    assert c.feed(rec(299.999, "a", A))["reason"] == "window_start"
    assert c.feed(rec(300, "b", E))["reason"] == "first_seen"          # exactly 300 s: out of the window start


def test_window_start_does_not_replace_the_other_reasons():
    c = zh.Classifier(window_start=0)
    c.feed(rec(10, "a", A, cached=0))
    assert c.feed(rec(20, "a", B))["reason"] == "evicted_or_missed"    # in the window start, but agent a is known


def test_returned_after_gap_at_601_not_at_600():
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A, cached=0))
    assert c.feed(rec(1600, "a", B))["reason"] == "evicted_or_missed"    # gap exactly 600: not a return, sharing decides
    assert c.feed(rec(1600 + 601, "a", B)) == {"reason": "agent_returned_after_gap", "gap_s": 601.0, "earlier_age_s": None}


def test_diverged_early_when_the_prefix_changed_at_the_start():
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A))
    r = c.feed(rec(1100, "a", E, cached=0))                              # same agent, different start: nothing shared
    assert r == {"reason": "diverged_early", "gap_s": 100.0, "earlier_age_s": None}


def test_diverged_early_when_the_shared_prefix_is_under_4096():
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A))
    assert c.feed(rec(1100, "a", C))["reason"] == "diverged_early"       # shares 4095 tokens: under the threshold


def test_evicted_or_missed_when_an_earlier_request_shares_4096_tokens():
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A))
    r = c.feed(rec(1100, "a", B, cached=0))
    assert r == {"reason": "evicted_or_missed", "gap_s": 100.0, "earlier_age_s": 100.0}


def test_evicted_from_another_agent_reports_the_age_of_the_sharing_request():
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A))
    r = c.feed(rec(2000, "b", B, cached=0))                              # agent b is new, but its prefix was cached by a
    assert r == {"reason": "evicted_or_missed", "gap_s": None, "earlier_age_s": 1000.0}


def test_other_when_the_first_block_was_seen_but_nothing_reaches_4096():
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A))
    # agent b is new, but the first block is known from a: not first_seen; the prefix under 4096 is no shared request: other
    assert c.feed(rec(2000, "b", D))["reason"] == "other"


def test_a_prompt_of_exactly_4096_tokens_is_long_and_4095_is_not():
    c = zh.Classifier(window_start=0)
    exact = ids(7, 4096)
    assert c.feed(rec(1000, "a", exact))["reason"] == "first_seen"
    assert c.feed(rec(1001, "b", exact[:4095])) is None                  # below the threshold: not a zero-hit request


def test_a_hit_or_an_unknown_cache_is_not_zero_hit():
    c = zh.Classifier(window_start=0)
    assert c.feed(rec(1000, "a", A, cached=512)) is None
    assert c.feed(rec(1001, "a", B, cached=None)) is None               # cached unknown (an old trace): not counted
    assert c.feed(rec(1002, "a", E, cached=0))["reason"] == "diverged_early"   # the hits above still count as history


def test_each_reason_has_exactly_one_label_and_the_order_holds():
    # a returning agent whose previous request also shared a prefix: the gap decides (2) before sharing (4)
    c = zh.Classifier(window_start=0)
    c.feed(rec(1000, "a", A, cached=0))
    assert c.feed(rec(9000, "a", B, cached=0))["reason"] == "agent_returned_after_gap"


# ------------------------------------------------------------------ the bounded state

def test_history_and_indexes_stay_within_the_cap():
    c = zh.Classifier(window_start=0, cap=3, agent_cap=2)
    for i in range(10):
        c.feed(rec(1000 + i, f"agent{i}", ids(100 + i, 5000)))
    assert len(c.hist) == 3 and len(c.h0) <= 3 and len(c.h15) <= 3 and len(c.pair) <= 3
    assert len(c.last) == 2


def test_an_earlier_request_that_left_the_cap_is_not_compared_any_more():
    c = zh.Classifier(window_start=0, cap=1)
    c.feed(rec(1000, "a", A))
    c.feed(rec(1001, "z", ids(99, 5000)))                                # pushes A out of the cap
    assert c.feed(rec(1002, "b", B, cached=0))["reason"] == "first_seen"  # A's first block is no longer known
    assert c.feed(rec(1003, "c", B, cached=0))["reason"] == "evicted_or_missed"  # b's B is in the cap now


# ------------------------------------------------------------------ the offline summary

def test_summarize_counts_shares_medians_top_agents_and_examples():
    recs = [rec(0, "a", A, cached=0),          # window start
            rec(100, "a", B, cached=0),        # shares 4096 with a's earlier request: evicted_or_missed, 100 s before
            rec(200, "a", C, cached=0),        # shares 4095 only: diverged_early
            rec(1000, "b", E, cached=0),       # a new agent and prefix: first_seen
            rec(1000, "a", B, cached=0),       # 800 s after a's last request: agent_returned_after_gap
            rec(2000, "a", B, cached=1)]       # a hit: not zero-hit
    recs[4]["restore"], recs[4]["restore_tier"] = "no_match", "ram"
    z = zh.summarize(recs, window_start=0)
    assert z["requests"] == 6 and z["zero_hit"] == 5 and z["long"] == 6 and z["zero_hit_share"] == 5 / 6
    got = {r: z["reasons"][r]["count"] for r in zh.REASONS}
    assert got == {"window_start": 1, "evicted_or_missed": 1, "diverged_early": 1, "first_seen": 1,
                   "agent_returned_after_gap": 1, "other": 0}
    assert z["reasons"]["evicted_or_missed"]["median_prompt"] == 6000
    assert z["reasons"]["evicted_or_missed"]["agents"] == [("a", 1)]
    ev = z["examples"]["evicted_or_missed"]
    assert len(ev) == 1 and ev[0]["gap_s"] == 100.0 and ev[0]["agent"] == "a" and ev[0]["prompt_tokens"] == 6000
    ret = z["examples"]["agent_returned_after_gap"]
    assert len(ret) == 1 and ret[0]["gap_s"] == 800.0 and ret[0]["restore"] == "no_match" and ret[0]["restore_tier"] == "ram"
    assert z["reasons"]["agent_returned_after_gap"]["restore"] == {"ram no_match": 1}


def test_summarize_big_subset_and_share():
    big = ids(50, 17000)
    recs = [rec(0, "a", big, cached=0), rec(1, "a", big[:5000], cached=0)]
    z = zh.summarize(recs, window_start=0)
    assert z["big"] == 1 and z["zero_hit_big"] == 1 and z["zero_hit_big_share"] == 1.0
    assert z["zero_hit"] == 2 and z["zero_hit_share"] == 1.0 and z["long"] == 2
    assert z["reasons"]["evicted_or_missed"]["big"] == 0


def test_summarize_window_start_defaults_to_the_first_record():
    z = zh.summarize([rec(500, "a", A, cached=0), rec(600, "b", E, cached=0)])
    assert z["window_start"] == 500 and z["reasons"]["window_start"]["count"] == 2


def test_median_of_even_and_odd_counts():
    from collections import Counter
    assert zh.median_of(Counter({4096: 1})) == 4096
    assert zh.median_of(Counter({4096: 1, 8192: 1})) == 6144
    assert zh.median_of(Counter({4096: 2, 8192: 1})) == 4096
    assert zh.median_of(Counter()) is None


def test_from_req_tolerates_missing_blocks():
    r = zh.from_req({"t": "req", "ts": 1, "agent": "a", "prompt_tokens": 100, "cached": 0})
    assert r["h0"] is None and r["h15"] is None and r["cached"] == 0
    assert zh.from_req({"t": "done", "ts": 1}) is None


# ------------------------------------------------------------------ the live view (the server's), and parity with the offline one

def test_live_view_matches_the_offline_summary_on_the_same_records():
    recs = [rec(1000, "a", A), rec(1100, "a", B), rec(1200, "a", C), rec(2000, "b", D), rec(3000, "c", E, cached=0),
            rec(9000, "a", B, cached=0)]
    off = zh.summarize(recs, window_start=1000)
    live = zh.Live(1000.0)
    for x in recs:
        live.observe(x)
    snap = live.snapshot(now=1000.0)
    for r in zh.REASONS:
        assert snap["totals"][r]["count"] == off["reasons"][r]["count"], r
        assert snap["totals"][r]["tokens"] == off["reasons"][r]["tokens"], r


def test_live_recent_ring_and_since_filter():
    live = zh.Live(0.0, ring=3)
    for i in range(5):
        live.observe(rec(100 + i, f"a{i}", ids(200 + i, 5000), cached=0))
    snap = live.snapshot(now=200.0)
    assert len(snap["recent"]) == 3 and [e["ts"] for e in snap["recent"]] == [102, 103, 104]
    assert [e["ts"] for e in live.snapshot(since=103.0, now=200.0)["recent"]] == [104]
    assert snap["window_s"] == 200.0 and snap["since_ts"] == 0.0 and snap["threshold_tokens"] == 4096
    assert set(snap["totals"]) == set(zh.REASONS)


def test_live_agents_top_and_metrics_names():
    live = zh.Live(0.0)
    for i in range(3):
        live.observe(rec(1000 + i, "busy", ids(300 + i, 5000), cached=0))
    live.observe(rec(1010, "quiet", ids(400, 5000), cached=0))
    top = live.snapshot(now=1010.0)["agents_top"]
    assert top[0]["agent"] == "busy" and top[0]["count"] == 3
    m = live.metrics()
    assert m['q27:zero_hit_requests_total{reason="first_seen"}'] == 2      # busy's first and quiet's first
    assert m['q27:zero_hit_requests_total{reason="diverged_early"}'] == 2  # busy's second and third: their own prefix changed
    assert len([k for k in m if k.startswith("q27:zero_hit_requests_total")]) == len(zh.REASONS)
    assert all(k.startswith(("q27:zero_hit_requests_total{", "q27:zero_hit_tokens_total{")) for k in m)


def test_live_metrics_include_zero_counts_for_every_reason():
    m = zh.Live(0.0).metrics()
    assert m['q27:zero_hit_requests_total{reason="evicted_or_missed"}'] == 0
    assert m['q27:zero_hit_tokens_total{reason="window_start"}'] == 0


def test_enabled_parses_like_request_trace():
    assert zh.enabled({}) is True
    assert zh.enabled({"ZERO_HIT_STATS": "1"}) is True
    for v in ("0", "off", "FALSE", "no", "none", "", "  0  "):
        assert zh.enabled({"ZERO_HIT_STATS": v}) is False, v


def test_head_hashes_are_the_first_16_chain_hashes_of_the_prompt():
    data = array.array("i", A).tobytes()
    assert rt.head_hashes(data) == rt.block_chain(A)[0][:16]
    assert rt.head_hashes(array.array("i", A[:300]).tobytes()) == rt.block_chain(A[:300])[0][:1]


def test_trace_summary_prints_the_section_and_the_json_block(tmp_path, capsys, monkeypatch):
    import json
    import trace_summary

    def line(ts, t, **kw):
        return json.dumps({"t": t, "ts": ts, **kw}, separators=(",", ":"))
    chain_a, chain_b = rt.block_chain(A)[0], rt.block_chain(B)[0]
    rows = [line(0, "hdr", version=1),
            line(10, "req", rid="r1", agent="a/s1", prompt_tokens=len(A), cached=0, reusable=0, blocks=chain_a, restore=None, restore_tier=None),
            line(110, "req", rid="r2", agent="a/s1", prompt_tokens=len(B), cached=0, reusable=0, blocks=chain_b, restore="no_match",
                 restore_tier="ssd"),
            line(120, "done", rid="r2", reason="stop")]
    p = tmp_path / "trace-x.jsonl"
    p.write_text("\n".join(rows) + "\n")
    assert trace_summary.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "零命中分析" in out and "evicted_or_missed" in out and "快取失效" in out
    assert trace_summary.main(["--json", str(p)]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["zero_hit"]["zero_hit"] == 2
    assert j["zero_hit"]["reasons"]["evicted_or_missed"]["count"] == 1
    assert j["zero_hit"]["examples"]["evicted_or_missed"][0]["restore_tier"] == "ssd"


def test_from_head_maps_the_engine_start_fields():
    zh_ = {"agent": "a/s1", "begun": 12.5, "prompt_tokens": 5000, "cached": 0, "h0": "x", "h15": "y", "restore": "no_match",
           "restore_tier": "none"}
    r = zh.from_head(zh_)
    assert r == {"ts": 12.5, "agent": "a/s1", "prompt_tokens": 5000, "cached": 0, "h0": "x", "h15": "y", "restore": "no_match",
                 "restore_tier": "none"}
    assert zh.from_head({"agent": "a", "begun": 1.0})["h15"] is None   # (a short prompt has no 16th block)
