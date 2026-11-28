#!/usr/bin/env python3
"""Summary of request trace files (frontend/request_trace.py, docs/request-trace.md). CPU only, no engine.

    python tools/trace_summary.py trace-*.jsonl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trace_io  # noqa: E402  (a file path or an http(s):// URL such as http://HOST:8080/admin/trace/all)
import zero_hit  # noqa: E402  (the zero-hit classification, shared with the server's live view)

BIG = 16384  # "long prompt" for the zero-hit share
BUCKETS = (0, 1024, 4096, 16384, 32768, 65536, 131072)


def load(paths):
    reqs, dones, snaps = [], [], []
    for p in paths:
        with trace_io.open_lines(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # (a torn last line)
                {"req": reqs, "done": dones, "snap": snaps}.get(r.get("t"), []).append(r)
    reqs.sort(key=lambda r: r["ts"])
    return reqs, dones, snaps


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def bucket(v):
    return max(b for b in BUCKETS if v >= b)


def summarize(reqs, dones=(), snaps=()):
    n = len(reqs)
    out = {"requests": n, "done": len(dones), "snaps": len(snaps)}
    if not n:
        return out
    counts = {}
    for r in reqs:
        counts[r["agent"]] = counts.get(r["agent"], 0) + 1
    out["per_agent"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    prompts = [r["prompt_tokens"] for r in reqs]
    out["prompt_tokens"] = {"min": min(prompts), "median": statistics.median(prompts), "p90": pct(prompts, 0.9), "max": max(prompts),
                            "mean": sum(prompts) / n}
    hist = {}
    for p in prompts:
        hist[bucket(p)] = hist.get(bucket(p), 0) + 1
    out["prompt_histogram"] = hist  # lower bound of the bucket -> requests
    out["mean_cached"] = sum(r["cached"] or 0 for r in reqs) / n
    out["mean_reusable"] = sum(r["reusable"] or 0 for r in reqs) / n
    big = [r for r in reqs if r["prompt_tokens"] > BIG]
    out["long_prompts"] = len(big)
    out["long_zero_hit_share"] = sum(1 for r in big if not r["cached"]) / len(big) if big else None
    div = [r["diverged_at"] for r in reqs if r.get("diverged_at") is not None]
    out["diverged_share"] = len(div) / n
    dh = {}
    for d in div:
        dh[bucket(d)] = dh.get(bucket(d), 0) + 1
    out["divergence_histogram"] = dh
    gaps, last = {}, {}
    for r in reqs:
        if r["agent"] in last:
            gaps.setdefault(r["agent"], []).append(r["ts"] - last[r["agent"]])
        last[r["agent"]] = r["ts"]
    out["gap_s"] = {a: {"median": statistics.median(g), "p90": pct(g, 0.9), "n": len(g)} for a, g in gaps.items()}
    out["reusable_not_cached_tokens"] = sum(max(0, (r["reusable"] or 0) - (r["cached"] or 0)) for r in reqs)
    # Cross-agent sharing (reusable_from: whose history entry the reusable prefix comes from; traces before it have none)
    src = [r for r in reqs if r.get("reusable_from") is not None]
    if src:
        by = {}
        for r in src:
            k = r["reusable_from"]
            b = by.setdefault(k, {"requests": 0, "reusable_tokens": 0})
            b["requests"] += 1
            b["reusable_tokens"] += r["reusable"] or 0
        total = sum(b["reusable_tokens"] for b in by.values())
        out["reusable_from"] = by
        out["reusable_from_other_share"] = by.get("other", {}).get("reusable_tokens", 0) / total if total else None
    # Restore decisions (req.restore / restore_tier; traces before them have none): "tier outcome" -> requests
    rd = {}
    for r in reqs:
        if r.get("restore") is not None:
            k = f"{r.get('restore_tier') or 'none'} {r['restore']}"
            rd[k] = rd.get(k, 0) + 1
    if rd:
        out["restore_decisions"] = dict(sorted(rd.items(), key=lambda kv: (-kv[1], kv[0])))
    out["anchors"] = anchor_stats(reqs)
    out["zero_hit"] = zero_hit.summarize(zero_hit.from_req(r) for r in reqs)  # (reqs are in time order: load() sorts them)
    return out


SHARE_BINS = ((0.0, 0.0, "0%"), (0.0, 0.25, "0-25%"), (0.25, 0.75, "25-75%"), (0.75, 1.0, "75-100%"), (1.0, 1.0, "100%"))


def share_bin(x):
    if x <= 0:
        return "0%"
    if x >= 1:
        return "100%"
    return "0-25%" if x <= 0.25 else "25-75%" if x <= 0.75 else "75-100%"


def anchor_stats(reqs):
    """Position-independent view (req.new_anchors, docs/request-trace.md): per request the share of the anchors of its 'new' tail
    (beyond `reusable`) that an EARLIER request (file/time order) already had in its own tail: any agent / the same agent / another
    agent. Estimated shareable tokens = share x new_tokens. None when the trace has no new_anchors (an older version)."""
    if not any("new_anchors" in r for r in reqs):
        return None
    seen = {}  # anchor hash -> agents whose earlier request had it
    shares, groups = [], {}
    tot_new = cov_new = 0
    est = {"any": 0.0, "same": 0.0, "other": 0.0}
    nanch = no_anchor = 0
    for r in reqs:
        a = r.get("new_anchors")
        if a is None:
            continue
        nt = r.get("new_tokens")
        if nt is None:
            nt = max(0, (r.get("prompt_tokens") or 0) - (r.get("reusable") or 0))
        tot_new += nt
        nanch += len(a)
        if a:
            ag = r["agent"]
            c = {"any": 0, "same": 0, "other": 0}
            for h in a:
                who = seen.get(h)
                if who:
                    c["any"] += 1
                    if ag in who:
                        c["same"] += 1
                    if len(who - {ag}):
                        c["other"] += 1
            f = {k: v / len(a) for k, v in c.items()}
            cov_new += nt
            for k in est:
                est[k] += f[k] * nt
            shares.append(f["any"])
            for g in ("cached 0" if not r.get("cached") else "cached >0", "reusable_from " + str(r.get("reusable_from"))):
                b = groups.setdefault(g, {"requests": 0, "new_tokens": 0, "shareable_est": 0.0})
                b["requests"] += 1
                b["new_tokens"] += nt
                b["shareable_est"] += f["any"] * nt
            for h in a:
                seen.setdefault(h, set()).add(ag)
        elif nt:
            no_anchor += 1
    hist = {}
    for x in shares:
        hist[share_bin(x)] = hist.get(share_bin(x), 0) + 1
    return {"requests": len(shares), "requests_without_anchor": no_anchor, "anchors": nanch, "new_tokens": tot_new,
            "new_tokens_with_anchors": cov_new,
            "shareable_est": est, "shareable_share": {k: (v / cov_new if cov_new else None) for k, v in est.items()},
            "share_p10": pct(shares, 0.1), "share_median": pct(shares, 0.5), "share_p90": pct(shares, 0.9),
            "share_histogram": hist, "groups": groups}


ANCHOR_CAVEAT = ("caveat: estimate only. One anchor per ~128 tokens (sampled), each a window of 32 tokens (see the header's anchor_every / "
                 "anchor_k); a window counts as seen only if its 32 tokens are identical to an earlier request's tail. Content-defined: "
                 "a changed header or offset shifts nothing. 'Seen' is an upper bound of what position-independent reuse could find; "
                 "the prefix cache cannot reuse it (KV and recurrent state depend on everything before).")


def render_anchors(a):
    if a is None:
        return "position-independent sharing: no new_anchors in this trace (older version or REQUEST_TRACE_ANCHORS=0), skipped"
    L = ["position-independent sharing of the 'new' tail (beyond reusable):",
         f"  requests with anchors {a['requests']}  without {a['requests_without_anchor']}  anchors {a['anchors']}  "
         f"new tokens {a['new_tokens']} (with anchors {a['new_tokens_with_anchors']})"]
    sh = a["shareable_share"]
    f = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    L.append(f"  estimated new tokens shareable by content: any agent {a['shareable_est']['any']:.0f} ({f(sh['any'])})  "
             f"same agent {a['shareable_est']['same']:.0f} ({f(sh['same'])})  other agent {a['shareable_est']['other']:.0f} ({f(sh['other'])})")
    L.append(f"  per-request share seen before: p10 {f(a['share_p10'])}  median {f(a['share_median'])}  p90 {f(a['share_p90'])}  histogram "
             + "  ".join(f"{lab}: {a['share_histogram'].get(lab, 0)}" for _, _, lab in SHARE_BINS))
    for g, b in sorted(a["groups"].items()):
        L.append(f"  {g}: {b['requests']} requests, new {b['new_tokens']}, shareable ~{b['shareable_est']:.0f} "
                 f"({f(b['shareable_est'] / b['new_tokens'] if b['new_tokens'] else None)})")
    L.append("  " + ANCHOR_CAVEAT)
    return "\n".join(L)


def render(s):
    L = [f"requests {s['requests']}  finished {s['done']}  snapshots {s['snaps']}"]
    if s["requests"] == 0:
        return "\n".join(L)
    L.append("requests per agent: " + ", ".join(f"{a} {c}" for a, c in s["per_agent"].items()))
    p = s["prompt_tokens"]
    L.append(f"prompt tokens: mean {p['mean']:.0f}  median {p['median']:.0f}  p90 {p['p90']}  min {p['min']}  max {p['max']}")
    L.append("prompt size (tokens >=): " + "  ".join(f"{k}: {v}" for k, v in sorted(s["prompt_histogram"].items())))
    L.append(f"mean cached {s['mean_cached']:.0f}  mean reusable {s['mean_reusable']:.0f}")
    z = s["long_zero_hit_share"]
    L.append(f"prompts > {BIG}: {s['long_prompts']}, with no cache hit at all: " + ("-" if z is None else f"{100 * z:.1f}%"))
    L.append(f"requests with diverged_at: {100 * s['diverged_share']:.1f}%  position (token >=): "
             + ("  ".join(f"{k}: {v}" for k, v in sorted(s["divergence_histogram"].items())) or "-"))
    L.append("gap between requests of an agent (s):")
    for a, g in s["gap_s"].items():
        L.append(f"  {a}: median {g['median']:.1f}  p90 {g['p90']:.1f}  (n {g['n']})")
    L.append(f"reusable but not cached: {s['reusable_not_cached_tokens']} tokens")
    if "reusable_from" in s:
        o = s["reusable_from_other_share"]
        L.append("reusable prefix from (requests / reusable tokens): " + "  ".join(
            f"{k} {v['requests']} / {v['reusable_tokens']}" for k, v in sorted(s["reusable_from"].items()))
            + "  -> share of reusable tokens from other agents: " + ("-" if o is None else f"{100 * o:.1f}%"))
    if s.get("restore_decisions"):
        L.append("restore decisions (tier outcome: requests): " + "  ".join(f"{k}: {v}" for k, v in s["restore_decisions"].items()))
    L.append(render_anchors(s.get("anchors")))
    if s.get("zero_hit"):
        L.append(render_zero_hit(s["zero_hit"]))
    return "\n".join(L)


REASON_ZH = {  # one line per reason (zh-TW), the same text as the status page's card
    "first_seen": "全新：agent 在窗口內首次出現，且第一個區塊的雜湊從未出現",
    "agent_returned_after_gap": "回頭的 agent：距上次請求超過 600 秒（每日或隔夜才回來的 agent）",
    "diverged_early": "開頭就變了：600 秒內的上一個請求與它的前綴不足 4096 token（system prompt 或工具定義變了）",
    "evicted_or_missed": "快取失效：與先前請求共享 >= 4096 token 的前綴，卻沒有命中（待查：被淘汰或沒有寫入）",
    "other": "其他：與先前請求的共同前綴不足 4096 token，且不屬於上述情況",
    "window_start": "窗口起點：窗口開始後 300 秒內的全新 agent，歷史未知，不計入 first_seen",
}


def render_zero_hit(z):
    L = ["零命中分析（prompt >= 4096 token 且 cached = 0；分類見 tools/zero_hit.py）"]
    if not z["zero_hit"]:
        L.append(f"  請求 {z['requests']}，沒有零命中請求")
        return "\n".join(L)
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    L.append(f"  請求 {z['requests']}，長 prompt（>= {z['threshold_tokens']}）{z['long']}，其中零命中 {z['zero_hit']}（{pc(z['zero_hit_share'])}）；"
             f">= {z['big_tokens']} 的長 prompt {z['big']}，其中零命中 {z['zero_hit_big']}（{pc(z['zero_hit_big_share'])}）")
    L.append(f"  窗口起點 ts {z['window_start']:.0f}；比對最近 {z['cap']} 個請求的雜湊（更早的請求不再比對）")
    L.append("  原因                       數量   占比    prompt token 合計   prompt 中位數   前 5 agent（數量）")
    R = z["reasons"]
    for r in zero_hit.REASONS:
        b = R[r]
        if not b["count"]:
            continue
        agents = ", ".join(f"{a} {c}" for a, c in b["agents"])
        L.append(f"  {r:<26} {b['count']:>6} {pc(b['share']):>6} {b['tokens']:>18} {b['median_prompt']:>15.0f}   {agents}")
    for r in zero_hit.REASONS:
        if R[r]["count"]:
            L.append(f"  {r}: {REASON_ZH[r]}")
    e = R["evicted_or_missed"]
    if e["count"]:
        L.append(f"  evicted_or_missed: 共享前綴的前一請求超過 600 秒前 {e['over_gap']} 筆（共 {e['count']} 筆）；還原結果（tier outcome: 數量）：")
        L.append("    " + "  ".join(f"{k}: {v}" for k, v in e["restore"].items()))
    for r in ("evicted_or_missed", "agent_returned_after_gap"):
        ex = z["examples"][r]
        if not ex:
            continue
        L.append(f"  {r} 範例（最多 {zero_hit.EXAMPLES} 筆；無 prompt 文字）：")
        L.append("    agent                          prompt tokens   " + ("前綴前一請求相隔 s" if r == "evicted_or_missed" else "與前次請求相隔 s")
                 + "   還原結果")
        for x in ex:
            gap = "-" if x["gap_s"] is None else f"{x['gap_s']:.1f}"
            L.append(f"    {x['agent']:<30} {x['prompt_tokens']:>13}   {gap:>20}   {x['restore_tier'] or 'none'} {x['restore'] or 'none'}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        s = summarize(*load(a.files))
    except trace_io.TraceSourceError as e:
        sys.exit(f"trace_summary: {e}")
    print(json.dumps(s, indent=1) if a.json else render(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
