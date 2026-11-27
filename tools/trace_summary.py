#!/usr/bin/env python3
"""Summary of request trace files (frontend/request_trace.py, docs/request-trace.md). CPU only, no engine.

    python tools/trace_summary.py trace-*.jsonl
"""

import argparse
import json
import statistics
import sys

BIG = 16384  # "long prompt" for the zero-hit share
BUCKETS = (0, 1024, 4096, 16384, 32768, 65536, 131072)


def load(paths):
    reqs, dones, snaps = [], [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
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
    return out


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
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = summarize(*load(a.files))
    print(json.dumps(s, indent=1) if a.json else render(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
