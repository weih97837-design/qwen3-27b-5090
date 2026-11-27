#!/usr/bin/env python3
"""Synthetic request traces for tools/cache_sim.py (the format of frontend/request_trace.py, docs/request-trace.md).

    python tools/cache_sim_gen.py out.jsonl [--workload mixed|continuation|diverging] [--requests 3000] [--seed 1] [...]

Block hashes are random 16-hex-digit strings: a prompt that continues or shares a prefix reuses the parent's list, so prefixes compare
exactly as the real chained hashes do. The workload (all knobs are options):

  * agents in groups that share a system prompt (9-40K tokens: --sys-min / --sys-max), so prefixes are shared across agents;
  * heavy agents (--heavy) whose conversations grow to 50-126K tokens, the rest to 20-40K; a conversation that would pass its limit
    restarts from the system prompt (a new conversation);
  * each request of an agent is a pure continuation of its conversation (prompt = the previous prompt + output + a new message), or with
    probability --diverge it leaves the conversation at a point 9-40K tokens in (a sub-task / a re-rendered history: the shared
    prefix, then new text) and that branch becomes the conversation (with probability --return it goes back to the old one next time);
  * arrivals: each agent sends its next request --gap-s (median, log-normal) after its previous one ended; the request lasts its
    prefill (the t0 / k model) plus its output at --decode-tps.

Workloads: "mixed" (the default: everything above), "continuation" (one agent class, no divergence, no shared system prompt: every
request a pure continuation), "diverging" (heavy agents with --diverge 0.35). `reusable` is computed like the worker's: the longest
prefix shared with the last 64 sequences (prompt + output).
"""

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import cache_policy as cp  # noqa: E402

BLOCK = 256


class Gen:
    def __init__(self, a):
        self.a = a
        self.rng = random.Random(a.seed)
        self.p = cp.SmartParams()
        self.hist = collections.deque()
        self.hist_blocks = collections.Counter()
        self.n = 0

    def blocks(self, n):
        return [f"{self.rng.getrandbits(64):016x}" for _ in range(n)]

    def reusable(self, blocks):
        lo, hi = 0, len(blocks)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.hist_blocks.get(blocks[mid - 1]):
                lo = mid
            else:
                hi = mid - 1
        return lo * BLOCK

    def remember(self, blocks):
        self.hist.append(blocks)
        self.hist_blocks.update(blocks)
        if len(self.hist) > 64:
            self.hist_blocks.subtract(self.hist.popleft())

    def run(self):
        a, rng = self.a, self.rng
        groups = []
        for g in range(a.groups):
            groups.append(self.blocks(rng.randint(a.sys_min, a.sys_max) // BLOCK) if a.workload != "continuation" else [])
        agents = []
        for i in range(a.agents):
            heavy = rng.random() < a.heavy
            sysp = groups[i % len(groups)]
            agents.append({"name": f"{'heavy' if heavy else 'agent'}{i:03d}", "heavy": heavy, "sys": sysp,
                           "conv": list(sysp) or self.blocks(rng.randint(2, 8) * 4), "old": None, "next": rng.random() * a.gap_s * 2,
                           "limit": (rng.randint(50_000, 126_000) if heavy else rng.randint(20_000, 40_000)) // BLOCK})
        heap = [(ag["next"], i) for i, ag in enumerate(agents)]
        import heapq
        heapq.heapify(heap)
        out = []
        diverge = a.diverge if a.workload != "continuation" else 0.0
        while self.n < a.requests:
            t, i = heapq.heappop(heap)
            ag = agents[i]
            msg = max(1, int(rng.lognormvariate(math.log(a.msg_tokens), 0.8)) // BLOCK)
            conv = ag["conv"]
            div = None
            if ag["old"] is not None and rng.random() < a.ret:
                conv, ag["old"] = ag["old"], None
            if len(conv) + msg > ag["limit"]:  # a new conversation from the system prompt
                conv = list(ag["sys"]) or self.blocks(rng.randint(2, 8) * 4)
            elif rng.random() < diverge and len(conv) * BLOCK > a.div_min + BLOCK:
                k = rng.randint(a.div_min, min(a.div_max, len(conv) * BLOCK - BLOCK)) // BLOCK
                ag["old"] = conv
                conv = conv[:k]
                div = k * BLOCK
            prompt = conv + self.blocks(msg)
            out_tok = max(16, int(rng.lognormvariate(math.log(a.out_tokens), 0.7)))
            ptoks = len(prompt) * BLOCK + rng.randint(0, BLOCK - 1)
            total_blocks = (ptoks + out_tok) // BLOCK
            end = self.blocks(max(0, total_blocks - len(prompt)))
            rid = f"r{self.n:06d}"
            reusable = self.reusable(prompt)
            if div is not None and reusable < div:
                div = None
            dur = cp.prefill_seconds(self.p, 0, ptoks) * a.prefill_share + out_tok / a.decode_tps
            out.append({"t": "req", "ts": round(t, 3), "rid": rid, "agent": ag["name"], "client": "sim", "prompt_tokens": ptoks,
                        "media_tokens": 0, "max_tokens": out_tok, "cached": None, "reusable": reusable, "diverged_at": div,
                        "restored_from": None, "queued_s": 0.0, "blocks": prompt})
            out.append({"t": "done", "ts": round(t + dur, 3), "rid": rid, "reason": "stop", "output_tokens": out_tok, "ttft_s": None,
                        "prefill_s": None, "computed_tokens": None, "preempted": 0, "total_tokens": ptoks + out_tok, "end_blocks": end})
            self.remember(prompt + end)
            ag["conv"] = prompt + end
            gap = rng.lognormvariate(math.log(a.gap_s * (1.0 if ag["heavy"] else a.light_gap_factor)), 1.0)
            heapq.heappush(heap, (t + dur + gap, i))
            self.n += 1
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--workload", choices=("mixed", "continuation", "diverging"), default="mixed")
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agents", type=int, default=40)
    ap.add_argument("--groups", type=int, default=6, help="system prompts shared by the agents (round robin)")
    ap.add_argument("--heavy", type=float, default=0.35, help="share of heavy (50-126K) agents")
    ap.add_argument("--diverge", type=float, default=0.18)
    ap.add_argument("--ret", type=float, default=0.3, help="after a divergence: chance the next request goes back to the old branch")
    ap.add_argument("--sys-min", type=int, default=9000)
    ap.add_argument("--sys-max", type=int, default=40000)
    ap.add_argument("--div-min", type=int, default=9000)
    ap.add_argument("--div-max", type=int, default=40000)
    ap.add_argument("--msg-tokens", type=float, default=1500)
    ap.add_argument("--out-tokens", type=float, default=900)
    ap.add_argument("--gap-s", type=float, default=40.0, help="median gap of a heavy agent between its requests")
    ap.add_argument("--light-gap-factor", type=float, default=4.0)
    ap.add_argument("--decode-tps", type=float, default=60.0)
    ap.add_argument("--prefill-share", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.workload == "diverging":
        a.heavy, a.diverge = max(a.heavy, 0.6), max(a.diverge, 0.35)
    rows = Gen(a).run()
    rows.sort(key=lambda r: r["ts"])
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"t": "hdr", "version": 1, "block_tokens": BLOCK, "engine": "cache_sim_gen", "settings": vars(a)}) + "\n")
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
