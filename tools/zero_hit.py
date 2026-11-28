"""Zero-hit requests: a prompt of at least THRESHOLD tokens whose prefix cache hit nothing (cached == 0), and why (docs/request-trace.md,
"零命中請求").

Shared by tools/trace_summary.py (offline: the req events of a trace, in time order) and the server (live: frontend/stats.py feeds
every request that began, with its first block hashes, when it finishes). Stdlib only, no engine.

Each zero-hit request gets exactly one reason, evaluated in this order:
  first_seen                the agent has no earlier request in the window and its first block hash was never seen there
  agent_returned_after_gap  the agent's previous request is more than GAP_S seconds earlier
  diverged_early            the agent's previous request is within GAP_S, but none of the agent's earlier requests shares a
                            THRESHOLD-token prefix with this one (the prompt changed at the start)
  evicted_or_missed         an earlier request shares a THRESHOLD-token prefix, yet nothing was cached: the cache failed
  other                     none of the above
window_start: a would-be first_seen within the first WINDOW_S seconds of the window. The history before the window is unknown, so
that is counted apart and not as first_seen.

Prefixes are compared with the chained block hashes (request_trace.block_chain, BLOCK tokens per block): a prefix of THRESHOLD tokens
is the same as its PREFIX_BLOCKS-th chain hash being equal, because each chain hash covers everything before it. Nothing else is
needed, so a record keeps only: agent, first block hash (h0), PREFIX_BLOCKS-th block hash (h15), prompt length, cached tokens, time.

Memory and time per request are bounded: the classifier keeps the last `cap` requests (hashes only), one last-seen time per agent
(LRU, agent_cap), and is O(1) per request. A request whose earlier sharing request left that window is classified without it.
"""

import collections
import threading

BLOCK = 256                      # tokens per block hash (request_trace.BLOCK)
THRESHOLD = 4096                 # "a long prompt" and "a shared prefix" in tokens
PREFIX_BLOCKS = THRESHOLD // BLOCK
BIG = 16384                      # the subset of very long prompts (trace_summary's long_zero_hit_share uses this)
GAP_S = 600.0                    # an agent's gap that makes it a returning one
WINDOW_S = 300.0                 # the start of a window whose earlier history is unknown
CAP = 4096                       # requests kept by the live classifier (the server)
OFFLINE_CAP = 65536              # requests kept by trace_summary (a whole trace is read, so it looks further back)
AGENT_CAP = 4096                 # agents with a last-seen time kept (LRU)
RING = 200                       # zero-hit requests kept by the live view
EXAMPLES = 10                    # examples per reason in the offline summary
TOP_AGENTS = 10                  # agents in the live agents_top
REASONS = ("first_seen", "agent_returned_after_gap", "diverged_early", "evicted_or_missed", "other", "window_start")
OFF = ("0", "off", "false", "no", "none")  # the ZERO_HIT_STATS values that turn the live view off (as REQUEST_TRACE, any case)


def enabled(env):
    """The live view (ZERO_HIT_STATS) is on unless the value is an off value or empty (request_trace.resolve_target's rule)."""
    v = env.get("ZERO_HIT_STATS")
    if v is None:
        return True
    return v.strip().lower() not in ("",) + OFF


def from_req(r):
    """The record of a trace "req" event (None for other events). Missing or short block lists give None hashes."""
    if r.get("t") != "req":
        return None
    blocks = r.get("blocks") or []
    return {"ts": float(r["ts"]), "agent": r.get("agent"), "prompt_tokens": int(r.get("prompt_tokens") or 0),
            "cached": r.get("cached"), "h0": blocks[0] if blocks else None,
            "h15": blocks[PREFIX_BLOCKS - 1] if len(blocks) >= PREFIX_BLOCKS else None,
            "restore": r.get("restore"), "restore_tier": r.get("restore_tier")}


def _row(reason, gap=None, earlier=None):
    return {"reason": reason, "gap_s": None if gap is None else round(gap, 3),
            "earlier_age_s": None if earlier is None else round(earlier, 3)}


def _dec(c, k):
    c[k] -= 1
    if c[k] <= 0:
        del c[k]


class Classifier:
    """Feed records in time order (feed returns the classification of a zero-hit request, else None; it always records the request
    as history). window_start: the start of the window (default: the first record's ts)."""

    def __init__(self, window_start=None, cap=CAP, agent_cap=AGENT_CAP, gap_s=GAP_S, window_s=WINDOW_S):
        self.t0, self.cap, self.agent_cap, self.gap_s, self.window_s = window_start, cap, agent_cap, gap_s, window_s
        self.hist = collections.deque()                  # (agent, h0, h15) of the last `cap` requests
        self.h0 = collections.Counter()                  # h0 -> requests in hist
        self.h15 = {}                                    # h15 -> [requests in hist, last ts]
        self.pair = collections.Counter()                # (agent, h15) -> requests in hist
        self.last = collections.OrderedDict()            # agent -> ts of its last request (LRU, agent_cap)

    def feed(self, rec):
        if self.t0 is None:
            self.t0 = rec["ts"]
        out = self._classify(rec) if rec.get("cached") == 0 and rec["prompt_tokens"] >= THRESHOLD else None
        self._add(rec)
        return out

    def _classify(self, rec):
        ts, ag, h0, h15 = rec["ts"], rec["agent"], rec["h0"], rec["h15"]
        prev = self.last.get(ag)
        if prev is None and (h0 is None or not self.h0.get(h0)):
            return _row("window_start" if ts - self.t0 < self.window_s else "first_seen")
        gap = None
        if prev is not None:
            gap = max(0.0, ts - prev)
            if ts - prev > self.gap_s:
                return _row("agent_returned_after_gap", gap)
            if h15 is None or not self.pair.get((ag, h15)):
                return _row("diverged_early", gap)
        m = self.h15.get(h15) if h15 is not None else None
        if m is not None and m[0] > 0:
            return _row("evicted_or_missed", gap, max(0.0, ts - m[1]))
        return _row("other", gap)

    def _add(self, rec):
        ts, ag, h0, h15 = rec["ts"], rec["agent"], rec["h0"], rec["h15"]
        self.hist.append((ag, h0, h15))
        if h0 is not None:
            self.h0[h0] += 1
        if h15 is not None:
            self.pair[(ag, h15)] += 1
            m = self.h15.get(h15)
            if m is None:
                self.h15[h15] = [1, ts]
            else:
                m[0] += 1
                m[1] = max(m[1], ts)
        prev = self.last.get(ag)
        self.last[ag] = ts if prev is None else max(prev, ts)
        self.last.move_to_end(ag)
        if len(self.last) > self.agent_cap:
            self.last.popitem(last=False)
        while len(self.hist) > self.cap:
            a, o, f = self.hist.popleft()
            if o is not None:
                _dec(self.h0, o)
            if f is not None:
                _dec(self.pair, (a, f))
                m = self.h15[f]
                m[0] -= 1
                if m[0] <= 0:
                    del self.h15[f]


def from_head(zh, prompt_tokens, cached):
    """The record of a finished request from the server's "zh" info (frontend/worker.py finish): the agent, the start time (begin),
    its first and PREFIX_BLOCKS-th block hashes, and the restore decision."""
    return {"ts": zh["begun"], "agent": zh["agent"], "prompt_tokens": int(prompt_tokens or 0), "cached": cached,
            "h0": zh.get("h0"), "h15": zh.get("h15"), "restore": zh.get("restore"), "restore_tier": zh.get("restore_tier")}


class Live:
    """The server's rolling view: counts per reason since `since_ts`, the last RING zero-hit requests, the agents with most zero-hit
    requests. Thread-safe (the stats reader thread writes, HTTP handler threads read). Stores no prompt text."""

    def __init__(self, since_ts, cap=CAP, ring=RING):
        self.lock = threading.Lock()
        self.since_ts = since_ts
        self.cls = Classifier(window_start=since_ts, cap=cap)
        self.totals = {r: [0, 0] for r in REASONS}       # reason -> [count, prompt tokens]
        self.recent = collections.deque(maxlen=ring)
        self.agents = collections.OrderedDict()          # agent -> [count, prompt tokens] (LRU, AGENT_CAP)

    def observe(self, rec):
        with self.lock:
            res = self.cls.feed(rec)
            if res is None:
                return None
            t = self.totals[res["reason"]]
            t[0] += 1
            t[1] += rec["prompt_tokens"]
            a = self.agents.setdefault(rec["agent"], [0, 0])
            a[0] += 1
            a[1] += rec["prompt_tokens"]
            self.agents.move_to_end(rec["agent"])
            if len(self.agents) > AGENT_CAP:
                self.agents.popitem(last=False)
            self.recent.append({"ts": rec["ts"], "agent": rec["agent"], "prompt_tokens": rec["prompt_tokens"], **res,
                                "restore": rec["restore"], "restore_tier": rec["restore_tier"]})
            return res

    def snapshot(self, since=None, now=0.0):
        with self.lock:
            recent = [dict(e) for e in self.recent if since is None or e["ts"] > since]
            totals = {r: {"count": c, "tokens": t} for r, (c, t) in self.totals.items()}
            top = sorted(self.agents.items(), key=lambda kv: (-kv[1][0], kv[0]))[:TOP_AGENTS]
        return {"window_s": round(max(0.0, now - self.since_ts), 1), "since_ts": self.since_ts, "threshold_tokens": THRESHOLD,
                "totals": totals, "recent": recent,
                "agents_top": [{"agent": a, "count": c, "tokens": t} for a, (c, t) in top]}

    def metrics(self):
        with self.lock:
            out = {}
            for r, (c, t) in self.totals.items():
                out[f'q27:zero_hit_requests_total{{reason="{r}"}}'] = c
                out[f'q27:zero_hit_tokens_total{{reason="{r}"}}'] = t
        return out


def median_of(counter):
    """The median of the values of a Counter (value -> count), the mean of the two middle values when even; None when empty."""
    n = sum(counter.values())
    if not n:
        return None
    lo, hi = (n - 1) // 2, n // 2
    seen, lo_v = 0, None
    for v in sorted(counter):
        c = counter[v]
        if lo_v is None and lo < seen + c:
            lo_v = v
        if hi < seen + c:
            return (lo_v + v) / 2
        seen += c


def summarize(records, window_start=None, cap=OFFLINE_CAP):
    """The offline summary of normalized records (from_req) in time order: per reason count, share of the zero-hit requests, prompt
    tokens, median prompt length, the top agents, the restore decisions, and up to EXAMPLES examples of the evicted_or_missed and the
    agent_returned_after_gap ones (numbers and ids only)."""
    cls = Classifier(window_start=window_start, cap=cap)
    st = {r: {"count": 0, "tokens": 0, "lengths": collections.Counter(), "agents": collections.Counter(),
              "restore": collections.Counter(), "over_gap": 0, "big": 0} for r in REASONS}
    examples = {"evicted_or_missed": [], "agent_returned_after_gap": []}
    long_n = big_n = zero = zero_big = n = 0
    for rec in records:
        n += 1
        p = rec["prompt_tokens"]
        long_n += p >= THRESHOLD
        big_n += p >= BIG
        res = cls.feed(rec)
        if res is None:
            continue
        r = res["reason"]
        s = st[r]
        zero += 1
        s["count"] += 1
        s["tokens"] += p
        s["lengths"][p] += 1
        s["agents"][rec["agent"]] += 1
        s["restore"][f"{rec.get('restore_tier') or 'none'} {rec.get('restore') or 'none'}"] += 1
        if (res["earlier_age_s"] or 0) > GAP_S:
            s["over_gap"] += 1
        if p >= BIG:
            s["big"] += 1
            zero_big += 1
        if r in examples and len(examples[r]) < EXAMPLES:
            examples[r].append({"ts": rec["ts"], "agent": rec["agent"], "prompt_tokens": p,
                                "gap_s": res["gap_s"] if r == "agent_returned_after_gap" else res["earlier_age_s"],
                                "restore": rec.get("restore"), "restore_tier": rec.get("restore_tier")})
    reasons = {}
    for r in REASONS:
        s = st[r]
        reasons[r] = {"count": s["count"], "share": s["count"] / zero if zero else None, "tokens": s["tokens"],
                      "median_prompt": median_of(s["lengths"]), "agents": s["agents"].most_common(5),
                      "restore": dict(s["restore"].most_common()), "over_gap": s["over_gap"], "big": s["big"]}
    return {"threshold_tokens": THRESHOLD, "big_tokens": BIG, "cap": cap, "window_start": cls.t0, "requests": n,
            "long": long_n, "zero_hit": zero, "zero_hit_share": zero / long_n if long_n else None,
            "big": big_n, "zero_hit_big": zero_big, "zero_hit_big_share": zero_big / big_n if big_n else None,
            "reasons": reasons, "examples": examples}
