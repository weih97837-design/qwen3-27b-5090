#!/usr/bin/env python3
"""Trace-driven simulator of the prefix cache's retention policies (VRAM checkpoints -> RAM tier -> SSD tier). CPU only, no engine.

    python tools/cache_sim.py trace-*.jsonl --policy legacy,smart [--ram-gib 64 --ssd-gib 200 --vram-pages 8207 --state-mib 190
                                                                  --kv-kib-per-token 18 --checkpoint-slots 43] [--json]

Replays the JSONL request traces of frontend/request_trace.py (docs/request-trace.md; tools/cache_sim_gen.py writes synthetic ones)
and reports per policy how much of the reusable prefix the cache supplied. The policies (docs/tiered-cache.md "CACHE_POLICY=smart"):

  legacy           the rules before smart became the default (CACHE_POLICY=legacy): VRAM plain LRU for pages, covered-first then LRU for a state slot; a checkpoint a deeper one covers
                   (> 16K tokens) is DROPPED when the pool evicts it (subsumed), and spilling a deeper one erases its spilled shallower
                   ones; RAM / SSD victims: cold agents first (--cold-first), the agent using the most bytes (--fair-agents), then
                   superseded, never-hit, the old rule (RAM: frees the most, LRU within 2x; SSD: mostly-shared last, oldest written).
                   An entry superseded by one on the SSD is dropped instead of written.
  smart            CACHE_POLICY=smart (frontend/cache_policy.py, the same code as engine/src/retention.h): GreedyDual priority =
                   stamp + prefill seconds saved over the parent / GiB x (1 + hits) x junction x superseded factor, leaf-first,
                   young-entry protection in RAM / SSD, everything spilled (nothing subsumed but equal copies), SSD admission when
                   the priority is above the SSD's watermark.
  lru              the same flow as smart (nothing subsumed), every victim the least recently used.
  sglang_statecost the same flow; GreedyDual with SGLang's StateCost restore cost, priority = stamp + (e - s) + (e^2 - s^2) / 2e5
                   (tokens; s = the parent's depth), leaf-first, no bytes, no hits (reimplemented from the formula).
  marconi_utility  the same flow; Marconi's utility = alpha * norm(gain / bytes) + norm(1 / (now - last use)) (min-max over the
                   candidates), candidates with at most one retained extender in the tier (reimplemented from the formula; alpha fixed,
                   --marconi-alpha, no tuning).
  ideal            no capacity limit anywhere (the upper bound of a checkpoint cache with these checkpoint positions).

Model (approximations, all in 256-token blocks: a checkpoint is identified by the chained hash of its last block):
  * Checkpoints are created where the engine plans them, block-rounded: at `reusable` (the worker plans one where the prompt leaves its
    best history), at the last 4 multiples of 4096 tokens inside the prefill (CHECKPOINT_INTERVAL), at the end of the prompt (the last
    full block: the generation prompt's start, plan_checkpoints' last im_start), and at the end of the request (prompt + output, from
    the `done` event's end_blocks). The first-message end and media ends are not in the trace and not modelled.
  * Lookup = the deepest retained checkpoint (any tier) on the request's block chain below the prompt's last token; a RAM / SSD match is
    restored into VRAM (no bandwidth or worth-loading test: every restore succeeds). `cached` is that depth.
  * VRAM: --vram-pages KV pages (64 tokens) shared by the running sequences (a request holds its prompt from its `req` to its `done`,
    its output from `done`) and the checkpoints (pages shared by reference along a chain); --checkpoint-slots state slots. A checkpoint
    frees the pages only it holds (a suffix of its chain: the holders of a block are a subset of the holders of the block before).
  * RAM / SSD: an entry costs --state-mib plus its KV (--kv-kib-per-token); full 4096-token chunks are shared by content (RAM dedup /
    SSD chunk files), the rest is the entry's own. Bytes per agent split a shared chunk between its agents (spill_agents).
  * No idle backups, no RAM retention tick (the engine's RAM cap writes the victims to the SSD, as measured), no TTLs, no restarts.
  * `reusable` is the trace's (rounded down to a block), else the longest prefix shared with the last 64 sequences (the worker's).

Reported: (reusable - cached) / reusable, tokens recomputed, zero-hit share of prompts > 16K (of all of them and of those with something
reusable), RAM / SSD restores, spills, SSD writes, VRAM evictions per request, prefill seconds wasted (the same t0 / k model).
"""

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cache_policy as cp  # noqa: E402
import trace_io  # noqa: E402  (a file path or an http(s):// URL such as http://HOST:8080/admin/trace/all)

BLOCK = 256
CHUNK_BLOCKS = 16           # 4096-token chunks (chunk_store.h kChunkPages * 64)
KEEP_SHORT = 16384          # retention.h kKeepShortTokens
INTERVAL_BLOCKS = 16        # CHECKPOINT_INTERVAL 4096
GIB = float(1 << 30)
POLICIES = ("legacy", "smart", "lru", "sglang_statecost", "marconi_utility", "ideal")
V, R, S = cp.TIER_VRAM, cp.TIER_RAM, cp.TIER_SSD
TIERS = (V, R, S)
TI = {V: 0, R: 1, S: 2}


class Node:
    __slots__ = ("key", "blocks", "nb", "owner", "hits", "last_use", "tiers", "born", "parent", "children", "desc", "forks", "own",
                 "sid")
    serial = 0

    def __init__(self, key, blocks, nb, owner):
        self.key, self.blocks, self.nb, self.owner = key, blocks, nb, owner
        Node.serial += 1
        self.sid = Node.serial  # (creation order: candidates are listed in it, so ties break the same way on every run)
        self.hits, self.last_use, self.tiers = 0, 0, 0
        self.born = [0.0, 0.0, 0.0]   # when it came to VRAM / RAM / SSD
        self.parent, self.children = None, set()
        self.desc = [0, 0, 0]         # strict descendants in VRAM / RAM / SSD
        self.forks = 0                # nodes of its subtree (itself included) with >= 2 children: > 0 is a branch point
        self.own = {}                 # owner -> strict descendants of that owner

    @property
    def depth(self):
        return self.nb * BLOCK


class ChunkStore:
    """Content-shared chunks of a tier (RAM dedup / SSD chunk files): refcounts and the fractional bytes per agent."""

    def __init__(self, chunk_bytes):
        self.chunk_bytes = chunk_bytes
        self.refs = {}       # chunk key -> refs
        self.agents = {}     # chunk key -> {agent: refs}
        self.use = collections.defaultdict(float)  # agent -> bytes (own bytes are added by the tier)

    def _share(self, key, sign):
        ag = self.agents.get(key)
        if ag:
            for a in ag:
                self.use[a] += sign * self.chunk_bytes / len(ag)

    def ref(self, key, agent):
        self._share(key, -1)
        self.refs[key] = self.refs.get(key, 0) + 1
        ag = self.agents.setdefault(key, {})
        ag[agent] = ag.get(agent, 0) + 1
        self._share(key, +1)

    def unref(self, key, agent):
        self._share(key, -1)
        self.refs[key] -= 1
        ag = self.agents[key]
        ag[agent] -= 1
        if not ag[agent]:
            del ag[agent]
        if not self.refs[key]:
            del self.refs[key], self.agents[key]
        else:
            self._share(key, +1)

    @property
    def bytes(self):
        return len(self.refs) * self.chunk_bytes


class Sim:
    def __init__(self, policy, ram_gib=64.0, ssd_gib=200.0, vram_pages=8207, state_mib=190.0, kv_kib_per_token=18.0,
                 checkpoint_slots=43, fair_agents=True, cold_first=True, cold_after_s=600.0, params=None, marconi_alpha=1.0,
                 smart_admit="watermark", smart_variant=()):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r} (one of {', '.join(POLICIES)})")
        self.policy = policy
        inf = policy == "ideal"
        self.ram_cap = math.inf if inf else ram_gib * GIB
        self.ssd_cap = math.inf if inf else ssd_gib * GIB
        self.vram_blocks = math.inf if inf else vram_pages // 4
        self.slots = math.inf if inf else checkpoint_slots
        self.state = state_mib * (1 << 20)
        self.kv_tok = kv_kib_per_token * 1024
        self.block_bytes = self.kv_tok * BLOCK
        self.fair, self.cold_first, self.cold_after = fair_agents, cold_first, cold_after_s
        self.p = params or cp.SmartParams()
        self.alpha = marconi_alpha
        self.smart_admit = smart_admit
        self.variant = set(smart_variant)  # sim-only ablations of smart: nohits, nobytes, nojunction, nosup, noleaf
        self.nodes = {}
        self.roots = set()
        self.vref = {}               # VRAM block -> holders (checkpoints + running sequences)
        self.n_vram = 0
        self.vram_nodes = set()
        self.ram_nodes, self.ssd_nodes = set(), set()
        self.ram_chunks = ChunkStore(CHUNK_BLOCKS * self.block_bytes)
        self.ssd_chunks = ChunkStore(CHUNK_BLOCKS * self.block_bytes)
        self.ram_own = 0.0           # states + tails in RAM
        self.ssd_own = 0.0
        self.ram_use = collections.defaultdict(float)  # agent -> own bytes in RAM
        self.ssd_use = collections.defaultdict(float)
        self.gd = {V: cp.GdTier(), R: cp.GdTier(), S: cp.GdTier()}
        self.clock = 0
        self.now = 0.0
        self.c = collections.Counter()
        self.running = {}            # rid -> (blocks, nb held)
        self.history = collections.deque()
        self.hist_blocks = collections.Counter()

    # ------------------------------------------------------------------ trie of the retained checkpoints (any tier)
    def _attach(self, n):
        p = None
        for k in range(n.nb - 1, 0, -1):
            q = self.nodes.get(n.blocks[k - 1])
            if q is not None:
                p = q
                break
        sibs = self.roots if p is None else p.children
        before = len(sibs) >= 2
        moved = [c for c in sibs if c.nb > n.nb and c.blocks[n.nb - 1] == n.key]
        for c in moved:
            sibs.discard(c)
            c.parent = n
            n.children.add(c)
            for t in TIERS:
                n.desc[TI[t]] += c.desc[TI[t]] + (1 if c.tiers & t else 0)
            n.forks += c.forks
            for o, k in c.own.items():
                n.own[o] = n.own.get(o, 0) + k
            n.own[c.owner] = n.own.get(c.owner, 0) + 1
        mine = 1 if len(moved) >= 2 else 0
        n.forks += mine
        n.parent = p
        sibs.add(n)
        self.nodes[n.key] = n
        if p is not None:  # p's subtree: n's own fork, and p's change of children
            self._up(p, mine + (1 if len(sibs) >= 2 else 0) - (1 if before else 0), n.owner, +1)

    def _detach(self, n):
        p = n.parent
        sibs = self.roots if p is None else p.children
        before = len(sibs) >= 2
        mine = 1 if len(n.children) >= 2 else 0
        sibs.discard(n)
        for c in n.children:
            c.parent = p
            sibs.add(c)
        n.children = set()
        del self.nodes[n.key]
        if p is not None:
            self._up(p, -mine + (1 if len(sibs) >= 2 else 0) - (1 if before else 0), n.owner, -1)

    @staticmethod
    def _up(a, forks, owner, d):
        """a and its ancestors: forks += forks, own[owner] += d."""
        while a is not None:
            a.forks += forks
            k = a.own.get(owner, 0) + d
            if k:
                a.own[owner] = k
            else:
                del a.own[owner]
            a = a.parent

    def _tier_add(self, n, t):
        n.tiers |= t
        a = n.parent
        while a is not None:
            a.desc[TI[t]] += 1
            a = a.parent

    def _tier_del(self, n, t):
        n.tiers &= ~t
        a = n.parent
        while a is not None:
            a.desc[TI[t]] -= 1
            a = a.parent
        if not n.tiers:
            self._detach(n)

    def _mask_count(self, n, mask):
        return sum(n.desc[TI[t]] + (1 if n.tiers & t else 0) for t in TIERS if mask & t)

    def junction(self, n, mask=V | R | S):
        """retention.h chain_info's branch / shared over the extenders of n in the tiers of `mask`."""
        if mask == V | R | S:  # (every retained node: kept incrementally)
            return n.forks > 0, len(n.own) - (1 if n.owner in n.own else 0) > 0
        branch = shared = False
        stack = [n]
        while stack:
            x = stack.pop()
            kids = [c for c in x.children if self._mask_count(c, mask)]
            if len(kids) >= 2:
                branch = True
            for c in kids:
                if c.tiers & mask and c.owner != n.owner:
                    shared = True
                stack.append(c)
            if branch and shared:
                break
        return branch, shared

    def parent_in(self, n, mask):
        a = n.parent
        while a is not None and not a.tiers & mask:
            a = a.parent
        return a

    def superseded(self, n):
        """retention.h superseded_flags (no draft ring in the trace): a deeper retained entry extends it, it is longer than 16K, and it
        is neither a branch point nor shared by two agents."""
        if n.depth <= KEEP_SHORT or not self._mask_count_desc(n):
            return False
        b, s = self.junction(n)
        return not b and not s

    def _mask_count_desc(self, n):
        return n.desc[0] + n.desc[1] + n.desc[2]

    # ------------------------------------------------------------------ bytes
    def _chunks(self, n):
        return [n.blocks[(j + 1) * CHUNK_BLOCKS - 1] for j in range(n.nb // CHUNK_BLOCKS)]

    def _own(self, n):
        return self.state + (n.nb % CHUNK_BLOCKS) * self.block_bytes

    def _sole_chunks(self, n, store):
        """Trailing chunks of n only it references (the holders of a chunk include the holders of every later one)."""
        ch = self._chunks(n)
        lo, hi = 0, len(ch)  # first index with refs == 1
        while lo < hi:
            mid = (lo + hi) // 2
            if store.refs.get(ch[mid], 0) <= 1:
                hi = mid
            else:
                lo = mid + 1
        return len(ch) - lo

    def freed(self, n, t):
        store = self.ram_chunks if t == R else self.ssd_chunks
        return self._own(n) + self._sole_chunks(n, store) * store.chunk_bytes

    def sole_blocks(self, n):
        lo, hi = 0, n.nb
        while lo < hi:
            mid = (lo + hi) // 2
            if self.vref.get(n.blocks[mid], 0) <= 1:
                hi = mid
            else:
                lo = mid + 1
        return n.nb - lo

    def ram_bytes(self):
        return self.ram_own + self.ram_chunks.bytes

    def ssd_bytes(self):
        return self.ssd_own + self.ssd_chunks.bytes

    # ------------------------------------------------------------------ VRAM
    def _vref(self, blocks, nb, d):
        for b in blocks[:nb]:
            k = self.vref.get(b, 0) + d
            if k:
                self.vref[b] = k
            else:
                del self.vref[b]

    def _vram_room(self, need):
        """Evicts checkpoints until `need` more blocks fit (page pressure)."""
        while len(self.vref) + need > self.vram_blocks:
            if not self.vram_nodes or not self._evict_vram(for_slot=False):
                self.c["kv_overflow"] += 1
                return

    def _evict_vram(self, for_slot):
        cands = sorted(self.vram_nodes, key=_sid)
        pol = self.policy
        if pol in ("legacy", "lru"):
            if pol == "legacy" and for_slot:
                cov = [n for n in cands if n.depth > KEEP_SHORT and n.desc[0] and not self.junction(n, V)[0]]
                if cov:
                    cands = cov
            victim = min(cands, key=lambda n: n.last_use)
        else:
            victim = self._policy_pick(cands, V, for_slot=for_slot)
        self.c["vram_evictions"] += 1
        self._vram_out(victim)
        return True

    def _vram_out(self, n):
        self.vram_nodes.discard(n)
        self.n_vram -= 1
        self._vref(n.blocks, n.nb, -1)
        spill = True
        if self.policy == "legacy":
            # Model::spill_subsumed: a deeper retained checkpoint (VRAM / RAM / SSD) covers it and it is long, or an equal copy exists
            if (n.depth > KEEP_SHORT and self._mask_count_desc(n)) or n.tiers & (R | S):
                spill = False
        elif n.tiers & (R | S):
            spill = False  # (an equal copy is spilled already)
        if not spill:
            self.c["subsumed"] += 1
            self._tier_del(n, V)
            return
        self._tier_del_keep(n, V)
        if self.policy == "legacy":  # spill_checkpoint: the spilled shallower entries it covers go (subsumed)
            a = n.parent
            while a is not None:
                nxt = a.parent
                if a.tiers & (R | S) and a.depth > KEEP_SHORT:
                    self.c["subsumed"] += 1
                    self._drop_spilled(a)
                a = nxt
        self._ram_in(n)
        self.c["spilled"] += 1
        self._enforce_ram()

    def _tier_del_keep(self, n, t):
        """Leaves tier t without detaching (it moves to another tier right after)."""
        n.tiers &= ~t
        a = n.parent
        while a is not None:
            a.desc[TI[t]] -= 1
            a = a.parent

    def _vram_in(self, n):
        if n.tiers & V:
            return
        self._vref(n.blocks, n.nb, +1)
        self.vram_nodes.add(n)
        self.n_vram += 1
        n.born[0] = self.now
        self._tier_add(n, V)

    # ------------------------------------------------------------------ RAM / SSD
    def _ram_in(self, n):
        n.born[1] = self.now
        for k in self._chunks(n):
            self.ram_chunks.ref(k, n.owner)
        self.ram_own += self._own(n)
        self.ram_use[n.owner] += self._own(n)
        self.ram_nodes.add(n)
        self._tier_add(n, R)

    def _ram_out(self, n, detach=True):
        for k in self._chunks(n):
            self.ram_chunks.unref(k, n.owner)
        self.ram_own -= self._own(n)
        self.ram_use[n.owner] -= self._own(n)
        self.ram_nodes.discard(n)
        (self._tier_del if detach else self._tier_del_keep)(n, R)

    def _ssd_in(self, n):
        n.born[2] = self.now
        for k in self._chunks(n):
            self.ssd_chunks.ref(k, n.owner)
        self.ssd_own += self._own(n)
        self.ssd_use[n.owner] += self._own(n)
        self.ssd_nodes.add(n)
        self._tier_add(n, S)

    def _ssd_out(self, n, detach=True):
        for k in self._chunks(n):
            self.ssd_chunks.unref(k, n.owner)
        self.ssd_own -= self._own(n)
        self.ssd_use[n.owner] -= self._own(n)
        self.ssd_nodes.discard(n)
        (self._tier_del if detach else self._tier_del_keep)(n, S)

    def _drop_spilled(self, n):
        if n.tiers & R:
            self._ram_out(n)
        elif n.tiers & S:
            self._ssd_out(n)

    def _enforce_ram(self):
        while self.ram_bytes() > self.ram_cap and self.ram_nodes:
            v = self._pick_ram()
            self.c["ram_evictions"] += 1
            if self.ssd_cap > 0 and self._admit_ssd(v):
                self._ram_out(v, detach=False)
                self._ssd_in(v)
                self.c["ssd_written"] += 1
                self._enforce_ssd()
            else:
                self.c["ram_dropped"] += 1
                self._ram_out(v)

    def _enforce_ssd(self):
        while self.ssd_bytes() > self.ssd_cap and self.ssd_nodes:
            v = self._pick_ssd()
            self.c["ssd_evictions"] += 1
            self._ssd_out(v)

    def _admit_ssd(self, n):
        if self.policy == "legacy":
            # superseded_durable: a deeper one that stands in for it is on the SSD -> dropped, not written
            return not (n.desc[2] and self.superseded(n))
        if self.policy == "smart":
            c = self._smart_cand(n, S, bytes_=self._own(n) + sum(1 for k in self._chunks(n) if k not in self.ssd_chunks.refs)
                                 * self.ssd_chunks.chunk_bytes)
            pr = cp.smart_priority(self.p, c)
            if self.smart_admit == "always":
                ok = True
            elif self.smart_admit == "victim":  # above the priority of the entry that would go now (sim-only variant)
                cs = [self._smart_cand(x, S) for x in self.ssd_nodes]
                k = cp.smart_pick(cs, self.p)
                ok = self.ssd_bytes() + c.bytes <= self.ssd_cap or k.idx < 0 or pr > k.priority
            else:  # "watermark": the implemented rule (tiercache.py _smart_admit)
                ok = pr > self.gd[S].L
            if not ok:
                self.c["ssd_not_admitted"] += 1
            return ok
        return True

    # ------------------------------------------------------------------ victims
    def _cold(self, nodes, ti):
        a = {}
        for n in nodes:
            warm, newest = a.get(n.owner, (False, -math.inf))
            a[n.owner] = (warm or n.hits > 0, max(newest, n.born[ti]))
        return {o for o, (w, t) in a.items() if not w and self.now - t >= self.cold_after}

    def _legacy_pick(self, cands, t):
        ti = TI[t]
        pool = cands
        if self.cold_first:
            cold = self._cold(cands, ti)
            if any(n.owner in cold for n in pool):
                pool = [n for n in pool if n.owner in cold]
        if self.fair:
            use = self._agent_use(t)
            top = max({n.owner for n in pool}, key=lambda o: (use.get(o, 0.0), _neg(o)))
            pool = [n for n in pool if n.owner == top]
        sup = {id(n): self.superseded(n) for n in pool}
        if t == R:
            rank = lambda n: (0 if sup[id(n)] else 2) + (1 if n.hits else 0)  # noqa: E731
            best = min(rank(n) for n in pool)
            pool = [n for n in pool if rank(n) == best]
            fr = {id(n): self.freed(n, R) for n in pool}
            top = max(fr.values())
            pool = [n for n in pool if fr[id(n)] * 2 >= top]
            return min(pool, key=lambda n: n.last_use)

        def key(n):  # tiercache._pick_victim
            size = self._own(n) + len(self._chunks(n)) * self.ssd_chunks.chunk_bytes
            return (not sup[id(n)], n.hits > 0, self.freed(n, S) < 0.25 * size, n.born[2])
        return min(pool, key=key)

    def _agent_use(self, t):
        own, store = (self.ram_use, self.ram_chunks) if t == R else (self.ssd_use, self.ssd_chunks)
        return {a: own.get(a, 0.0) + store.use.get(a, 0.0) for a in set(own) | set(store.use)}

    def _smart_cand(self, n, t, for_slot=False, bytes_=None):
        ti = TI[t]
        if t == V:
            par = self.parent_in(n, V)
            b, s = self.junction(n, V)
            nbytes = self.state if for_slot else self.sole_blocks(n) * self.block_bytes
            leaf = not n.desc[0]
            young = False
            sup = False
        else:
            par = n.parent
            b, s = self.junction(n)
            nbytes = self.freed(n, t) if bytes_ is None else bytes_
            leaf = not n.desc[ti]
            young = self.now - n.born[ti] < self.p.min_resident_s
            sup = n.depth > KEEP_SHORT and self._mask_count_desc(n) > 0 and not b and not s
        if self.variant:
            v = self.variant
            return cp.Cand(depth=n.depth, parent_depth=par.depth if par else 0, bytes=GIB if "nobytes" in v else nbytes,
                           hits=0 if "nohits" in v else n.hits, junction=(b or s) and "nojunction" not in v,
                           superseded=sup and "nosup" not in v, leaf=leaf or "noleaf" in v, young=young,
                           stamp=self.gd[t].stamp_at(n.last_use), age_key=n.born[ti] if t != V else n.last_use, ref=n)
        return cp.Cand(depth=n.depth, parent_depth=par.depth if par else 0, bytes=nbytes, hits=n.hits, junction=b or s,
                       superseded=sup, leaf=leaf, young=young, stamp=self.gd[t].stamp_at(n.last_use), age_key=n.born[ti] if t != V
                       else n.last_use, ref=n)

    def _policy_pick(self, cands, t, for_slot=False):
        pol = self.policy
        if pol == "smart" or pol == "ideal":
            cs = [self._smart_cand(n, t, for_slot) for n in cands]
            k = cp.smart_pick(cs, self.p)
            self.gd[t].evicted(k, self.clock)
            self.c[f"smart_{'vram' if t == V else 'ram' if t == R else 'ssd'}_{cp.REASONS[k.reason]}"] += 1
            return cs[k.idx].ref
        if pol == "lru":
            return min(cands, key=lambda n: n.last_use)
        ti = TI[t]
        if pol == "sglang_statecost":
            best, bp = None, None
            leaves = [n for n in cands if not n.desc[ti]] or cands
            for n in leaves:
                par = self.parent_in(n, t) if t == V else n.parent
                s0, e = (par.depth if par else 0), n.depth
                pr = self.gd[t].stamp_at(n.last_use) + (e - s0) + (e * e - s0 * s0) / 2e5
                if bp is None or pr < bp:
                    best, bp = n, pr
            self.gd[t].evicted(cp.Pick(0, bp, cp.LEAF), self.clock)
            return best
        # marconi_utility
        pool = [n for n in cands if n.desc[ti] <= 1] or cands
        eff, rec = [], []
        for n in pool:
            par = self.parent_in(n, t) if t == V else n.parent
            gain = cp.prefill_seconds(self.p, par.depth if par else 0, n.depth)
            nbytes = (self.sole_blocks(n) * self.block_bytes if not for_slot else self.state) if t == V else self.freed(n, t)
            eff.append(gain / max(nbytes, 1 << 20))
            rec.append(1.0 / max(1, self.clock - n.last_use))

        def norm(xs):
            lo, hi = min(xs), max(xs)
            return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]
        ue, ur = norm(eff), norm(rec)
        i = min(range(len(pool)), key=lambda i: (self.alpha * ue[i] + ur[i], pool[i].last_use))
        return pool[i]

    def _pick_ram(self):
        cands = sorted(self.ram_nodes, key=_sid)
        if self.policy == "legacy":
            return self._legacy_pick(cands, R)
        return self._policy_pick(cands, R)

    def _pick_ssd(self):
        cands = sorted(self.ssd_nodes, key=_sid)
        if self.policy == "legacy":
            return self._legacy_pick(cands, S)
        return self._policy_pick(cands, S)

    # ------------------------------------------------------------------ requests
    def lookup(self, blocks, prompt_tokens):
        top = min(len(blocks), (prompt_tokens - 1) // BLOCK)
        for k in range(top, 0, -1):
            n = self.nodes.get(blocks[k - 1])
            if n is not None:
                return n
        return None

    def reusable_of(self, blocks):
        lo, hi = 0, len(blocks)  # the longest prefix in the history window (a prefix property: binary search)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.hist_blocks.get(blocks[mid - 1]):
                lo = mid
            else:
                hi = mid - 1
        return lo * BLOCK

    def _history_add(self, blocks):
        self.history.append(blocks)
        self.hist_blocks.update(blocks)
        if len(self.history) > 64:
            old = self.history.popleft()
            self.hist_blocks.subtract(old)

    def checkpoint(self, blocks, nb, owner, hits):
        if nb <= 0:
            return
        self.clock += 1
        key = blocks[nb - 1]
        n = self.nodes.get(key)
        if n is not None and n.tiers & V:
            n.last_use = self.clock
            return
        while self.n_vram >= self.slots and self.vram_nodes:
            self._evict_vram(for_slot=True)
        if n is None:
            n = Node(key, blocks, nb, owner)
            self._attach(n)
        n.hits = max(n.hits, hits)
        n.last_use = self.clock
        self._vram_in(n)
        self.c["checkpoints"] += 1

    def begin(self, r):
        blocks = r["blocks"]
        P = r["prompt_tokens"]
        reusable = r.get("reusable")
        reusable = self.reusable_of(blocks) if reusable is None else min(reusable, len(blocks) * BLOCK)
        reusable = min(reusable // BLOCK * BLOCK, (P - 1) // BLOCK * BLOCK)
        hit = self.lookup(blocks, P)
        cached = 0
        seq_hits = 0
        self.clock += 1
        if hit is not None:
            if hit.tiers & V:
                self.c["vram_hits"] += 1
            elif hit.tiers & R:
                self.c["ram_restored"] += 1
                self._ram_out(hit, detach=False)
                self._vram_in(hit)
            else:
                self.c["ssd_restored"] += 1
                self._ssd_out(hit, detach=False)
                self._vram_in(hit)
            hit.hits += 1
            hit.last_use = self.clock
            cached = hit.depth
            seq_hits = hit.hits if cached > KEEP_SHORT else 0
        nb = len(blocks)
        self._vref(blocks, nb, +1)  # (the sequence holds its pages first: evicting a checkpoint it shares frees nothing)
        self._vram_room(0)
        self.running[r["rid"]] = (blocks, nb, r["agent"], seq_hits, P)
        # the planned checkpoints of this prefill
        cb = cached // BLOCK
        at = set()
        if reusable // BLOCK > cb:
            at.add(reusable // BLOCK)
        top = (P - 1) // BLOCK
        k = (top // INTERVAL_BLOCKS) * INTERVAL_BLOCKS
        for _ in range(4):
            if k <= cb:
                break
            at.add(k)
            k -= INTERVAL_BLOCKS
        if min(nb, top) > cb:
            at.add(min(nb, top))
        for k in sorted(at):
            if 0 < k <= nb:
                self.checkpoint(blocks, k, r["agent"], seq_hits)
        # metrics
        c = self.c
        c["requests"] += 1
        c["prompt_tokens"] += P
        c["cached_tokens"] += cached
        c["reusable_tokens"] += reusable
        c["cached_of_reusable"] += min(cached, reusable)
        c["recomputed_tokens"] += P - cached
        c["wasted_prefill_us"] += int(1e6 * cp.prefill_seconds(self.p, min(cached, reusable), reusable))
        if P > KEEP_SHORT:
            c["long"] += 1
            c["long_zero"] += cached == 0
            if reusable > 0:
                c["long_reusable"] += 1
                c["long_reusable_zero"] += cached == 0

    def done(self, d):
        run = self.running.pop(d["rid"], None)
        if run is None:
            return
        blocks, nb, agent, seq_hits, P = run
        allb = blocks + list(d.get("end_blocks") or [])
        extra = len(allb) - nb
        if extra > 0:
            self._vref(allb[nb:], extra, +1)
            self._vram_room(0)
        if d.get("end_blocks"):
            self.checkpoint(allb, len(allb), agent, seq_hits)
        self._vref(allb, len(allb), -1)
        self._history_add(allb)

    def run(self, events):
        for ts, kind, rec in events:
            self.now = ts
            (self.begin if kind == 0 else self.done)(rec)
        # requests that never ended: release their pages
        for rid in list(self.running):
            blocks, nb, *_ = self.running.pop(rid)
            self._vref(blocks, nb, -1)
        return self.report()

    def report(self):
        c = self.c
        n = max(1, c["requests"])
        out = {"policy": self.policy, "requests": c["requests"],
               "missed_share": (c["reusable_tokens"] - c["cached_of_reusable"]) / c["reusable_tokens"] if c["reusable_tokens"] else None,
               "hit_share": c["cached_tokens"] / c["prompt_tokens"] if c["prompt_tokens"] else None,
               "recomputed_tokens": c["recomputed_tokens"], "reusable_tokens": c["reusable_tokens"], "cached_tokens": c["cached_tokens"],
               "long_zero_hit_share": c["long_zero"] / c["long"] if c["long"] else None,
               "long_reusable_zero_hit_share": c["long_reusable_zero"] / c["long_reusable"] if c["long_reusable"] else None,
               "vram_hits": c["vram_hits"], "ram_restored": c["ram_restored"], "ssd_restored": c["ssd_restored"],
               "spilled": c["spilled"], "subsumed": c["subsumed"], "ssd_written": c["ssd_written"], "ram_dropped": c["ram_dropped"],
               "ssd_evictions": c["ssd_evictions"], "ssd_not_admitted": c["ssd_not_admitted"],
               "vram_evictions_per_request": c["vram_evictions"] / n, "spills_per_request": c["spilled"] / n,
               "state_bytes_spilled_gib": c["spilled"] * self.state / GIB,
               "ram_reuse_rate": c["ram_restored"] / c["spilled"] if c["spilled"] else None,
               "ssd_reuse_rate": c["ssd_restored"] / c["ssd_written"] if c["ssd_written"] else None,
               "wasted_prefill_s": c["wasted_prefill_us"] / 1e6, "kv_overflow": c["kv_overflow"],
               "ram_bytes_end": self.ram_bytes(), "ssd_bytes_end": self.ssd_bytes()}
        if self.policy in ("smart", "ideal"):
            out["smart"] = {k: v for k, v in c.items() if k.startswith("smart_")}
            out["watermarks"] = {"vram": self.gd[V].L, "ram": self.gd[R].L, "ssd": self.gd[S].L}
        return out


def _sid(n):
    return n.sid


def _neg(o):
    return tuple(-ord(ch) for ch in o) if isinstance(o, str) else -o


def load(paths):
    """The events of trace files in time order: (ts, 0, req) and (ts, 1, done) (a done without its req is ignored)."""
    reqs, dones = {}, {}
    for p in paths:
        with trace_io.open_lines(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("t") == "req" and r.get("blocks") is not None:
                    reqs[r["rid"]] = r
                elif r.get("t") == "done":
                    dones[r["rid"]] = r
    ev = [(r["ts"], 0, r) for r in reqs.values()] + [(d["ts"], 1, d) for rid, d in dones.items() if rid in reqs]
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def simulate(events, policy, **kw):
    return Sim(policy, **kw).run(events)


def render(rows):
    cols = (("policy", "{}"), ("missed_share", "{:.2%}"), ("hit_share", "{:.2%}"), ("recomputed_tokens", "{:,}"),
            ("long_zero_hit_share", "{:.1%}"), ("long_reusable_zero_hit_share", "{:.1%}"), ("ram_restored", "{}"), ("ssd_restored", "{}"),
            ("spilled", "{}"), ("ssd_written", "{}"), ("vram_evictions_per_request", "{:.2f}"), ("spills_per_request", "{:.2f}"),
            ("ram_reuse_rate", "{:.1%}"), ("ssd_reuse_rate", "{:.1%}"), ("wasted_prefill_s", "{:,.0f}"))
    head = ["policy", "missed", "hit", "recomputed", "zero>16K", "zero>16K(reusable)", "RAM rest", "SSD rest", "spilled",
            "SSD writes", "VRAM ev/req", "spills/req", "RAM reuse", "SSD reuse", "wasted s"]
    lines = [head]
    for r in rows:
        lines.append([("-" if r.get(k) is None else f.format(r[k])) for k, f in cols])
    w = [max(len(l[i]) for l in lines) for i in range(len(head))]
    return "\n".join("  ".join(x.rjust(w[i]) for i, x in enumerate(l)) for l in lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--policy", default="legacy,smart", help=f"comma-separated: {', '.join(POLICIES)}; legacy-nofair: legacy with --fair-agents 0")
    ap.add_argument("--ram-gib", type=float, default=64.0)
    ap.add_argument("--ssd-gib", type=float, default=200.0)
    ap.add_argument("--vram-pages", type=int, default=8207, help="KV pool pages of 64 tokens (the snap's kv_pool_pages)")
    ap.add_argument("--state-mib", type=float, default=190.0)
    ap.add_argument("--kv-kib-per-token", type=float, default=18.0)
    ap.add_argument("--checkpoint-slots", type=int, default=43)
    ap.add_argument("--fair-agents", type=int, default=1)
    ap.add_argument("--cold-first", type=int, default=1)
    ap.add_argument("--marconi-alpha", type=float, default=1.0)
    ap.add_argument("--smart-admit", choices=("watermark", "victim", "always"), default="watermark",
                    help="smart's SSD admission: above the SSD's watermark (implemented), above the current victim, always (sim only)")
    ap.add_argument("--smart-variant", default="", help="sim-only ablations of smart, comma-separated: nohits, nobytes, nojunction, "
                    "nosup (no superseded factor), noleaf (no leaf-first)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        events = load(a.files)
    except trace_io.TraceSourceError as e:
        sys.exit(f"cache_sim: {e}")
    params = cp.SmartParams.from_env(os.environ)
    rows = []
    for pol in a.policy.split(","):
        pol = pol.strip()
        fair = a.fair_agents
        if pol == "legacy-nofair":
            pol, fair = "legacy", 0
        r = simulate(events, pol, ram_gib=a.ram_gib, ssd_gib=a.ssd_gib, vram_pages=a.vram_pages, state_mib=a.state_mib,
                     kv_kib_per_token=a.kv_kib_per_token, checkpoint_slots=a.checkpoint_slots, fair_agents=bool(fair),
                     cold_first=bool(a.cold_first), params=params, marconi_alpha=a.marconi_alpha,
                     smart_admit=a.smart_admit, smart_variant=[v for v in a.smart_variant.split(",") if v])
        if pol == "legacy" and not fair:
            r["policy"] = "legacy-nofair"
        rows.append(r)
    print(json.dumps(rows, indent=1) if a.json else render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
