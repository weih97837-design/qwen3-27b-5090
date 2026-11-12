"""Engine process (Phase 6): owns libq27 and schedules the requests (docs/architecture.md §10).

The API server (server.py) starts this in its own process and talks to it over a multiprocessing pipe,
so a crash of the engine (a CUDA error) leaves the server up: it answers 503 and starts a new engine.

Scheduling: up to MAX_SEQS requests run at once (the rest wait in arrival order). They take turns on the
GPU: the engine parks the running sequence's state in a device area and resumes another one (~0.5 ms,
Model::seq_swap), so a turn is a prefill chunk or a few decode steps. Prefills go first (shortest rest
first: a short warm turn of an agent is not stuck behind a long cold prompt), but each second of prefill
gives the decoding requests DECODE_SHARE seconds, so their output does not stop during a long prompt.
Decoding requests take turns of SCHED_QUANTUM_MS, or with BATCH_DECODE (P6-10, speculative decoding) step
together: one batched step drafts and verifies all of them (Model::spec_step_lanes; each gets the tokens
it would get alone), so the GEMMs read the weights once for up to MAX_SEQS requests. When the KV pool runs
out, the latest request (the one that ran out, if it is the latest) is suspended to the prefix cache (its
state becomes a checkpoint, its pages go back) and goes back to the head of the queue.

Admission (P7-14): the engine has room for MAX_SEQS requests (parking areas, lanes); how many run is decided
while it runs. A waiting request starts only while fewer than the target (max_seqs, MAX_SEQS at the start,
set by /admin/concurrency) run, the KV pool has room for it (ADMIT_KV: free pages and pages only checkpoints
hold cover the rest of every running prompt, KV_RESERVE_TOKENS of output for each request, and one more
parking area), and one more request would not slow each one below MIN_SEQ_TPS (measured tok/s per request
for each number of requests decoding together). With nothing running, the first waiting request always
starts. A request waiting for room rather than one suspended later keeps its turn and its KV.

Delay hits (SPILL_ASYNC_PROMOTE=1, off by default; docs/tiered-cache.md): when a waiting request's prompt matches a
checkpoint the tiered cache holds in host RAM or on the SSD, the engine restores it in the background
(Engine.spill_restore_begin) and the request waits in state "restoring" (keeping its place in the queue) while the
requests behind it are admitted and the engine goes on. Each loop iteration polls the restores; a finished one makes the
request admissible, and its prefix_begin hits the restored checkpoint. A restore is started only when loading is clearly
faster than computing the prompt again (RESTORE_MARGIN); after RESTORE_TIMEOUT_S the request stops waiting.

Messages (pickled tuples). In: ("submit", rid, request), ("cancel", rid), ("config", changes, n), ("stop",).
Out: ("ready", info), ("start", rid, info), ("tok", rid, [ids][, logprobs: per id (logprob, [(id, logprob)])]), ("done", rid, finish, info),
("prefilled", rid, info: the prompt side of "done"'s info after a prefill ends, for a request a stop string ends),
("error", rid, message[, HTTP status: 400 for an image that cannot be decoded]), ("metrics", dict), ("config", settings, n: the change answered, or None),
("memory", the pools: once after "ready"), ("stats", snapshot: the requests, KV pool and device memory, every STATS_INTERVAL s while busy),
("event", rid, kind, text: an admission wait, a preemption, an engine error).
"""

import array
import collections
import ctypes
import os
import random
import sys
import time
import traceback

PREFILL, DECODE = "prefill", "decode"
PAGE = 64  # tokens of a KV page (kernels.h kPageTokens)
ADMIT_RECHECK = 0.05  # s between checks of a request kept waiting by the admission
RATE_WINDOW = 10.0    # s of decoding the measured tok/s per request averages over
STATS_INTERVAL = 0.5  # s between stats snapshots (the stats UI) while requests run
STATS_IDLE = 5.0      # s between them while idle

# ---- delay-hit scheduling (SPILL_ASYNC_PROMOTE): when loading a spilled checkpoint beats computing it again
RESTORE_TIMEOUT_S = 30.0   # a request waits at most this long for its checkpoint, then it is admitted (prefix_begin falls back)
RESTORE_MARGIN = 0.8       # restore only if the estimated load time is below this fraction of the recompute time
RESTORE_OVERHEAD_S = 0.05  # fixed cost of a restore: thread start, slot and page allocation, the polls that notice the end
# Load speeds in bytes/s until measured (a finished restore: bytes / seconds, begin to visible, smoothed). A RAM entry is
# a host -> device copy (pinned memory is about twice as fast as pageable); an SSD entry is a file read plus that copy.
RESTORE_BW_RAM_PINNED, RESTORE_BW_RAM_PAGEABLE, RESTORE_BW_SSD = 12e9, 6e9, 2e9
RESTORE_BW_CLAMP = (0.25, 4.0)  # a measurement counts within these multiples of the default (a stalled loop must not lock restores out)
RESTORE_BW_SMOOTH = 0.3         # weight of a new measurement
# Prompt tokens/s until a prefill chunk was measured: docs/architecture.md measures 11,295 tok/s at 19K and 5,875 tok/s at
# 194K context; this is in between and leans toward loading.
PREFILL_TPS_DEFAULT = 8000.0
PREFILL_TPS_SMOOTH = 0.2        # weight of a new prefill chunk's speed
RESTORE_POLL_S = 0.005          # the loop's wait for a message while only restores are pending


def step_ms(n):
    """Time of a batched step for n requests (short contexts, gdnfp4 + KV nvfp4, P7-11: 12.40 / 13.76 / 16.39 /
    21.99 ms for 1 / 2 / 4 / 8): scales a measured rate per request to a number of requests not measured yet."""
    return 11.1 + 1.36 * n


class ImageError(Exception):
    """An image of the request cannot be decoded (the client's error: 400)."""


def log(msg):
    print(f"[sched] {msg}", file=sys.stderr, flush=True)


def _env_flag(name):
    """An engine switch as capi.cc reads it: atoi(value) != 0."""
    try:
        return int(os.environ.get(name) or 0) != 0
    except ValueError:
        return False


def _env_float(name):
    try:
        return float(os.environ.get(name))
    except (TypeError, ValueError):
        return None


class Seq:
    """A request: its prompt, sampling settings, progress and place in the engine."""

    def __init__(self, rid, req, now):
        self.rid = rid
        self.prompt = list(req["ids"])
        self.max_new = req["max_new"]
        self.temperature, self.top_k, self.top_p = req["temperature"], req["top_k"], req["top_p"]
        self.min_p = req.get("min_p", 0.0)
        self.presence, self.frequency = req.get("presence_penalty", 0.0), req.get("frequency_penalty", 0.0)
        self.logprobs = req.get("logprobs", -1)  # top logprobs per token (0..20; -1: none)
        # logit_bias ({token id: value}), min_tokens (the end-of-sequence ids are banned until that many tokens are
        # out) and ignore_eos (they are ordinary tokens): the engine's logit bias (Engine.bias_set) and its steps.
        self.bias = {int(k): float(v) for k, v in (req.get("logit_bias") or {}).items()}
        self.min_tokens, self.ignore_eos = req.get("min_tokens", 0), bool(req.get("ignore_eos"))
        self.rng = random.Random(req.get("seed"))
        self.images = req.get("images")  # (spans [(start, gh, gw)], [(encoded image, resized size, norm)]) or None
        self.decoding = {}       # image index -> Future of its pixels (imageproc.rgb), started on arrival
        self.arrival = now
        self.out = []            # generated tokens (EOS excluded)
        self.eng_ids = self.prompt  # what the engine runs (prompt + earlier output after a preemption)
        # Forced output (tool_choice): (token id, ids) continues the output with ids after that token (the end of the
        # reasoning); with no token id, the output starts with them. unsent: forced tokens the server has not got yet
        # (they go out with the next generated ones).
        self.force, self.unsent = req.get("force"), []
        if self.force is not None and self.force[0] is None:
            self.out, self.force = list(self.force[1]), None
            self.eng_ids, self.unsent = self.prompt + self.out, list(self.out)
        # Structured output: the llguidance grammar, and the token after which it applies (the end of the
        # reasoning; None: from the start). matcher: made on arrival (Scheduler.submit); gm: the same once the grammar
        # applies (every output token after that goes through it, and the engine masks the logits by it).
        self.grammar, self.grammar_after = req.get("grammar"), req.get("grammar_after")
        self.matcher = self.gm = None
        self.started = False     # prefix_begin done (the engine holds its state: running or parked)
        self.area = None         # parking area while parked
        self.phase = PREFILL
        self.fed = 0             # tokens of eng_ids the engine has run
        self.est_cached = 0      # expected cache hit (to order prefills before they start)
        self.at = []             # checkpoint positions of the prompt
        self.cached = None       # tokens restored by the first prefix_begin
        self.reusable = 0
        self.last_run = 0.0
        self.first_token = None
        self.preempted = 0
        self.dead = False
        self.share_with, self.share_len = None, 0  # an earlier request still computing a prefix this one shares
        self.held = False        # kept waiting by the admission (counted once)
        self._bytes = None
        self.begun = None        # time of the first prefix_begin
        self.restoring = None    # the restore record this request waits for (Scheduler.maybe_restore), or None
        self.restore_checked = False  # spill_match asked for this stay in the queue
        self.restore_id = None   # the spilled entry whose restore it waited for (engine.spill_release after begin)
        self.restore_pos = 0     # tokens of its prompt that entry covers (their pages are counted in its need, not as room)
        self.no_sync = False     # recompute chosen over loading: prefix_begin must not promote a spilled match itself
        self.computed = 0        # prompt tokens prefilled (not restored from the cache)
        self.prefill_s = self.decode_s = 0.0  # time in its prefill turns / decode turns (a batched turn counts for each)
        self.steps = 0           # decode steps (speculative: verify steps)
        self.drafted = self.draft_accepted = 0  # draft tokens verified, and those accepted (as llama.cpp counts them)
        self.marks = [0, 0, 0]   # end of the first message, starts of the last message and the generation prompt

    @property
    def biased(self):
        return bool(self.bias) or self.min_tokens > 0

    def rest(self):
        return len(self.eng_ids) - (self.fed if self.started else self.est_cached)

    def ids_bytes(self):
        if self._bytes is None or len(self._bytes) != 4 * len(self.eng_ids):
            self._bytes = array.array("i", self.eng_ids).tobytes()
        return self._bytes


class Scheduler:
    # (class defaults: a Scheduler made without __init__, as some tests do, has no tiers and no backup state)
    tiers = None
    last_active = 0.0
    backup_done = False
    pin_full = False  # the engine's pinned spill pool is full / off (SPILL_PINNED_GIB; poll)
    # Delay hits (SPILL_ASYNC_PROMOTE): off unless the engine was started with it and has the calls.
    async_promote = False
    restores = None       # spilled entry id -> record {"t0", "bytes", "on_disk", "seqs", "late"} while it loads
    restore_bw = None     # kind ("ram" / "ssd") -> measured bytes/s
    restore_pinned = False
    prefill_tps = None    # measured prompt tokens/s (smoothed)
    restore_released = 0  # requests a poll released (count, for the logs and tests)
    restore_completed = 0  # restores a poll saw end (any outcome; _guarded retries a KvFull turn when it grew)
    last_restore_poll = 0.0
    last_poll_error = 0.0  # (monotonic) when a failing restore poll was last logged

    def __init__(self, conn, cfg):
        from q27lib import Engine, KvFull

        self.conn, self.cfg, self.KvFull = conn, cfg, KvFull
        for k, v in cfg.get("env", {}).items():  # (the start settings, launch.py; None: unset)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.engine = Engine(cfg["model"], max_tokens=cfg["max_tokens"], prefill_chunk=cfg["prefill_chunk"],
                             kv_dtype=cfg["kv_dtype"], draft=cfg.get("draft"))
        from tiercache import TierPolicy  # (tiered cache: SPILL_RAM_GIB; the frontend dir is on sys.path)
        self.tiers = TierPolicy.from_env(self.engine, log)
        if self.tiers is not None:
            self.tiers.start()
        self.spec = cfg.get("draft") is not None and self.engine.has_draft
        self.restores, self.restore_bw = {}, {}
        self.async_promote = (self.tiers is not None and getattr(self.engine, "has_spill_async", False)
                              and _env_flag("SPILL_ASYNC_PROMOTE"))
        self.restore_pinned = (_env_float("SPILL_PINNED_GIB") or 0.0) > 0
        if self.async_promote:
            log("delay hits: spilled checkpoints are restored in the background (SPILL_ASYNC_PROMOTE=1)")
        # Draft tokens of a verify step: a chain of 7. Several paths verify a tree: not reported.
        self.draft_block = 7 if int(os.environ.get("Q27_SPEC_PATHS") or 1) <= 1 else 0
        self.engine_seqs = max(1, min(cfg["max_seqs"], self.engine.park_slots + 1))
        self.max_seqs = self.engine_seqs  # the target number of running requests (<= engine_seqs)
        self.admit_kv, self.kv_reserve, self.min_tps = True, 4096, 0.0
        self.configure({k: cfg[k] for k in ("target_seqs", "admit_kv", "kv_reserve", "min_seq_tps") if k in cfg})
        self.rates = {}          # n -> [tokens per request, seconds] with n requests decoding together (decayed sums)
        self.blocked = None      # why the first waiting request waits ("kv", "speed") or None
        self.held_until = (None, 0.0)  # (request, time): skip checking it again until then
        self.batch = self.spec and cfg.get("batch_decode", True) and self.engine.max_lanes > 1
        self.chunk = cfg["prefill_chunk"]
        self.share = cfg["decode_share"]
        self.quantum = cfg["quantum_ms"] / 1000.0
        self.eos = set(cfg["eos"])
        self.im_start = cfg["im_start"]
        self.waiting = collections.deque()
        self.alive = []          # admitted, unfinished
        self.slot_lanes = {}     # rid -> UI lane (1-based) in the last stats snapshot (slot_rows keeps lanes still)
        self.by_id = {}
        self.cur = None          # the sequence in the engine's working state (may be finished)
        self.credit = 0.0        # decode time owed to decoding requests by the prefills (DECODE_SHARE)
        self.history = collections.deque(maxlen=64)  # recent sequences (prompt + output) as int32 bytes
        self.decoder = None      # threads that decode images (prefetch)
        self.m = collections.Counter()
        self.last_metrics = self.last_stats = 0.0
        self.cache_full = False  # every state slot of the prefix cache allocated (poll)
        self.lltok = None        # llguidance's tokenizer (structured output; made with the first grammar)
        self.last_active = time.monotonic()  # the idle backup (touch, backup_due)

    # ------------------------------------------------------------ messages

    def send(self, *msg):
        self.conn.send(msg)

    def poll(self, block, wait=STATS_IDLE):
        # Idle: allocate the prefix cache's state slots one by one (not at startup, so the engine is ready sooner),
        # then wake up now and then for a stats snapshot (wait seconds at most: less when a backup is due).
        if block and not self.cache_full:
            self.cache_full = self.engine.cache_grow(1) == 0
            block = self.cache_full
        # Same for the pinned pool of the tiered cache (SPILL_PINNED_GIB, off by default): one slab per idle pass, since
        # pinning is slow (other CUDA calls wait) and must never happen under load.
        if block and not self.pin_full:
            self.pin_full = self.grow_pinned() == 0
            block = self.pin_full
        if block and wait > 0:
            self.conn.poll(RESTORE_POLL_S if self.restores else wait)  # (restores pending: never a long wait)
        while self.conn.poll(0):
            msg = self.conn.recv()
            if msg[0] == "submit":
                self.submit(msg[1], msg[2])
            elif msg[0] == "cancel":
                self.cancel(msg[1])
            elif msg[0] == "config":
                self.configure(msg[1])
                self.send("config", self.settings(), msg[2])
            elif msg[0] == "stop":
                self.flush_on_stop()
                raise SystemExit(0)

    def grow_pinned(self):
        """Pins one more slab of the engine's spill pool; 0 when it is full or off, the engine predates it, or it failed."""
        if self.tiers is None or not hasattr(self.engine, "spill_grow_pinned"):
            return 0
        try:
            return self.engine.spill_grow_pinned(1)
        except Exception as e:  # noqa: BLE001 - the pool is an optimisation: the entries stay pageable
            log(f"pinned spill pool: {e}")
            return 0

    def touch(self):
        """Activity (a request in or out, a turn): the idle clock restarts and the idle backup starts over."""
        self.last_active = time.monotonic()
        self.backup_done = False

    def backup_ready(self):
        """Whether the idle backup may run at all now: tiers on, not exhausted, the engine can back up, and idle."""
        t = self.tiers
        return (t is not None and t.backup_idle_s > 0 and not self.backup_done and not self.alive and not self.waiting
                and hasattr(self.engine, "spill_backup_one"))

    def backup_due(self):
        return self.backup_ready() and time.monotonic() - self.last_active >= self.tiers.backup_idle_s

    def backup_wait(self):
        """Seconds the idle wait may last before a backup is due (STATS_IDLE when none is coming)."""
        if not self.backup_ready():
            return STATS_IDLE
        return min(STATS_IDLE, max(0.0, self.tiers.backup_idle_s - (time.monotonic() - self.last_active)))

    def backup_one(self):
        """One idle backup: the least recently used VRAM checkpoint without a RAM copy is copied to RAM, and stays in
        VRAM. False when there is nothing left or it failed; the backup then waits for the next activity."""
        try:
            if self.engine.spill_backup_one():
                return True
            log("idle backup: every checkpoint is backed up")
        except Exception as e:  # noqa: BLE001 - the backup must not stop the engine
            log(f"idle backup: {e}")
        self.backup_done = True
        return False

    def flush_on_stop(self):
        """On a stop (idle unload, restart, shutdown): the VRAM checkpoints without a RAM copy are backed up, then the
        RAM entries are written to SSD, within SPILL_FLUSH_ON_STOP_S (at most half of it for the backups, which take
        0.3-1 s each). Only with an SSD directory: nothing else survives the stop. Never raises."""
        t = self.tiers
        if t is None or t.directory is None or t.flush_on_stop_s <= 0:
            return
        t0 = time.monotonic()
        try:
            if hasattr(self.engine, "spill_backup_one"):
                backed = 0
                while time.monotonic() - t0 < t.flush_on_stop_s / 2 and self.engine.spill_backup_one():
                    backed += 1
                log(f"stop: backed up {backed} VRAM checkpoints")
            t.flush(max(0.0, t.flush_on_stop_s - (time.monotonic() - t0)))
        except Exception as e:  # noqa: BLE001 - the stop must go on
            log(f"stop flush: {e}")

    def submit(self, rid, req):
        self.touch()
        s = Seq(rid, req, time.time())
        if s.grammar is not None:
            try:
                s.matcher = self.matcher(s.grammar)
            except ValueError as e:
                self.send("error", rid, str(e), 400)
                return
            s.gm = s.matcher if s.grammar_after is None else None
            self.m["q27:structured_requests_total"] += 1
        s.reusable = self.reusable(s.prompt)
        # Requests that arrive together (an agent and its sub-agents) often share a long prefix (the system
        # prompt and tools) that no checkpoint holds yet: the later one waits until the earlier one has run
        # past it (with a checkpoint planned there) instead of computing it again.
        p = array.array("i", s.prompt).tobytes()
        for o in list(self.alive) + list(self.waiting):
            n = lcp_bytes(p, array.array("i", o.eng_ids).tobytes())
            n = min(n, len(s.prompt) - 1) // 64 * 64
            if n > max(s.reusable, s.share_len) + 256 and not (o.started and o.fed >= n):
                s.share_with, s.share_len = o, n
        if s.share_with is not None:
            o = s.share_with
            if not o.started:
                o.at = sorted(set(o.at) | {s.share_len})
            s.reusable = s.share_len
        s.at = self.checkpoints(s.prompt, s.reusable)
        # (stats: where the first message ends, the last message and the generation prompt start)
        starts = [i for i, t in enumerate(s.prompt) if t == self.im_start]
        s.marks = (starts[1:2] or [0]) + (starts[-2:-1] or [0]) + (starts[-1:] or [0])
        s.est_cached = s.reusable
        if s.images:
            self.prefetch(s)
        self.by_id[rid] = s
        self.waiting.append(s)
        self.m["q27:requests_total"] += 1

    def matcher(self, grammar):
        """An llguidance matcher of a grammar (structured output); ValueError when it cannot be used."""
        import llguidance

        if self.lltok is None:
            t0 = time.time()
            words = self.engine.mask_row(0)[1] * 8  # (the engine's mask rows: one bit per token of its vocabulary)
            self.lltok = llguidance.LLTokenizer(self.cfg["tokenizer"], n_vocab=words, eos_token=sorted(self.eos))
            log(f"grammar tokenizer ready ({time.time() - t0:.2f} s)")
        m = llguidance.LLMatcher(self.lltok, grammar, log_level=0)
        if m.is_error():
            raise ValueError(f"the output format cannot be used: {m.get_error()}")
        return m

    def prefetch(self, s):
        """Starts decoding the images that lie past the reusable prefix, which begin() will likely encode, on
        threads while the request waits (images inside the prefix are decoded only if the cache misses them)."""
        import imageproc

        if self.decoder is None:
            from concurrent.futures import ThreadPoolExecutor

            self.decoder = ThreadPoolExecutor(2, thread_name_prefix="image")
        for i, ((start, gh, gw), (raw, size, _)) in enumerate(zip(*s.images)):
            if start + gh * gw > s.reusable:
                s.decoding[i] = self.decoder.submit(imageproc.rgb, raw, size)

    def drop_decoding(self, s):
        for f in s.decoding.values():
            f.cancel()
        s.decoding.clear()

    def cancel(self, rid):
        s = self.by_id.get(rid)
        if s is None:
            return
        if s in self.waiting:
            self.waiting.remove(s)
        self.finish(s, "cancelled", save=False)

    def configure(self, c):
        """Admission settings (cfg at the start, then /admin/concurrency); values are clamped to valid ones."""
        if c.get("target_seqs") is not None:
            self.max_seqs = max(1, min(int(c["target_seqs"]), self.engine_seqs))
        if c.get("admit_kv") is not None:
            self.admit_kv = bool(c["admit_kv"])
        if c.get("kv_reserve") is not None:
            self.kv_reserve = max(0, int(c["kv_reserve"]))
        if c.get("min_seq_tps") is not None:
            self.min_tps = max(0.0, float(c["min_seq_tps"]))

    def settings(self):
        return {"target_seqs": self.max_seqs, "engine_seqs": self.engine_seqs, "admit_kv": self.admit_kv,
                "kv_reserve": self.kv_reserve, "min_seq_tps": self.min_tps,
                "rates": {n: round(self.rate(n), 1) for n in sorted(self.rates)},
                "running": len(self.alive), "waiting": len(self.waiting), "blocked": self.blocked}

    # ------------------------------------------------------------ admission (P7-14)

    def admit(self, s):
        """Whether the first waiting request may start now (below the target number of requests)."""
        held, until = self.held_until
        if held is s and time.time() < until:
            return False
        reason = None
        if self.alive:
            # (a suspended request waits for room even without ADMIT_KV: let in at once, it would run out again)
            # (a request whose restore has finished still holds its restored prefix: that is no extra need)
            cached = s.restore_pos if s.restore_id is not None else 0
            if (self.admit_kv or s.preempted) and not self.kv_room(s, cached=cached):
                reason = "kv"
            elif self.min_tps > 0 and self.predicted_rate(len(self.alive) + 1) < self.min_tps:
                reason = "speed"
        if reason and not s.held:
            s.held = True
            self.m[f'q27:admission_waits_total{{reason="{reason}"}}'] += 1
            log(f"{s.rid}: waits for room ({reason}), {len(self.alive)} running")
            self.send("event", s.rid, "wait", f"waits for room ({reason}), {len(self.alive)} running")
        self.blocked = reason
        self.held_until = (s, time.time() + ADMIT_RECHECK) if reason else (None, 0.0)
        return reason is None

    def head(self):
        """The waiting request the admission looks at: the first one that is not waiting for a restore (the one admit()
        keeps its held_until / blocked bookkeeping for). Without delay hits that is waiting[0]."""
        return next((q for q in self.waiting if getattr(q, "restoring", None) is None), None)

    def admit_waiting(self):
        """Moves the waiting requests that may start into alive, in arrival order. Without SPILL_ASYNC_PROMOTE only the
        head is ever looked at. With it, a request whose checkpoint is loading (state "restoring") is skipped: the
        next ones are admitted meanwhile (Strata's delay hit), and the first one the admission does not let in stops
        the pass, as before."""
        if not self.async_promote:
            while self.waiting and len(self.alive) < self.max_seqs and self.admit(self.waiting[0]):
                self.alive.append(self.waiting.popleft())
            return
        if self.restores is None:
            self.restores, self.restore_bw = {}, {}
        self.poll_restores()
        for s in list(self.waiting):
            if len(self.alive) >= self.max_seqs:
                break
            if s.restoring is not None:
                continue
            if self.maybe_restore(s):
                continue
            if not self.admit(s):
                break
            self.waiting.remove(s)
            self.alive.append(s)

    # ------------------------------------------------------------ delay hits (SPILL_ASYNC_PROMOTE)

    def bandwidth(self, on_disk):
        """Bytes/s a restore is expected to load at: measured (within RESTORE_BW_CLAMP of the default), else the default."""
        default = RESTORE_BW_SSD if on_disk else RESTORE_BW_RAM_PINNED if self.restore_pinned else RESTORE_BW_RAM_PAGEABLE
        got = (self.restore_bw or {}).get("ssd" if on_disk else "ram")
        if got is None:
            return default
        lo, hi = RESTORE_BW_CLAMP
        return min(max(got, default * lo), default * hi)

    def est_load_s(self, match):
        return match["bytes"] / self.bandwidth(match["on_disk"]) + RESTORE_OVERHEAD_S

    def est_recompute_s(self, tokens):
        return tokens / (self.prefill_tps or PREFILL_TPS_DEFAULT)

    def measure_prefill(self, tokens, dt):
        """A prefill chunk of `tokens` took dt seconds in the engine."""
        if tokens < 256 or dt <= 0:
            return
        tps = tokens / dt
        self.prefill_tps = tps if self.prefill_tps is None else (
            (1 - PREFILL_TPS_SMOOTH) * self.prefill_tps + PREFILL_TPS_SMOOTH * tps)

    def worth_loading(self, match):
        """Load only if it is clearly quicker than computing the matched prefix again."""
        return self.est_load_s(match) < RESTORE_MARGIN * self.est_recompute_s(match["pos"])

    def maybe_restore(self, s):
        """Before s is admitted: starts the background restore of a spilled checkpoint that matches its prompt and puts
        s in state "restoring" (True). False: nothing to wait for (no match, not worth it, or it could not start); s is
        admitted as usual, and prefix_begin does what it did before (a match judged not worth loading is not promoted)."""
        if s.restore_checked:
            return False
        s.restore_checked = True
        try:
            m = self.engine.spill_match(s.eng_ids, self.spec)
            if m is None:
                return False
            rec = self.restores.get(m["id"])
            if rec is not None:  # (another request already waits for this entry; a late one is still loading: not waited for)
                if rec["late"]:
                    return False
                rec["seqs"].append(s)
                s.restoring, s.restore_id, s.restore_pos = rec, rec["id"], m["pos"]
                return True
            if not self.worth_loading(m):
                s.no_sync = True
                log(f"{s.rid}: spilled checkpoint of {m['pos']} tokens not loaded (load ~{self.est_load_s(m):.2f} s, "
                    f"recompute ~{self.est_recompute_s(m['pos']):.2f} s)")
                return False
            # (the restored pages must fit next to what the running requests still need: else nothing is started, the
            # request is admitted as usual and prefix_begin recomputes or promotes as it did without delay hits)
            if not self.kv_room(s, cached=m["pos"], extra=-(-m["pos"] // PAGE)):
                log(f"{s.rid}: spilled checkpoint of {m['pos']} tokens not loaded (the KV pool has no room for it)")
                return False
            if not self.engine.spill_restore_begin(m["id"]):
                return False
        except Exception as e:  # noqa: BLE001 - delay hits are an optimisation: the request takes the synchronous path
            log(f"{s.rid}: spilled checkpoint lookup failed ({e})")
            return False
        rec = {"id": m["id"], "t0": time.time(), "bytes": m["bytes"], "on_disk": m["on_disk"], "pos": m["pos"],
               "seqs": [s], "late": False}
        self.restores[m["id"]] = rec
        s.restoring, s.restore_id, s.restore_pos = rec, m["id"], m["pos"]
        self.m["q27:restores_started_total"] += 1
        log(f"{s.rid}: loading a spilled checkpoint of {m['pos']} tokens ({m['bytes'] / 1e9:.2f} GB from "
            f"{'SSD' if m['on_disk'] else 'RAM'}, ~{self.est_load_s(m):.2f} s vs ~{self.est_recompute_s(m['pos']):.2f} s to compute)")
        self.send("event", s.rid, "restore", f"loading a cached prefix of {m['pos']} tokens in the background")
        return True

    def release(self, rec, why):
        n = 0
        for q in rec["seqs"]:
            if q.restoring is rec:
                q.restoring = None
                n += 1
        self.restore_released += n
        return n

    def engine_release(self, entry_id):
        """Tells the engine nobody waits for the restored checkpoint of `entry_id` any more (it stops being held against
        eviction). An older engine build has no such call; failures are not worth stopping for."""
        if not entry_id or not hasattr(self.engine, "spill_release"):
            return
        try:
            self.engine.spill_release(entry_id)
        except Exception as e:  # noqa: BLE001 - an optimisation hint
            log(f"spill_release: {e}")

    def release_restored(self, s):
        """s no longer needs the checkpoint restored for it (it began, or ended before): released in the engine unless the
        restore is still loading for another request, or another request that shares it has not begun yet."""
        eid = getattr(s, "restore_id", None)
        if eid is None:
            return
        s.restore_id, s.restore_pos = None, 0
        if any(q is not s and not q.dead and getattr(q, "restore_id", None) == eid for q in self.alive + list(self.waiting)):
            return
        self.engine_release(eid)

    def detach_restore(self, s):
        """s stops waiting for its restore (cancelled, finished, preempted). The last one to leave a restore that is still
        loading releases it in the engine, which then does not hold the checkpoint for nobody."""
        rec = getattr(s, "restoring", None)
        if rec is None:
            return
        s.restoring, s.restore_id, s.restore_pos = None, None, 0
        if s in rec["seqs"]:
            rec["seqs"].remove(s)
        if not rec["seqs"] and not rec["late"]:
            rec["late"] = True  # (no new request waits for it either; the engine goes on loading it)
            self.engine_release(rec["id"])

    def poll_restores(self):
        """Advances the engine's restores; requests whose checkpoint is visible (or failed, or took longer than
        RESTORE_TIMEOUT_S) become admissible. Never blocks. Returns how many requests it released. A failing engine poll
        is logged (at most once a second) and tried again by the next call: the restores stay queued, and the timeout
        still releases their requests."""
        if not self.restores:
            return 0
        now = time.time()
        freed = 0
        try:
            done = self.engine.spill_restore_poll()
        except Exception as e:  # noqa: BLE001 - the restores stay; a fatal engine error kills the worker at its next call
            done = []
            if time.monotonic() - self.last_poll_error > 1.0:
                self.last_poll_error = time.monotonic()
                log(f"restore poll failed ({e}); trying again")
        self.restore_completed += len(done)
        for sid, ok in done:
            rec = self.restores.pop(sid, None)
            if rec is None:
                continue
            dt = now - rec["t0"]
            if ok and not rec["late"] and dt > 0:  # (only a restore that worked, and that was waited for, teaches the speed)
                kind = "ssd" if rec["on_disk"] else "ram"
                bw = rec["bytes"] / dt
                old = self.restore_bw.get(kind)
                self.restore_bw[kind] = bw if old is None else (1 - RESTORE_BW_SMOOTH) * old + RESTORE_BW_SMOOTH * bw
            log(f"restore of {rec['pos']} tokens ended after {dt:.2f} s" + ("" if ok else " (failed)"))
            freed += self.release(rec, "done")
        for rec in self.restores.values():
            if not rec["late"] and now - rec["t0"] > RESTORE_TIMEOUT_S:
                rec["late"] = True  # (the engine goes on loading it; nobody waits for it any more)
                self.m["q27:restore_timeouts_total"] += 1
                log(f"restore of {rec['pos']} tokens took over {RESTORE_TIMEOUT_S:.0f} s: its requests start without it")
                freed += self.release(rec, "timeout")
                self.engine_release(rec["id"])  # (its checkpoint must not stay held for nobody)
        return freed

    def restore_due(self):
        """For the decode loops: whether a restore ended and released a request, so the turn should end and let the admission
        run (cheap and false without SPILL_ASYNC_PROMOTE)."""
        if not self.restores:
            return False
        now = time.monotonic()
        if now - self.last_restore_poll < RESTORE_POLL_S:
            return False
        self.last_restore_poll = now
        return self.poll_restores() > 0

    def live_prefix(self, s):
        """Tokens of s's prompt whose KV pages a running request holds already (a shared prefix)."""
        p = s.ids_bytes()
        n = max((lcp_bytes(p, o.ids_bytes()[:4 * o.fed]) for o in self.alive if o.started and o is not s), default=0)
        return n // PAGE * PAGE

    def kv_room(self, s, cached=0, extra=0):
        """Whether the KV pool holds s next to the running requests: the free pages and the pages only
        checkpoints hold (evicted when needed) cover the rest of each prompt, kv_reserve tokens of output of
        each request, and the parking areas of one more request. For a restore (maybe_restore): `cached` tokens of s's
        prompt come from the restored checkpoint, whose `extra` pages are taken from the pool first."""
        st = self.engine.cache_stats()
        room = st["free_pages"] + st["checkpoint_only_pages"]
        parked = sum(1 for o in self.alive if o.area is not None)
        room -= max(0, len(self.alive) - parked) * st["park_pages"]  # (n + 1 requests park n at most)
        need = 0
        for o in self.alive + [s]:
            done = o.fed if o.started else self.live_prefix(o)
            if o is s and cached:
                done = max(done, cached)
            prompt = len(o.eng_ids) - done
            out = min(max(0, o.max_new - len(o.out)), self.kv_reserve)
            need += -(-max(0, prompt) // PAGE) + -(-out // PAGE)
        return need + extra <= room

    def predicted_rate(self, n):
        """tok/s per request with n requests decoding together: measured, or scaled from the largest measured
        number below n by the step time; no limit before anything was measured."""
        if n in self.rates:
            return self.rate(n)
        below = [m for m in self.rates if m < n]
        if not below:
            return float("inf")
        m = max(below)
        return self.rate(m) * step_ms(m) / step_ms(n)

    def rate(self, n):
        tokens, seconds = self.rates[n]
        return tokens / seconds

    def measure(self, n, tokens, dt):
        """A decode turn of n requests: tokens in all, dt seconds. The sums decay with a time constant of about
        RATE_WINDOW seconds of decoding with n requests, so a slow turn (a graph capture) weighs by its time."""
        if dt <= 0:
            return
        k = max(0.0, 1.0 - dt / RATE_WINDOW)
        t, sec = self.rates.get(n, (0.0, 0.0))
        self.rates[n] = [t * k + tokens / n, sec * k + dt]

    # ------------------------------------------------------------ prefix cache helpers

    def reusable(self, prompt):
        """Tokens of the prompt an ideal prefix cache could reuse: the longest common prefix with a recent
        prompt + output (the rest is new to the server). cached / reusable measures the cache itself."""
        p = array.array("i", prompt).tobytes()
        best = 0
        for h in self.history:
            lo, hi = best, min(len(p), len(h)) // 4  # binary search on memcmp'd slices
            if p[:4 * lo] != h[:4 * lo]:
                continue
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if p[4 * lo:4 * mid] == h[4 * lo:4 * mid] else (lo, mid - 1)
            best = lo
        return max(0, min(best, len(prompt) - 1))  # (the engine always runs the last prompt token)

    def checkpoints(self, prompt, reusable):
        """Checkpoints of a prompt (docs/architecture.md §9): the end of the first message (the shared system prompt
        and tools), the starts of the last message and of the generation prompt (a retried request, or a
        client that rewrites the last turn), and where the prompt leaves the longest earlier sequence (a
        prefix shared inside a message). The end of the generation is saved after it."""
        starts = [i for i, t in enumerate(prompt) if t == self.im_start]
        at = set(starts[1:2] + starts[-2:])
        if reusable > 0:
            at.add(reusable)
        return sorted(at)

    # ------------------------------------------------------------ scheduling

    def run(self):
        self.send("ready", {"max_tokens": self.engine.max_tokens, "max_seqs": self.engine_seqs, "spec": self.spec,
                            "vision": self.engine.has_vision})
        self.send("memory", self.engine.memory_report())
        self.send("config", self.settings(), None)
        log(f"{self.engine_seqs} sequences at once (target {self.max_seqs}, KV admission {'on' if self.admit_kv else 'off'}, "
            f"min {self.min_tps:g} tok/s per request), decode share {self.share}, quantum {self.quantum * 1000:.0f} ms, "
            f"{'DFlash2' if self.spec else 'no speculative decoding'}, "
            f"{'batched decode steps' if self.batch else 'decode in turns'}")
        while True:
            idle = not self.alive and not self.waiting and not self.restores
            # A backup due now is one copy per iteration, so the poll does not wait (a request is admitted after it).
            self.poll(block=idle, wait=0.0 if self.backup_due() else self.backup_wait())
            head = self.head()
            if head is None or self.held_until[0] is not head:
                self.blocked = None
            self.admit_waiting()
            job = self.pick()
            if job:
                self.guarded(*job)
            elif self.restores:  # (only restores to wait for: a message wakes this up early)
                self.conn.poll(RESTORE_POLL_S)
            elif self.backup_due():  # (still idle after the poll: no request arrived)
                self.backup_one()
            # (stats also once more when the work is done, and after each idle wait)
            self.report(force=not (self.alive or self.waiting))
            if self.tiers is not None:
                try:
                    self.tiers.tick(idle=not self.alive and not self.waiting)
                except Exception as e:  # noqa: BLE001 - the cache policy must not stop the engine
                    log(f"tiered cache: {e}")

    def report(self, force=False):
        """Metrics and admission settings every 2 s, stats snapshots every STATS_INTERVAL (also from inside a
        long decode turn: one request alone decodes until it ends or a message arrives)."""
        now = time.time()
        if now - self.last_metrics > 2.0:
            self.send("metrics", self.metrics())
            self.send("config", self.settings(), None)
        if force or now - self.last_stats > STATS_INTERVAL:
            self.send("stats", self.stats())

    def can_prefill(self, s):
        o = s.share_with
        if not s.started and o is not None and not o.dead and o.phase == PREFILL and not (o.started and o.fed >= s.share_len):
            return False
        # Image embeddings of a request live in one shared pinned table until its prompt has run, so only one
        # request with images may be part-way through its prompt.
        if s.images and not s.started:
            return not any(o.images and o.started and o.phase == PREFILL for o in self.alive)
        return True

    def pick(self):
        prefills = [s for s in self.alive if s.phase == PREFILL and self.can_prefill(s)]
        decodes = [s for s in self.alive if s.phase == DECODE]
        if not prefills:
            self.credit = 0.0
        if prefills and (not decodes or self.credit <= 0):
            return ("prefill", min(prefills, key=lambda s: (s.rest(), s.arrival)))
        if self.batch and len(decodes) >= 2:
            return ("batch", decodes)
        if decodes:
            return ("decode", min(decodes, key=lambda s: s.last_run))
        return None

    def guarded(self, kind, s):
        """Runs a turn; KV pool full: preempts another request and retries; engine errors end the request."""
        t0 = time.time()
        self.touch()
        try:
            return self._guarded(kind, s)
        finally:
            self.touch()
            self.m["q27:engine_busy_seconds_total"] += time.time() - t0

    def _guarded(self, kind, s):
        while True:
            if kind == "batch":  # (s: the decoding requests; a retry after a preemption keeps the rest)
                s = [o for o in s if o.phase == DECODE and not o.dead]
                if len(s) < 2:
                    if not s:
                        return None
                    kind, s = "decode", s[0]
            try:
                if kind == "batch":
                    return self.batch_turn(s)
                return self.prefill_turn(s) if kind == "prefill" else self.decode_turn(s)
            except self.KvFull:
                if kind != "batch" and s.started and s is self.cur:
                    s.fed = self.engine.pos  # (a prompt may have run part of the way)
                # A restore that completed meanwhile makes its checkpoint visible: poll once, and retry the turn without
                # preempting anyone if any ended. (The engine evicts a loading checkpoint only when nothing else can go:
                # evict_checkpoint takes non-held ones first, then the oldest held, then finishes the oldest loading one
                # synchronously and evicts it.)
                if self.restores:
                    before = self.restore_completed
                    self.poll_restores()
                    if self.restore_completed > before:
                        continue
                # The latest request gives way, also when it is the one that ran out: suspending an earlier
                # one instead lets the two take turns suspending each other (P7-14).
                started = [o for o in self.alive if o.started and not o.dead]
                victim = max(started, key=lambda o: o.arrival, default=None)
                if victim is None or (kind != "batch" and started == [s]):
                    log(f"{s.rid}: the KV pool cannot hold it ({self.engine.pos} tokens)")
                    self.finish(s, "length")
                    return None
                self.suspend(victim)
                if victim is s:
                    return None
            except ImageError as e:  # (an image the server could read the size of but not decode)
                log(f"{s.rid}: {e}")
                self.send("error", s.rid, str(e), 400)
                self.finish(s, "error", save=False, notify=False)
                self.engine.reset()
                self.cur = None
                return None
            except RuntimeError as e:
                msg = str(e)
                if "check failed" not in msg:  # a CUDA error: the context is unusable, restart the process
                    raise
                for o in s if kind == "batch" else [s]:
                    log(f"{o.rid}: {msg}")
                    self.send("event", o.rid, "error", msg)
                    self.send("error", o.rid, msg)
                    self.finish(o, "error", save=False, notify=False)
                self.engine.reset()
                self.cur = None
                return None

    def load(self, s):
        """Makes s the engine's running sequence (parks the running one, if it is still alive)."""
        if self.cur is s:
            return
        cur = self.cur
        t0 = time.time()
        if cur is not None and not cur.dead and cur.started:
            if s.area is not None:
                a = s.area
                self.engine.seq_swap(a)
                cur.area, s.area = a, None
            else:
                cur.area = self.engine.seq_park()
            self.m["q27:sequence_switches_total"] += 1
            self.m["q27:switch_seconds_total"] += time.time() - t0
        elif s.area is not None:
            self.engine.seq_unpark(s.area)
            s.area = None
        self.cur = s
        if not s.started:
            self.begin(s)

    def begin(self, s):
        """prefix_begin: restores the deepest cached prefix (the engine's state is reset first)."""
        self.cur = None  # (prefix_begin drops whatever the engine held)
        if s.no_sync:  # (loading its spilled match was judged slower than computing it: prefix_begin must not load it)
            self.engine.spill_sync_lookup(False)
        try:
            cached = self.engine.prefix_begin(s.eng_ids, at=s.at, spec=self.spec)
        finally:
            if s.no_sync:
                self.engine.spill_sync_lookup(True)
        self.release_restored(s)  # (a hit has cleared it; if prefix_begin took another checkpoint, release this one)
        self.cur = s
        s.started, s.fed = True, cached
        if s.cached is None:
            s.cached, s.begun = cached, time.time()
            self.m["q27:prompt_tokens_total"] += len(s.prompt)
            self.m["q27:cached_tokens_total"] += cached
            self.m["q27:reusable_tokens_total"] += s.reusable
            log(f"{s.rid}: prompt {len(s.prompt)} tokens, cached {cached} of {s.reusable} reusable, "
                f"waited {time.time() - s.arrival:.2f} s, checkpoints at {s.at}")
            self.send("start", s.rid, {"cached": cached})
        else:  # (after a preemption or a forced output)
            log(f"{s.rid}: resumed at {len(s.eng_ids)} tokens, {cached} restored from the prefix cache")
        if s.images:
            import imageproc

            spans, srcs = s.images
            self.engine.mm_set(spans)
            t0, encoded, waited = time.time(), 0, 0.0
            for i, (start, gh, gw) in enumerate(spans):
                if start + gh * gw > cached:  # (images inside the restored prefix need no embeddings)
                    raw, size, norm = srcs[i]
                    t1, f = time.time(), s.decoding.pop(i, None)
                    try:
                        rgb = f.result() if f is not None else imageproc.rgb(raw, size)
                    except Exception as e:  # noqa: BLE001 - any decoder error is the client's
                        raise ImageError(f"cannot decode image: {e}") from e
                    waited += time.time() - t1
                    self.engine.mm_encode_rgb(i, rgb, norm)
                    encoded += 1
            self.drop_decoding(s)  # (the pixels are not kept: a suspended request that needs them decodes again)
            log(f"{s.rid}: images {len(spans)} ({sum(gh * gw for _, gh, gw in spans)} tokens), encoded {encoded} "
                f"in {time.time() - t0:.3f} s ({waited:.3f} s decoding or waiting for it)")

    def prefill_turn(self, s):
        self.load(s)
        t0, fed0 = time.time(), s.fed
        rest = s.eng_ids[s.fed:]
        if len(rest) > self.chunk:
            s.fed += self.engine.prefill(rest, self.chunk, spec=self.spec)
            self.measure_prefill(s.fed - fed0, time.time() - t0)
        else:
            left = s.max_new - len(s.out)
            seed = s.rng.getrandbits(63)
            if s.presence or s.frequency:
                self.engine.penalty_init(s.out)  # (after a preemption the earlier output counts too)
            if s.biased:  # (min_tokens: output token j is at position len(prompt) + j)
                self.engine.bias_set(s.bias, len(s.prompt) + s.min_tokens if s.min_tokens > len(s.out) else 0)
            extra = dict(min_p=s.min_p, presence=s.presence, frequency=s.frequency, logprobs=s.logprobs,
                         masked=s.gm is not None, biased=s.biased)
            # A step ends at the end of the reasoning when a forced output or a grammar follows it, so the engine holds
            # no tokens past it (sampled without the forced tokens, or the masks).
            until = s.force[0] if s.force is not None else s.grammar_after if s.matcher is not None and s.gm is None \
                else None
            if s.gm is not None:
                write_mask(self.engine, s.gm, 0)
            if self.spec:
                tok = self.engine.spec_begin(rest, s.temperature, s.top_k, s.top_p, seed=seed,
                                             uniform=s.rng.random(), max_new=left, until=until,
                                             ignore_eos=s.ignore_eos, **extra)
            else:
                tok = self.engine.forward_sample(rest, s.temperature, s.top_k, s.top_p, s.rng.random(), **extra)
            s.fed = len(s.eng_ids)
            s.phase = DECODE
            if s.first_token is None:
                s.first_token = time.time()
        fed = s.fed  # (a forced output in the emit starts the request again: s.fed is 0 after it)
        if s.phase == DECODE:
            self.emit(s, [tok], self.engine.logprobs() if s.logprobs >= 0 else None)
        dt = time.time() - t0
        s.last_run = time.time()
        s.prefill_s += dt
        s.computed += fed - fed0
        if s.phase == DECODE and not s.dead:
            # The server answers a request its stop string ends with this and its own count of the steps after
            # it, rather than wait for "done", which follows the cancel only after the engine's next step.
            self.send("prefilled", s.rid, {"ttft": s.first_token - s.arrival, "prefill_s": s.prefill_s,
                                           "computed_tokens": s.computed, "drafted": s.drafted,
                                           "draft_accepted": s.draft_accepted, "draft_block": self.draft_block if self.spec else 0})
        self.m["q27:prefill_tokens_total"] += fed - fed0
        if any(o.phase == DECODE for o in self.alive if o is not s):
            self.credit += self.share * dt

    def decode_turn(self, s):
        self.load(s)
        t0 = time.time()
        others = sum(1 for o in self.alive if o is not s) > 0
        has_prefill = any(o.phase == PREFILL for o in self.alive)
        turn = min(self.quantum, max(self.credit, 0.0)) if has_prefill else self.quantum
        step, n0 = 0.0, len(s.out)
        while not s.dead and s.phase == DECODE:  # (a forced output makes it a prefill again)
            now = time.time()
            last = (others and now + step >= t0 + turn) or self.conn.poll(0) or self.restore_due()
            ts = time.time()
            if self.spec:
                if s.gm is not None:
                    write_masks(self.engine, s.gm, self.engine.spec_draft(), 0, self.eos)
                toks = self.engine.spec_step(ahead=not last)
                self.m["q27:spec_verify_calls_total"] += 1
                self.count_draft(s, toks)
            else:
                if s.gm is not None:
                    write_mask(self.engine, s.gm, 0)
                toks = [self.engine.forward_sample([s.out[-1]], s.temperature, s.top_k, s.top_p, s.rng.random(),
                                                   min_p=s.min_p, presence=s.presence, frequency=s.frequency,
                                                   logprobs=s.logprobs, masked=s.gm is not None,
                                                   biased=s.biased)]
            step = max(step * 0.8, time.time() - ts)
            s.steps += 1
            self.emit(s, toks, self.engine.logprobs() if s.logprobs >= 0 else None)
            if last:
                break
            self.report()
        dt = time.time() - t0
        s.last_run = time.time()
        s.decode_s += dt
        if not others and self.spec:
            self.measure(1, len(s.out) - n0, dt)
        if has_prefill:
            self.credit -= dt

    def batch_turn(self, seqs):
        """Batched speculative steps of the decoding requests (each one parked or the running one) until the
        turn ends, a request finishes, or a message arrives."""
        t0 = time.time()
        has_prefill = any(o.phase == PREFILL for o in self.alive)
        turn = min(self.quantum, max(self.credit, 0.0)) if has_prefill else self.quantum
        lanes = [None if s is self.cur else s.area for s in seqs]
        n0 = sum(len(s.out) for s in seqs)
        while True:
            if any(s.gm is not None for s in seqs):  # (structured output: drafted, masked by the grammars, verified)
                for z, (s, chain) in enumerate(zip(seqs, self.engine.spec_draft_lanes(lanes))):
                    if s.gm is not None:
                        write_masks(self.engine, s.gm, chain, 8 * z, self.eos)
            new = self.engine.spec_step_lanes(lanes)
            self.m["q27:spec_verify_calls_total"] += len(seqs)
            self.m["q27:batched_steps_total"] += 1
            for z, (s, toks) in enumerate(zip(seqs, new)):
                s.steps += 1
                self.count_draft(s, toks)
                self.emit(s, toks, self.engine.logprobs(z) if s.logprobs >= 0 else None)
            if (any(s.dead or s.phase != DECODE for s in seqs) or time.time() >= t0 + turn or self.conn.poll(0)
                    or self.restore_due()):
                break
        dt = time.time() - t0
        self.measure(len(seqs), sum(len(s.out) for s in seqs) - n0, dt)
        for s in seqs:
            s.last_run = time.time()
            s.decode_s += dt
        if has_prefill:
            self.credit -= dt

    def count_draft(self, s, toks):
        """A verify step's tokens are the accepted drafts and the bonus token."""
        s.drafted += self.draft_block
        s.draft_accepted += len(toks) - 1

    def emit(self, s, toks, logprobs=None):
        """Tokens of a step: stops at EOS and at max_new, and at the token a forced output follows (force); the rest goes
        to the server (with their logprobs, when the request asked for them), after forced tokens not sent yet.
        Structured output: the tokens go through the grammar; at the end of the reasoning it starts to apply, and the
        rest of that step (sampled without the masks) stays only as far as the grammar allows it."""
        out, finish, force, began, cut = [], None, None, False, False
        for t in toks:
            if s.gm is not None and not allows(s.gm, t, self.eos):
                cut = True  # (only right after the grammar began: masked steps output what it allows)
                break
            if t in self.eos and not s.ignore_eos:
                finish = "stop"
                break
            out.append(t)
            if s.gm is not None:
                s.gm.consume_token(t)
            if len(s.out) + len(out) >= s.max_new:
                finish = "length"
                break
            if s.force is not None and t == s.force[0]:
                force = s.force[1]
                break
            if s.matcher is not None and s.gm is None and t == s.grammar_after:
                s.gm, began = s.matcher, True
        if out:
            s.out += out
            self.m["q27:generation_tokens_total"] += len(out)
        if out or s.unsent:
            self.send_toks(s, out, logprobs)
        if finish:
            self.finish(s, finish)
        elif force is not None:
            self.force_output(s, force)
        elif cut:
            # The engine holds tokens past the grammar's: s runs its prompt + output again from the prefix cache.
            self.m["q27:grammar_restarts_total"] += 1
            self.restart(s)
            s.est_cached = len(s.prompt) - 1
        elif began and self.spec:
            self.engine.spec_set_masked(None if s is self.cur else s.area)

    def send_toks(self, s, out, logprobs=None):
        """Sends output tokens, after the forced ones not sent yet (their logprob: 0, no top entries)."""
        forced, s.unsent = s.unsent, []
        if s.logprobs < 0:
            self.send("tok", s.rid, forced + out)
        else:
            self.send("tok", s.rid, forced + out, [(0.0, [])] * len(forced) + list((logprobs or [])[:len(out)]))

    def force_output(self, s, ids):
        """The output continues with ids (the forced start of a tool call after the reasoning, tool_choice): s's state
        becomes a prefix cache checkpoint and s runs its prompt + output + ids from the deepest one (its own if the
        tokens committed past the reasoning's end are ids', else the start of the generation prompt: the reasoning
        is computed again)."""
        s.force = None
        ids = list(ids[:s.max_new - len(s.out)])
        s.out += ids
        s.unsent += ids
        self.m["q27:forced_outputs_total"] += 1
        if len(s.out) >= s.max_new:
            self.send_toks(s, [])
            self.finish(s, "length")
            return
        self.restart(s)
        s.est_cached = len(s.eng_ids) - len(ids)  # (prompt + output up to the reasoning's end: its own checkpoint)

    def suspend(self, v):
        """Preemption (KV pool full): v's state becomes a prefix cache checkpoint and its pages go back; it
        waits at the head of the queue (the admission lets it in again) as a prompt of its prompt + output."""
        saved = len(v.eng_ids) if v.phase == DECODE else v.fed
        log(f"{v.rid}: preempted at {saved} tokens (KV pool full)")
        self.send("event", v.rid, "preempt", f"preempted at {saved} tokens (KV pool full)")
        self.restart(v)
        v.est_cached = min(saved, len(v.eng_ids) - 1)  # (the checkpoint it resumes from)
        v.preempted += 1
        self.m["q27:preemptions_total"] += 1
        self.alive.remove(v)
        self.waiting.appendleft(v)
        v.held = False

    def restart(self, v):
        """v's state becomes a prefix cache checkpoint and leaves the engine; v runs again as a prompt of its prompt +
        output (prefix_begin restores the deepest checkpoint of that)."""
        if v is self.cur:
            self.engine.checkpoint()
            self.engine.reset()
            self.cur = None
        elif v.area is not None:
            self.engine.seq_suspend(v.area)
            v.area = None
        v.eng_ids = v.prompt + v.out  # (ids_bytes follows the new length)
        v.started, v.fed, v.phase = False, 0, PREFILL
        # (its own checkpoint is in the pool now; a deeper spilled one would be new: ask again, and let prefix_begin load)
        v.restore_checked = v.no_sync = False
        self.detach_restore(v)
        v.restore_id, v.restore_pos = None, 0  # (a preempted request has begun: released then)

    def finish(self, s, reason, save=True, notify=True):
        self.touch()
        if s.dead:
            return
        s.dead = True
        self.detach_restore(s)
        self.release_restored(s)  # (it never began: the checkpoint restored for it is not held for nothing)
        self.drop_decoding(s)
        if s in self.alive:
            self.alive.remove(s)
        self.by_id.pop(s.rid, None)
        # The end of the generation: the next turn resumes here when it re-renders this output as is (a parked
        # request, e.g. one of a batched step, is saved from its parking area).
        end = save and reason in ("stop", "length")
        if s.area is not None:
            (self.engine.seq_suspend if end else self.engine.seq_drop)(s.area)
            s.area = None
        elif s is self.cur and end:
            self.engine.checkpoint()
        if s.started or s.out:
            self.history.append(array.array("i", s.prompt + s.out).tobytes())
        now = time.time()
        info = {"prompt_tokens": len(s.prompt), "completion_tokens": len(s.out), "cached_tokens": s.cached or 0,
                "ttft": (s.first_token - s.arrival) if s.first_token else None, "total": now - s.arrival,
                "preempted": s.preempted, "reusable_tokens": s.reusable, "computed_tokens": s.computed,
                "queued": (s.begun - s.arrival) if s.begun else None, "prefill_s": s.prefill_s,
                "decode_s": s.decode_s, "steps": s.steps, "drafted": s.drafted, "draft_accepted": s.draft_accepted}
        self.m[f'q27:requests_finished_total{{reason="{reason}"}}'] += 1
        if notify:
            self.send("done", s.rid, reason, dict(info, metrics=self.metrics()))
        if s.first_token and len(s.out) > 1:
            rate = (len(s.out) - 1) / max(now - s.first_token, 1e-6)
            log(f"{s.rid}: {reason}, {len(s.out)} tokens, ttft {info['ttft']:.2f} s, {rate:.1f} tok/s"
                + (f", preempted {s.preempted}x" if s.preempted else ""))

    def metrics(self):
        self.last_metrics = time.time()
        st = self.engine.cache_stats()
        out = dict(self.m)
        out.update({"q27:num_requests_running": len(self.alive), "q27:num_requests_waiting": len(self.waiting),
                    "q27:target_seqs": self.max_seqs, "q27:engine_seqs": self.engine_seqs,
                    'q27:admission_blocked{reason="kv"}': int(self.blocked == "kv"),
                    'q27:admission_blocked{reason="speed"}': int(self.blocked == "speed"),
                    "q27:kv_free_pages": st["free_pages"], "q27:kv_checkpoint_only_pages": st["checkpoint_only_pages"],
                    "q27:prefix_cache_checkpoints": st["checkpoints"], "q27:prefix_cache_evictions_total": st["evicted"]})
        if self.tiers is not None:
            sp, ts = self.engine.spill_stats(), self.tiers.status()
            out.update({"q27:tier_ram_bytes": sp["ram_bytes"], "q27:tier_in_ram": sp["in_ram"],
                        "q27:tier_on_disk": sp["on_disk"], "q27:tier_spilled": sp["spilled"],
                        "q27:tier_restored": sp["restored"], "q27:tier_written": sp["written"],
                        "q27:tier_dropped": sp["dropped"], "q27:tier_subsumed": sp["subsumed"],
                        "q27:tier_failed": sp["failed"], "q27:tier_restoring": sp.get("restoring", 0),
                        "q27:tier_restored_async": sp.get("restored_async", 0),
                        "q27:tier_restore_failed": sp.get("restore_failed", 0), "q27:tier_ssd_bytes": ts["ssd_bytes"],
                        "q27:tier_ssd_budget_bytes": ts["ssd_budget_bytes"]})
        return out


    def stats(self):
        """Snapshot for the stats UI: each request (where its context stands), the KV pool, device memory."""
        from stats import slot_rows, tiers_block  # (the frontend dir is on sys.path)
        self.last_stats = now = time.time()
        if not hasattr(self, "slot_lanes"):  # (a Scheduler built without __init__, as the tests do)
            self.slot_lanes = {}
        st = self.engine.cache_stats()
        seqs, lanes = [], []
        for s in self.alive + list(self.waiting):
            if s.dead:
                continue
            state = ("restoring" if s.restoring is not None else "waiting") if s not in self.alive else (
                "running" if s is self.cur else "parked" if s.area is not None else "admitted")
            seqs.append({"rid": s.rid, "state": state, "phase": s.phase, "prompt": len(s.prompt),
                         "cached": s.cached if s.cached is not None else s.est_cached,
                         "fed": min(s.fed, len(s.prompt)) if s.started else 0, "out": len(s.out),
                         "max_new": s.max_new, "computed": s.computed, "reusable": s.reusable,
                         "age": now - s.arrival, "queued": (s.begun or now) - s.arrival,
                         "ttft": (s.first_token - s.arrival) if s.first_token else None,
                         "decode_wall": (now - s.first_token) if s.first_token else 0.0,
                         "prefill_s": s.prefill_s, "decode_s": s.decode_s, "steps": s.steps,
                         "preempted": s.preempted, "held": s.held and s in self.waiting,
                         "images": sum(gh * gw for _, gh, gw in s.images[0]) if s.images else 0,
                         "marks": s.marks})
            if s.restoring is not None:
                rec = s.restoring
                seqs[-1]["restore"] = {"tokens": rec["pos"], "bytes": rec["bytes"], "on_disk": rec["on_disk"],
                                       "elapsed": now - rec["t0"]}
            if state not in ("waiting", "restoring"):
                lanes.append((state, s.area, seqs[-1]))
        return {"t": now, "seqs": seqs, "cache": st, "device": self.engine.device_memory(),
                "running": len(self.alive), "waiting": len(self.waiting), "blocked": self.blocked,
                "restoring": sum(1 for q in seqs if q["state"] == "restoring"),
                "target_seqs": self.max_seqs, "busy_s": self.m["q27:engine_busy_seconds_total"],
                "generated": self.m["q27:generation_tokens_total"], "prefilled": self.m["q27:prefill_tokens_total"],
                "slots": slot_rows(lanes, self.engine_seqs, target=self.max_seqs, last_lanes=self.slot_lanes),
                "tiers": tiers_block(self.engine, getattr(self, "tiers", None))}


def allows(m, t, eos):
    """Whether llguidance matcher m takes token t next (its validate_tokens does not count an end-of-sequence id)."""
    return m.is_accepting() if t in eos else m.validate_tokens([t]) == 1


def write_mask(engine, m, row):
    """Writes engine mask row `row`: what llguidance matcher m allows next."""
    addr, size = engine.mask_row(row)
    m.unsafe_compute_mask_ptr(addr, size)


def write_masks(engine, m, chain, row0, eos):
    """Grammar masks of a speculative step's verify rows row0 .. row0 + 7 (chain: their tokens, Engine.spec_draft; the
    first, the last output token, has been through matcher m): row i masks by what follows chain[1..i]. Rows past a
    drafted token the grammar does not allow, or an end-of-sequence id (eos), are left open: that token is rejected,
    or ends the step. (A matcher that consumed an end-of-sequence id does not roll back to the state before it.)"""
    k = m.validate_tokens(chain[1:])
    k = next((i for i, t in enumerate(chain[1:1 + k]) if t in eos), k)
    for i in range(8):
        addr, size = engine.mask_row(row0 + i)
        if i and i <= k:
            m.consume_token(chain[i])
        if i <= k:
            m.unsafe_compute_mask_ptr(addr, size)
        else:
            ctypes.memset(addr, 0xFF, size)
    if k:
        m.rollback(k)


def lcp_bytes(a, b):
    """Common prefix of two int32 byte strings, in tokens."""
    lo, hi = 0, min(len(a), len(b)) // 4
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if a[4 * lo:4 * mid] == b[4 * lo:4 * mid] else (lo, mid - 1)
    return lo


def run(conn, cfg):
    """Process entry point (multiprocessing, spawn)."""
    sys.path.insert(0, cfg["engine_python"])
    try:
        Scheduler(conn, cfg).run()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 - report and exit; the server starts a new engine
        refused = "not enough VRAM" in str(e)
        if not refused:
            traceback.print_exc()
        try:
            conn.send(("refused", str(e)) if refused else ("fatal", traceback.format_exc(limit=3)))
        except OSError:
            pass
        os._exit(1)
