"""Tiered KV cache policy (SPILL_RAM_GIB > 0): what happens to the checkpoints the engine spilled to host RAM.

When the GPU KV pool is full, the engine copies an evicted prefix-cache checkpoint to host RAM (a spill). This
policy keeps that RAM bounded: a checkpoint that has been in RAM for ram_retention_s (2x that when the server is
busy) is written to a file in the directory (engine.spill_export). Files stay within an SSD budget (oldest
written first) and a free-space reserve, and are deleted after ttl_days. A file an earlier run wrote is indexed at
start() (engine.spill_import). A checkpoint that is reused leaves the engine's list by itself.

The SSD tier is ON by default (SPILL_SSD_GIB=300). SPILL_SSD_GIB=0 (explicit) turns it off: TierPolicy then has no
directory, and the engine is started with SPILL_DIR empty (ssd_off_env, called by worker.Scheduler before the engine is made;
launch.py does the same in the engine's settings), so the engine DROPS the victims of a full RAM tier instead of writing them.
Nothing is written to, read from, scanned or deleted in any SPILL_DIR then (no write at a stop, no import at a start, no sweep): a
folder an earlier run left is not touched.

Idle backups (VRAM_BACKUP_IDLE_S, worker.Scheduler): while the engine is idle, the engine copies VRAM checkpoints to
RAM without removing them from VRAM (engine.spill_backup_one); their entries have "backup": True. Retention never writes
them to the SSD (tick skips them: the checkpoint is still in VRAM). flush() does write them on a stop
(SPILL_FLUSH_ON_STOP_S), since VRAM is lost when the engine unloads.

SSD layout. An engine with SSD dedup (engine.spill_disk_usage() is not None) writes each entry as a manifest in
SPILL_DIR/entries (SUFFIX_MANIFEST) plus chunk files in SPILL_DIR/chunks that entries with the same prompt prefix share (the
engine owns chunks/: this module never lists or deletes there). Budget, reserve and make_room count PHYSICAL bytes (each chunk
once, engine.spill_disk_usage) and free space by an entry's unique_disk_bytes (what dropping it really frees). The first start with
such an engine deletes the flat *.q27kv files an older build wrote (it is only a cache). An older engine, without the API, keeps the
flat layout (one SUFFIX file per entry, sizes from the files).

Retention (value-based, engine/src/retention.h; docs/tiered-cache.md "Retention"). ONE victim order per tier, used here for the SSD
(make_room, _pick_victim) and by the engine for RAM (spill_enforce_ram): the agent using the most bytes of the tier (SPILL_FAIR_AGENTS;
elastic, no guaranteed share), then within it superseded entries (a deeper checkpoint of the same conversation exists), never-hit before
hit entries, and the older / bigger (the pre-retention rule) last. Superseded entries are never written to the SSD: past the RAM
retention they are dropped. The engine reports per entry "owner" (the agent's id, 0 unknown), "hits", "superseded" and "superseded_durable" (spill_list; the latter counts only deeper entries already on the SSD and is what
decides whether an entry is dropped instead of written) and
per agent the bytes it holds in each tier (spill_agents); an engine without them gives the defaults (owner 0, no hits, nothing
superseded) and the order is the old "oldest written first".

Three more rules for entries and agents that never prove their value (docs/tiered-cache.md "Retention"; they only act on entries the
engine reports "hits" for, so an engine without the retention fields keeps the old behaviour):
  A. SPILL_UNUSED_TTL_S (0: off, the default): an entry with no hit for that long since it came to RAM / was written is dropped from RAM and
     the SSD by tick(), whatever the free capacity (not: hit entries, entries of <= KEEP_SHORT_TOKENS tokens, backups, busy ones). "expired".
  B. cold agents (SPILL_COLD_FIRST, SPILL_COLD_AFTER_S): see _pick_victim. The engine's retention.h cold_agents is the same rule.
  C. SPILL_SSD_WARM_ONLY (0: off, the default): an entry is written to the SSD only when it was hit or its agent has a hit in some reported entry
     ("warm"). An unproven entry of an agent with an explicit id (not `auto:`) is written within the PROBATION quota
     (SPILL_PROBATION_GIB, 4: all unproven entries on the SSD together; the oldest unproven one goes first); one of an `auto:` agent
     (no X-Agent-Id: a new id per prompt) is dropped instead. "not_written".

CACHE_POLICY=smart (frontend/cache_policy.py; docs/tiered-cache.md "CACHE_POLICY=smart"; default legacy: everything above unchanged).
For the SSD tier: the victim is the minimum GreedyDual priority (stamp + prefill seconds saved over the parent / GiB x (1 + hits) x
junction x superseded factor) among the SSD leaves, young entries (written < CACHE_MIN_RESIDENT_S ago) protected; no cold-agent step, no
per-agent step. A superseded entry is written like any other (with its reduced value), never dropped for being superseded. SSD admission
replaces SPILL_SSD_WARM_ONLY / the probation quota: an entry goes to the SSD only when its priority there is above the SSD's watermark L
(the priority of the last SSD victim: below it, it would be the next victim). The tick path checks it before writing; an entry the
engine's RAM cap wrote is checked at the first tick that sees it on the SSD (and dropped when it fails). The per-entry numbers come from
the engine's spill_meta (parent_pos, desc_tiers, junction, appended under smart); SPILL_SSD_MIN_TOKENS and rule A still apply.

Everything runs on the caller's thread: the engine is not thread-safe.
"""

import math
import os
import shutil
import time
import uuid
from pathlib import Path

import cache_policy as cp

GiB = 1 << 30
DAY = 86400.0
DEFAULT_SSD_GIB = 300.0   # SPILL_SSD_GIB when unset or empty (compose.yaml, launch.TIER_SETTINGS); an explicit 0 turns the SSD tier off
SUFFIX = ".q27kv"                  # the flat layout (older engine); in the dedup layout: a legacy file, deleted at start
TMP_SUFFIX = ".q27kv.tmp"
SUFFIX_MANIFEST = ".q27m"          # SSD dedup layout: SPILL_DIR/entries/<id>.q27m (+ the engine's SPILL_DIR/chunks/*.kvc)
ENTRIES_DIR = "entries"
MOSTLY_SHARED = 0.25               # an entry whose unique bytes are under this fraction of its size frees little when dropped
# The longest a stop may spend writing (SPILL_FLUSH_ON_STOP_S is cut to it). server.py gives the engine process
# 15 s to exit on an unload or restart and 10 s on shutdown (EngineClient._stop, stop); a flush cut off by the
# server's terminate would leave a file half written.
STOP_FLUSH_MAX_S = 8.0
KEEP_SHORT_TOKENS = 16384          # engine/src/retention.h kKeepShortTokens (a test compares): the shared system-prompt prefixes are never expired / gated
DEFAULT_WRITE_BPS = 1e9            # flush() assumes this write speed until an export has been timed (conservative: an NVMe SSD does more)
WRITE_RATE_MIN_BYTES = 64 << 20    # an export smaller than this does not update the measured speed


def _float(name, raw):
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a number >= 0, got {raw!r}")
    return v


def ssd_gib_of_env(environ=None):
    """SPILL_SSD_GIB as a number of GiB: unset or empty is 300 (the default; an explicit 0 means the SSD tier is off). Raises ValueError for a bad value."""
    env = os.environ if environ is None else environ
    text = env.get("SPILL_SSD_GIB")
    return DEFAULT_SSD_GIB if text is None or not text.strip() else _float("SPILL_SSD_GIB", text)


def ssd_off_env(environ=None):
    """Makes an environment the engine will read say "no SSD tier" when the SSD cap is 0 (SPILL_SSD_GIB=0 set explicitly): SPILL_DIR is set to the empty
    string, which is what makes the engine drop the victims of a full RAM tier instead of writing them to a folder (the policy would
    delete them again). Returns True when the SSD tier is off. An unreadable SPILL_SSD_GIB changes nothing (TierPolicy.from_env raises)."""
    env = os.environ if environ is None else environ
    try:
        off = not ssd_gib_of_env(env) > 0
    except ValueError:
        return False
    if off:
        env["SPILL_DIR"] = ""
    return off


def _size(path):
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _mtime(path, default):
    try:
        return os.path.getmtime(path)
    except (OSError, TypeError):
        return default


def _unlink(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _hit(e):
    return (e.get("hits") or 0) > 0


def _sup_durable(e):
    """Superseded by a deeper entry that is ON THE SSD (the engine's superseded_durable; an engine without it: the plain flag). Only
    such an entry may be dropped instead of written: a deeper one that is in VRAM / RAM only may not survive a stop."""
    return bool(e.get("superseded_durable", e.get("superseded")))


def _owner(e):
    return e.get("owner") or 0


def _known(e):
    """The engine reports this entry's hits (an engine without the retention fields does not: then no rule below applies to it)."""
    return "hits" in e


def _cold_owners(entries, times, now, after):
    """The cold agents of a tier (engine/src/retention.h cold_agents): none of the agent's entries has hits, and its newest entry is
    at least `after` s old. entries: the whole tier; times: id -> when the entry came to the tier. Agent 0 (unknown) is never cold; an
    agent with an entry the engine reports no hits for, or no time for, is not cold."""
    agents = {}  # owner -> [any hit / unknown, newest time]
    for e in entries:
        t = times.get(e["id"])
        a = agents.setdefault(_owner(e), [False, None])
        if not _known(e) or _hit(e) or t is None:
            a[0] = True
        elif a[1] is None or t > a[1]:
            a[1] = t
    return {o for o, (warm, newest) in agents.items() if o != 0 and not warm and newest is not None and now - newest >= after}


class TierPolicy:
    def __init__(self, engine, directory, *, ram_retention_s=3600, ssd_gib=300, ssd_min_free_gib=10, ttl_days=7,
                 tick_s=30, clock=time.time, disk_free=None, log=print, backup_idle_s=60, flush_on_stop_s=8,
                 backup_victims=8, ssd_min_tokens=32768, fair_agents=True, unused_ttl_s=0, cold_after_s=600,
                 cold_first=True, ssd_warm_only=False, probation_gib=4, cache_policy=cp.LEGACY, smart_params=None):
        self.engine = engine
        self.cache_policy = cp.policy_of(cache_policy)  # CACHE_POLICY: legacy (the rules above) or smart (cache_policy.py)
        self.smart = self.cache_policy == cp.SMART
        self.smart_params = smart_params or cp.SmartParams()
        self.gd = cp.GdTier()                           # smart: the SSD's watermark, stamps and counters
        self.not_admitted = 0                           # smart: entries refused by the SSD admission (below the watermark)
        self._pick = None                               # smart: (entry id, Pick) of the last victim chosen, until its drop succeeds
        self.budget = float(ssd_gib) * GiB
        # None: no SSD tier. Also when the cap is 0 (set explicitly): "0 GiB" means off, not "a zero budget that deletes what the engine writes".
        self.directory = Path(directory) if directory and self.budget > 0 else None
        self.ram_retention_s = float(ram_retention_s)
        self.reserve = float(ssd_min_free_gib) * GiB
        self.ttl_s = float(ttl_days) * DAY
        self.tick_s = float(tick_s)
        self.backup_idle_s = float(backup_idle_s)      # idle seconds before a VRAM checkpoint is backed up (0: off)
        self.flush_on_stop_s = float(flush_on_stop_s)  # the time a stop may spend writing (0: off)
        self.backup_victims = int(backup_victims)      # the idle backup copies only one of the next N checkpoints the pool would evict (0: any)
        # Spilled entries shorter than this many tokens (the engine's "pos") are never written to the SSD: every entry
        # carries the same ~190 MiB recurrent state, and a short prefix recomputes in about the time a read takes (0: off).
        self.ssd_min_tokens = max(0, int(ssd_min_tokens))
        self.fair_agents = bool(fair_agents)           # SPILL_FAIR_AGENTS: the SSD victim comes from the agent using the most SSD
        self.unused_ttl_s = max(0.0, float(unused_ttl_s))   # SPILL_UNUSED_TTL_S: never-hit entries older than this are dropped (0: off)
        self.cold_after_s = max(0.0, float(cold_after_s))   # SPILL_COLD_AFTER_S: an agent without hits whose newest entry is this old is cold
        self.cold_first = bool(cold_first)                  # SPILL_COLD_FIRST: cold agents' entries are the SSD victims first
        self.ssd_warm_only = bool(ssd_warm_only)            # SPILL_SSD_WARM_ONLY: no unproven entry goes to the SSD (but the probation quota)
        self.probation_bytes = max(0.0, float(probation_gib)) * GiB  # SPILL_PROBATION_GIB: the SSD room of all unproven entries together
        self.auto_owners = set()                            # owner ids of `auto:` agents (note_agent): the worker tells; lost at a restart
        self.expired = 0                                    # entries dropped by the never-hit TTL (A)
        self.not_written = 0                                # unproven entries refused / removed by the warm-only rule (C)
        self.skipped_short = 0                         # short entries dropped instead of written
        self.skipped_superseded = 0                    # superseded entries dropped instead of written
        self.agent_rows = []                           # the engine's per-agent bytes / entries at the last look (spill_agents), for status()
        self.ssd_use = {}                              # owner -> bytes in the SSD tier, as make_room last read it
        self.write_bps = None                          # measured throughput of the last exports (bytes / s; None: none yet), for flush()
        self.clock = clock
        self.disk_free = disk_free or (lambda d: shutil.disk_usage(d).free)
        self.log = log
        self.dedup = None    # True: the engine has the SSD dedup layout (decided at start()); False: flat files; None: not known yet
        self.usage = None    # the engine's last spill_disk_usage() (dedup layout), for status()
        self.written = {}    # id -> time the entry was written to a file (epoch s)
        self.ram_since = {}  # id -> time the entry was first seen in RAM
        self.listed = []     # the engine's list at the last look
        self.last_tick = None
        self.last_error = None

    @classmethod
    def from_env(cls, engine, log=print):
        """None unless SPILL_RAM_GIB > 0. Bad values raise ValueError naming the variable."""
        raw = os.environ.get("SPILL_RAM_GIB") or "0"
        ram = _float("SPILL_RAM_GIB", raw)
        if not ram > 0:
            return None
        opts = {}
        for env, key, default in (("SPILL_RAM_RETENTION_S", "ram_retention_s", 3600),
                                  ("SPILL_SSD_GIB", "ssd_gib", DEFAULT_SSD_GIB),
                                  ("SPILL_SSD_MIN_FREE_GIB", "ssd_min_free_gib", 10),
                                  ("SPILL_TTL_DAYS", "ttl_days", 7),
                                  ("SPILL_TICK_S", "tick_s", 30),
                                  ("VRAM_BACKUP_IDLE_S", "backup_idle_s", 60),
                                  ("SPILL_FLUSH_ON_STOP_S", "flush_on_stop_s", 8),
                                  ("VRAM_BACKUP_VICTIMS", "backup_victims", 8),
                                  ("SPILL_SSD_MIN_TOKENS", "ssd_min_tokens", 32768),
                                  ("SPILL_UNUSED_TTL_S", "unused_ttl_s", 0),
                                  ("SPILL_COLD_AFTER_S", "cold_after_s", 600),
                                  ("SPILL_PROBATION_GIB", "probation_gib", 4)):
            text = os.environ.get(env)
            opts[key] = default if text is None or not text.strip() else _float(env, text)
        for env, key, default in (("SPILL_FAIR_AGENTS", "fair_agents", True), ("SPILL_COLD_FIRST", "cold_first", True),
                                  ("SPILL_SSD_WARM_ONLY", "ssd_warm_only", False)):  # (flags: 1 / 0; unset or empty: the default)
            text = os.environ.get(env)
            opts[key] = default if text is None or not text.strip() else text.strip().lower() not in ("0", "false", "no", "off")
        opts["cache_policy"] = cp.policy_of(os.environ.get("CACHE_POLICY"))  # (ValueError for anything but legacy / smart)
        opts["smart_params"] = cp.SmartParams.from_env(os.environ)
        if opts["flush_on_stop_s"] > STOP_FLUSH_MAX_S:
            log(f"tiered cache: SPILL_FLUSH_ON_STOP_S {opts['flush_on_stop_s']:g} cut to {STOP_FLUSH_MAX_S:g} s "
                "(the server stops the engine within 10-15 s)")
            opts["flush_on_stop_s"] = STOP_FLUSH_MAX_S
        return cls(engine, os.environ.get("SPILL_DIR", ""), log=log, **opts)

    def note_agent(self, agent, owner):
        """The worker tells which agent an owner id is (agent.owner_id): `auto:` agents (no X-Agent-Id) are remembered, since the owner
        id is a hash. Lost at a restart: after it the agent counts as one with an explicit id (probation quota, not dropped at once)."""
        if owner and isinstance(agent, str) and agent.startswith("auto:"):
            self.auto_owners.add(owner)

    # ------------------------------------------------------------ SSD layout

    def _read_usage(self):
        """The engine's SSD usage {"physical_bytes", ...} or None (an engine without the dedup API, or the call failed)."""
        fn = getattr(self.engine, "spill_disk_usage", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - treated as an engine without the API
            self._error(f"spill_disk_usage: {e}")
            return None

    def _layout(self):
        """True for the dedup layout. Decided once, by asking the engine."""
        if self.dedup is None:
            self.usage = self._read_usage()
            self.dedup = self.usage is not None
        return self.dedup

    @property
    def suffix(self):
        return SUFFIX_MANIFEST if self._layout() else SUFFIX

    @property
    def files_dir(self):
        """Where entry files go: SPILL_DIR/entries (dedup layout) or SPILL_DIR itself."""
        return self.directory / ENTRIES_DIR if self._layout() else self.directory

    def _error(self, msg):
        self.last_error = msg
        self.log(f"tiered cache: {msg}")

    # ------------------------------------------------------------ startup

    def start(self):
        if self.directory is None:
            return
        # No try here: a directory that cannot be created is a config error the user must see.
        self.directory.mkdir(parents=True, exist_ok=True)
        probe = self.directory / ".write-test"
        try:
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as e:
            raise RuntimeError(
                f"SPILL_DIR {self.directory} is not writable ({e}). If it is the kvcache volume, it was probably "
                "created owned by root: remove the kvcache volume (docker volume rm) and recreate it, or fix the "
                "directory's permissions.") from e
        files = self.files_dir
        if self.dedup:
            files.mkdir(parents=True, exist_ok=True)
            # The flat files of an older build are not read by this layout: it is only a cache, so they go (one line).
            legacy = [p for pat in ("*" + SUFFIX, "*" + TMP_SUFFIX) for p in self.directory.glob(pat)]
            if legacy:
                for p in legacy:
                    _unlink(p)
                self.log(f"tiered cache: deleted {len(legacy)} legacy {SUFFIX} file(s) (the SSD layout changed to "
                         f"{ENTRIES_DIR}/ + chunks/; it is only a cache)")
        for p in list(files.glob("*.tmp")):  # left by a write that did not finish
            _unlink(p)
        for p in sorted(files.glob("*" + self.suffix)):
            try:
                try:
                    entry_id = self.engine.spill_import(str(p))
                except Exception as e:  # noqa: BLE001 - the call failed: nothing says the file is not ours, so it stays
                    self._error(f"import {p.name}: {e}")
                    continue
                if not entry_id:  # (0: the engine rejected it: another model / KV type, damaged, a chunk file missing)
                    self.log(f"tiered cache: deleting {p.name} (not a checkpoint of this model)")
                    _unlink(p)
                    continue
                # A file that vanished or cannot be stat'ed gets the clock as its write time.
                self.written[entry_id] = _mtime(p, self.clock())
            except OSError as e:  # a file locked or removed mid-import must not stop the worker
                self._error(f"start {p.name}: {e}")
        if self.dedup:
            # Every manifest is imported: the chunk files nobody lists (a crash between a chunk and its manifest) and any .tmp go.
            try:
                n = self.engine.spill_sweep()
                if n:
                    self.log(f"tiered cache: deleted {n} leftover chunk / temporary file(s)")
            except Exception as e:  # noqa: BLE001
                self._error(f"sweep: {e}")
            self.usage = self._read_usage()
        self.listed = self.engine.spill_list()

    # ------------------------------------------------------------ periodic work

    def tick(self, idle, now=None):
        now = self.clock() if now is None else now
        if self.last_tick is not None and now - self.last_tick < self.tick_s:
            return
        self.last_tick = now

        entries = self.engine.spill_list()
        self.listed = entries
        self.agent_rows = self._read_agents()
        listed = {e["id"] for e in entries}
        for book in (self.written, self.ram_since):
            for entry_id in [i for i in book if i not in listed]:
                del book[entry_id]
        fresh = []  # smart: entries first seen on the SSD (the engine's RAM cap wrote them): their admission is checked below
        for e in entries:
            entry_id = e["id"]
            if e["ram_bytes"] > 0:
                # Retention is counted from the first tick that sees the entry in RAM, so a write-out can be up to
                # SPILL_TICK_S late.
                self.ram_since.setdefault(entry_id, now)
            else:
                self.ram_since.pop(entry_id, None)
                if e["on_disk"] and entry_id not in self.written:
                    self.written[entry_id] = _mtime(e["path"], now)
                    fresh.append(entry_id)

        entries = self._expire(entries, now)  # A (also without an SSD: RAM entries nobody used go too)
        if self.directory is None:  # no SSD tier: RAM entries stay until the engine's own cap drops them
            return
        if self.smart:
            entries = self._admit_fresh(entries, fresh, now)
            probation = None  # (no warm-only test under smart: the admission replaces it)
        else:
            probation = self._probation_state(entries, now)  # C: also removes unproven entries that reached the SSD some other way

        # RAM -> SSD: the oldest checkpoints first; when busy, only after twice the retention. An idle backup is not
        # written: its checkpoint is still in VRAM, so the file would be a copy of data that is not lost.
        # (an entry being restored in the background, "busy", is neither written out nor dropped: the engine refuses)
        # Entries that were reused go first (so under SSD pressure the never-reused ones written after them are the first victims),
        # then the oldest. A superseded entry is not written at all: it is dropped once it is past the retention.
        ram = sorted((e for e in entries if e["ram_bytes"] > 0 and not e.get("backup") and not e.get("busy")),
                     key=lambda e: (not _hit(e), self.ram_since.get(e["id"], now)))
        for e in ram:
            age = now - self.ram_since.get(e["id"], now)
            if age < self.ram_retention_s or not (idle or age >= 2 * self.ram_retention_s):
                continue
            if not self.smart and _sup_durable(e):
                self.skipped_superseded += 1
                self._drop(e["id"], f"superseded: a deeper checkpoint of its conversation is on the SSD ({e['pos']} tokens)")
                continue
            if self._short(e):
                self.skipped_short += 1
                self._drop(e["id"], f"shorter than SPILL_SSD_MIN_TOKENS ({e['pos']} < {self.ssd_min_tokens} tokens)")
                continue
            if not self._gate(e, probation):
                continue
            if self.smart and not self._smart_admit(e, now):
                if self._drop(e["id"], "below the SSD's watermark (CACHE_POLICY=smart admission)"):
                    self.not_admitted += 1
                continue
            self._to_disk(e, now)

        # TTL: files older than ttl_days go, and so do stray files no entry refers to.
        entries = self.engine.spill_list()
        self.listed = entries
        for e in entries:
            entry_id = e["id"]
            if not e["on_disk"] or e.get("busy"):
                continue
            if entry_id not in self.written:
                self.written[entry_id] = _mtime(e["path"], now)
            if now - self.written[entry_id] >= self.ttl_s:
                self._drop(entry_id, "older than ttl_days")
        referenced = {Path(e["path"]).name for e in entries if e["path"]}
        for p in self.files_dir.glob("*" + self.suffix):  # (dedup layout: entries/ only; chunks/ is the engine's)
            if p.name in referenced:
                continue
            mtime = _mtime(p, None)
            if mtime is not None and now - mtime >= self.ttl_s:
                self.log(f"tiered cache: deleting stray {p.name}")
                _unlink(p)

        # Keep the SSD within its budget and reserve.
        try:
            self.make_room(0)
        except RuntimeError as ex:
            self._error(f"SSD budget: {ex}")
        if self.dedup:
            self.usage = self._read_usage()
            if self.usage and self.usage.get("pending_files"):  # chunk files whose deletion failed (a lock): the engine retries
                try:
                    self.engine.spill_sweep()
                except Exception as e:  # noqa: BLE001
                    self._error(f"sweep: {e}")
                self.usage = self._read_usage()

    def _short(self, e):
        return self.ssd_min_tokens > 0 and e.get("pos", 0) < self.ssd_min_tokens

    def _drop(self, entry_id, why):
        """True when the entry was dropped."""
        try:
            self.engine.spill_drop(entry_id)
        except RuntimeError as ex:
            self._error(f"drop {entry_id} ({why}): {ex}")
            return False
        self.written.pop(entry_id, None)
        self.ram_since.pop(entry_id, None)
        self.log(f"tiered cache: entry {entry_id} dropped ({why})")
        return True

    # ------------------------------------------------------------ never-hit entries (A, C)

    def _expire(self, entries, now):
        """A: drops the entries that have had no hit for unused_ttl_s since they came to RAM / were written, from RAM and from the SSD,
        whatever the free room. Not: entries with hits, the shared short prefixes (<= KEEP_SHORT_TOKENS, as "superseded"), idle backups
        (their checkpoint is still in VRAM, and the idle backup would copy it again), busy ones, and entries the engine reports no hits
        for. Returns the engine's list again when something went."""
        if self.unused_ttl_s <= 0:
            return entries
        gone = False
        for e in entries:
            if e.get("busy") or e.get("backup") or not _known(e) or _hit(e) or (e.get("pos") or 0) <= KEEP_SHORT_TOKENS:
                continue
            since = self.ram_since.get(e["id"]) if e["ram_bytes"] > 0 else self.written.get(e["id"])
            if since is None or now - since < self.unused_ttl_s:
                continue
            if self._drop(e["id"], f"never hit for {now - since:.0f} s (SPILL_UNUSED_TTL_S {self.unused_ttl_s:g})"):
                self.expired += 1
                gone = True
        if gone:
            entries = self.engine.spill_list()
            self.listed = entries
        return entries

    def _ssd_size(self, e):
        return e.get("logical_bytes") or _size(e.get("path")) or e.get("ram_bytes") or 0

    def _unproven(self, e, warm):
        """C: no hit on the entry and none on any entry of its agent (agent 0, unknown, is not an agent: never unproven), and not one of
        the short shared prefixes."""
        return (_known(e) and not _hit(e) and _owner(e) != 0 and _owner(e) not in warm
                and (e.get("pos") or 0) > KEEP_SHORT_TOKENS)

    def _probation_state(self, entries, now):
        """C: the unproven entries on the SSD ({"warm": agents with a hit, "disk": [(time written, id, bytes)], "used"}), after removing
        the ones that must not be there: those of `auto:` agents, and the oldest ones beyond SPILL_PROBATION_GIB. None: rule off."""
        if not self.ssd_warm_only:
            return None
        warm = {_owner(e) for e in entries if _known(e) and _hit(e)}
        disk = sorted((self.written.get(e["id"], now), e["id"], self._ssd_size(e)) for e in entries
                      if e["on_disk"] and not e.get("busy") and self._unproven(e, warm))
        st = {"warm": warm, "disk": [], "used": 0}
        owner = {e["id"]: _owner(e) for e in entries}
        for t, i, size in disk:
            if owner[i] in self.auto_owners and self._drop(i, "unproven agent without an id is not kept on the SSD (SPILL_SSD_WARM_ONLY)"):
                self.not_written += 1
                continue
            st["disk"].append((t, i, size))
            st["used"] += size
        while st["used"] > self.probation_bytes and st["disk"]:
            t, i, size = st["disk"].pop(0)
            st["used"] -= size  # (a failed drop stays on the SSD but no longer counts: the loop must end)
            if self._drop(i, "SPILL_PROBATION_GIB: the oldest unproven entry goes"):
                self.not_written += 1
        return st

    def _gate(self, e, st):
        """C: whether a RAM entry may be written to the SSD now. A hit entry or one of a warm agent: yes. An unproven one of an `auto:`
        agent: no (dropped, "not_written"). An unproven one of an agent with an id: yes within the probation quota, after dropping the
        oldest unproven entries on the SSD to make room; larger than the whole quota: no (dropped)."""
        if st is None or not self._unproven(e, st["warm"]):
            return True
        size = self._ssd_size(e)
        if _owner(e) in self.auto_owners or size > self.probation_bytes:
            if self._drop(e["id"], "unproven agent: not written to the SSD (SPILL_SSD_WARM_ONLY)"):
                self.not_written += 1
            return False
        while st["used"] + size > self.probation_bytes and st["disk"]:
            t, i, old = st["disk"].pop(0)
            st["used"] -= old
            if self._drop(i, "SPILL_PROBATION_GIB: the oldest unproven entry goes"):
                self.not_written += 1
        st["disk"].append((self.clock(), e["id"], size))
        st["used"] += size
        return True

    def _to_disk(self, e, now):
        """Writes one RAM entry to a file in the directory. Returns "written", "kept" (the SSD room could not be
        checked now: retried later) or "dropped" (no room on the SSD, or the export failed)."""
        entry_id = e["id"]
        # (dedup layout: what the export WRITES now: its manifest + the chunk files that are not on the SSD yet)
        incoming = e.get("new_disk_bytes", e["ram_bytes"]) if self._layout() else e["ram_bytes"]
        try:
            room = self.make_room(incoming, entry_id)
        except RuntimeError as ex:
            # Unknown whether there is room, so this is not "no room": keep the entry for a later tick.
            self._error(f"make room for {entry_id}: {ex}")
            return "kept"
        if not room:
            self._drop(entry_id, "no room on SSD")
            return "dropped"
        path = self.files_dir / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}{self.suffix}"
        t0 = time.monotonic()
        try:
            self.engine.spill_export(entry_id, str(path))
        except RuntimeError as ex:
            self._error(f"export {entry_id}: {ex}")
            _unlink(path)
            self._drop(entry_id, "export failed")
            return "dropped"
        dt = time.monotonic() - t0
        if incoming >= WRITE_RATE_MIN_BYTES and dt > 0:  # (small writes measure the latency, not the throughput)
            self.write_bps = incoming / dt
        self.written[entry_id] = now
        self.ram_since.pop(entry_id, None)
        if self.dedup:
            self.usage = self._read_usage()
        self.log(f"tiered cache: entry {entry_id} ({e['ram_bytes']} bytes) written to {path.name}")
        return "written"

    def flush(self, budget_s, clock=None):
        """On a stop (worker.Scheduler): writes the RAM entries to the directory, oldest first, until budget_s seconds
        have passed (clock: time.monotonic). Retention and idle do not apply. Each entry starts only inside the
        budget, so the stop can overrun it by one export. Returns {"written": n, "left": m}, m being the entries
        not tried. Without a directory nothing is written."""
        clock = clock or time.monotonic
        result = {"written": 0, "left": 0}
        if self.directory is None or budget_s <= 0:
            return result
        deadline = clock() + budget_s
        now = self.clock()
        ram = sorted((e for e in self.engine.spill_list()  # (short ones stay in RAM, and so do those whose deeper checkpoint is already on the SSD; deeper ones first)
                      if e["ram_bytes"] > 0 and not e.get("busy") and not self._short(e) and (self.smart or not _sup_durable(e))),
                     key=lambda e: (not _hit(e), -(e.get("pos") or 0), self.ram_since.get(e["id"], now)))
        rate = self.write_bps or DEFAULT_WRITE_BPS
        skipped = 0
        probation = None if self.smart else self._probation_state(self.engine.spill_list(), now)
        for i, e in enumerate(ram):
            remaining = deadline - clock()
            if remaining <= 0:
                result["left"] = len(ram) - i + skipped
                break
            # An export that would not finish in the time left is not started (a file cut off by the server's terminate is lost,
            # and the stop would overrun): a smaller one later in the list may still fit.
            incoming = e.get("new_disk_bytes", e["ram_bytes"]) if self._layout() else e["ram_bytes"]
            if incoming / rate > remaining:
                skipped += 1
                continue
            if not self._gate(e, probation):
                continue
            if self._to_disk(e, now) == "written":
                result["written"] += 1
        else:
            result["left"] = skipped
        self.log(f"tiered cache: stop flush wrote {result['written']} of {len(ram)} RAM entries "
                 f"({result['left']} left in RAM)")
        return result

    def make_room(self, incoming, entry_id=None):
        """True when `incoming` bytes fit the SSD budget and reserve, after deleting the oldest files if needed.
        False when they do not: deleting nothing if `incoming` cannot fit even with every file deleted, otherwise
        after deleting what it could.

        A file whose drop fails (e.g. the entry was promoted back to the GPU since the list was read) is skipped,
        and its bytes are not counted as freed; the other files are still tried.

        With incoming == 0 (the tick's cap pass) an infeasible reserve is not a refusal: the budget is still
        enforced, deleting the oldest files until used <= budget and no further. The cap only ever deletes this
        tier's own files down to the cap, so it is safe even when other files fill the disk; the reserve is only
        pursued when it can actually be reached.

        Dedup layout: `used` is the PHYSICAL size (each chunk file once, engine.spill_disk_usage), a drop frees the entry's
        unique_disk_bytes (see _pick_victim), and the entry to be written (entry_id) may share chunk files with the
        victims: its new_disk_bytes is read again after each drop."""
        if self.directory is None:
            return False
        if self._layout():
            return self._make_room_dedup(incoming, entry_id)
        entries = self.engine.spill_list()
        self.listed = entries
        disk = [e for e in entries if e["on_disk"]]
        sizes = {e["id"]: _size(e["path"]) for e in disk}
        used = sum(sizes.values())
        free = self.disk_free(str(self.directory))
        reserve_reachable = free + used - incoming >= self.reserve
        if incoming > self.budget or (incoming > 0 and not reserve_reachable):
            return False

        def fits():
            return used + incoming <= self.budget and (free - incoming >= self.reserve or not reserve_reachable)

        self.ssd_use = self._use_of(disk, lambda e: sizes[e["id"]], "ssd_bytes")
        skipped = set()
        while not fits():
            e = self._pick_victim([x for x in disk if not x.get("busy") and x["id"] not in skipped], 0, disk)  # (busy: its file is read now)
            if e is None:
                break
            entry_id = e["id"]
            skipped.add(entry_id)
            try:
                self.engine.spill_drop(entry_id)
            except RuntimeError as ex:
                self._error(f"make room, drop {entry_id}: {ex}")
                continue  # its bytes are still on disk: not freed
            self._evicted(entry_id, entries)
            self.written.pop(entry_id, None)
            self.log(f"tiered cache: entry {entry_id} deleted to make room ({sizes[entry_id]} bytes)")
            used -= sizes[entry_id]
            free += sizes[entry_id]
            self.ssd_use[_owner(e)] = max(0.0, self.ssd_use.get(_owner(e), 0.0) - sizes[entry_id])
        return fits()


    def _pick_victim(self, disk, need, tier=None):
        """The entry to delete next, among the SSD entries that can be dropped (a small function, so the policy can be swapped).
        The retention order (the engine applies the same to RAM): the agent using the most SSD bytes (self.ssd_use; with
        fair_agents), then in it superseded entries, never-hit before hit ones, then the old rule: an entry that is mostly
        shared (little of it would be freed, MOSTLY_SHARED) only after the ones that free real bytes, and among those the oldest
        written. `need`: bytes still to free.
        Before the agent step (B): entries of COLD agents (self.cold_first): the agent has no hit in any entry of the tier (`tier`: all
        entries on the SSD, busy ones too; default `disk`) and its newest entry was written at least cold_after_s ago. If any candidate
        belongs to one the candidates are restricted to them; the agent step then takes the cold agent with the MOST bytes (so the order
        among cold agents is "most bytes first"; with fair_agents off: all cold entries, then the entry order below)."""
        if not disk:
            return None
        if self.smart:
            return self._smart_victim(disk)
        cand = disk
        if self.cold_first:
            now = self.clock()
            whole = disk if tier is None else tier
            cold = _cold_owners(whole, {e["id"]: self.written.get(e["id"], now) for e in whole}, now, self.cold_after_s)
            if cold and any(_owner(e) in cold for e in cand):
                cand = [e for e in cand if _owner(e) in cold]
        if self.fair_agents:
            use = self.ssd_use
            top = max({_owner(e) for e in cand}, key=lambda o: (use.get(o, 0.0), -o))  # (ties: the smaller id, as the engine)
            cand = [e for e in cand if _owner(e) == top]

        def key(e):
            unique = e.get("unique_disk_bytes", 0)
            size = e.get("logical_bytes") or unique
            return (not e.get("superseded"), _hit(e), unique < MOSTLY_SHARED * size, self.written.get(e["id"], 0.0))
        return min(cand, key=key)

    # ------------------------------------------------------------ CACHE_POLICY=smart (the SSD tier)

    def _cand(self, e, now, size=None):
        """cache_policy.Cand of an entry for the SSD tier, from the engine's spill_meta numbers (an engine without them: no parent, a
        leaf, no junction). size: the bytes it costs on the SSD (default: what dropping it frees there)."""
        if size is None:
            size = e.get("unique_disk_bytes") or e.get("logical_bytes") or _size(e.get("path")) or e.get("ram_bytes") or 0
        written = self.written.get(e["id"], now)
        return cp.Cand(depth=e.get("pos") or 0, parent_depth=e.get("parent_pos") or 0, bytes=size, hits=e.get("hits") or 0,
                       junction=bool(e.get("junction")), superseded=bool(e.get("superseded")),
                       leaf=not (int(e.get("desc_tiers") or 0) & cp.TIER_SSD), young=now - written < self.smart_params.min_resident_s,
                       stamp=self.gd.stamp_at(e.get("last_use") or 0), age_key=written, ref=e["id"])

    def _smart_victim(self, disk):
        now = self.clock()
        cands = [self._cand(e, now) for e in disk]
        k = cp.smart_pick(cands, self.smart_params)
        if k.idx < 0:
            return None
        self._pick = (cands[k.idx].ref, k)
        return disk[k.idx]

    def _evicted(self, entry_id, entries):
        """A victim of make_room was dropped: under smart the watermark moves to its priority (legacy: nothing)."""
        if self.smart and self._pick is not None and self._pick[0] == entry_id:
            self.gd.evicted(self._pick[1], max((e.get("last_use") or 0 for e in entries), default=0))
        self._pick = None

    def _smart_admit(self, e, now, size=None):
        """Whether entry e may stay on / go to the SSD (smart): its priority there is above the SSD's watermark L. size: what it costs
        there (default for a RAM entry: new_disk_bytes, the chunk files and manifest not on the SSD yet)."""
        if self.gd.L <= 0:
            return True  # (nothing was evicted from the SSD yet: there is room)
        if size is None:
            size = (e.get("new_disk_bytes", e.get("ram_bytes")) if self._layout() else e.get("ram_bytes")) or 0
        return cp.smart_priority(self.smart_params, self._cand(e, now, size=size)) > self.gd.L

    def _admit_fresh(self, entries, fresh, now):
        """Entries the engine's RAM cap wrote to the SSD since the last tick get the admission test now (dropped when they fail).
        Returns the engine's list again when something went."""
        if not fresh or self.gd.L <= 0:
            return entries
        byid = {e["id"]: e for e in entries}
        gone = False
        for i in fresh:
            e = byid.get(i)
            if e is None or e.get("busy") or not e["on_disk"] or e["ram_bytes"] > 0:
                continue
            if not self._smart_admit(e, now, size=e.get("unique_disk_bytes") or e.get("logical_bytes") or _size(e.get("path"))):
                if self._drop(i, "below the SSD's watermark (CACHE_POLICY=smart admission)"):
                    self.not_admitted += 1
                    gone = True
        if gone:
            entries = self.engine.spill_list()
            self.listed = entries
        return entries

    def _read_agents(self):
        """The engine's per-agent bytes [{"owner", "ram_bytes", "ssd_bytes", "ram_entries", "ssd_entries"}], [] without the call."""
        fn = getattr(self.engine, "spill_agents", None)
        if fn is None:
            return []
        try:
            return fn() or []
        except Exception as e:  # noqa: BLE001 - an engine without the call, or it failed: the per-entry sizes are the fallback
            self._error(f"spill_agents: {e}")
            return []

    def _use_of(self, entries, size, key):
        """owner -> bytes in a tier: the engine's fractional figure (shared chunks split between the agents referencing them) when
        it reports one, else the entries' own sizes."""
        rows = self._read_agents()
        self.agent_rows = rows or self.agent_rows
        if rows:
            return {r["owner"]: float(r.get(key, 0)) for r in rows}
        use = {}
        for e in entries:
            use[_owner(e)] = use.get(_owner(e), 0.0) + size(e)
        return use

    def _make_room_dedup(self, incoming, entry_id):
        entries = self.engine.spill_list()
        self.listed = entries
        usage = self._read_usage()
        if usage is None:  # (the engine stopped answering: no decision)
            raise RuntimeError("spill_disk_usage failed")
        used = usage["physical_bytes"]
        free = self.disk_free(str(self.directory))
        # Whether it can EVER fit is judged by its full size: once exported its manifest and all its chunk files are on the SSD
        # (new_disk_bytes is only what is missing now, and drops may make it grow). So nothing is deleted for an entry that
        # cannot fit even when every other file is gone.
        mine = next((x for x in entries if x["id"] == entry_id), None) if entry_id is not None else None
        full = max(incoming, (mine or {}).get("logical_bytes", 0))
        reserve_reachable = free + used - full >= self.reserve
        if full > self.budget or (full > 0 and not reserve_reachable):
            return False

        def fits():
            return used + incoming <= self.budget and (free - incoming >= self.reserve or not reserve_reachable)

        skipped = set()  # entries tried (a drop that failed stays on the SSD): the loop is bounded by the number of entries
        self.ssd_use = self._use_of([e for e in entries if e["on_disk"]], lambda e: e.get("logical_bytes", 0), "ssd_bytes")
        while not fits():
            disk = [e for e in entries if e["on_disk"] and not e.get("busy") and e["id"] not in skipped]
            need = max(used + incoming - self.budget, self.reserve - (free - incoming) if reserve_reachable else 0)
            e = self._pick_victim(disk, need, [x for x in entries if x["on_disk"]])
            if e is None:
                break
            victim = e["id"]
            skipped.add(victim)
            try:
                self.engine.spill_drop(victim)
            except RuntimeError as ex:
                self._error(f"make room, drop {victim}: {ex}")
                continue  # its bytes are still on disk: not freed
            self._evicted(victim, entries)
            self.written.pop(victim, None)
            # The siblings' unique bytes and the incoming entry's new bytes changed with this drop: ask again.
            entries = self.engine.spill_list()
            self.listed = entries
            self.ssd_use = self._use_of([x for x in entries if x["on_disk"]], lambda x: x.get("logical_bytes", 0), "ssd_bytes")
            before, usage = used, self._read_usage()
            if usage is None:
                raise RuntimeError("spill_disk_usage failed")
            used = usage["physical_bytes"]
            free += before - used
            if entry_id is not None:
                now_e = next((x for x in entries if x["id"] == entry_id), None)
                if now_e is not None and "new_disk_bytes" in now_e:
                    incoming = now_e["new_disk_bytes"]
            self.log(f"tiered cache: entry {victim} deleted to make room ({before - used} bytes freed)")
        self.usage = usage
        return fits()

    # ------------------------------------------------------------ metrics

    def status(self):
        """For the metrics, from the last look at the engine's list (no engine call)."""
        disk = [e for e in self.listed if e["on_disk"]]
        ram = [e for e in self.listed if e["ram_bytes"] > 0]
        # (ram_bytes here is the entries' LOGICAL size: what an entry takes in a file and what the SSD budget counts. The
        # engine's RAM tier holds less when entries share chunks: stats.tiers_block has the physical number.)
        # Dedup layout: an entry's "size" is its logical size (what its file set would take alone); the SSD's real use is the
        # engine's physical figure (each shared chunk file once), taken at the last tick / write (self.usage).
        dedup = bool(self.dedup) and self.usage is not None
        if dedup:
            sizes = {id(e): e.get("logical_bytes", 0) for e in disk}
            ssd_physical = self.usage["physical_bytes"]
            ssd_logical = sum(sizes.values())
        else:
            sizes = {id(e): _size(e["path"]) for e in disk}
            ssd_physical = ssd_logical = sum(sizes.values())
        # Per tier: entries and bytes, and how many of them are live backups (copies of checkpoints the VRAM pool still
        # holds) rather than evicted ones (their only copy).
        split = {}
        for tier, entries, size in (("ram", ram, lambda e: e["ram_bytes"]), ("disk", disk, lambda e: sizes[id(e)])):
            for kind, sel in (("backup", True), ("evicted", False)):
                part = [e for e in entries if bool(e.get("backup")) == sel]
                split[f"{tier}_{kind}_entries"], split[f"{tier}_{kind}_bytes"] = len(part), sum(size(e) for e in part)
        # Reuse (retention): per tier the entries / bytes that were restored at least once ("reused") vs never ("never"), and the
        # superseded ones. The reuse RATE (restored / spilled) is stats.tiers_block's, from the engine's counters.
        reuse = {}
        for tier, entries, size in (("ram", ram, lambda e: e["ram_bytes"]), ("disk", disk, lambda e: sizes[id(e)])):
            for kind, sel in (("reused", _hit), ("never", lambda e: not _hit(e)), ("superseded", lambda e: bool(e.get("superseded")))):
                part = [e for e in entries if sel(e)]
                reuse[f"{tier}_{kind}_entries"], reuse[f"{tier}_{kind}_bytes"] = len(part), sum(size(e) for e in part)
        return {"ram_entries": len(ram), "ram_bytes": sum(e["ram_bytes"] for e in ram),
                "ram_logical_bytes": sum(e["ram_bytes"] for e in ram),
                "disk_entries": len(disk), "ssd_bytes": ssd_physical, "ssd_logical_bytes": ssd_logical,
                "ssd_physical_bytes": ssd_physical,
                "ssd_dedup_ratio": ssd_logical / ssd_physical if dedup and ssd_logical and ssd_physical else None,
                "backup_entries": sum(1 for e in self.listed if e.get("backup")), **split, **reuse,
                "fair_agents": self.fair_agents, "ssd_skipped_superseded": self.skipped_superseded,
                "expired": self.expired, "not_written": self.not_written, "cold_first": self.cold_first,
                "unused_ttl_s": self.unused_ttl_s, "cold_after_s": self.cold_after_s, "ssd_warm_only": self.ssd_warm_only,
                "probation_bytes": int(self.probation_bytes),
                "agent_rows": list(self.agent_rows),
                "ssd_enabled": self.directory is not None,
                "ssd_off_reason": None if self.directory is not None else ("gib" if not self.budget > 0 else "dir"),
                "ssd_budget_bytes": int(self.budget) if self.directory is not None else 0, "ssd_min_tokens": self.ssd_min_tokens,
                "ssd_skipped_short": self.skipped_short, "last_error": self.last_error,
                "cache_policy": self.cache_policy, "ssd_not_admitted": self.not_admitted,
                **{f"ssd_smart_{k}": v for k, v in self.gd.stats().items()}}
