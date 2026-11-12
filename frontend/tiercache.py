"""Tiered KV cache policy (SPILL_RAM_GIB > 0): what happens to the checkpoints the engine spilled to host RAM.

When the GPU KV pool is full, the engine copies an evicted prefix-cache checkpoint to host RAM (a spill). This
policy keeps that RAM bounded: a checkpoint that has been in RAM for ram_retention_s (2x that when the server is
busy) is written to a file in the directory (engine.spill_export). Files stay within an SSD budget (oldest
written first) and a free-space reserve, and are deleted after ttl_days. A file an earlier run wrote is indexed at
start() (engine.spill_import). A checkpoint that is reused leaves the engine's list by itself.

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

Everything runs on the caller's thread: the engine is not thread-safe.
"""

import math
import os
import shutil
import time
import uuid
from pathlib import Path

GiB = 1 << 30
DAY = 86400.0
SUFFIX = ".q27kv"                  # the flat layout (older engine); in the dedup layout: a legacy file, deleted at start
TMP_SUFFIX = ".q27kv.tmp"
SUFFIX_MANIFEST = ".q27m"          # SSD dedup layout: SPILL_DIR/entries/<id>.q27m (+ the engine's SPILL_DIR/chunks/*.kvc)
ENTRIES_DIR = "entries"
MOSTLY_SHARED = 0.25               # an entry whose unique bytes are under this fraction of its size frees little when dropped
# The longest a stop may spend writing (SPILL_FLUSH_ON_STOP_S is cut to it). server.py gives the engine process
# 15 s to exit on an unload or restart and 10 s on shutdown (EngineClient._stop, stop); a flush cut off by the
# server's terminate would leave a file half written.
STOP_FLUSH_MAX_S = 8.0


def _float(name, raw):
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a number >= 0, got {raw!r}")
    return v


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


class TierPolicy:
    def __init__(self, engine, directory, *, ram_retention_s=3600, ssd_gib=100, ssd_min_free_gib=10, ttl_days=7,
                 tick_s=30, clock=time.time, disk_free=None, log=print, backup_idle_s=60, flush_on_stop_s=8,
                 backup_victims=8, ssd_min_tokens=8192):
        self.engine = engine
        self.directory = Path(directory) if directory else None  # None: no SSD tier
        self.ram_retention_s = float(ram_retention_s)
        self.budget = float(ssd_gib) * GiB
        self.reserve = float(ssd_min_free_gib) * GiB
        self.ttl_s = float(ttl_days) * DAY
        self.tick_s = float(tick_s)
        self.backup_idle_s = float(backup_idle_s)      # idle seconds before a VRAM checkpoint is backed up (0: off)
        self.flush_on_stop_s = float(flush_on_stop_s)  # the time a stop may spend writing (0: off)
        self.backup_victims = int(backup_victims)      # the idle backup copies only one of the next N checkpoints the pool would evict (0: any)
        # Spilled entries shorter than this many tokens (the engine's "pos") are never written to the SSD: every entry
        # carries the same ~190 MiB recurrent state, and a short prefix recomputes in about the time a read takes (0: off).
        self.ssd_min_tokens = max(0, int(ssd_min_tokens))
        self.skipped_short = 0                         # short entries dropped instead of written
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
                                  ("SPILL_SSD_GIB", "ssd_gib", 100),
                                  ("SPILL_SSD_MIN_FREE_GIB", "ssd_min_free_gib", 10),
                                  ("SPILL_TTL_DAYS", "ttl_days", 7),
                                  ("SPILL_TICK_S", "tick_s", 30),
                                  ("VRAM_BACKUP_IDLE_S", "backup_idle_s", 60),
                                  ("SPILL_FLUSH_ON_STOP_S", "flush_on_stop_s", 8),
                                  ("VRAM_BACKUP_VICTIMS", "backup_victims", 8),
                                  ("SPILL_SSD_MIN_TOKENS", "ssd_min_tokens", 32768)):
            text = os.environ.get(env)
            opts[key] = default if text is None or not text.strip() else _float(env, text)
        if opts["flush_on_stop_s"] > STOP_FLUSH_MAX_S:
            log(f"tiered cache: SPILL_FLUSH_ON_STOP_S {opts['flush_on_stop_s']:g} cut to {STOP_FLUSH_MAX_S:g} s "
                "(the server stops the engine within 10-15 s)")
            opts["flush_on_stop_s"] = STOP_FLUSH_MAX_S
        return cls(engine, os.environ.get("SPILL_DIR", ""), log=log, **opts)

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
                except Exception as e:  # noqa: BLE001 - an unreadable file is just not ours
                    self._error(f"import {p.name}: {e}")
                    entry_id = 0
                if not entry_id:
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
        listed = {e["id"] for e in entries}
        for book in (self.written, self.ram_since):
            for entry_id in [i for i in book if i not in listed]:
                del book[entry_id]
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

        if self.directory is None:  # no SSD tier: RAM entries stay until the engine's own cap drops them
            return

        # RAM -> SSD: the oldest checkpoints first; when busy, only after twice the retention. An idle backup is not
        # written: its checkpoint is still in VRAM, so the file would be a copy of data that is not lost.
        # (an entry being restored in the background, "busy", is neither written out nor dropped: the engine refuses)
        ram = sorted((e for e in entries if e["ram_bytes"] > 0 and not e.get("backup") and not e.get("busy")),
                     key=lambda e: self.ram_since.get(e["id"], now))
        for e in ram:
            age = now - self.ram_since.get(e["id"], now)
            if age < self.ram_retention_s or not (idle or age >= 2 * self.ram_retention_s):
                continue
            if self._short(e):
                self.skipped_short += 1
                self._drop(e["id"], f"shorter than SPILL_SSD_MIN_TOKENS ({e['pos']} < {self.ssd_min_tokens} tokens)")
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

    def _short(self, e):
        return self.ssd_min_tokens > 0 and e.get("pos", 0) < self.ssd_min_tokens

    def _drop(self, entry_id, why):
        try:
            self.engine.spill_drop(entry_id)
        except RuntimeError as ex:
            self._error(f"drop {entry_id} ({why}): {ex}")
            return
        self.written.pop(entry_id, None)
        self.ram_since.pop(entry_id, None)
        self.log(f"tiered cache: entry {entry_id} dropped ({why})")

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
        try:
            self.engine.spill_export(entry_id, str(path))
        except RuntimeError as ex:
            self._error(f"export {entry_id}: {ex}")
            _unlink(path)
            self._drop(entry_id, "export failed")
            return "dropped"
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
        ram = sorted((e for e in self.engine.spill_list()
                      if e["ram_bytes"] > 0 and not e.get("busy") and not self._short(e)),  # (short ones stay in RAM)
                     key=lambda e: self.ram_since.get(e["id"], now))
        for i, e in enumerate(ram):
            if clock() >= deadline:
                result["left"] = len(ram) - i
                break
            if self._to_disk(e, now) == "written":
                result["written"] += 1
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

        for e in sorted(disk, key=lambda e: self.written.get(e["id"], 0.0)):
            if fits():
                break
            entry_id = e["id"]
            if e.get("busy"):  # (being restored: its file is read right now)
                continue
            try:
                self.engine.spill_drop(entry_id)
            except RuntimeError as ex:
                self._error(f"make room, drop {entry_id}: {ex}")
                continue  # its bytes are still on disk: not freed
            self.written.pop(entry_id, None)
            self.log(f"tiered cache: entry {entry_id} deleted to make room ({sizes[entry_id]} bytes)")
            used -= sizes[entry_id]
            free += sizes[entry_id]
        return fits()


    def _pick_victim(self, disk, need):
        """The entry to delete next, among the SSD entries that can be dropped (a small function, so the policy can be swapped).
        Oldest written first, except that an entry that is mostly shared (little of it would be freed, MOSTLY_SHARED) goes only
        after the ones that free real bytes, and among those the oldest. `need`: bytes still to free."""
        def key(e):
            unique = e.get("unique_disk_bytes", 0)
            size = e.get("logical_bytes") or unique
            return (unique < MOSTLY_SHARED * size, self.written.get(e["id"], 0.0))
        return min(disk, key=key) if disk else None

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
        while not fits():
            disk = [e for e in entries if e["on_disk"] and not e.get("busy") and e["id"] not in skipped]
            need = max(used + incoming - self.budget, self.reserve - (free - incoming) if reserve_reachable else 0)
            e = self._pick_victim(disk, need)
            if e is None:
                break
            victim = e["id"]
            skipped.add(victim)
            try:
                self.engine.spill_drop(victim)
            except RuntimeError as ex:
                self._error(f"make room, drop {victim}: {ex}")
                continue  # its bytes are still on disk: not freed
            self.written.pop(victim, None)
            # The siblings' unique bytes and the incoming entry's new bytes changed with this drop: ask again.
            entries = self.engine.spill_list()
            self.listed = entries
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
        return {"ram_entries": len(ram), "ram_bytes": sum(e["ram_bytes"] for e in ram),
                "ram_logical_bytes": sum(e["ram_bytes"] for e in ram),
                "disk_entries": len(disk), "ssd_bytes": ssd_physical, "ssd_logical_bytes": ssd_logical,
                "ssd_physical_bytes": ssd_physical,
                "ssd_dedup_ratio": ssd_logical / ssd_physical if dedup and ssd_logical and ssd_physical else None,
                "backup_entries": sum(1 for e in self.listed if e.get("backup")), **split,
                "ssd_budget_bytes": int(self.budget), "ssd_min_tokens": self.ssd_min_tokens,
                "ssd_skipped_short": self.skipped_short, "last_error": self.last_error}
