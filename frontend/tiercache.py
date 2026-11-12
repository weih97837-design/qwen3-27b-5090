"""Tiered KV cache policy (SPILL_RAM_GIB > 0): what happens to the checkpoints the engine spilled to host RAM.

When the GPU KV pool is full, the engine copies an evicted prefix-cache checkpoint to host RAM (a spill). This
policy keeps that RAM bounded: a checkpoint that has been in RAM for ram_retention_s (2x that when the server is
busy) is written to a file in the directory (engine.spill_export). Files stay within an SSD budget (oldest
written first) and a free-space reserve, and are deleted after ttl_days. A file an earlier run wrote is indexed at
start() (engine.spill_import). A checkpoint that is reused leaves the engine's list by itself.

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
SUFFIX = ".q27kv"
TMP_SUFFIX = ".q27kv.tmp"


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
                 tick_s=30, clock=time.time, disk_free=None, log=print):
        self.engine = engine
        self.directory = Path(directory) if directory else None  # None: no SSD tier
        self.ram_retention_s = float(ram_retention_s)
        self.budget = float(ssd_gib) * GiB
        self.reserve = float(ssd_min_free_gib) * GiB
        self.ttl_s = float(ttl_days) * DAY
        self.tick_s = float(tick_s)
        self.clock = clock
        self.disk_free = disk_free or (lambda d: shutil.disk_usage(d).free)
        self.log = log
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
                                  ("SPILL_TICK_S", "tick_s", 30)):
            text = os.environ.get(env)
            opts[key] = default if text is None or not text.strip() else _float(env, text)
        return cls(engine, os.environ.get("SPILL_DIR", ""), log=log, **opts)

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
        for p in self.directory.glob("*" + TMP_SUFFIX):  # left by a write that did not finish
            _unlink(p)
        for p in sorted(self.directory.glob("*" + SUFFIX)):
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

        # RAM -> SSD: the oldest checkpoints first; when busy, only after twice the retention.
        ram = sorted((e for e in entries if e["ram_bytes"] > 0),
                     key=lambda e: self.ram_since.get(e["id"], now))
        for e in ram:
            entry_id = e["id"]
            age = now - self.ram_since.get(entry_id, now)
            if age < self.ram_retention_s or not (idle or age >= 2 * self.ram_retention_s):
                continue
            try:
                room = self.make_room(e["ram_bytes"])
            except RuntimeError as ex:
                # Unknown whether there is room, so this is not "no room": keep the entry for a later tick.
                self._error(f"make room for {entry_id}: {ex}")
                continue
            if not room:
                self._drop(entry_id, "no room on SSD")
                continue
            path = self.directory / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}{SUFFIX}"
            try:
                self.engine.spill_export(entry_id, str(path))
            except RuntimeError as ex:
                self._error(f"export {entry_id}: {ex}")
                _unlink(path)
                self._drop(entry_id, "export failed")
                continue
            self.written[entry_id] = now
            self.ram_since.pop(entry_id, None)
            self.log(f"tiered cache: entry {entry_id} ({e['ram_bytes']} bytes) written to {path.name}")

        # TTL: files older than ttl_days go, and so do stray files no entry refers to.
        entries = self.engine.spill_list()
        self.listed = entries
        for e in entries:
            entry_id = e["id"]
            if not e["on_disk"]:
                continue
            if entry_id not in self.written:
                self.written[entry_id] = _mtime(e["path"], now)
            if now - self.written[entry_id] >= self.ttl_s:
                self._drop(entry_id, "older than ttl_days")
        referenced = {Path(e["path"]).name for e in entries if e["path"]}
        for p in self.directory.glob("*" + SUFFIX):
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

    def _drop(self, entry_id, why):
        try:
            self.engine.spill_drop(entry_id)
        except RuntimeError as ex:
            self._error(f"drop {entry_id} ({why}): {ex}")
            return
        self.written.pop(entry_id, None)
        self.ram_since.pop(entry_id, None)
        self.log(f"tiered cache: entry {entry_id} dropped ({why})")

    def make_room(self, incoming):
        """True when `incoming` bytes fit the SSD budget and reserve, after deleting the oldest files if needed.
        False when they do not: deleting nothing if `incoming` cannot fit even with every file deleted, otherwise
        after deleting what it could.

        A file whose drop fails (e.g. the entry was promoted back to the GPU since the list was read) is skipped,
        and its bytes are not counted as freed; the other files are still tried.

        With incoming == 0 (the tick's cap pass) an infeasible reserve is not a refusal: the budget is still
        enforced, deleting the oldest files until used <= budget and no further. The cap only ever deletes this
        tier's own files down to the cap, so it is safe even when other files fill the disk; the reserve is only
        pursued when it can actually be reached."""
        if self.directory is None:
            return False
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

    # ------------------------------------------------------------ metrics

    def status(self):
        """For the metrics, from the last look at the engine's list (no engine call)."""
        disk = [e for e in self.listed if e["on_disk"]]
        ram = [e for e in self.listed if e["ram_bytes"] > 0]
        return {"ram_entries": len(ram), "ram_bytes": sum(e["ram_bytes"] for e in ram),
                "disk_entries": len(disk), "ssd_bytes": sum(_size(e["path"]) for e in disk),
                "ssd_budget_bytes": int(self.budget), "last_error": self.last_error}
