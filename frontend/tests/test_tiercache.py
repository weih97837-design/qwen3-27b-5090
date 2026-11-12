"""The tiered cache policy (frontend/tiercache.py) against a fake engine with real files: no GPU."""

import os
import pathlib
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tiercache  # noqa: E402
from tiercache import TierPolicy  # noqa: E402

GiB = tiercache.GiB
DAY = tiercache.DAY
B = 1 / GiB  # one byte, in the GiB units the policy takes (exact in binary floats)
MAGIC = b"Q27KV\n"  # the fake engine's files start with it; anything else is "not this model"


class FakeEngine:
    def __init__(self):
        self.entries = {}
        self.next_id = 1
        self.exports, self.drops, self.lists = [], [], 0
        self.fail_export = set()
        self.fail_drop = set()  # ids whose drop raises, as when the entry was promoted back to the GPU

    def add_ram(self, size, pos=10_000):  # (pos: the token count; the default is above SPILL_SSD_MIN_TOKENS)
        entry_id = self.next_id
        self.next_id += 1
        self.entries[entry_id] = {"id": entry_id, "pos": pos, "ram_bytes": size, "on_disk": False,
                                  "last_use": 0, "path": ""}
        return entry_id

    def spill_list(self):
        self.lists += 1
        return [dict(e) for _, e in sorted(self.entries.items())]

    def spill_export(self, entry_id, path):
        self.exports.append(entry_id)
        if entry_id in self.fail_export:
            raise RuntimeError("export failed")
        e = self.entries[entry_id]
        Path(path).write_bytes(MAGIC + b"x" * (e["ram_bytes"] - len(MAGIC)))
        e.update(ram_bytes=0, on_disk=True, path=path)

    def spill_drop(self, entry_id):
        self.drops.append(entry_id)
        if entry_id not in self.entries:
            raise RuntimeError("no such entry")
        if entry_id in self.fail_drop:
            raise RuntimeError("entry is on the GPU")
        e = self.entries.pop(entry_id)
        if e["path"]:
            Path(e["path"]).unlink(missing_ok=True)

    def spill_import(self, path):
        if Path(path).name.startswith("boom"):
            raise RuntimeError("unreadable")
        if not Path(path).read_bytes().startswith(MAGIC):
            return 0
        entry_id = self.next_id
        self.next_id += 1
        self.entries[entry_id] = {"id": entry_id, "pos": 0, "ram_bytes": 0, "on_disk": True,
                                  "last_use": 0, "path": path}
        return entry_id

    def spill_stats(self):
        es = self.entries.values()
        return {"entries": len(self.entries), "in_ram": sum(e["ram_bytes"] > 0 for e in es),
                "on_disk": sum(e["on_disk"] for e in es), "ram_bytes": sum(e["ram_bytes"] for e in es),
                "spilled": len(self.exports), "restored": 0, "written": 0, "dropped": len(self.drops),
                "subsumed": 0, "failed": len(self.fail_export)}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    eng = FakeEngine()
    clock = Clock()
    free = {"bytes": 10 ** 12}
    logs = []
    d = tmp_path / "spill"
    tier = TierPolicy(eng, str(d), ram_retention_s=3600, ssd_gib=1000 * B, ssd_min_free_gib=100 * B,
                      ttl_days=7, tick_s=0, clock=clock, disk_free=lambda _d: free["bytes"], log=logs.append)
    tier.start()  # (the worker does this first: it makes the directory)
    eng.lists = 0
    return types.SimpleNamespace(eng=eng, tier=tier, dir=d, clock=clock, free=free, logs=logs)


def test_export_after_retention_only_when_idle(env):
    rid = env.eng.add_ram(400)
    env.clock.t = 0
    env.tier.tick(idle=True)
    env.clock.t = 3599
    env.tier.tick(idle=True)
    assert env.eng.exports == []
    env.clock.t = 3600
    env.tier.tick(idle=True)
    assert env.eng.exports == [rid]
    entry = env.eng.entries[rid]
    assert entry["on_disk"] and entry["ram_bytes"] == 0
    assert Path(entry["path"]).exists() and Path(entry["path"]).parent == env.dir


def test_busy_server_exports_only_after_double_retention(env):
    rid = env.eng.add_ram(400)
    for t, idle in ((0, False), (3600, False), (7199, False)):
        env.clock.t = t
        env.tier.tick(idle=idle)
    assert env.eng.exports == []
    env.clock.t = 7200
    env.tier.tick(idle=False)
    assert env.eng.exports == [rid]


def test_ttl_deletes_files_after_seven_days_and_strays_too(env):
    env.tier.ram_retention_s = 0
    rid = env.eng.add_ram(400)
    env.clock.t = 0
    env.tier.tick(idle=True)
    path = Path(env.eng.entries[rid]["path"])
    env.dir.mkdir(exist_ok=True)
    old_stray = env.dir / "stray-old.q27kv"
    old_stray.write_bytes(b"junk")
    os.utime(old_stray, (0, 0))
    new_stray = env.dir / "stray-new.q27kv"
    new_stray.write_bytes(b"junk")
    os.utime(new_stray, (5 * DAY, 5 * DAY))

    env.clock.t = 7 * DAY - 1
    env.tier.tick(idle=True)
    assert rid in env.eng.entries and path.exists() and old_stray.exists()

    env.clock.t = 7 * DAY
    env.tier.tick(idle=True)
    assert rid not in env.eng.entries and not path.exists()
    assert not old_stray.exists()
    assert new_stray.exists()


def test_ssd_budget_deletes_oldest_first(env):
    env.tier.ram_retention_s = 0
    ids = []
    for t in (0, 10, 20):
        env.clock.t = t
        ids.append(env.eng.add_ram(400))
        env.tier.tick(idle=True)
    # budget 1000 bytes: 400 + 400 on disk, the third 400 needs the oldest one gone
    assert env.eng.drops == [ids[0]]
    assert set(env.eng.entries) == {ids[1], ids[2]}
    assert all(env.eng.entries[i]["on_disk"] for i in (ids[1], ids[2]))


def test_reserve_impossible_deletes_nothing_and_drops_ram_entry(env):
    env.tier.ram_retention_s = 0
    kept = env.eng.add_ram(400)
    env.tier.tick(idle=True)
    assert env.eng.exports == [kept]
    env.free["bytes"] = 50  # free + used - incoming = 50 < 100 bytes reserve, even deleting everything
    rid = env.eng.add_ram(400)
    assert env.tier.make_room(400) is False
    assert env.eng.drops == [] and kept in env.eng.entries  # nothing deleted by the refusal
    assert env.tier.make_room(2000) is False  # bigger than the whole budget
    env.clock.t = 1
    env.tier.tick(idle=True)
    assert env.eng.exports == [kept]  # rid is not exported
    assert rid not in env.eng.entries and rid in env.eng.drops
    # the tick's budget pass (make_room(0)) then deletes the oldest file to get free space back to the reserve
    assert env.eng.drops == [rid, kept]


def test_reserve_infeasible_cap_still_deletes_oldest_until_under_budget(env):
    env.tier.ram_retention_s = 0
    env.tier.budget = 10 ** 6  # no cap while the files are written
    ids = []
    for t in (0, 10, 20):
        env.clock.t = t
        ids.append(env.eng.add_ram(400))
        env.tier.tick(idle=True)
    assert all(env.eng.entries[i]["on_disk"] for i in ids)
    oldest_path = Path(env.eng.entries[ids[0]]["path"])
    # other files fill the disk: the reserve cannot be met even if all 1200 bytes of ours go (50 + 1200 < 1500),
    # but the 1000-byte cap must still hold
    env.tier.budget = 1000
    env.tier.reserve = 1500
    env.free["bytes"] = 50
    env.clock.t = 30
    env.tier.tick(idle=True)
    assert env.eng.drops == [ids[0]]  # one file is enough (1200 -> 800), so nothing else goes
    assert set(env.eng.entries) == {ids[1], ids[2]}
    assert all(env.eng.entries[i]["on_disk"] for i in (ids[1], ids[2]))
    assert not oldest_path.exists() and len(list(env.dir.glob("*.q27kv"))) == 2


def test_reserve_infeasible_under_budget_deletes_nothing(env):
    env.tier.ram_retention_s = 0
    ids = []
    for t in (0, 10):
        env.clock.t = t
        ids.append(env.eng.add_ram(400))
        env.tier.tick(idle=True)
    assert all(env.eng.entries[i]["on_disk"] for i in ids)  # 800 bytes, under the 1000-byte budget
    env.tier.reserve = 1500
    env.free["bytes"] = 50  # 50 + 800 < 1500: the reserve cannot be met
    env.clock.t = 20
    env.tier.tick(idle=True)
    assert env.eng.drops == []
    assert set(env.eng.entries) == set(ids)


def test_export_failure_drops_the_ram_entry(env):
    env.tier.ram_retention_s = 0
    rid = env.eng.add_ram(400)
    env.eng.fail_export.add(rid)
    env.tier.tick(idle=True)
    assert rid not in env.eng.entries
    assert env.tier.status()["last_error"]
    assert not any(env.dir.glob("*.q27kv"))


def test_status_reports_logical_ram_bytes(env):
    # (the engine's physical RAM use, which dedup lowers, is in stats.tiers_block; here the entries' logical sizes)
    env.eng.add_ram(300)
    env.eng.add_ram(500)
    env.tier.tick(idle=False)
    st = env.tier.status()
    assert st["ram_bytes"] == 800 and st["ram_logical_bytes"] == 800 and st["ram_entries"] == 2


def test_reused_entry_leaves_bookkeeping(env):
    rid = env.eng.add_ram(400)
    env.tier.tick(idle=False)
    assert rid in env.tier.ram_since
    del env.eng.entries[rid]  # reused by the engine
    env.clock.t = 100
    env.tier.tick(idle=False)
    assert env.tier.ram_since == {} and env.tier.written == {}


def test_start_imports_valid_files_and_deletes_the_rest(env):
    good = env.dir / "good.q27kv"
    good.write_bytes(MAGIC + b"x" * 100)
    os.utime(good, (12345, 12345))
    bad = env.dir / "bad.q27kv"
    bad.write_bytes(b"not a checkpoint")
    boom = env.dir / "boom.q27kv"
    boom.write_bytes(MAGIC + b"x" * 100)
    tmp = env.dir / "half.q27kv.tmp"
    tmp.write_bytes(b"partial")

    env.tier.start()

    assert good.exists() and not bad.exists() and not boom.exists() and not tmp.exists()
    assert len(env.eng.entries) == 1
    (entry_id,) = env.eng.entries
    assert env.tier.written[entry_id] == 12345


def _to_disk(env, size, t):
    """Write one RAM entry to a file at time t (the budget must be large enough while this runs)."""
    env.clock.t = t
    entry_id = env.eng.add_ram(size)
    env.tier.tick(idle=True)
    assert env.eng.entries[entry_id]["on_disk"]
    return entry_id


def test_make_room_skips_an_entry_whose_drop_raises(env):
    env.tier.ram_retention_s = 0
    env.tier.budget = 10 ** 6
    old, mid, new = (_to_disk(env, 400, t) for t in (0, 10, 20))
    env.tier.budget = 1000  # used 1200; incoming 400 needs used <= 600
    env.eng.fail_drop.add(old)
    assert env.tier.make_room(400) is True
    # the oldest raised and its 400 bytes were not counted: the next two went, which is enough
    assert env.eng.drops == [old, mid, new]
    assert old in env.eng.entries and mid not in env.eng.entries and new not in env.eng.entries
    assert env.tier.written.get(old) is not None  # still on disk, still known
    assert "entry is on the GPU" in env.tier.last_error


def test_make_room_false_when_failed_drops_leave_no_room(env):
    env.tier.ram_retention_s = 0
    env.tier.budget = 10 ** 6
    ids = [_to_disk(env, 400, t) for t in (0, 10)]
    env.tier.budget = 1000  # used 800; incoming 400 needs used <= 600
    env.eng.fail_drop.update(ids)
    assert env.tier.make_room(400) is False
    assert set(env.eng.entries) == set(ids)


def test_export_path_one_failed_drop_still_exports_when_room_exists(env):
    env.tier.ram_retention_s = 0
    env.tier.budget = 10 ** 6
    old, mid = _to_disk(env, 400, 0), _to_disk(env, 400, 10)
    env.tier.budget = 1000
    env.eng.fail_drop.add(old)
    kept = env.eng.add_ram(400)
    env.clock.t = 20
    env.tier.tick(idle=True)
    assert env.eng.exports[-1] == kept
    assert env.eng.entries[kept]["on_disk"]  # exported, not dropped as "no room"
    assert kept not in env.eng.drops
    assert old in env.eng.entries and mid not in env.eng.entries
    assert not any("no room" in m for m in env.logs)


def test_export_path_drops_the_ram_entry_only_when_no_room_is_left(env):
    env.tier.ram_retention_s = 0
    env.tier.budget = 10 ** 6
    old, mid = _to_disk(env, 400, 0), _to_disk(env, 400, 10)
    env.tier.budget = 1000
    env.eng.fail_drop.update({old, mid})
    kept = env.eng.add_ram(400)
    env.clock.t = 20
    env.tier.tick(idle=True)
    assert env.eng.exports == [old, mid]  # kept was not exported
    assert kept not in env.eng.entries  # really no room: the RAM entry goes
    assert any("no room on SSD" in m for m in env.logs)


def test_start_survives_a_file_that_vanishes_during_import(env):
    env.clock.t = 777
    gone = env.dir / "gone.q27kv"
    gone.write_bytes(MAGIC + b"x" * 100)
    real_import = env.eng.spill_import

    def import_then_vanish(path):
        entry_id = real_import(path)
        os.unlink(path)
        return entry_id

    env.eng.spill_import = import_then_vanish
    env.tier.start()  # must not raise
    (entry_id,) = env.eng.entries
    assert env.tier.written[entry_id] == 777  # no mtime: the clock


def test_start_uses_the_clock_when_stat_fails(env, monkeypatch):
    env.clock.t = 555
    locked = env.dir / "locked.q27kv"
    locked.write_bytes(MAGIC + b"x" * 100)
    real_getmtime = os.path.getmtime

    def flaky_getmtime(path):
        if Path(path).name == "locked.q27kv":
            raise PermissionError("locked")
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", flaky_getmtime)
    env.tier.start()
    (entry_id,) = env.eng.entries
    assert env.tier.written[entry_id] == 555
    assert locked.exists()


def test_engine_written_entry_without_write_time_uses_file_mtime(env):
    f = env.dir / "engine.q27kv"
    f.write_bytes(MAGIC + b"x" * 100)
    os.utime(f, (5000, 5000))
    entry_id = env.eng.next_id
    env.eng.next_id += 1
    env.eng.entries[entry_id] = {"id": entry_id, "pos": 0, "ram_bytes": 0, "on_disk": True,
                                 "last_use": 0, "path": str(f)}
    env.clock.t = 6000
    env.tier.tick(idle=True)
    assert env.tier.written[entry_id] == 5000
    assert entry_id in env.eng.entries
    env.clock.t = 5000 + 7 * DAY  # the ttl counts from the file's mtime, not from the first sighting
    env.tier.tick(idle=True)
    assert entry_id not in env.eng.entries and not f.exists()


def test_thousand_entries_complete(env):
    env.tier.ram_retention_s = 0
    env.tier.budget = 10 ** 6
    ids = [env.eng.add_ram(6) for _ in range(1000)]
    started = time.perf_counter()
    env.tier.tick(idle=True)
    elapsed = time.perf_counter() - started
    assert env.eng.exports == ids
    assert all(env.eng.entries[i]["on_disk"] for i in ids)
    assert elapsed < 20  # generous: this only catches a pathological blow-up


def test_tick_is_rate_limited(env):
    env.tier.tick_s = 30
    env.clock.t = 100
    env.tier.tick(idle=True)
    ran = env.eng.lists
    assert ran > 0
    env.clock.t = 120
    env.tier.tick(idle=True)
    env.clock.t = 129.9
    env.tier.tick(idle=True)
    assert env.eng.lists == ran  # skipped: the engine was not asked
    env.clock.t = 130
    env.tier.tick(idle=True)
    assert env.eng.lists > ran


def test_from_env_none_when_ram_spill_not_enabled(monkeypatch):
    monkeypatch.delenv("SPILL_RAM_GIB", raising=False)
    assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None) is None
    monkeypatch.setenv("SPILL_RAM_GIB", "0")
    assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None) is None


def test_from_env_reads_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", str(tmp_path))
    monkeypatch.setenv("SPILL_RAM_RETENTION_S", "60")
    monkeypatch.setenv("SPILL_SSD_GIB", "2")
    monkeypatch.setenv("SPILL_SSD_MIN_FREE_GIB", "1")
    monkeypatch.setenv("SPILL_TTL_DAYS", "3")
    monkeypatch.setenv("SPILL_TICK_S", "5")
    tier = TierPolicy.from_env(FakeEngine(), log=lambda _m: None)
    assert tier.directory == tmp_path
    assert tier.ram_retention_s == 60
    assert tier.budget == 2 * GiB and tier.reserve == GiB
    assert tier.ttl_s == 3 * DAY and tier.tick_s == 5


@pytest.mark.parametrize("name", ["SPILL_TTL_DAYS", "SPILL_RAM_GIB", "SPILL_SSD_GIB"])
def test_from_env_bad_value_names_the_variable(monkeypatch, name):
    monkeypatch.setenv("SPILL_RAM_GIB", "1")
    monkeypatch.setenv(name, "lots")
    with pytest.raises(ValueError, match=name):
        TierPolicy.from_env(FakeEngine(), log=lambda _m: None)


def test_empty_spill_dir_never_exports(monkeypatch):
    monkeypatch.setenv("SPILL_RAM_GIB", "1")
    monkeypatch.setenv("SPILL_DIR", "")
    eng = FakeEngine()
    tier = TierPolicy.from_env(eng, log=lambda _m: None)
    assert tier is not None and tier.directory is None
    rid = eng.add_ram(400)
    tier.start()
    tier.tick(idle=True, now=10 ** 9)
    assert eng.exports == [] and eng.entries[rid]["ram_bytes"] == 400


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_start_fails_loudly_when_spill_dir_is_not_writable(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o555)
    try:
        tier = TierPolicy(FakeEngine(), str(d), log=lambda _m: None)
        with pytest.raises(RuntimeError, match="SPILL_DIR"):
            tier.start()
        assert not any(d.iterdir())  # the probe file is not left behind
    finally:
        d.chmod(0o755)


def _retention(env):
    env.clock.t = 0
    env.tier.tick(idle=True)
    env.clock.t = 3600
    env.tier.tick(idle=True)


def test_short_entry_past_retention_is_dropped_not_exported(env):
    short, long_ = env.eng.add_ram(400, pos=100), env.eng.add_ram(400, pos=8192)
    _retention(env)
    assert env.eng.exports == [long_]
    assert env.eng.drops == [short] and short not in env.eng.entries
    assert env.eng.entries[long_]["on_disk"]
    assert env.tier.status()["ssd_skipped_short"] == 1
    assert list(env.dir.glob("*" + tiercache.SUFFIX)) == [pathlib.Path(env.eng.entries[long_]["path"])]


def test_short_entry_is_kept_in_ram_before_retention(env):
    short = env.eng.add_ram(400, pos=100)
    env.clock.t = 0
    env.tier.tick(idle=True)
    env.clock.t = 3599
    env.tier.tick(idle=True)
    assert env.eng.drops == [] and short in env.eng.entries


def test_min_tokens_zero_exports_everything(env):
    env.tier.ssd_min_tokens = 0
    rid = env.eng.add_ram(400, pos=1)
    _retention(env)
    assert env.eng.exports == [rid] and env.eng.drops == []
    assert env.tier.status()["ssd_skipped_short"] == 0


def test_short_entry_on_disk_is_unaffected(env):
    rid = env.eng.add_ram(400, pos=100)
    env.tier.ssd_min_tokens = 0
    _retention(env)
    assert env.eng.entries[rid]["on_disk"]
    env.tier.ssd_min_tokens = 8192
    env.clock.t = 3700
    env.tier.tick(idle=True)
    assert rid in env.eng.entries and env.eng.entries[rid]["on_disk"] and env.eng.drops == []
    assert env.tier.status()["ssd_skipped_short"] == 0


def test_flush_on_stop_skips_short_entries(env):
    short, long_ = env.eng.add_ram(400, pos=5), env.eng.add_ram(400, pos=9000)
    assert env.tier.flush(10)["written"] == 1
    assert env.eng.exports == [long_] and short in env.eng.entries


def test_from_env_reads_min_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", str(tmp_path))
    monkeypatch.delenv("SPILL_SSD_MIN_TOKENS", raising=False)
    assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None).ssd_min_tokens == 32768
    monkeypatch.setenv("SPILL_SSD_MIN_TOKENS", "0")
    assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None).ssd_min_tokens == 0
    monkeypatch.setenv("SPILL_SSD_MIN_TOKENS", "2000.9")
    assert TierPolicy.from_env(FakeEngine(), log=lambda _m: None).ssd_min_tokens == 2000
    monkeypatch.setenv("SPILL_SSD_MIN_TOKENS", "many")
    with pytest.raises(ValueError, match="SPILL_SSD_MIN_TOKENS"):
        TierPolicy.from_env(FakeEngine(), log=lambda _m: None)


# ---------------------------------------------------------------- SSD dedup layout (an engine with spill_disk_usage)

import json  # noqa: E402

MANIFEST = 10  # bytes of a fake manifest


class FakeDedupEngine:
    """An engine with the SSD dedup API: entries are chunk lists ([(name, bytes)], tail bytes); a chunk file is shared by the
    manifests that list it (refs) and exists while refs > 0, as the real engine's DiskIndex."""

    def __init__(self):
        self.entries, self.next_id = {}, 1
        self.refs = {}   # chunk name -> manifests listing it
        self.size = {}   # chunk name -> bytes
        self.exports, self.drops, self.sweeps = [], [], 0
        self.fail_drop = set()

    def add_ram(self, chunks, tail=50, pos=10_000):
        entry_id = self.next_id
        self.next_id += 1
        logical = sum(b for _, b in chunks) + tail + MANIFEST
        self.entries[entry_id] = {"id": entry_id, "pos": pos, "ram_bytes": logical, "on_disk": False, "last_use": 0,
                                  "path": "", "chunks": list(chunks), "tail": tail, "logical_bytes": logical}
        return entry_id

    def _unique(self, e):
        if not e["on_disk"]:
            return 0
        return MANIFEST + e["tail"] + sum(b for n, b in e["chunks"] if self.refs.get(n) == 1)

    def _new(self, e):
        if e["on_disk"]:
            return 0
        return MANIFEST + e["tail"] + sum(b for n, b in e["chunks"] if not self.refs.get(n))

    def spill_list(self):
        return [{k: v for k, v in e.items() if k not in ("chunks", "tail")} | {
            "backup": False, "busy": False, "unique_disk_bytes": self._unique(e), "new_disk_bytes": self._new(e)}
                for _, e in sorted(self.entries.items())]

    def spill_export(self, entry_id, path):
        self.exports.append(entry_id)
        e = self.entries[entry_id]
        Path(path).write_text(json.dumps({"chunks": e["chunks"], "tail": e["tail"]}))
        for n, b in e["chunks"]:
            self.refs[n] = self.refs.get(n, 0) + 1
            self.size[n] = b
        e.update(ram_bytes=0, on_disk=True, path=path)

    def _forget(self, e):
        Path(e["path"]).unlink(missing_ok=True)
        for n, _ in e["chunks"]:
            self.refs[n] -= 1
            if not self.refs[n]:
                del self.refs[n]

    def spill_drop(self, entry_id):
        self.drops.append(entry_id)
        if entry_id in self.fail_drop:
            raise RuntimeError("entry is on the GPU")
        e = self.entries.pop(entry_id)
        if e["on_disk"]:
            self._forget(e)

    def spill_import(self, path):
        try:
            data = json.loads(Path(path).read_text())
        except ValueError:
            return 0
        entry_id = self.next_id
        self.next_id += 1
        chunks = [tuple(c) for c in data["chunks"]]
        logical = sum(b for _, b in chunks) + data["tail"] + MANIFEST
        self.entries[entry_id] = {"id": entry_id, "pos": 20_000, "ram_bytes": 0, "on_disk": True, "last_use": 0,
                                  "path": path, "chunks": chunks, "tail": data["tail"], "logical_bytes": logical}
        for n, b in chunks:
            self.refs[n] = self.refs.get(n, 0) + 1
            self.size[n] = b
        return entry_id

    def spill_disk_usage(self):
        disk = [e for e in self.entries.values() if e["on_disk"]]
        physical = sum(self.size[n] for n in self.refs) + sum(MANIFEST + e["tail"] for e in disk)
        return {"physical_bytes": physical, "logical_bytes": sum(e["logical_bytes"] for e in disk),
                "chunk_files": len(self.refs), "manifests": len(disk)}

    def spill_sweep(self):
        self.sweeps += 1
        return 2

    def spill_stats(self):
        return {"entries": len(self.entries), "in_ram": 0, "on_disk": 0, "ram_bytes": 0, "spilled": 0, "restored": 0,
                "written": 0, "dropped": 0, "subsumed": 0, "failed": 0}


@pytest.fixture
def denv(tmp_path):
    eng = FakeDedupEngine()
    clock = Clock()
    free = {"bytes": 10 ** 12}
    logs = []
    d = tmp_path / "spill"
    # budget 1500 bytes
    tier = TierPolicy(eng, str(d), ram_retention_s=0, ssd_gib=1500 * B, ssd_min_free_gib=100 * B, ttl_days=7, tick_s=0,
                      clock=clock, disk_free=lambda _d: free["bytes"], log=logs.append)
    return types.SimpleNamespace(eng=eng, tier=tier, dir=d, clock=clock, free=free, logs=logs)


def _shared(n, unique, size=400):
    return [("P1", size), ("P2", size), (f"U{n}", unique)]


def test_dedup_exports_into_entries_dir_and_counts_physical_bytes(denv):
    denv.tier.start()
    assert denv.tier.dedup is True and (denv.dir / "entries").is_dir()
    ids = []
    for t in range(3):  # each entry: 800 shared + 100 unique + 60 own = 960 logical; physical total 800 + 3 * 160 = 1280
        denv.clock.t = t
        ids.append(denv.eng.add_ram(_shared(t, 100)))
        denv.tier.tick(idle=True)
    assert denv.eng.exports == ids and denv.eng.drops == []  # logical 2880 > the 1500 budget, physical 1280 fits
    assert all(Path(denv.eng.entries[i]["path"]).parent == denv.dir / "entries" for i in ids)
    assert all(Path(denv.eng.entries[i]["path"]).suffix == ".q27m" for i in ids)
    st = denv.tier.status()
    assert st["ssd_physical_bytes"] == 1280 and st["ssd_bytes"] == 1280
    assert st["ssd_logical_bytes"] == 3 * 960
    assert st["ssd_dedup_ratio"] == pytest.approx(2880 / 1280)


def test_dedup_make_room_frees_unique_bytes_and_redoes_the_incoming_size(denv):
    denv.tier.start()
    ids = []
    for t in range(3):
        denv.clock.t = t
        ids.append(denv.eng.add_ram(_shared(t, 100)))
        denv.tier.tick(idle=True)
    # a 4th entry with 400 unique bytes: it writes 460 new bytes (the shared prefix is on the SSD already); physical 1280 + 460 > 1500,
    # so victims must go. Each of the oldest frees only 160 (the prefix is shared): one is not enough (1120 + 460), two are (960 + 460).
    new = denv.eng.add_ram(_shared(9, 400))
    denv.clock.t = 100
    entry = next(e for e in denv.eng.spill_list() if e["id"] == new)
    assert entry["new_disk_bytes"] == 460
    assert denv.tier.make_room(entry["new_disk_bytes"], new) is True
    assert denv.eng.drops == [ids[0], ids[1]]  # the oldest first; no more than needed
    assert denv.eng.spill_disk_usage()["physical_bytes"] == 800 + 160 and denv.tier.usage["physical_bytes"] == 960


def test_dedup_make_room_asks_the_incoming_size_again_after_a_drop(denv):
    denv.tier.start()
    denv.tier.budget = 10 ** 6
    a = denv.eng.add_ram([("X", 700)], tail=10)
    denv.tier.tick(idle=True)
    z = denv.eng.add_ram([("Z", 300)], tail=10)
    denv.clock.t = 1
    denv.tier.tick(idle=True)
    # b shares X with a, which is on the SSD: it writes 20 bytes now. Once a is dropped X is gone and b must write it (720).
    b = denv.eng.add_ram([("X", 700)], tail=10)
    assert next(e for e in denv.eng.spill_list() if e["id"] == b)["new_disk_bytes"] == 20
    denv.tier.budget = 800  # used 1040 + 20 > 800: a goes (used 320); the stale 20 would fit now, but b needs 720 + 320 > 800: z goes too
    assert denv.tier.make_room(20, b) is True
    assert denv.eng.drops == [a, z]
    assert denv.tier.usage["physical_bytes"] == 0


def test_dedup_an_entry_that_cannot_fit_even_alone_deletes_nothing(denv):
    denv.tier.start()
    denv.tier.budget = 10 ** 6
    a = denv.eng.add_ram([("X", 700)], tail=10)
    denv.tier.tick(idle=True)
    big = denv.eng.add_ram([("X", 700), ("Y", 700)], tail=10)  # shares X: it writes 720 bytes now, but its full size is 1420
    entry = next(e for e in denv.eng.spill_list() if e["id"] == big)
    assert entry["new_disk_bytes"] == 720 and entry["logical_bytes"] == 1420
    denv.tier.budget = 1000
    assert denv.tier.make_room(entry["new_disk_bytes"], big) is False  # its 720 new bytes alone would need a drop, but 1420 (its full size) never fits, 1420 never does
    assert denv.eng.drops == [] and a in denv.eng.entries


def test_dedup_drops_entries_that_free_bytes_before_mostly_shared_ones(denv):
    denv.tier.start()
    denv.tier.budget = 10 ** 6
    denv.clock.t = 0
    a = denv.eng.add_ram(_shared(0, 0), tail=10)   # oldest: all shared with b / c, so it would free only 20 bytes
    denv.tier.tick(idle=True)
    denv.clock.t = 1
    b = denv.eng.add_ram(_shared(1, 0), tail=10)
    denv.tier.tick(idle=True)
    denv.clock.t = 2
    c = denv.eng.add_ram([("P1", 400), ("P2", 400), ("BIG", 900)], tail=10)  # youngest, but 900 unique bytes
    denv.tier.tick(idle=True)
    assert denv.eng.drops == []
    denv.tier.budget = 1500  # physical now 800 + 900 + 3 * 20 = 1760 > 1500: someone must go
    assert denv.tier.make_room(0) is True
    assert denv.eng.drops == [c]  # the youngest frees 920 at once; dropping a, b (20 bytes each) would not have been enough


def test_dedup_a_failed_drop_is_skipped(denv):
    denv.tier.start()
    denv.tier.budget = 10 ** 6
    ids = []
    for t in range(2):
        denv.clock.t = t
        ids.append(denv.eng.add_ram([(f"X{t}", 900)], tail=10))
        denv.tier.tick(idle=True)
    denv.eng.fail_drop.add(ids[0])
    denv.tier.budget = 1000
    assert denv.tier.make_room(0) is True
    assert denv.eng.drops == [ids[0], ids[1]] and ids[0] in denv.eng.entries and ids[1] not in denv.eng.entries


def test_dedup_incoming_bigger_than_budget_deletes_nothing(denv):
    denv.tier.start()
    denv.eng.add_ram(_shared(0, 100))
    denv.tier.tick(idle=True)
    assert denv.tier.make_room(2000) is False and denv.eng.drops == []


def test_dedup_ttl_counts_from_the_manifest_write_time(denv):
    denv.tier.start()
    rid = denv.eng.add_ram(_shared(0, 100))
    denv.clock.t = 0
    denv.tier.tick(idle=True)
    path = Path(denv.eng.entries[rid]["path"])
    denv.clock.t = 7 * DAY - 1
    denv.tier.tick(idle=True)
    assert rid in denv.eng.entries
    denv.clock.t = 7 * DAY
    denv.tier.tick(idle=True)
    assert rid not in denv.eng.entries and not path.exists()


def test_dedup_start_deletes_legacy_files_imports_manifests_and_sweeps(denv):
    denv.dir.mkdir()
    (denv.dir / "entries").mkdir()
    (denv.dir / "old1.q27kv").write_bytes(b"x")
    (denv.dir / "old2.q27kv").write_bytes(b"x")
    (denv.dir / "half.q27kv.tmp").write_bytes(b"x")
    (denv.dir / "entries" / "half.q27m.tmp").write_bytes(b"x")
    good = denv.dir / "entries" / "a.q27m"
    good.write_text(json.dumps({"chunks": [["P1", 400]], "tail": 5}))
    os.utime(good, (123.0, 123.0))
    (denv.dir / "entries" / "bad.q27m").write_text("not json")
    denv.tier.start()
    assert sorted(p.name for p in denv.dir.iterdir()) == ["entries"]
    assert [p.name for p in (denv.dir / "entries").iterdir()] == ["a.q27m"]  # (the damaged manifest and the .tmp went)
    legacy_lines = [m for m in denv.logs if "legacy" in m]
    assert len(legacy_lines) == 1 and "3" in legacy_lines[0]  # one line for all of them
    (rid,) = denv.eng.entries
    assert denv.tier.written[rid] == 123.0  # the write time is the manifest's mtime
    assert denv.eng.sweeps == 1  # after the imports
    assert denv.tier.usage["physical_bytes"] == 400 + MANIFEST + 5


def test_dedup_stray_manifest_sweep_stays_in_entries(denv):
    denv.tier.start()
    chunks = denv.dir / "chunks"
    chunks.mkdir()
    old_chunk = chunks / "abc.kvc"
    old_chunk.write_bytes(b"x")
    os.utime(old_chunk, (0, 0))
    stray = denv.dir / "entries" / "stray.q27m"
    stray.write_bytes(b"x")
    os.utime(stray, (0, 0))
    denv.clock.t = 8 * DAY
    denv.tier.tick(idle=True)
    assert old_chunk.exists()  # chunks/ belongs to the engine: never touched from here
    assert not stray.exists()


def test_old_engine_without_the_api_keeps_the_flat_layout(env):
    # (FakeEngine has no spill_disk_usage: the whole legacy suite above runs on it. Here the specifics.)
    assert env.tier.dedup is False
    (env.dir).mkdir(exist_ok=True)
    flat = env.dir / "keep.q27kv"
    flat.write_bytes(MAGIC + b"x" * 10)
    env.tier.start()
    assert flat.exists()  # not deleted as "legacy": it is this engine's own format
    assert any(e["path"] == str(flat) for e in env.eng.entries.values())
    rid = env.eng.add_ram(400)
    env.tier.tick(idle=True)
    env.clock.t = 3600
    env.tier.tick(idle=True)
    assert Path(env.eng.entries[rid]["path"]).parent == env.dir and not (env.dir / "entries").exists()
    st = env.tier.status()
    assert st["ssd_bytes"] == st["ssd_physical_bytes"] == st["ssd_logical_bytes"] and st["ssd_dedup_ratio"] is None


def test_engine_whose_usage_call_fails_falls_back_to_flat(env):
    def boom():
        raise RuntimeError("no such symbol")
    env.eng.spill_disk_usage = boom
    env.tier.dedup = None
    env.tier.start()
    assert env.tier.dedup is False
