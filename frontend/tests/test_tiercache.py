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

    def add_ram(self, size):
        entry_id = self.next_id
        self.next_id += 1
        self.entries[entry_id] = {"id": entry_id, "pos": 0, "ram_bytes": size, "on_disk": False,
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
