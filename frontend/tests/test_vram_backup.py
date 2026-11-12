"""Idle VRAM backup (VRAM_BACKUP_IDLE_S), the stop flush (SPILL_FLUSH_ON_STOP_S) and the backup keys of the tiers, against
fake engines and real files: no GPU."""

import collections
import pathlib
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import stats  # noqa: E402
import worker  # noqa: E402
from tiercache import TierPolicy  # noqa: E402

MAGIC = b"Q27KV\n"


class Clock:
    """A fake time.monotonic; the fake engine advances it for each copy and export."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Base:
    """A fake engine without spill_backup_one (an engine that predates the idle backup)."""

    def __init__(self, vram=0, clock=None, copy_s=0.0, export_s=0.0):
        self.max_tokens, self.park_slots, self.has_vision = 8, 0, False
        self.vram, self.clock, self.copy_s, self.export_s = vram, clock, copy_s, export_s
        self.entries, self.next_id = {}, 1
        self.exports, self.conn = [], None
        self.backup_calls, self.marks, self.backed_up = [], [], 0
        self.victims = []  # the `victims` argument of each spill_backup_one call (0: any checkpoint)

    def _advance(self, seconds):
        if self.clock is not None:
            self.clock.t += seconds

    def add_ram(self, size=100, backup=False):
        i = self.next_id
        self.next_id += 1
        self.entries[i] = {"id": i, "pos": 0, "ram_bytes": size, "on_disk": False, "path": "", "last_use": 0,
                           "backup": backup}
        return i

    def cache_grow(self, n):
        return 0

    def memory_report(self):
        return {}

    def spill_list(self):
        return [dict(e) for _, e in sorted(self.entries.items())]

    def spill_export(self, entry_id, path):
        self._advance(self.export_s)
        self.exports.append(entry_id)
        e = self.entries[entry_id]
        Path(path).write_bytes(MAGIC + b"x" * max(0, e["ram_bytes"] - len(MAGIC)))
        e.update(ram_bytes=0, on_disk=True, path=path)

    def spill_drop(self, entry_id):
        e = self.entries.pop(entry_id)
        if e["path"]:
            Path(e["path"]).unlink(missing_ok=True)

    def spill_import(self, path):
        return 0

    def spill_stats(self):
        es = list(self.entries.values())
        return {"entries": len(es), "in_ram": sum(e["ram_bytes"] > 0 for e in es),
                "on_disk": sum(e["on_disk"] for e in es), "ram_bytes": sum(e["ram_bytes"] for e in es),
                "spilled": len(self.exports), "restored": 0, "written": 0, "dropped": 0, "subsumed": 0,
                "failed": 0, "backed_up": self.backed_up}


class Engine(Base):
    """The engine with spill_backup_one: copies the next VRAM checkpoint to RAM (it stays in VRAM)."""

    on_backup = None  # hook(index of the call), run inside spill_backup_one
    fail_backup = None

    def spill_backup_one(self, victims=0):
        n = len(self.backup_calls)
        self.backup_calls.append(n)
        self.victims.append(victims)
        self.marks.append(len(self.conn.polls) if self.conn else 0)
        if self.on_backup:
            self.on_backup(n)
        if self.fail_backup:
            raise RuntimeError(self.fail_backup)
        if self.vram <= 0:
            return False
        self.vram -= 1
        self.backed_up += 1
        self._advance(self.copy_s)
        self.add_ram(100, backup=True)
        return True


class Conn:
    """The pipe to the server: messages go in the inbox; each poll is recorded with its timeout."""

    def __init__(self):
        self.inbox, self.polls, self.sent = collections.deque(), [], []

    def poll(self, timeout=None):
        self.polls.append(timeout)
        return bool(self.inbox)

    def recv(self):
        return self.inbox.popleft()

    def send(self, *msg):
        self.sent.append(msg)


def tiers_for(engine, directory=None, backup_idle_s=60, flush_on_stop_s=8, **kw):
    return TierPolicy(engine, str(directory) if directory else None, backup_idle_s=backup_idle_s,
                      flush_on_stop_s=flush_on_stop_s, disk_free=lambda _d: 10 ** 12, log=lambda _m: None, **kw)


def make_sched(engine, tiers, conn=None, iterations=None, on_iteration=None):
    """A Scheduler without __init__ (no GPU). run() ends with SystemExit: on the stop message, which is sent after
    `iterations` loop iterations (report() is called once per iteration)."""
    conn = conn or Conn()
    engine.conn = conn
    s = object.__new__(worker.Scheduler)
    s.conn, s.engine, s.tiers = conn, engine, tiers
    s.alive, s.waiting = [], collections.deque()
    s.max_seqs = s.engine_seqs = 4
    s.admit_kv, s.kv_reserve, s.min_tps = True, 4096, 0.0
    s.rates, s.blocked, s.held_until = {}, None, (None, 0.0)
    s.spec = s.batch = False
    s.share, s.quantum = 0.5, 0.05
    s.m, s.last_metrics, s.last_stats = collections.Counter(), 0.0, 0.0
    s.cache_full, s.history, s.credit = False, collections.deque(maxlen=64), 0.0
    s.last_active = time.monotonic()
    s.backup_done = False
    s.metrics = lambda: {}
    s.stats = lambda: {}
    # A request "submit" becomes a waiting request that never prefills or decodes: it stays admitted.
    s.submit = lambda rid, req: s.waiting.append(
        types.SimpleNamespace(rid=rid, phase="wait", arrival=0.0, last_run=0.0))
    count = {"n": 0}

    def report(force=False):
        count["n"] += 1
        if on_iteration:
            on_iteration(count["n"], s)
        if iterations is not None and count["n"] >= iterations:
            conn.inbox.append(("stop",))

    s.report = report
    return s


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("SPILL_RAM_GIB", "SPILL_DIR", "VRAM_BACKUP_IDLE_S", "SPILL_FLUSH_ON_STOP_S"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ------------------------------------------------------------ settings

def test_from_env_defaults(clean_env):
    clean_env.setenv("SPILL_RAM_GIB", "32")
    t = TierPolicy.from_env(Base())
    assert t.backup_idle_s == 60 and t.flush_on_stop_s == 8


def test_no_tier_policy_means_no_backup(clean_env):
    assert TierPolicy.from_env(Base()) is None  # SPILL_RAM_GIB unset: the backup needs the tiers


def test_zero_turns_the_backup_off(clean_env):
    clean_env.setenv("SPILL_RAM_GIB", "32")
    clean_env.setenv("VRAM_BACKUP_IDLE_S", "0")
    assert TierPolicy.from_env(Base()).backup_idle_s == 0


@pytest.mark.parametrize("raw", ["-1", "abc", "nan"])
def test_bad_backup_values_name_the_variable(clean_env, raw):
    clean_env.setenv("SPILL_RAM_GIB", "32")
    clean_env.setenv("VRAM_BACKUP_IDLE_S", raw)
    with pytest.raises(ValueError, match="VRAM_BACKUP_IDLE_S"):
        TierPolicy.from_env(Base())


def test_flush_budget_is_cut_to_the_stop_limit(clean_env):
    clean_env.setenv("SPILL_RAM_GIB", "32")
    clean_env.setenv("SPILL_FLUSH_ON_STOP_S", "30")
    logs = []
    t = TierPolicy.from_env(Base(), log=logs.append)
    assert t.flush_on_stop_s == 8
    assert any("cut to 8" in m for m in logs)


# ------------------------------------------------------------ idle backup

def test_backup_due_only_after_the_idle_threshold():
    eng = Engine(vram=3)
    s = make_sched(eng, tiers_for(eng, backup_idle_s=60))
    s.last_active = time.monotonic() - 59
    assert not s.backup_due()
    assert 0 < s.backup_wait() <= 1.5  # the idle wait ends when the backup is due
    s.last_active = time.monotonic() - 61
    assert s.backup_due()
    assert s.backup_wait() == 0.0


def test_no_backup_while_active(capsys):
    eng = Engine(vram=3)
    s = make_sched(eng, tiers_for(eng), iterations=3)  # last_active is now
    with pytest.raises(SystemExit):
        s.run()
    assert eng.backup_calls == []


def test_one_backup_per_iteration_and_polls_do_not_wait():
    eng = Engine(vram=3)
    s = make_sched(eng, tiers_for(eng), iterations=6)
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    assert len(eng.backup_calls) == 4  # three copies, then one False: nothing left
    assert eng.backed_up == 3
    # Between two copies the loop only drains the pipe (timeout 0): a request is read at once.
    for a, b in zip(eng.marks, eng.marks[1:]):
        assert all(p == 0 for p in s.conn.polls[a:b])


def test_a_request_stops_the_backups_and_is_admitted_first():
    eng = Engine(vram=10)
    conn = Conn()
    s = make_sched(eng, tiers_for(eng), conn=conn, iterations=5)
    s.last_active -= 1000
    eng.on_backup = lambda n: conn.inbox.append(("submit", "r1", {})) if n == 1 else None
    with pytest.raises(SystemExit):
        s.run()
    assert len(eng.backup_calls) == 2  # the request arrived during the second copy: no third
    assert [r.rid for r in s.alive] == ["r1"]


def test_exhausted_until_the_next_activity():
    eng = Engine(vram=1)
    counts = {}

    def on_iteration(n, s):
        counts[n] = len(eng.backup_calls)
        if n == 3:  # a request or a turn: activity
            s.touch()
            s.last_active -= 1000  # (the idle time that follows)

    s = make_sched(eng, tiers_for(eng), iterations=5, on_iteration=on_iteration)
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    assert counts[2] == 2  # one copy, then False: exhausted
    assert counts[3] == 2  # no call while exhausted
    assert len(eng.backup_calls) == 3  # after the activity, one more try (also False)


def test_a_failing_backup_is_logged_and_the_loop_goes_on(capsys):
    eng = Engine(vram=3)
    eng.fail_backup = "copy failed"
    s = make_sched(eng, tiers_for(eng), iterations=3)
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    assert "idle backup: copy failed" in capsys.readouterr().err
    assert len(eng.backup_calls) == 1  # exhausted after the failure, not retried at once


def test_an_engine_without_spill_backup_one_is_left_alone():
    eng = Base(vram=3)
    s = make_sched(eng, tiers_for(eng), iterations=3)
    s.last_active -= 1000
    assert not s.backup_due()
    assert s.backup_wait() == worker.STATS_IDLE
    with pytest.raises(SystemExit):
        s.run()


def test_backup_off_when_the_threshold_is_zero():
    eng = Engine(vram=3)
    s = make_sched(eng, tiers_for(eng, backup_idle_s=0), iterations=3)
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    assert eng.backup_calls == []


# ------------------------------------------------------------ stop flush

def test_stop_flush_respects_the_budget(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)  # (the worker and the tiers read it at call time)
    eng = Engine(vram=10, clock=clock, copy_s=3.0, export_s=1.5)
    for _ in range(4):
        eng.add_ram(100)  # RAM entries that were there before the stop
    t = tiers_for(eng, tmp_path / "spill")
    t.start()
    conn = Conn()
    conn.inbox.append(("stop",))
    s = make_sched(eng, t, conn=conn)
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    # Backups get half of the 8 s (copies of 3 s: two of them); the RAM entries get the rest (exports of 1.5 s).
    assert len(eng.backup_calls) == 2
    assert eng.exports == [1, 2]  # the budget ends the flush: the other four stay in RAM
    assert clock.t <= 1000 + 8 + 1.5  # the budget, plus the one export that started inside it
    assert len(list((tmp_path / "spill").glob("*.q27kv"))) == 2


def test_stop_flush_without_spill_backup_one_still_writes(tmp_path):
    eng = Base()
    eng.add_ram(100)
    eng.add_ram(100)
    t = tiers_for(eng, tmp_path / "spill")
    t.start()
    conn = Conn()
    conn.inbox.append(("stop",))
    s = make_sched(eng, t, conn=conn)
    with pytest.raises(SystemExit):
        s.run()
    assert eng.exports == [1, 2]


def test_no_ssd_directory_means_no_stop_flush():
    eng = Engine(vram=3)
    conn = Conn()
    conn.inbox.append(("stop",))
    s = make_sched(eng, tiers_for(eng), conn=conn)  # no directory: nothing survives the stop anyway
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    assert eng.backup_calls == [] and eng.exports == []


def test_zero_flush_budget_means_no_stop_flush(tmp_path):
    eng = Engine(vram=3)
    eng.add_ram(100)
    t = tiers_for(eng, tmp_path / "spill", flush_on_stop_s=0)
    t.start()
    conn = Conn()
    conn.inbox.append(("stop",))
    s = make_sched(eng, t, conn=conn)
    s.last_active -= 1000
    with pytest.raises(SystemExit):
        s.run()
    assert eng.backup_calls == [] and eng.exports == []


def test_a_failing_flush_does_not_stop_the_stop(tmp_path, capsys):
    eng = Engine(vram=3)
    eng.add_ram(100)
    t = tiers_for(eng, tmp_path / "spill")
    t.start()

    def broken_flush(budget_s, clock=None):
        raise RuntimeError("disk gone")

    t.flush = broken_flush
    conn = Conn()
    conn.inbox.append(("stop",))
    s = make_sched(eng, t, conn=conn)
    with pytest.raises(SystemExit):
        s.run()
    assert "stop flush: disk gone" in capsys.readouterr().err


# ------------------------------------------------------------ tiers: backup entries

def test_status_counts_backup_entries():
    eng = Base()
    t = tiers_for(eng)
    t.listed = [{"id": 1, "ram_bytes": 100, "on_disk": False, "path": "", "backup": True},
                {"id": 2, "ram_bytes": 100, "on_disk": False, "path": ""}]
    st = t.status()
    assert st["backup_entries"] == 1 and st["ram_entries"] == 2


def test_status_splits_each_tier_into_backups_and_evicted(tmp_path):
    f = tmp_path / "e3"
    f.write_bytes(b"x" * 40)
    eng = Base()
    t = tiers_for(eng)
    t.listed = [{"id": 1, "ram_bytes": 100, "on_disk": False, "path": "", "backup": True},
                {"id": 2, "ram_bytes": 300, "on_disk": False, "path": ""},
                {"id": 3, "ram_bytes": 0, "on_disk": True, "path": str(f)}]
    st = t.status()
    assert (st["ram_backup_entries"], st["ram_backup_bytes"]) == (1, 100)
    assert (st["ram_evicted_entries"], st["ram_evicted_bytes"]) == (1, 300)
    assert (st["disk_evicted_entries"], st["disk_evicted_bytes"]) == (1, 40)
    assert (st["disk_backup_entries"], st["disk_backup_bytes"]) == (0, 0)


def test_retention_never_writes_backup_entries(tmp_path):
    # A backup's checkpoint is still in VRAM: past twice the retention, idle or busy, it stays in RAM (no SSD copy).
    eng = Base()
    backup_id = eng.add_ram(400, backup=True)
    t = tiers_for(eng, tmp_path / "spill", tick_s=0, ram_retention_s=3600)
    t.start()
    t.tick(idle=True, now=0.0)
    t.tick(idle=True, now=3600.0)
    t.tick(idle=True, now=7200.0)
    t.tick(idle=False, now=10800.0)
    assert eng.exports == []
    assert eng.entries[backup_id]["ram_bytes"] > 0 and not eng.entries[backup_id]["on_disk"]
    assert list((tmp_path / "spill").glob("*.q27kv")) == []


def test_retention_still_writes_a_sole_copy_of_the_same_size(tmp_path):
    eng = Base()
    plain_id = eng.add_ram(400, backup=False)
    t = tiers_for(eng, tmp_path / "spill", tick_s=0, ram_retention_s=3600)
    t.start()
    t.tick(idle=True, now=0.0)
    t.tick(idle=True, now=3600.0)
    assert eng.exports == [plain_id]


def test_a_backup_that_is_evicted_from_vram_is_retained_as_usual(tmp_path):
    # Once the pool gives its checkpoint up, the entry is the only copy: live is off, backup goes False.
    eng = Base()
    eid = eng.add_ram(400, backup=True)
    t = tiers_for(eng, tmp_path / "spill", tick_s=0, ram_retention_s=3600)
    t.start()
    t.tick(idle=True, now=0.0)
    eng.entries[eid]["backup"] = False
    t.tick(idle=True, now=3600.0)
    t.tick(idle=True, now=3601.0)
    assert eng.exports == [eid]


def test_flush_writes_ram_entries_to_files(tmp_path):
    eng = Base()
    eng.add_ram(100)
    eng.add_ram(100, backup=True)
    t = tiers_for(eng, tmp_path / "spill")
    t.start()
    assert t.flush(10.0) == {"written": 2, "left": 0}
    assert all(e["on_disk"] for e in eng.spill_list())


def test_flush_writes_backup_entries_too(tmp_path):
    # The stop flush is the one place a backup is written: the engine's VRAM is lost when it unloads.
    eng = Base()
    backup_id = eng.add_ram(100, backup=True)
    t = tiers_for(eng, tmp_path / "spill")
    t.start()
    assert t.flush(10.0) == {"written": 1, "left": 0}
    assert eng.exports == [backup_id]
    assert eng.entries[backup_id]["on_disk"]


def test_flush_without_directory_writes_nothing():
    eng = Base()
    eng.add_ram(100)
    assert tiers_for(eng).flush(10.0) == {"written": 0, "left": 0}
    assert eng.exports == []


# ------------------------------------------------------------ stats and UI keys

def test_tiers_block_reports_backups():
    eng = Engine(vram=0)
    eng.backed_up = 5
    eng.add_ram(100, backup=True)
    eng.add_ram(100)
    t = tiers_for(eng)
    t.listed = eng.spill_list()  # (status() reads the list of the last look, as the worker's tick leaves it)
    out = stats.tiers_block(eng, t, environ={})
    assert out["backed_up"] == 5
    assert out["backup_entries"] == 1


def test_tiers_block_tolerates_an_engine_without_the_keys():
    class OldStats(Base):
        def spill_stats(self):
            d = super().spill_stats()
            d.pop("backed_up")
            return d

    eng = OldStats()
    out = stats.tiers_block(eng, tiers_for(eng), environ={})
    assert out["backed_up"] == 0 and out["backup_entries"] == 0
    assert "error" not in out


def test_ui_has_the_backup_counter_and_entry_count():
    html = (pathlib.Path(__file__).resolve().parents[1] / "ui" / "index.html").read_text(encoding="utf-8")
    assert '["已備份 backed_up", "backed_up"]' in html
    assert "VRAM 備份" in html
    assert "t.backup_entries" in html  # (a missing key shows "–", as the other counters)


# ------------------------------------------------------------ restrained backup: only the next victims

def test_idle_backup_asks_for_the_next_victims_only(tmp_path):
    eng = Engine(vram=2)
    t = tiers_for(eng, backup_victims=5)
    s = make_sched(eng, t)
    s.backup_one()
    assert eng.victims == [5]


def test_stop_flush_backs_up_any_checkpoint(tmp_path):
    eng = Engine(vram=2)
    t = tiers_for(eng, tmp_path / "spill", backup_victims=5)
    t.start()
    s = make_sched(eng, t)
    s.flush_on_stop()
    assert eng.victims and set(eng.victims) == {0}  # (everything that is left, not only the next victims)


def test_backup_victims_from_the_environment(monkeypatch):
    monkeypatch.setenv("SPILL_RAM_GIB", "8")
    assert TierPolicy.from_env(Base(), log=lambda _m: None).backup_victims == 8
    monkeypatch.setenv("VRAM_BACKUP_VICTIMS", "3")
    assert TierPolicy.from_env(Base(), log=lambda _m: None).backup_victims == 3
