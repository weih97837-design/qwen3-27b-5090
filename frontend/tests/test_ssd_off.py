"""The SSD tier is OFF by default (SPILL_SSD_GIB=0, also when set explicitly) and the per-agent step (SPILL_FAIR_AGENTS) is off by
default. With the SSD off the engine gets SPILL_DIR="" (it drops RAM victims instead of writing them) and TierPolicy has no
directory: nothing is written, read, scanned or deleted in any SPILL_DIR. No GPU."""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "reference"))

import launch  # noqa: E402
import stats  # noqa: E402
import tiercache  # noqa: E402
from tiercache import TierPolicy  # noqa: E402
from test_tiercache import FakeEngine, MAGIC  # noqa: E402


def LOG(_m):
    pass


class StrictEngine(FakeEngine):
    """Every SSD-side call is recorded: with the SSD off none may happen."""

    def __init__(self):
        super().__init__()
        self.ssd_calls = []

    def spill_import(self, path):
        self.ssd_calls.append(("import", path))
        return super().spill_import(path)

    def spill_sweep(self):
        self.ssd_calls.append(("sweep",))
        return 0

    def spill_disk_usage(self):
        self.ssd_calls.append(("usage",))
        return None


def ssd_off_tier(tmp_path, engine=None, **kw):
    d = tmp_path / "kvcache"
    return TierPolicy(engine or StrictEngine(), str(d), ssd_gib=0, log=LOG, tick_s=0, **kw), d


def default_settings():
    return {k: (launch.spill_dir_default() if v[2] == "path" else v[1]) for k, v in launch.TIER_SETTINGS.items()}


# ---------------------------------------------------------------- the environment of the engine

def test_ssd_off_env_unset_empty_and_zero_all_mean_off():
    for value in (None, "", "  ", "0", "0.0", " 0 "):
        env = {"SPILL_DIR": "/cache/kv"}
        if value is not None:
            env["SPILL_SSD_GIB"] = value
        assert tiercache.ssd_off_env(env) is True and env["SPILL_DIR"] == "", value


def test_ssd_off_env_leaves_an_enabled_ssd_alone():
    env = {"SPILL_SSD_GIB": "10", "SPILL_DIR": "/cache/kv"}
    assert tiercache.ssd_off_env(env) is False and env["SPILL_DIR"] == "/cache/kv"
    env = {"SPILL_SSD_GIB": "0.5", "SPILL_DIR": "/x"}
    assert tiercache.ssd_off_env(env) is False and env["SPILL_DIR"] == "/x"


def test_ssd_off_env_bad_value_changes_nothing():  # (TierPolicy.from_env names the variable)
    env = {"SPILL_SSD_GIB": "lots", "SPILL_DIR": "/cache/kv"}
    assert tiercache.ssd_off_env(env) is False and env["SPILL_DIR"] == "/cache/kv"


def test_tier_env_is_explicit_and_empties_spill_dir_when_the_ssd_is_off():
    s = default_settings()
    env = launch.tier_env(s)
    assert set(env) == launch.TIER_ENV  # every variable, so the engine's own defaults never decide
    assert env["SPILL_SSD_GIB"] == "0" and env["SPILL_DIR"] == "" and env["SPILL_FAIR_AGENTS"] == "0"
    assert launch.tier_env(dict(s, spill_ssd_gib=0))["SPILL_DIR"] == ""
    on = launch.tier_env(dict(s, spill_ssd_gib=10, spill_fair_agents=True))
    assert on["SPILL_SSD_GIB"] == "10" and on["SPILL_DIR"] == launch.spill_dir_default() and on["SPILL_FAIR_AGENTS"] == "1"
    # the folder stays a setting (turning the SSD on again finds it); only the engine's variable is empty
    assert s["spill_dir"] == launch.spill_dir_default()
    assert launch.tier_env(dict(s, spill_dir="", spill_ssd_gib=10))["SPILL_DIR"] == ""


def test_the_engine_gets_the_environment_before_it_is_made():
    """worker.Scheduler.__init__: the start settings, then ssd_off_env, then the Engine (which reads SPILL_DIR when it is made)."""
    src = (ROOT / "frontend/worker.py").read_text(encoding="utf-8")
    assert src.index("os.environ[k] = v") < src.index("ssd_off_env(os.environ)") < src.index("self.engine = Engine(")


# ---------------------------------------------------------------- the defaults

def test_defaults_are_ssd_off_and_fair_agents_off(monkeypatch):
    assert launch.TIER_SETTINGS["spill_ssd_gib"][1] == 0 and launch.TIER_SETTINGS["spill_fair_agents"][1] is False
    for name in ("SPILL_SSD_GIB", "SPILL_FAIR_AGENTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", "/cache/kv")
    t = TierPolicy.from_env(FakeEngine(), log=LOG)
    assert t.directory is None and t.budget == 0 and t.fair_agents is False
    assert TierPolicy(FakeEngine(), "/x").directory is None and TierPolicy(FakeEngine(), "/x").fair_agents is False


def test_explicit_zero_means_off_and_a_cap_turns_it_on(monkeypatch, tmp_path):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", str(tmp_path))
    monkeypatch.setenv("SPILL_SSD_GIB", "0")
    assert TierPolicy.from_env(FakeEngine(), log=LOG).directory is None
    monkeypatch.setenv("SPILL_SSD_GIB", "100")
    assert TierPolicy.from_env(FakeEngine(), log=LOG).directory == tmp_path
    monkeypatch.setenv("SPILL_FAIR_AGENTS", "1")
    assert TierPolicy.from_env(FakeEngine(), log=LOG).fair_agents is True


def test_the_defaults_in_compose_and_docs():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert re.search(r"^\s+SPILL_SSD_GIB: \$\{SPILL_SSD_GIB:-0\}", compose, re.M)
    assert re.search(r"^\s+SPILL_FAIR_AGENTS: \$\{SPILL_FAIR_AGENTS:-0\}", compose, re.M)
    conf = (ROOT / "docs/configuration.md").read_text(encoding="utf-8")
    assert re.search(r"^\| `SPILL_SSD_GIB` \| 0 \|", conf, re.M) and re.search(r"^\| `SPILL_FAIR_AGENTS` \| 0 \|", conf, re.M)
    assert re.search(r"^\| `spill_ssd_gib` \|.*\(0: the SSD tier is off\) \|", conf, re.M) and re.search(r"^\| `spill_fair_agents` \|.*\(`false`\) \|", conf, re.M)


def test_start_package_adds_no_tier_default_the_server_passes_them_all(tmp_path):
    """start_package.py (the tray's start) adds no SPILL_* variable: the server's settings (launch.tier_env) pass every one."""
    sys.path.insert(0, str(ROOT / "scripts/windows"))
    import start_package
    env, _ = start_package.server_env({}, {}, tmp_path)
    assert not [k for k in env if k.startswith("SPILL_")]
    assert launch.tier_env(default_settings())["SPILL_FAIR_AGENTS"] == "0"


# ---------------------------------------------------------------- nothing touches the SSD

def test_start_makes_no_folder_imports_nothing_and_deletes_nothing(tmp_path):
    t, d = ssd_off_tier(tmp_path)
    (d / "entries").mkdir(parents=True)
    keep = [d / "entries" / "1-1.q27m", d / "old.q27kv", d / "stray.tmp", d / "note.txt"]
    for p in keep:
        p.write_bytes(MAGIC + b"data")
    t.start()
    assert t.engine.ssd_calls == [] and all(p.exists() for p in keep) and t.directory is None
    assert not (d / ".write-test").exists() and not (d / "chunks").exists()


def test_start_does_not_create_the_folder(tmp_path):
    t, d = ssd_off_tier(tmp_path)
    t.start()
    assert not d.exists() and t.engine.ssd_calls == []


def test_tick_writes_nothing_scans_nothing_and_leaves_an_old_folder_alone(tmp_path, monkeypatch):
    t, d = ssd_off_tier(tmp_path, ram_retention_s=0)
    ids = [t.engine.add_ram(400) for _ in range(3)]
    d.mkdir()
    old = d / "1-1.q27m"
    old.write_bytes(MAGIC + b"data")
    scanned = []
    monkeypatch.setattr(pathlib.Path, "glob", lambda self, pat: scanned.append(str(self)) or iter(()))
    monkeypatch.setattr(pathlib.Path, "iterdir", lambda self: scanned.append(str(self)) or iter(()))
    t.start()
    t.tick(idle=True, now=10 ** 9)
    t.tick(idle=True, now=10 ** 9 + 10 ** 6)
    assert t.engine.exports == [] and t.engine.drops == [] and t.engine.ssd_calls == [] and scanned == []
    assert all(t.engine.entries[i]["ram_bytes"] == 400 for i in ids)  # RAM tier unchanged (the engine's cap drops, not the policy)
    assert old.exists()
    assert t.make_room(0) is False and t.make_room(10 ** 9) is False


def test_no_flush_on_stop(tmp_path):
    t, d = ssd_off_tier(tmp_path, flush_on_stop_s=8)
    t.engine.add_ram(400)
    t.start()
    assert t.flush(8) == {"written": 0, "left": 0}
    assert t.engine.exports == [] and t.engine.ssd_calls == [] and not d.exists()


def test_status_counters_stay_zero_and_nothing_raises(tmp_path):
    t, d = ssd_off_tier(tmp_path, ram_retention_s=0)
    t.engine.add_ram(400)
    t.start()
    t.tick(idle=True, now=10 ** 9)
    st = t.status()
    assert st["ssd_enabled"] is False and st["ssd_off_reason"] == "gib"
    assert st["ssd_budget_bytes"] == 0 and st["ssd_bytes"] == 0 and st["disk_entries"] == 0 and st["ssd_logical_bytes"] == 0
    assert st["ssd_skipped_short"] == 0 and st["ssd_skipped_superseded"] == 0 and st["not_written"] == 0
    assert st["ram_entries"] == 1 and st["ram_bytes"] == 400 and st["last_error"] is None


def test_an_enabled_ssd_still_works(tmp_path):
    eng = StrictEngine()
    t = TierPolicy(eng, str(tmp_path / "kv"), ssd_gib=1, ssd_min_free_gib=0, ram_retention_s=0, tick_s=0, log=LOG,
                   disk_free=lambda _d: 10 ** 12, ssd_min_tokens=0)
    rid = eng.add_ram(400)
    t.start()
    t.tick(idle=True, now=10 ** 9)
    assert eng.exports == [rid] and t.status()["ssd_enabled"] is True and t.status()["ssd_budget_bytes"] == tiercache.GiB


def test_spill_dir_empty_with_a_cap_is_reported_as_dir():
    t = TierPolicy(FakeEngine(), "", ssd_gib=10, log=LOG)
    assert t.directory is None and t.status()["ssd_off_reason"] == "dir" and t.status()["ssd_budget_bytes"] == 0


# ---------------------------------------------------------------- the stats and the page

def test_tiers_block_of_an_off_ssd(tmp_path):
    t, _ = ssd_off_tier(tmp_path)
    t.engine.add_ram(400)
    t.start()
    t.tick(idle=True, now=10 ** 9)
    out = stats.tiers_block(t.engine, t, environ={"SPILL_RAM_GIB": "4"})
    assert "error" not in out
    assert out["ssd_enabled"] is False and out["ssd_cap_bytes"] == 0 and out["ssd_bytes"] == 0
    assert out["dir"] is None and out["disk_free_bytes"] is None and out["on_disk"] == 0 and out["written"] == 0


def test_deploy_info_shows_the_new_defaults():
    out = stats.deploy_info(0.0, "ready", environ={})
    assert out["settings"]["SPILL_SSD_GIB"] == "0" and out["settings"]["SPILL_FAIR_AGENTS"] == "0"
    out = stats.deploy_info(0.0, "ready", environ={"SPILL_SSD_GIB": "200", "SPILL_FAIR_AGENTS": "1"})
    assert out["settings"]["SPILL_SSD_GIB"] == "200" and out["settings"]["SPILL_FAIR_AGENTS"] == "1"
    assert stats.deploy_info(0.0, "ready", environ={"SPILL_SSD_GIB": ""})["settings"]["SPILL_SSD_GIB"] == "0"


def test_ui_shows_the_off_text_and_guards_the_division():
    html = (ROOT / "frontend/ui/index.html").read_text(encoding="utf-8")
    assert "SSD 層：已關閉（" in html and "SPILL_SSD_GIB=0" in html
    assert "ssdOffOf(" in html and "const pct = cap ?" in html  # (tierCol: no division by a zero or absent cap)
