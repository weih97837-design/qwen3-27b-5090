"""The SSD tier is ON by default (SPILL_SSD_GIB=300), the per-agent step (SPILL_FAIR_AGENTS) is on, SPILL_SSD_WARM_ONLY and
SPILL_UNUSED_TTL_S are off. An explicit SPILL_SSD_GIB=0 turns the SSD tier off: the engine gets SPILL_DIR="" (it drops RAM victims
instead of writing them) and TierPolicy has no directory: nothing is written, read, scanned or deleted in any SPILL_DIR. No GPU."""

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

def test_ssd_off_env_only_an_explicit_zero_means_off():
    for value in ("0", "0.0", " 0 "):
        env = {"SPILL_DIR": "/cache/kv", "SPILL_SSD_GIB": value}
        assert tiercache.ssd_off_env(env) is True and env["SPILL_DIR"] == "", value
    for value in (None, "", "  "):  # unset or empty: the default, 300 GiB, the tier is on
        env = {"SPILL_DIR": "/cache/kv"}
        if value is not None:
            env["SPILL_SSD_GIB"] = value
        assert tiercache.ssd_off_env(env) is False and env["SPILL_DIR"] == "/cache/kv", value


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
    assert env["SPILL_SSD_GIB"] == "300" and env["SPILL_DIR"] == launch.spill_dir_default() and env["SPILL_FAIR_AGENTS"] == "1"
    assert env["SPILL_SSD_WARM_ONLY"] == "0" and env["SPILL_UNUSED_TTL_S"] == "0"
    assert launch.tier_env(dict(s, spill_ssd_gib=0))["SPILL_DIR"] == ""  # an explicit 0: the tier is off, the engine gets no folder
    on = launch.tier_env(dict(s, spill_ssd_gib=10, spill_fair_agents=False))
    assert on["SPILL_SSD_GIB"] == "10" and on["SPILL_DIR"] == launch.spill_dir_default() and on["SPILL_FAIR_AGENTS"] == "0"
    # the folder stays a setting (turning the SSD on again finds it); only the engine's variable is empty
    assert s["spill_dir"] == launch.spill_dir_default()
    assert launch.tier_env(dict(s, spill_dir="", spill_ssd_gib=10))["SPILL_DIR"] == ""


def test_the_engine_gets_the_environment_before_it_is_made():
    """worker.Scheduler.__init__: the start settings, then ssd_off_env, then the Engine (which reads SPILL_DIR when it is made)."""
    src = (ROOT / "frontend/worker.py").read_text(encoding="utf-8")
    assert src.index("os.environ[k] = v") < src.index("ssd_off_env(os.environ)") < src.index("self.engine = Engine(")


# ---------------------------------------------------------------- the defaults

def test_defaults_are_ssd_300_fair_on_warm_only_and_ttl_off(monkeypatch):
    ts = launch.TIER_SETTINGS
    assert ts["spill_ssd_gib"][1] == 300 and ts["spill_fair_agents"][1] is True
    assert ts["spill_ssd_warm_only"][1] is False and ts["spill_unused_ttl_s"][1] == 0
    for name in ("SPILL_SSD_GIB", "SPILL_FAIR_AGENTS", "SPILL_SSD_WARM_ONLY", "SPILL_UNUSED_TTL_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", "/cache/kv")
    t = TierPolicy.from_env(FakeEngine(), log=LOG)
    assert str(t.directory) == "/cache/kv" and t.budget == 300 * tiercache.GiB and t.fair_agents is True
    assert t.ssd_warm_only is False and t.unused_ttl_s == 0
    d = TierPolicy(FakeEngine(), "/x")  # (the constructor's own defaults are the same)
    assert str(d.directory) == "/x" and d.budget == 300 * tiercache.GiB and d.fair_agents is True
    assert d.ssd_warm_only is False and d.unused_ttl_s == 0
    assert tiercache.DEFAULT_SSD_GIB == 300 and tiercache.ssd_gib_of_env({}) == 300 and tiercache.ssd_gib_of_env({"SPILL_SSD_GIB": ""}) == 300


def test_empty_env_values_take_the_defaults(monkeypatch):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", "/cache/kv")
    for name in ("SPILL_SSD_GIB", "SPILL_FAIR_AGENTS", "SPILL_SSD_WARM_ONLY", "SPILL_UNUSED_TTL_S"):
        monkeypatch.setenv(name, "")
    t = TierPolicy.from_env(FakeEngine(), log=LOG)
    assert t.directory is not None and t.budget == 300 * tiercache.GiB and t.fair_agents is True
    assert t.ssd_warm_only is False and t.unused_ttl_s == 0


def test_unused_ttl_zero_expires_nothing(tmp_path):
    """SPILL_UNUSED_TTL_S=0 (the default) is off: a never-hit entry is not dropped however long it sits."""
    eng = FakeEngine()
    t = TierPolicy(eng, str(tmp_path / "kv"), ram_retention_s=10 ** 9, tick_s=0, log=LOG)
    rid = eng.add_ram(400)
    t.start()
    t.tick(idle=True, now=10 ** 9)
    t.tick(idle=True, now=10 ** 9 + 10 ** 7)
    assert rid not in eng.drops and t.status()["expired"] == 0


def test_explicit_zero_means_off_and_a_cap_turns_it_on(monkeypatch, tmp_path):
    monkeypatch.setenv("SPILL_RAM_GIB", "4")
    monkeypatch.setenv("SPILL_DIR", str(tmp_path))
    monkeypatch.setenv("SPILL_SSD_GIB", "0")
    assert TierPolicy.from_env(FakeEngine(), log=LOG).directory is None
    monkeypatch.setenv("SPILL_SSD_GIB", "100")
    assert TierPolicy.from_env(FakeEngine(), log=LOG).directory == tmp_path
    monkeypatch.setenv("SPILL_FAIR_AGENTS", "0")
    assert TierPolicy.from_env(FakeEngine(), log=LOG).fair_agents is False


def test_explicit_zero_in_the_settings_switches_the_ssd_off_for_the_engine():
    """launch.tier_env: SPILL_SSD_GIB=0 from settings.ini or the environment gives the engine an empty SPILL_DIR."""
    s = default_settings()
    s["spill_ssd_gib"] = 0
    env = launch.tier_env(s)
    assert env["SPILL_SSD_GIB"] == "0" and env["SPILL_DIR"] == ""


def test_the_defaults_in_compose_and_docs():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    for name, value in (("SPILL_SSD_GIB", "300"), ("SPILL_FAIR_AGENTS", "1"), ("SPILL_SSD_WARM_ONLY", "0"), ("SPILL_UNUSED_TTL_S", "0")):
        assert re.search(rf"^\s+{name}: \$\{{{name}:-{value}\}}", compose, re.M), name
    conf = (ROOT / "docs/configuration.md").read_text(encoding="utf-8")
    for name, value in (("SPILL_SSD_GIB", "300"), ("SPILL_FAIR_AGENTS", "1"), ("SPILL_SSD_WARM_ONLY", "0"), ("SPILL_UNUSED_TTL_S", "0")):
        assert re.search(rf"^\| `{name}` \| {value} \|", conf, re.M), name
    assert re.search(r"^\| `spill_ssd_gib` \|.*\(300; 0: the SSD tier is off\) \|", conf, re.M)
    assert re.search(r"^\| `spill_fair_agents` \|.*\(`true`\) \|", conf, re.M) and re.search(r"^\| `spill_ssd_warm_only` \|.*\(`false`\) \|", conf, re.M)
    assert re.search(r"^\| `spill_unused_ttl_s` \|.*\(0: off\) \|", conf, re.M)
    ini = (ROOT / "scripts/windows/settings.example.ini").read_text(encoding="utf-8")
    assert ";SPILL_SSD_GIB=300" in ini and ";SPILL_FAIR_AGENTS=1" in ini


def test_start_package_adds_no_tier_default_the_server_passes_them_all(tmp_path):
    """start_package.py (the tray's start) adds no SPILL_* variable: the server's settings (launch.tier_env) pass every one."""
    sys.path.insert(0, str(ROOT / "scripts/windows"))
    import start_package
    env, _ = start_package.server_env({}, {}, tmp_path)
    assert not [k for k in env if k.startswith("SPILL_")]
    assert launch.tier_env(default_settings())["SPILL_FAIR_AGENTS"] == "1"


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
    assert stats.DEPLOY_DEFAULTS == {"SPILL_SSD_GIB": "300", "SPILL_FAIR_AGENTS": "1", "SPILL_SSD_WARM_ONLY": "0", "SPILL_UNUSED_TTL_S": "0"}
    assert [out["settings"][k] for k in stats.DEPLOY_DEFAULTS] == ["300", "1", "0", "0"]
    out = stats.deploy_info(0.0, "ready", environ={"SPILL_SSD_GIB": "200", "SPILL_FAIR_AGENTS": "0", "SPILL_UNUSED_TTL_S": "900"})
    assert out["settings"]["SPILL_SSD_GIB"] == "200" and out["settings"]["SPILL_FAIR_AGENTS"] == "0" and out["settings"]["SPILL_UNUSED_TTL_S"] == "900"
    assert stats.deploy_info(0.0, "ready", environ={"SPILL_SSD_GIB": ""})["settings"]["SPILL_SSD_GIB"] == "300"
    assert stats.deploy_info(0.0, "ready", environ={"SPILL_SSD_GIB": "0"})["settings"]["SPILL_SSD_GIB"] == "0"  # an explicit 0 stays 0


def test_ui_shows_the_off_text_and_guards_the_division():
    html = (ROOT / "frontend/ui/index.html").read_text(encoding="utf-8")
    assert "SSD 層：已關閉（" in html and "SPILL_SSD_GIB=0" in html
    assert "ssdOffOf(" in html and "const pct = cap ?" in html  # (tierCol: no division by a zero or absent cap)
