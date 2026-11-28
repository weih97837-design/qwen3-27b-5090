"""Pure helpers of scripts/tiered_cache_test.py (scenario l's plan and chunk arithmetic); no GPU, no torch."""
import importlib.util
import pathlib
import subprocess
import sys
import types

PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tiered_cache_test.py"
spec = importlib.util.spec_from_file_location("tiered_cache_test_under_test", PATH)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def test_l_chunks_of_the_failing_run():
    assert t.l_chunks([], 12288) == [2048] * 6
    assert t.l_chunks([8192, 12125, 12286], 12288) == [2048] * 5 + [1856, 128, 64]
    assert t.fmt_chunks(t.l_chunks([8192, 12125, 12286], 12288)) == "2048x5 + 1856 + 128 + 64"
    assert t.l_chunks([8192, 12125], 12288, 12096) == [192]
    assert not t.plan_final_chunk_ok([8192, 12125, 12286], 12288)
    assert t.plan_final_chunk_ok([8192, 12125], 12288)


def test_l_chunks_interval_checkpoints():
    # an interval checkpoint more than 64 tokens from the others is a cut; one within 64 of the end is not
    assert t.l_chunks([], 10000, iv=4096) == [2048, 2048, 2048, 2048, 1808]
    assert 4096 in {sum(t.l_chunks([], 10000, iv=4096)[:i]) for i in range(1, 6)}
    assert t.l_chunks([], 4100, iv=4096) == t.l_chunks([], 4100)  # (4096 is within 64 of the end 4100)


def test_l_plan_ok():
    assert t.l_plan_ok([8192, 12125, 12286], 12288) == ([8192, 12125], [12286])
    assert t.l_plan_ok([8192, 12125, 12286], 12288, 12096) == ([8192, 12125], [12286])
    assert t.l_plan_ok([8192, 12125], 12288) == ([8192, 12125], [])


def test_l_plan_ok_never_drops_the_resume_point():
    assert t.l_plan_ok([12100], 12160) == ([], [12100])  # (without a floor everything goes)
    plan, dropped = t.l_plan_ok([12100], 12160, 12096)
    assert plan == [12100] and dropped == []  # (still not ok: the caller reports it)
    assert not t.plan_final_chunk_ok(plan, 12160, 0, resume=12096)


def test_b_is_checked_too():
    # A from 0 is fine (the cut at 12000 leaves 288), B resumed at 12224 would end in a 64-token chunk
    assert t.plan_final_chunk_ok([12000], 12288)
    assert not t.plan_final_chunk_ok([12000], 12288, resume=12224)


def test_first_diff():
    assert t.first_diff([1, 2, 3], [1, 2, 3]) is None
    assert t.first_diff([1, 2, 3], [1, 9, 3]) == 1
    assert t.first_diff([1, 2], [1, 2, 3]) == 2
    assert "different at token 2 (got 198, ref 271" in t.diff_text([198, 1, 198], [198, 1, 271])
    assert t.diff_text([1], [1]).startswith("equal")


# ---- the server's SSD tier is on by default (300 GiB), 0 turns it off: the SSD scenarios and the bench must turn it on, or fail loudly

def test_require_ssd_fails_loudly_with_the_ssd_off():
    import pytest
    for bad in (0, 0.0, "0", "", None, "x", -1):
        with pytest.raises(SystemExit, match="needs the SSD tier on"):
            t.require_ssd(bad, "scenario b")
    with pytest.raises(SystemExit, match="SPILL_DIR"):
        t.require_ssd("10", "scenario b worker", "")
    with pytest.raises(SystemExit, match="SPILL_DIR"):
        t.require_ssd("10", "scenario b worker", None)
    t.require_ssd(10, "scenario b")
    t.require_ssd("0.5", "scenario b worker", "/cache/kv/x")


def test_the_ssd_scenarios_are_the_ones_that_use_spill_dir():
    assert t.SSD_SCENARIOS == set("bcfghijkl") and not t.SSD_SCENARIOS & set("ae") and "d" not in t.SSD_SCENARIOS  # (i runs scen_h: export_ram too)
    assert t.SCEN_ENV["k"]["SPILL_FAIR_AGENTS"] == "1"
    assert t.SSD_SCENARIOS <= set(t.SCENARIOS) and t.DEFAULT_SSD_GIB == 10


def _args(**kw):
    base = dict(prompt_len=3665, tokens=48, pool_tokens=0, kv_dtype="nvfp4", weights="nvidia", ram_gib=2.0, spec="none", ssd_gib=10.0)
    return types.SimpleNamespace(**dict(base, **kw))


def test_the_tiered_workers_get_the_ssd_cap_and_dir_explicitly(tmp_path):
    for scen in sorted(t.SSD_SCENARIOS):
        env, cmd = t.worker_invocation(_args(), scen, "tiered", "full", tmp_path / "spill", tmp_path / "o.json")
        assert env["SPILL_SSD_GIB"] == "10" and env["SPILL_DIR"] == str(tmp_path / "spill"), scen
        assert cmd[cmd.index("--ssd-gib") + 1] == "10"
    env, cmd = t.worker_invocation(_args(ssd_gib=0.5), "b", "tiered", "full", tmp_path, tmp_path / "o.json")
    assert env["SPILL_SSD_GIB"] == "0.5"
    env, _ = t.worker_invocation(_args(), "b", "ref", "full", "", tmp_path / "o.json")
    assert "SPILL_SSD_GIB" not in env and env["STATE_HOST_CACHE"] == "0"  # (the reference has no cache at all)


def test_the_driver_refuses_ssd_scenarios_with_the_ssd_off():
    import subprocess
    import sys
    for scen in ("b", "a,c", "j"):
        r = subprocess.run([sys.executable, str(PATH), "--ssd-gib", "0", "--scenarios", scen], capture_output=True, text=True, timeout=60)
        assert r.returncode != 0 and "needs the SSD tier on" in (r.stdout + r.stderr), (scen, r.stdout, r.stderr)


def test_the_windows_scripts_turn_the_ssd_on_explicitly():
    root = PATH.parents[1]
    for name, script in (("test.cmd", "tiered_cache_test.py"), ("bench.cmd", "tiered_cache_bench.py")):
        text = (root / "scripts/windows" / name).read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if script in ln and not ln.startswith("rem"))
        assert "--ssd-gib 10" in line and line.index("--ssd-gib 10") < line.index("%*"), name  # (the user's arguments come later and win)
    bench = (root / "scripts/tiered_cache_bench.py").read_text(encoding="utf-8")
    assert 'require_ssd(args.ssd_gib' in bench and '"--ssd-gib"' in bench


def test_scenario_k_under_smart_turns_the_residency_guard_off(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHE_POLICY", raising=False)
    monkeypatch.delenv("CACHE_MIN_RESIDENT_S", raising=False)
    env, _ = t.worker_invocation(_args(cache_policy="smart"), "k", "tiered", "run", tmp_path / "spill", tmp_path / "o.json")
    assert env["CACHE_MIN_RESIDENT_S"] == "0" and env["SPILL_FAIR_AGENTS"] == "1" and env["CACHE_POLICY"] == "smart"
    monkeypatch.setenv("CACHE_MIN_RESIDENT_S", "30")  # (an explicit value is the user's: passed on visibly)
    env, _ = t.worker_invocation(_args(cache_policy="smart"), "k", "tiered", "run", tmp_path / "spill", tmp_path / "o.json")
    assert env["CACHE_MIN_RESIDENT_S"] == "30"
    monkeypatch.delenv("CACHE_MIN_RESIDENT_S")
    env, _ = t.worker_invocation(_args(cache_policy="legacy"), "k", "tiered", "run", tmp_path / "spill", tmp_path / "o.json")
    assert "CACHE_MIN_RESIDENT_S" not in env and env["CACHE_POLICY"] == "legacy"


def test_every_worker_gets_cache_policy_explicitly_default_smart(tmp_path, monkeypatch):
    monkeypatch.delenv("CACHE_POLICY", raising=False)
    for scen in "abcdefghijkl":
        for mode in ("ref", "tiered"):
            env, _ = t.worker_invocation(_args(), scen, mode, "full", tmp_path / "spill", tmp_path / "o.json")
            assert env["CACHE_POLICY"] == "smart", (scen, mode)  # (the server default; not left to the engine's C++ default, legacy)
            env, _ = t.worker_invocation(_args(cache_policy="legacy"), scen, mode, "full", tmp_path / "spill", tmp_path / "o.json")
            assert env["CACHE_POLICY"] == "legacy", (scen, mode)


def test_the_driver_clears_the_inherited_cache_variables_and_scenario_k_runs_both_policies(tmp_path, monkeypatch):
    assert t.policies_for("k", None) == ["smart", "legacy"]
    assert t.policies_for("k", "legacy") == ["legacy"] and t.policies_for("k", "smart") == ["smart"]
    assert t.policies_for("a", None) == ["smart"] and t.policies_for("l", "legacy") == ["legacy"]
    seen = {}

    def fake_run(cmd, env=None, **_kw):
        seen.update(env)
        out = cmd[cmd.index("--out") + 1]
        pathlib.Path(out).write_text("{}")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(t.subprocess, "run", fake_run)
    monkeypatch.setenv("CACHE_POLICY", "legacy")  # (inherited: the explicit value of the worker wins)
    monkeypatch.setenv("CACHE_PREFILL_T0_US", "99")
    monkeypatch.setenv("SPILL_FAIR_AGENTS", "0")
    monkeypatch.delenv("CACHE_MIN_RESIDENT_S", raising=False)
    t.run_worker(_args(cache_policy="smart"), "a", "tiered", "full", tmp_path / "spill", tmp_path / "o.json")
    assert seen["CACHE_POLICY"] == "smart" and seen["CACHE_PREFILL_T0_US"] == "99"  # (passed on by worker_invocation, not inherited)
    assert "SPILL_FAIR_AGENTS" not in seen


def test_the_cache_policy_option_and_the_windows_scripts_document_it():
    root = PATH.parents[1]
    for script in ("tiered_cache_test.py", "tiered_cache_bench.py"):
        text = (root / "scripts" / script).read_text(encoding="utf-8")
        assert '"--cache-policy"' in text and "smart" in text and "legacy" in text, script
    r = subprocess.run([sys.executable, str(PATH), "--help"], capture_output=True, text=True, timeout=60)
    assert "--cache-policy" in r.stdout
    for name in ("test.cmd", "bench.cmd"):
        assert "--cache-policy" in (root / "scripts/windows" / name).read_text(encoding="utf-8"), name
