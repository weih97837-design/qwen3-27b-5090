"""Start settings of the engine (launch.py) and the device memory their plans assume (stats.Hub.plan_device). The
plans themselves need libq27 and the weights' headers (no GPU); without them those tests are skipped."""

import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "reference"))

import launch  # noqa: E402
from test_stats import hub  # noqa: E402

GiB = 1 << 30
DEFAULTS = {"weights": "gdnfp4", "kv_dtype": "nvfp4", "spec": "dspine", "vision": "host", "max_seqs": 8,
            "max_tokens": 262144, "vram_reserve_gib": None,
            **{k: (launch.spill_dir_default() if v[2] == "path" else v[1]) for k, v in launch.TIER_SETTINGS.items()}}
DEVICE = (int(30.20 * GiB), int(31.82 * GiB))  # (free at the start, total) of an engine on the RTX 5090 under WSL2


def settings():
    return launch.Launch({"prefill_chunk": 2048, "engine_python": str(ROOT / "engine/python")}, dict(DEFAULTS))


def test_auto_budget_is_the_engines():
    # memory.cc budget_for: max(0, free - reserve), the reserve 1.5 GiB unless given.
    assert abs(launch.auto_budget(*DEVICE) / GiB - 28.70) < 0.01
    assert abs(launch.auto_budget(*DEVICE, reserve=0.5) / GiB - 29.70) < 0.01
    assert abs(launch.auto_budget(*DEVICE, reserve=0) / GiB - 30.20) < 0.01
    assert launch.auto_budget(int(1 * GiB), DEVICE[1]) == 0 and launch.auto_budget(int(2 * GiB), reserve=3) == 0


def test_validate():
    lc = settings()
    assert lc.validate({"kv_dtype": "fp8", "vram_reserve_gib": 2}) == dict(DEFAULTS, kv_dtype="fp8", vram_reserve_gib=2.0)
    assert lc.settings == DEFAULTS  # (validate does not change them)
    for bad in ({"kv_dtype": "int4"}, {"max_seqs": 9}, {"max_seqs": 2.0}, {"max_tokens": 1 << 20}, {"vision": True},
                {"vram_reserve_gib": -0.5}, {"vram_reserve_gib": "auto"}, {"vram_reserve_gib": True},
                {"vram_reserve_gib": float("nan")}, {"vram_reserve_gib": float("inf")}, {"vram_budget_gib": 20},
                {"what": 1}, {"weights": "/elsewhere"}):
        with pytest.raises(ValueError):
            lc.validate(bad)
    assert lc.validate({"vram_reserve_gib": 0})["vram_reserve_gib"] == 0.0
    assert lc.validate({"vram_reserve_gib": 1.234})["vram_reserve_gib"] == 1.23
    assert lc.validate({"vram_reserve_gib": None})["vram_reserve_gib"] is None
    with pytest.raises(ValueError, match="at most the device's memory"):
        lc.validate({"vram_reserve_gib": 40}, device=DEVICE)
    assert lc.validate({"vram_reserve_gib": 40}, device=DEVICE, check_budget=False)["vram_reserve_gib"] == 40.0
    assert lc.validate({"vram_reserve_gib": 4}, device=DEVICE)["vram_reserve_gib"] == 4.0


def test_reserve_default_is_the_compose_and_engine_one():
    import re
    compose = (ROOT / "compose.yaml").read_text()
    assert float(re.search(r"^\s+VRAM_RESERVE_GIB: \$\{VRAM_RESERVE_GIB:-([^}]*)\}", compose, re.M).group(1)) == launch.DEFAULT_RESERVE_GIB
    assert re.search(r"double reserve = %s;" % launch.DEFAULT_RESERVE_GIB, (ROOT / "engine/src/memory.cc").read_text())
    assert launch.ENV_NAMES["vram_reserve_gib"] == "VRAM_RESERVE_GIB"
    assert launch.env_settings()["vram_reserve_gib"] is None or os.environ.get("VRAM_RESERVE_GIB")


def test_reserve_env_is_read_like_the_engine():
    assert launch.parse_reserve("") is None and launch.parse_reserve(None) is None
    assert launch.parse_reserve("2.5") == 2.5
    assert launch.parse_reserve("-1") == 0.0 and launch.parse_reserve("abc") == 0.0 and launch.parse_reserve("nan") == 0.0


def test_the_launcher_clears_a_legacy_budget():
    import inspect
    assert '"VRAM_BUDGET_GIB": None' in inspect.getsource(launch.Launch.config)


def test_env_changes_name_what_changed_since_the_start():
    lc = object.__new__(launch.Launch)
    lc.settings = lc.at_start = dict(DEFAULTS, vram_reserve_gib=3.0)
    assert lc.env_changes() == {}
    lc.settings = dict(lc.at_start, kv_dtype="fp8", max_seqs=4, vram_reserve_gib=None)
    assert lc.env_changes() == {"KV_DTYPE": "fp8", "MAX_SEQS": "4", "VRAM_RESERVE_GIB": ""}


def test_plan_device():
    h = hub()
    assert h.plan_device() == (None, None)
    h.samples.append({"t": 0, "used": 1 * GiB, "total": 32 * GiB})
    h.engine_state = "stopped"
    assert h.plan_device() == ((31 * GiB - h.context_bytes, 32 * GiB), "nvml")
    # A started engine: its free memory at the start, and the CUDA context measured against NVML before it.
    h.baseline_used, h.nvml_total = 1 * GiB, 32 * GiB
    h.engine_state = "ready"
    h.set_snapshot({"t": 1.0, "seqs": [], "device": {"free_at_start": 30 * GiB, "total": 31.9 * GiB}})
    assert h.plan_device() == ((30 * GiB, 31.9 * GiB), "engine")
    assert h.context_bytes == 1 * GiB
    h.engine_state = "stopped"  # (another process may take memory meanwhile: NVML now)
    h.samples.append({"t": 2, "used": 3 * GiB, "total": 32 * GiB})
    assert h.plan_device() == ((28 * GiB, 31.9 * GiB), "nvml")


def test_device_free_now_counts_other_programs():
    h = hub()
    assert h.device_free_now() is None  # (no NVML)

    class Nvml:
        @staticmethod
        def sample():
            return {"used": 6 * GiB, "total": 32 * GiB}

    h.nvml, h.context_bytes = Nvml, 1 * GiB
    assert h.device_free_now() == 25 * GiB


def planner_or_skip():
    lc = settings()
    if not pathlib.Path(os.environ.get("Q27_LIB", "/build/engine/libq27.so")).exists():
        pytest.skip("no libq27.so")
    if lc.weights_choices()["gdnfp4"] is not None:
        pytest.skip("the gdnfp4 weights are not downloaded")
    return lc


def test_plans_match_the_engine():
    lc = planner_or_skip()
    os.environ.setdefault("MAX_TOKENS_FIT", "1")
    os.environ.setdefault("STATE_HOST_CACHE", "8")
    out = lc.plans(lc.settings, DEVICE)
    p = out["plan"]
    # The serve default's startup log on this device: kv pool 9461 pages (605504 tokens), device 28.377 GiB
    # (28.378 until the own GEMMs replaced CUTLASS and its 7.8 MiB workspace, then 28.370 until the penalty
    # counts of the 8 sequences, 7.6 MiB). With the DSpine draft (the default since; window 2048): 601856 tokens,
    # 28.3924 GiB (its codebook is in host memory, its sender table of the draft vocabulary, 256 MB, on the device).
    assert (p["max_tokens"], p["kv_tokens"], p["fits"]) == (262144, 601856, True)
    assert abs(p["device"] / GiB - 28.3924) < 0.001 and abs(p["budget"] / GiB - 28.68) < 0.01
    kv = {v["value"]: v["plan"] for v in out["variants"]["kv_dtype"]}
    assert kv["nvfp4"] == p and kv["fp8"]["kv_tokens"] < kv["nvfp4"]["kv_tokens"]
    big = {v["value"]: v["plan"] for v in out["variants"]["vram_reserve_gib"]}
    assert big[4.0]["kv_tokens"] < p["kv_tokens"] <= big[0.5]["kv_tokens"] and big[None] == p
    seqs = [v["plan"]["kv_tokens"] for v in out["variants"]["max_seqs"]]
    assert seqs == sorted(seqs, reverse=True)  # (each sequence's buffers take some of the pool)


def test_plan_errors():
    lc = planner_or_skip()
    p = lc.plan_many([dict(DEFAULTS, vram_reserve_gib=20.0)], DEVICE)[0]
    assert "error" in p and "VRAM budget" in p["error"]


def test_tokenizer_fingerprint(tmp_path):
    # The ids decide: a tokenizer.json rewritten by another transformers (merges as strings, another pre-tokenizer
    # pattern, decoder flags; Huihui's abliterated checkpoint) is the same tokenizer, and so is one with another chat
    # template (unsloth's: prompts are then rendered with it); another vocabulary is not.
    from tokenizer import fingerprint

    def write(name, merges, pattern, vocab=None, template="{{ eos_token }}"):
        d = tmp_path / name
        d.mkdir()
        tok = {"model": {"type": "BPE", "vocab": vocab or {"a": 0, "b": 1, "ab": 2}, "merges": merges},
               "added_tokens": [{"id": 3, "content": "<|im_start|>", "special": True}],
               "pre_tokenizer": {"type": "Split", "pattern": {"Regex": pattern}}, "decoder": {"type": "ByteLevel"}}
        (d / "tokenizer.json").write_text(json.dumps(tok))
        (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": template, "eos_token": "<|im_end|>"}))
        return d

    base = fingerprint(write("base", [["a", "b"]], r"\p{L}\p{M}+"))
    assert fingerprint(write("rewritten", ["a b"], r"\p{L}+")) == base
    assert fingerprint(write("template", [["a", "b"]], r"\p{L}\p{M}+", template="{{ bos_token }}")) == base
    assert fingerprint(write("other", [["a", "b"]], r"\p{L}\p{M}+", {"a": 0, "b": 1, "ab": 5})) != base


def test_too_little_free_vram_lowers_the_pool_and_a_big_reserve_does_too():
    lc = planner_or_skip()
    full = lc.plan_many([dict(DEFAULTS)], DEVICE)[0]
    busy, reserved = lc.plan_many([dict(DEFAULTS), dict(DEFAULTS, vram_reserve_gib=6.2)], (int(25 * GiB), DEVICE[1]))
    assert busy["kv_tokens"] < full["kv_tokens"] and busy["fits"]
    assert busy["budget"] == launch.auto_budget(int(25 * GiB), DEVICE[1])
    assert reserved["budget"] == launch.auto_budget(int(25 * GiB), DEVICE[1], reserve=6.2) < busy["budget"]
    assert reserved["device"] <= reserved["budget"]


def test_every_shortage_is_named_as_such():
    # (worker.run tells these from a crash by this text: no retries with ENGINE_IDLE_UNLOAD_S)
    lc = planner_or_skip()
    for free in (21, 20, 19, 18, 5):
        plan = lc.plan_many([dict(DEFAULTS)], (int(free * GiB), DEVICE[1]))[0]
        assert "not enough VRAM" in plan.get("error", "") or plan["fits"], (free, plan)


def test_min_kv_tokens_refuses(monkeypatch):
    lc = planner_or_skip()
    tokens = lc.plan_many([dict(DEFAULTS)], DEVICE)[0]["kv_tokens"]
    monkeypatch.setenv("MIN_KV_TOKENS", str(tokens))
    lc = planner_or_skip()
    assert lc.plan_many([dict(DEFAULTS)], DEVICE)[0]["kv_tokens"] == tokens
    plan = lc.plan_many([dict(DEFAULTS)], (int(25 * GiB), DEVICE[1]))[0]
    assert "not enough VRAM" in plan["error"] and "MIN_KV_TOKENS" in plan["error"]


def test_tokenizer_template_of_other_weights():
    # The server's tokenizer with the chat template of the weights the engine runs (Server.use_template_of).
    from tokenizer import Tokenizer

    try:
        own, other = launch.model_dir("gdnfp4"), launch.model_dir("unsloth/Qwen3.8-27B-NVFP4")
    except (FileNotFoundError, KeyError, ValueError) as e:
        pytest.skip(str(e))
    msgs = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, {"role": "user", "content": "c"}]
    mixed, theirs = Tokenizer(own, template=other), Tokenizer(other)
    assert mixed.path == own and mixed.template_path == other
    assert mixed.render(msgs) == theirs.render(msgs)  # (unsloth's template merges the system messages)
    with pytest.raises(Exception, match="System message must be at the beginning"):
        Tokenizer(own).render(msgs)
    assert mixed.encode("na\u0308ive text") == Tokenizer(own).encode("na\u0308ive text")  # (a combining mark)


ENGINE_KEYS = ("KV_POOL_TOKENS", "SPILL_RAM_GIB", "SPILL_DIR", "MAX_TOKENS_FIT", "STATE_HOST_CACHE", "Q27_DRAFT_VOCAB")


def _clean_engine_env(monkeypatch):
    # setenv then delenv: monkeypatch restores the variables after the test, whatever set_engine_env writes.
    for k in ENGINE_KEYS:
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)


def test_engine_defaults_applied_when_unset(monkeypatch, tmp_path):
    _clean_engine_env(monkeypatch)
    launch.set_engine_env(tmp_path)
    assert os.environ["KV_POOL_TOKENS"] == "524288"
    assert os.environ["SPILL_RAM_GIB"] == "64"
    assert os.environ["SPILL_DIR"] == ("/cache/kv" if os.name != "nt" else launch.spill_dir_default())
    assert os.environ["MAX_TOKENS_FIT"] == "1" and os.environ["STATE_HOST_CACHE"] == "8"


def test_engine_defaults_do_not_override_set_values(monkeypatch, tmp_path):
    _clean_engine_env(monkeypatch)
    monkeypatch.setenv("KV_POOL_TOKENS", "0")
    monkeypatch.setenv("SPILL_RAM_GIB", "0")
    monkeypatch.setenv("SPILL_DIR", "/elsewhere")
    launch.set_engine_env(tmp_path)
    assert os.environ["KV_POOL_TOKENS"] == "0"
    assert os.environ["SPILL_RAM_GIB"] == "0"
    assert os.environ["SPILL_DIR"] == "/elsewhere"


def test_spill_dir_on_windows_is_a_user_folder():
    local = r"C:\Users\someone\AppData\Local"
    assert launch.spill_dir_default("nt", {"LOCALAPPDATA": local}) == os.path.join(local, "qwen3-27b-5090", "kvcache")
    home_only = launch.spill_dir_default("nt", {})  # no LOCALAPPDATA: the home folder
    assert home_only == os.path.join(os.path.expanduser("~"), "qwen3-27b-5090", "kvcache")
    assert launch.spill_dir_default("posix", {"LOCALAPPDATA": local}) == "/cache/kv"
    assert launch.engine_env_defaults("nt", {"LOCALAPPDATA": local})["SPILL_DIR"].endswith(
        os.path.join("qwen3-27b-5090", "kvcache"))
    assert launch.engine_env_defaults("posix", {})["SPILL_DIR"] == "/cache/kv"


def test_tier_policy_sees_the_defaults(monkeypatch, tmp_path):
    from tiercache import TierPolicy

    _clean_engine_env(monkeypatch)
    launch.set_engine_env(tmp_path)
    tiers = TierPolicy.from_env(object(), log=lambda *a: None)  # (the worker's read, after set_engine_env)
    assert tiers is not None and tiers.directory == pathlib.Path(launch.spill_dir_default())
    assert tiers.budget == 100 * (1 << 30) and tiers.ttl_s == 7 * 86400.0  # (the other SPILL_* keep their defaults)


# ---------------------------------------------------------------- the tiered cache settings

def test_tier_defaults_are_the_compose_ones():
    import re
    compose = (ROOT / "compose.yaml").read_text()
    for key, (var, default, kind) in launch.TIER_SETTINGS.items():
        m = re.search(rf"^\s+{var}: \$\{{{var}:-([^}}]*)\}}", compose, re.M)
        assert m, var
        if kind == "path":
            assert m.group(1) == launch.ENGINE_ENV_DEFAULTS[var]
        else:
            assert float(m.group(1)) == float(default), var
        assert launch.ENV_NAMES[key] == var
    assert launch.ENGINE_ENV_DEFAULTS["SPILL_RAM_GIB"] == str(launch.TIER_SETTINGS["spill_ram_gib"][1])


def test_tier_validate():
    lc = settings()
    s = lc.validate({"spill_ram_gib": 64, "spill_ssd_gib": 0.5, "spill_async_io": True, "spill_dir": ""})
    assert s["spill_ram_gib"] == 64 and s["spill_ssd_gib"] == 0.5 and s["spill_async_io"] is True and s["spill_dir"] == ""
    assert lc.validate({"spill_ram_gib": 0})["spill_ram_gib"] == 0       # (tiered cache off)
    assert lc.validate({"spill_ram_gib": 4096})["spill_ram_gib"] == 4096  # (no artificial upper limit)
    assert lc.validate({"spill_ssd_gib": 1e6})["spill_ssd_gib"] == 1000000
    assert isinstance(lc.validate({"spill_ttl_days": 7.0})["spill_ttl_days"], int)
    for bad in ({"spill_ram_gib": -1}, {"spill_ram_gib": "32"}, {"spill_ram_gib": True}, {"spill_ssd_gib": float("nan")},
                {"spill_ttl_days": float("inf")}, {"vram_backup_idle_s": None}, {"spill_async_io": 1},
                {"spill_async_promote": "yes"}, {"spill_dir": 3}, {"spill_dir": " /x"}, {"spill_dir": "a\0b"}):
        with pytest.raises(ValueError):
            lc.validate(bad)


def test_ssd_min_tokens_setting():
    lc = settings()
    assert launch.TIER_SETTINGS["spill_ssd_min_tokens"][:2] == ("SPILL_SSD_MIN_TOKENS", 32768)
    assert lc.validate({})["spill_ssd_min_tokens"] == 32768
    assert lc.validate({"spill_ssd_min_tokens": 0})["spill_ssd_min_tokens"] == 0
    v = lc.validate({"spill_ssd_min_tokens": 4096.7})["spill_ssd_min_tokens"]
    assert v == 4096 and isinstance(v, int)  # (a count: rounded down)
    for bad in (-1, "8192", True, float("nan"), None):
        with pytest.raises(ValueError):
            lc.validate({"spill_ssd_min_tokens": bad})
    assert launch.parse_tier("spill_ssd_min_tokens", "1000.5") == 1000
    assert launch.parse_tier("spill_ssd_min_tokens", None) == 32768
    assert launch.parse_tier("spill_ssd_min_tokens", "-3", warn=lambda m: None) == 32768


def test_tier_validate_dir_is_a_folder(tmp_path):
    lc = settings()
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(ValueError, match="is a file"):
        lc.validate({"spill_dir": str(f)})
    assert lc.validate({"spill_dir": str(f)}, check_budget=False)["spill_dir"] == str(f)  # (a save, the server's start)
    assert lc.validate({"spill_dir": str(tmp_path / "new")})["spill_dir"] == str(tmp_path / "new")


def test_tier_env_settings_are_read_like_tierpolicy_and_the_engine(monkeypatch):
    for var in launch.TIER_ENV:
        monkeypatch.delenv(var, raising=False)
    s = launch.env_settings()
    assert {k: s[k] for k in launch.TIER_SETTINGS if k != "spill_dir"} == \
        {k: v[1] for k, v in launch.TIER_SETTINGS.items() if k != "spill_dir"}
    monkeypatch.setenv("SPILL_RAM_GIB", "48.5")
    monkeypatch.setenv("SPILL_TTL_DAYS", "")
    monkeypatch.setenv("SPILL_ASYNC_PROMOTE", "1")
    monkeypatch.setenv("SPILL_ASYNC_IO", "0")
    monkeypatch.setenv("VRAM_BACKUP_IDLE_S", "-5")  # (TierPolicy refuses it: the default, with a warning)
    monkeypatch.setenv("SPILL_DIR", "/data/kv")
    s = launch.env_settings()
    assert s["spill_ram_gib"] == 48.5 and s["spill_ttl_days"] == 7 and s["spill_async_promote"] is True
    assert s["spill_async_io"] is False and s["vram_backup_idle_s"] == 60 and s["spill_dir"] == "/data/kv"
    assert launch.Launch({"prefill_chunk": 1}, dict(DEFAULTS, **{k: s[k] for k in launch.TIER_SETTINGS})).settings["spill_ram_gib"] == 48.5


def test_tier_flags_are_read_like_atoi(monkeypatch):
    for text, want in [("1", True), ("0", False), ("1.0", True), ("0.9", False), (" 2", True), ("\t-1", True), ("+1", True),
                       ("true", False), ("yes", False), ("00", False), ("-0", False), ("1abc", True), ("x1", False), ("- 1", False)]:
        assert launch.parse_tier("spill_async_io", text) is want, text
        monkeypatch.setenv("SPILL_ASYNC_PROMOTE", text)
        assert launch.env_settings()["spill_async_promote"] is want, text
    assert launch.parse_tier("spill_async_io", "") is True and launch.parse_tier("spill_async_io", None) is True  # (the default: on)


def test_tier_settings_reach_the_engine_env_and_the_policy(monkeypatch, tmp_path):
    """Launch.config's env is what worker.Scheduler puts in os.environ before TierPolicy.from_env reads it."""
    lc = settings()
    monkeypatch.setattr(launch, "model_dir", lambda w: tmp_path)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 1}')
    s = lc.validate({"spec": "none", "spill_ram_gib": 16, "spill_ram_retention_s": 120, "spill_ssd_gib": 50, "spill_ssd_min_free_gib": 2.5,
                     "spill_ttl_days": 3, "vram_backup_idle_s": 0, "spill_pinned_gib": 4, "spill_async_io": True, "spill_async_promote": False,
                     "spill_ssd_min_tokens": 2048, "spill_dir": str(tmp_path / "kv")})
    env = lc.config(s)["env"]
    assert env["SPILL_RAM_GIB"] == "16" and env["SPILL_PINNED_GIB"] == "4" and env["SPILL_ASYNC_IO"] == "1"
    assert env["SPILL_SSD_MIN_TOKENS"] == "2048" and env["SPILL_ASYNC_PROMOTE"] == "0" and env["SPILL_DIR"] == str(tmp_path / "kv")
    for var in launch.TIER_ENV:
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():  # (what the worker does)
        if v is not None:
            monkeypatch.setenv(k, v)
    from tiercache import TierPolicy
    policy = TierPolicy.from_env(object())
    assert policy.ram_retention_s == 120 and policy.budget == 50 * GiB and policy.reserve == 2.5 * GiB
    assert policy.ssd_min_tokens == 2048 and policy.ttl_s == 3 * 86400 and policy.backup_idle_s == 0 and str(policy.directory) == str(tmp_path / "kv")
    monkeypatch.setenv("SPILL_RAM_GIB", lc.config(dict(s, spill_ram_gib=0))["env"]["SPILL_RAM_GIB"])
    assert TierPolicy.from_env(object()) is None  # (0: off)


def test_tier_env_is_not_part_of_a_plan_job(monkeypatch, tmp_path):
    lc = settings()
    monkeypatch.setattr(launch, "model_dir", lambda w: tmp_path)
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": 1}')
    jobs = []
    monkeypatch.setattr(lc, "_run", lambda j: jobs.append(j) or [{"fits": True}] * len(j))
    base = lc.validate({"spec": "none"})
    lc.plan_many([base, dict(base, spill_ram_gib=64, spill_async_io=True)], DEVICE)
    assert len(jobs) == 1 and len(jobs[0]) == 1  # (one plan for both)
    assert not set(jobs[0][0]["env"]) & launch.TIER_ENV


def test_tier_env_changes():
    lc = object.__new__(launch.Launch)
    lc.settings = lc.at_start = dict(DEFAULTS)
    lc.settings = dict(lc.at_start, spill_ram_gib=48, spill_async_io=False, spill_ttl_days=0.5, spill_dir="")
    assert lc.env_changes() == {"SPILL_RAM_GIB": "48", "SPILL_ASYNC_IO": "0", "SPILL_TTL_DAYS": "0.5", "SPILL_DIR": ""}


def test_the_stats_page_settings_follow_the_applied_ones(monkeypatch):
    h = hub()
    monkeypatch.setenv("SPILL_RAM_GIB", "32")
    assert h.deploy_environ()["SPILL_RAM_GIB"] == "32"
    h.deploy_env = lambda: {"SPILL_RAM_GIB": "64", "VRAM_RESERVE_GIB": None}
    assert h.deploy_environ()["SPILL_RAM_GIB"] == "64"
    assert h.tick()["deploy"]["settings"]["SPILL_RAM_GIB"] == "64"


def test_spill_fair_agents_setting(monkeypatch):
    assert launch.TIER_SETTINGS["spill_fair_agents"][:3] == ("SPILL_FAIR_AGENTS", True, "flag")
    assert launch.ENV_NAMES["spill_fair_agents"] == "SPILL_FAIR_AGENTS"
    monkeypatch.delenv("SPILL_FAIR_AGENTS", raising=False)
    assert launch.env_settings()["spill_fair_agents"] is True
    monkeypatch.setenv("SPILL_FAIR_AGENTS", "0")
    assert launch.env_settings()["spill_fair_agents"] is False
    assert launch.tier_env_value("spill_fair_agents", False) == "0" and launch.tier_env_value("spill_fair_agents", True) == "1"


def test_default_spec_falls_back_to_a_downloaded_dflash2(monkeypatch):
    # A checkout updated from before DSpine became the default has DFlash2 alone: the server still starts.
    have = set()
    monkeypatch.setattr(launch, "downloaded", lambda pin: pin in have)
    monkeypatch.delenv("SPEC", raising=False)
    assert launch.default_spec() == "dspine"  # (nothing downloaded: the error names DSpine)
    have.add(launch.DFLASH2)
    assert launch.default_spec() == "dflash2"
    have.add(launch.DSPINE)
    assert launch.default_spec() == "dspine"
    monkeypatch.setenv("SPEC", "")  # (compose.yaml passes an empty SPEC when unset)
    assert launch.default_spec() == "dspine"
    have.discard(launch.DSPINE)
    monkeypatch.setenv("SPEC", "dspine")  # (asked for: no fallback)
    assert launch.default_spec() == "dspine"
    assert launch.env_settings()["spec"] == "dspine"
