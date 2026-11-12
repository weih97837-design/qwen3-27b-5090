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
DEFAULTS = {"weights": "gdnfp4", "kv_dtype": "nvfp4", "spec": "dflash2", "vision": "host", "max_seqs": 8,
            "max_tokens": 262144, "vram_reserve_gib": None}
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
    os.environ.setdefault("STATE_HOST_CACHE", "4")
    out = lc.plans(lc.settings, DEVICE)
    p = out["plan"]
    # The serve default's startup log on this device: kv pool 9461 pages (605504 tokens), device 28.377 GiB
    # (28.378 until the own GEMMs replaced CUTLASS and its 7.8 MiB workspace, then 28.370 until the penalty
    # counts of the 8 sequences, 7.6 MiB).
    assert (p["max_tokens"], p["kv_tokens"], p["fits"]) == (262144, 605504, True)
    assert abs(p["device"] / GiB - 28.3775) < 0.001 and abs(p["budget"] / GiB - 28.70) < 0.01
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
    assert os.environ["SPILL_RAM_GIB"] == "32"
    assert os.environ["SPILL_DIR"] == ("/cache/kv" if os.name != "nt" else launch.spill_dir_default())
    assert os.environ["MAX_TOKENS_FIT"] == "1" and os.environ["STATE_HOST_CACHE"] == "4"


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
