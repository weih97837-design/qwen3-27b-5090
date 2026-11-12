"""Tests for tools/recommend_config.py (run from the repo root: python -m pytest tools/tests)."""

import codecs
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import recommend_config as rc  # noqa: E402


def detected(**overrides):
    base = {
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "vram_total_gib": 32607 / 1024,
        "vram_used_gib": 500 / 1024,
        "ram_gib": 48.0,
        "is_wsl": True,
        "host_ram_gib": 96.0,
        "disk_path": "/tmp/x",
        "disk_free_gib": 500.0,
        "cache_used_gib": 0.0,
        "notes": [],
    }
    base.update(overrides)
    return base


def test_5090_wsl_vm_48_gib_recommends_full_pool_and_wslconfig_hint():
    rec = rc.recommend(detected())
    assert rec["env"]["KV_POOL_TOKENS"] == "524288"
    assert rec["env"]["MAX_TOKENS"] == "262144"
    assert rec["env"]["SPILL_RAM_GIB"] == "24"  # (48 - 8 - 4) * 0.75 = 27 -> 24
    assert len(rec["hints"]) == 1
    assert "memory=80GB" in rec["hints"][0]  # floor(96) - 16
    assert "SPILL_RAM_GIB 會是 48" in rec["hints"][0]  # spill at 80 GiB
    assert rec["env"]["MAX_SEQS"] == "4"
    assert rec["env"]["KV_DTYPE"] == "nvfp4"
    assert rec["env"]["SPILL_RAM_RETENTION_S"] == "3600"
    assert rec["env"]["SPILL_TTL_DAYS"] == "7"
    assert rec["env"]["SPILL_TICK_S"] == "30"


def test_96_gib_gpu_pool_capped_but_budget_is_real():
    rec = rc.recommend(detected(vram_total_gib=96.0, vram_used_gib=0.5))
    assert rec["env"]["KV_POOL_TOKENS"] == "524288"  # still capped at POOL_CAP_TOKENS
    assert rec["env"]["MAX_TOKENS"] == "262144"
    assert "預算 94.90 GiB" in rec["reasons"]["KV_POOL_TOKENS"]  # 96 - 0.5 - 0.6, no 30.3 cap
    assert "上限" not in rec["reasons"]["KV_POOL_TOKENS"]


def test_wsl_hint_rounds_host_ram_before_reserve():
    rec = rc.recommend(detected(host_ram_gib=95.8))
    assert len(rec["hints"]) == 1
    assert "memory=80GB" in rec["hints"][0]  # round(95.8) - 16 = 80 (floor would give 79)


def test_vm_80_gib_gives_spill_48_and_no_hint():
    rec = rc.recommend(detected(ram_gib=80.0))
    assert rec["env"]["SPILL_RAM_GIB"] == "48"  # (80 - 12) * 0.75 = 51 -> 48
    assert rec["hints"] == []  # 80 is within 4 GiB of the 80 GiB target


def test_spill_ram_is_clamped():
    assert rc.spill_ram_gib(200) == 64
    assert rc.spill_ram_gib(8) == 0
    assert rc.spill_ram_gib(4) == 0


def test_24_gib_gpu_lowers_pool_and_max_tokens_with_warning():
    rec = rc.recommend(detected(vram_total_gib=24.0, vram_used_gib=0.0))
    assert rec["env"]["KV_POOL_TOKENS"] == "131072"  # 23.4 - 19.9 - 0.57 = 2.93 GiB -> 2 steps
    assert rec["env"]["MAX_TOKENS"] == "131072"
    assert any("262,144" in w for w in rec["warnings"])


def test_gpu_too_small_warns_and_never_emits_zero():
    rec = rc.recommend(detected(vram_total_gib=16.0, vram_used_gib=0.0))
    assert rec["env"]["KV_POOL_TOKENS"] != "0"
    assert rec["env"]["MAX_TOKENS"] != "0"
    assert any("GPU 可用 VRAM 不足" in w for w in rec["warnings"])


def test_unknown_gpu_ram_disk_fall_back_to_compose_defaults_without_crashing():
    d = detected(
        vram_total_gib=None,
        vram_used_gib=None,
        ram_gib=None,
        is_wsl=False,
        host_ram_gib=None,
        disk_free_gib=None,
        cache_used_gib=None,
        notes=["GPU 未偵測"],
    )
    rec = rc.recommend(d)
    assert rec["env"]["KV_POOL_TOKENS"] == "524288"
    assert rec["env"]["MAX_TOKENS"] == "262144"
    assert rec["env"]["SPILL_RAM_GIB"] == "32"
    assert rec["env"]["SPILL_SSD_GIB"] == "100"
    assert rec["hints"] == []
    assert "GPU 未偵測" in rec["notes"]
    assert rec["reasons"]["SPILL_RAM_GIB"].startswith("RAM 未偵測")
    assert "未知" in rc.render_report(d, rec)


def test_detect_survives_missing_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    monkeypatch.delenv("SPILL_DIR", raising=False)
    d = rc.detect(spill_dir=str(tmp_path), measure_cache=True)
    assert d["gpu_name"] is None
    assert d["host_ram_gib"] is None
    assert d["disk_free_gib"] and d["disk_free_gib"] > 0
    assert rc.recommend(d)["env"]["MAX_TOKENS"]


def test_parsers_reject_garbage():
    assert rc.parse_nvidia_smi("NVIDIA GeForce RTX 5090, [N/A], [N/A]") is None
    assert rc.parse_nvidia_smi("") is None
    assert rc.parse_nvidia_smi("RTX, 32607, 500\n")["total_gib"] == pytest.approx(32607 / 1024)
    assert rc.parse_meminfo_gib("MemTotal:       50331648 kB\n") == pytest.approx(48.0)
    assert rc.parse_meminfo_gib("nothing") is None
    assert rc.parse_powershell_bytes_gib("103079215104\r\n") == pytest.approx(96.0)
    assert rc.parse_powershell_bytes_gib("") is None


def test_write_keeps_unrelated_lines_and_comments_and_makes_bak(tmp_path):
    env_file = tmp_path / ".env"
    original = (
        "# my settings\n"
        "HOST=0.0.0.0\n"
        "KV_POOL_TOKENS=0   # old\n"
        "#MAX_TOKENS=1\n"
        "\n"
        "SPILL_RAM_GIB = 16\n"
        "PORT=8080\n"
    )
    env_file.write_text(original, encoding="utf-8", newline="")
    rec = rc.recommend(detected())
    message = rc.write_env(env_file, rec["env"])
    assert "備份" in message

    bak = tmp_path / ".env.bak"
    assert bak.read_text(encoding="utf-8") == original

    text = env_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# my settings"
    assert "HOST=0.0.0.0" in lines
    assert "PORT=8080" in lines
    assert "#MAX_TOKENS=1" in lines
    assert "KV_POOL_TOKENS=524288" in lines  # replaced; the old inline comment is dropped with the line
    assert "SPILL_RAM_GIB=24" in lines
    assert "MAX_TOKENS=262144" in lines  # missing key appended
    assert "SPILL_SSD_MIN_FREE_GIB=10" not in text  # not a managed key


def test_write_preserves_crlf(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"# keep\r\nPORT=8080\r\nMAX_SEQS=8\r\n")
    rc.write_env(env_file, {"MAX_SEQS": "4"})
    assert env_file.read_bytes() == b"# keep\r\nPORT=8080\r\nMAX_SEQS=4\r\n"


def test_write_creates_missing_file_without_bak(tmp_path):
    env_file = tmp_path / ".env"
    message = rc.write_env(env_file, {"MAX_SEQS": "4"})
    assert "新建" in message
    assert env_file.read_text(encoding="utf-8").endswith("MAX_SEQS=4\n")
    assert not (tmp_path / ".env.bak").exists()


def test_write_existing_empty_file_says_backup_not_new(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"")
    message = rc.write_env(env_file, {"MAX_SEQS": "4"})
    assert "備份" in message
    assert "新建" not in message
    assert (tmp_path / ".env.bak").exists()
    assert env_file.read_bytes().endswith(b"MAX_SEQS=4\n")


def test_export_prefixed_lines_are_recognised_and_keep_prefix(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"export MAX_SEQS=8\nexport  KV_DTYPE = fp8\nexport=1\n")
    rc.write_env(env_file, {"MAX_SEQS": "4", "KV_DTYPE": "nvfp4"})
    assert env_file.read_bytes() == b"export MAX_SEQS=4\nexport  KV_DTYPE=nvfp4\nexport=1\n"


def test_bom_file_is_read_and_first_line_replaced_bom_preserved(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(codecs.BOM_UTF8 + b"MAX_SEQS=8\nPORT=8080\n")
    rc.write_env(env_file, {"MAX_SEQS": "4"})
    data = env_file.read_bytes()
    assert data == codecs.BOM_UTF8 + b"MAX_SEQS=4\nPORT=8080\n"


def test_no_bom_is_added_when_original_had_none(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"PORT=8080\n")
    rc.write_env(env_file, {"MAX_SEQS": "4"})
    assert not env_file.read_bytes().startswith(codecs.BOM_UTF8)


def test_main_runs_without_gpu_and_exits_zero(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    monkeypatch.delenv("SPILL_DIR", raising=False)
    assert rc.main(["--spill-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "== .env 區塊 ==" in out
    assert "KV_POOL_TOKENS=524288" in out


def test_main_json_output_is_json_only(monkeypatch, capsys, tmp_path):
    import json

    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    monkeypatch.delenv("SPILL_DIR", raising=False)
    assert rc.main(["--json", "--spill-dir", str(tmp_path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["recommendation"]["env"]["MAX_SEQS"] == "4"
