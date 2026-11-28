"""Tests for tools/ssd_bench.py (run from the repo root: python -m pytest tools/tests)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import ssd_bench  # noqa: E402


def leftovers(d):
    return [p for p in os.listdir(d) if p.startswith("q27_ssd_bench_")]


def test_bench_prints_results_and_leaves_no_files(tmp_path):
    proc = subprocess.run([sys.executable, str(TOOLS / "ssd_bench.py"), "--dir", str(tmp_path), "--gib", "0.02",
                           "--threads", "1,2", "--block-mib", "1"], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "順序讀取" in out and "GB/s" in out
    assert "1 執行緒順序讀取" in out and " 1 執行緒:" in out and " 2 執行緒:" in out
    assert "快取" in out  # the cache caveat is always printed
    assert leftovers(tmp_path) == []


def test_files_are_deleted_when_a_pass_fails(tmp_path, monkeypatch, capsys):
    def broken(*_args, **_kw):
        raise OSError("disk gone")

    monkeypatch.setattr(ssd_bench, "read_files", broken)
    with pytest.raises(OSError):
        ssd_bench.main(["--dir", str(tmp_path), "--gib", "0.01", "--threads", "2", "--block-mib", "1"])
    assert leftovers(tmp_path) == []


def test_bad_arguments_exit_before_anything_is_written(tmp_path):
    with pytest.raises(SystemExit):
        ssd_bench.main(["--dir", str(tmp_path), "--gib", "0"])
    with pytest.raises(SystemExit):
        ssd_bench.main(["--dir", str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        ssd_bench.main(["--dir", str(tmp_path), "--threads", "0"])
    assert leftovers(tmp_path) == []


def test_read_files_checks_the_size(tmp_path):
    p = tmp_path / "q27_ssd_bench_x_0.bin"
    p.write_bytes(b"abc" * 1000)
    assert ssd_bench.read_files([str(p)], 512) == 3000
