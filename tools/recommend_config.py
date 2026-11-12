#!/usr/bin/env python3
"""Recommend the memory and KV-cache settings of the `serve` service (compose.yaml) for this machine.

    python3 tools/recommend_config.py                     # 繁體中文 report and a .env block
    python3 tools/recommend_config.py --spill-dir /path   # measure the disk that holds the SSD tier
    python3 tools/recommend_config.py --write .env        # update the managed keys (.env.bak is made first)
    python3 tools/recommend_config.py --json              # machine-readable output on stdout

Stdlib only, Python 3.10+. Runs on the WSL/Linux host or in the dev container
(`docker compose --profile dev run --rm dev-cpu python tools/recommend_config.py`; the serve image does not ship tools/).
Every detection step fails soft: an unknown value is None plus a note, and the recommendation then keeps the
compose.yaml default for the affected keys. The exit code is 0 even when detection failed (1 only if --write fails).

The disk that holds the SSD tier is measured at --spill-dir (default: env SPILL_DIR, else the current directory).
Inside WSL2 the docker named volume `kvcache` lives on the WSL virtual disk, so pass that volume's host-side path
(e.g. /var/lib/docker/volumes/<project>_kvcache/_data) to measure it. The emitted SPILL_DIR is always the container
path /cache/kv (or env SPILL_DIR when set), never the measured host path.
"""

from __future__ import annotations

import argparse
import codecs
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------------------------------------
# Constants. GiB unless the name says otherwise. Sources: docs/architecture.md section 5, docs/tiered-cache.md,
# compose.yaml.
# ---------------------------------------------------------------------------------------------------------------
GiB = 2**30
MiB = 2**20

# VRAM (section 5: the engine budget is the free VRAM minus VRAM_RESERVE_GIB, the default of memory.cc).
VRAM_RESERVE_GIB = 1.5
# Weights + draft + state + activations, derived from the documented 605,504-token pool at the 30.3 GiB budget of a 5090 with
# 18 KiB/token NVFP4 KV.
NON_KV_GIB = 19.9
PARKING_GIB = 0.57  # 3 parking areas for MAX_SEQS = 4
KV_BYTES_PER_TOKEN = 18432  # 18 KiB, NVFP4 KV
POOL_STEP_TOKENS = 65536  # the pool is laid out in steps of this many tokens
POOL_CAP_TOKENS = 524288  # spec default KV_POOL_TOKENS (512K)
FULL_CONTEXT_TOKENS = 262144  # MAX_TOKENS when the pool holds it
POOL_FLOOR_TOKENS = 1024  # never emit 0 for KV_POOL_TOKENS / MAX_TOKENS (0 means "whatever VRAM leaves")
TOO_SMALL_POOL_TOKENS = 65536  # warn that the GPU is too small below this

# RAM (pinned host memory of the engine: embeddings 2.4 + prefix state cache 4 + vision 0.9 + frontend).
ENGINE_HOST_GIB = 8
SYSTEM_RESERVE_GIB = 4
SPILL_RAM_FRACTION = 0.75
SPILL_RAM_STEP_GIB = 4
SPILL_RAM_MAX_GIB = 64

# WSL2: the RAM tier counts against the WSL VM memory, so the VM should be the host RAM minus a system reserve.
WSL_HOST_RESERVE_GIB = 16
WSL_TOLERANCE_GIB = 4

# SSD tier.
SSD_CAP_GIB = 100
SSD_MIN_FREE_GIB = 10
SSD_FRACTION = 0.5
SSD_WARN_GIB = 20

# Fixed recommendations.
MAX_SEQS = 4
KV_DTYPE = "nvfp4"
SPILL_RAM_RETENTION_S = 3600
SPILL_TTL_DAYS = 7
SPILL_TICK_S = 30
SPILL_DIR_DEFAULT = "/cache/kv"

# compose.yaml defaults, used when a value cannot be detected.
DEFAULT_KV_POOL_TOKENS = 524288
DEFAULT_MAX_TOKENS = 262144
DEFAULT_SPILL_RAM_GIB = 32
DEFAULT_SPILL_SSD_GIB = 100

# nvidia-smi memory.used above this means something (usually the engine itself) already holds VRAM.
VRAM_USED_WARN_GIB = 4.0

NVIDIA_SMI_TIMEOUT_S = 10
POWERSHELL_TIMEOUT_S = 10

# Keys this tool manages in .env. SPILL_SSD_MIN_FREE_GIB is deliberately not managed: it is only used in the
# arithmetic, so a user's value is left alone.
MANAGED_KEYS = (
    "KV_POOL_TOKENS",
    "MAX_TOKENS",
    "MAX_SEQS",
    "KV_DTYPE",
    "SPILL_RAM_GIB",
    "SPILL_SSD_GIB",
    "SPILL_DIR",
    "SPILL_RAM_RETENTION_S",
    "SPILL_TTL_DAYS",
    "SPILL_TICK_S",
)


# ---------------------------------------------------------------------------------------------------------------
# Detection. Each function returns None (and appends a note) instead of raising.
# ---------------------------------------------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float) -> str | None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_nvidia_smi(text: str | None) -> dict[str, Any] | None:
    """Parse `name,memory.total,memory.used` (MiB, noheader, nounits); the first line is GPU 0."""
    if not text:
        return None
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) != 3 or not parts[0]:
            return None
        try:
            total_mib = int(parts[1])
            used_mib = int(parts[2])
        except ValueError:  # "[N/A]" and similar
            return None
        return {"name": parts[0], "total_gib": total_mib * MiB / GiB, "used_gib": used_mib * MiB / GiB}
    return None


def detect_gpu(notes: list[str]) -> dict[str, Any] | None:
    query = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]
    gpu = parse_nvidia_smi(_run(query, NVIDIA_SMI_TIMEOUT_S))
    if gpu is None:
        notes.append("GPU 未偵測：nvidia-smi 不存在、失敗或輸出無法解析。VRAM 相關值使用 compose.yaml 預設值。")
    return gpu


def parse_meminfo_gib(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1)) * 1024 / GiB


def detect_ram(notes: list[str], meminfo: Path = Path("/proc/meminfo")) -> float | None:
    try:
        ram = parse_meminfo_gib(meminfo.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        ram = None
    if ram is None:
        notes.append("RAM 未偵測：無法讀取 /proc/meminfo。SPILL_RAM_GIB 使用 compose.yaml 預設值。")
    return ram


def is_wsl(osrelease: Path = Path("/proc/sys/kernel/osrelease")) -> bool:
    try:
        return "microsoft" in osrelease.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


def parse_powershell_bytes_gib(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0)) / GiB


def detect_host_ram(notes: list[str]) -> float | None:
    cmd = ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
    host = parse_powershell_bytes_gib(_run(cmd, POWERSHELL_TIMEOUT_S))
    if host is None:
        notes.append("Windows 主機 RAM 未偵測（powershell.exe 不可用，例如在容器內）。無法檢查 .wslconfig。")
    return host


def _existing_parent(path: Path) -> tuple[Path, bool]:
    """The path itself, or its nearest existing parent (and whether the path exists)."""
    current = path
    exists = current.exists()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current, exists


def detect_disk(path: Path, notes: list[str]) -> float | None:
    probe, exists = _existing_parent(path)
    if not exists:
        notes.append(f"磁碟：{path} 不存在，改為測量其上層 {probe} 的可用空間。")
    try:
        free = shutil.disk_usage(probe).free / GiB
    except OSError:
        notes.append(f"磁碟未偵測：無法讀取 {probe} 的可用空間。SPILL_SSD_GIB 使用 compose.yaml 預設值。")
        return None
    return free


def detect_cache_used(path: Path, measure: bool, notes: list[str]) -> float | None:
    """Size of the existing SSD-tier files at `path`. Only measured when the path was given explicitly."""
    if not measure:
        notes.append("快取目錄大小未計入（未指定 --spill-dir 或 SPILL_DIR，以目前目錄代替會誤算 hf/ 權重）。視為 0。")
        return 0.0
    if not path.is_dir():
        return 0.0
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.stat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    except OSError:
        notes.append(f"快取目錄大小未偵測：無法讀取 {path}。視為 0。")
        return None
    return total / GiB


def detect(spill_dir: str | None = None, measure_cache: bool = True) -> dict[str, Any]:
    """Collect everything the recommendation needs. Unknown values are None; `notes` explains each gap."""
    notes: list[str] = []
    measured = Path(spill_dir) if spill_dir else Path(os.getcwd())

    gpu = detect_gpu(notes)
    ram = detect_ram(notes)
    wsl = is_wsl()
    host = detect_host_ram(notes) if wsl else None
    free = detect_disk(measured, notes)
    cache = detect_cache_used(measured, measure_cache, notes)

    return {
        "gpu_name": gpu["name"] if gpu else None,
        "vram_total_gib": gpu["total_gib"] if gpu else None,
        "vram_used_gib": gpu["used_gib"] if gpu else None,
        "ram_gib": ram,
        "is_wsl": wsl,
        "host_ram_gib": host,
        "disk_path": str(measured),
        "disk_free_gib": free,
        "cache_used_gib": cache,
        "notes": notes,
    }


# ---------------------------------------------------------------------------------------------------------------
# Recommendation. Pure functions of the detected dict.
# ---------------------------------------------------------------------------------------------------------------
def kv_pool_tokens(kv_gib: float) -> int:
    """Tokens of KV pool that fit in kv_gib, in POOL_STEP_TOKENS steps, capped at POOL_CAP_TOKENS."""
    if kv_gib <= 0:
        return 0
    steps = math.floor(kv_gib * GiB / KV_BYTES_PER_TOKEN / POOL_STEP_TOKENS)
    return min(steps * POOL_STEP_TOKENS, POOL_CAP_TOKENS)


def spill_ram_gib(vm_gib: float) -> int:
    """SPILL_RAM_GIB for a VM (or host) of vm_gib: what is left after the engine and system reserve, in steps of 4."""
    usable = (vm_gib - ENGINE_HOST_GIB - SYSTEM_RESERVE_GIB) * SPILL_RAM_FRACTION
    value = math.floor(usable / SPILL_RAM_STEP_GIB) * SPILL_RAM_STEP_GIB
    return int(max(0, min(SPILL_RAM_MAX_GIB, value)))


def recommend(detected: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = list(detected.get("notes") or [])
    warnings: list[str] = []
    hints: list[str] = []
    reasons: dict[str, str] = {}
    env: dict[str, str] = {}

    # --- VRAM: KV_POOL_TOKENS and MAX_TOKENS ---
    total = detected.get("vram_total_gib")
    used = detected.get("vram_used_gib")
    if total is None or used is None:
        env["KV_POOL_TOKENS"] = str(DEFAULT_KV_POOL_TOKENS)
        env["MAX_TOKENS"] = str(DEFAULT_MAX_TOKENS)
        reasons["KV_POOL_TOKENS"] = "GPU 未偵測，使用 compose.yaml 預設值。"
        reasons["MAX_TOKENS"] = "GPU 未偵測，使用 compose.yaml 預設值。"
    else:
        budget = max(0.0, total - used - VRAM_RESERVE_GIB)  # (the pool is still capped at POOL_CAP_TOKENS)
        kv_gib = budget - NON_KV_GIB - PARKING_GIB
        pool = kv_pool_tokens(kv_gib)
        if used > VRAM_USED_WARN_GIB:
            warnings.append(
                f"VRAM 已用 {used:.1f} GiB。若 serve 正在執行，這包含引擎自己的用量；"
                "要以空 GPU 計算，請先停止 serve 再重跑。"
            )
        if pool < TOO_SMALL_POOL_TOKENS:
            warnings.append(
                f"GPU 可用 VRAM 不足（預算 {budget:.1f} GiB，KV 池只能容納 {pool:,} tokens）。"
                "引擎很可能無法啟動；以下數值僅為下限，不可直接使用。"
            )
            pool = max(pool, POOL_FLOOR_TOKENS)
        if pool < FULL_CONTEXT_TOKENS:
            max_tokens = (pool // POOL_FLOOR_TOKENS) * POOL_FLOOR_TOKENS
            warnings.append(
                f"KV 池只有 {pool:,} tokens，低於 {FULL_CONTEXT_TOKENS:,}，因此 MAX_TOKENS 下調為 {max_tokens:,}。"
            )
        else:
            max_tokens = FULL_CONTEXT_TOKENS
        env["KV_POOL_TOKENS"] = str(pool)
        env["MAX_TOKENS"] = str(max_tokens)
        reasons["KV_POOL_TOKENS"] = (
            f"{detected.get('gpu_name') or 'GPU'}：預算 {budget:.2f} GiB"
            + f"（可用 VRAM 扣除保留 {VRAM_RESERVE_GIB} GiB），扣除非 KV {NON_KV_GIB} 與 3 個停放區 {PARKING_GIB} 後可容納 {pool:,} tokens。"
        )
        reasons["MAX_TOKENS"] = (
            f"池大於等於 {FULL_CONTEXT_TOKENS:,} tokens 時維持 {FULL_CONTEXT_TOKENS:,}，否則為池大小（向下取整到 1024）。"
        )

    # --- RAM: SPILL_RAM_GIB ---
    ram = detected.get("ram_gib")
    if ram is None:
        env["SPILL_RAM_GIB"] = str(DEFAULT_SPILL_RAM_GIB)
        reasons["SPILL_RAM_GIB"] = "RAM 未偵測，使用 compose.yaml 預設值。"
    else:
        spill = spill_ram_gib(ram)
        env["SPILL_RAM_GIB"] = str(spill)
        reasons["SPILL_RAM_GIB"] = (
            f"RAM {ram:.1f} GiB 減去引擎 {ENGINE_HOST_GIB} 與系統保留 {SYSTEM_RESERVE_GIB}，"
            f"取 {int(SPILL_RAM_FRACTION * 100)}%，以 {SPILL_RAM_STEP_GIB} GiB 為單位，上限 {SPILL_RAM_MAX_GIB}。"
        )
        if spill == 0:
            warnings.append("RAM 太少，RAM 層為 0（關閉）。")

        host = detected.get("host_ram_gib")
        if detected.get("is_wsl") and host is not None:
            wsl_gib = round(host) - WSL_HOST_RESERVE_GIB
            if ram < wsl_gib - WSL_TOLERANCE_GIB:
                wsl_spill = spill_ram_gib(wsl_gib)
                hints.append(
                    f"WSL VM 只有 {ram:.0f} GiB，建議 {wsl_gib} GiB（主機 {host:.0f} GiB 減去保留 {WSL_HOST_RESERVE_GIB}）。"
                    f"請在 %UserProfile%\\.wslconfig 的 [wsl2] 區段設定 memory={wsl_gib}GB，"
                    "然後在 Windows 執行 wsl --shutdown 重新啟動。"
                    f"設定後 SPILL_RAM_GIB 會是 {wsl_spill}（目前 {spill}）。"
                )

    # --- SSD: SPILL_SSD_GIB ---
    free = detected.get("disk_free_gib")
    cache = detected.get("cache_used_gib") or 0.0
    if free is None:
        env["SPILL_SSD_GIB"] = str(DEFAULT_SPILL_SSD_GIB)
        reasons["SPILL_SSD_GIB"] = "磁碟未偵測，使用 compose.yaml 預設值。"
    else:
        ssd = min(SSD_CAP_GIB, max(0, math.floor((free + cache - SSD_MIN_FREE_GIB) * SSD_FRACTION)))
        env["SPILL_SSD_GIB"] = str(ssd)
        reasons["SPILL_SSD_GIB"] = (
            f"可用 {free:.1f} GiB + 既有快取 {cache:.1f} GiB - 保留 {SSD_MIN_FREE_GIB} 後的一半，上限 {SSD_CAP_GIB}。"
        )
        if ssd < SSD_WARN_GIB:
            warnings.append(f"SSD 層只能 {ssd} GiB（低於 {SSD_WARN_GIB}）。請騰出磁碟空間，或接受較小的快取。")
    spill_dir_path = detected.get("disk_path") or ""
    if spill_dir_path.startswith("/mnt/"):
        warnings.append("測量的路徑在 /mnt/（Windows 磁碟，drvfs 很慢）。SSD 層請使用 WSL 的具名磁碟區。")

    # --- fixed values ---
    env["MAX_SEQS"] = str(MAX_SEQS)
    env["KV_DTYPE"] = KV_DTYPE
    env["SPILL_DIR"] = os.environ.get("SPILL_DIR") or SPILL_DIR_DEFAULT
    env["SPILL_RAM_RETENTION_S"] = str(SPILL_RAM_RETENTION_S)
    env["SPILL_TTL_DAYS"] = str(SPILL_TTL_DAYS)
    env["SPILL_TICK_S"] = str(SPILL_TICK_S)
    reasons["MAX_SEQS"] = "固定值（4 個並行請求，3 個停放區）。"
    reasons["KV_DTYPE"] = "固定值（NVFP4 KV，預設）。"
    reasons["SPILL_DIR"] = "容器內路徑（kvcache 具名磁碟區）；不是主機路徑。"
    reasons["SPILL_RAM_RETENTION_S"] = "固定值（閒置 1 小時後寫入 SSD）。"
    reasons["SPILL_TTL_DAYS"] = "固定值（7 天未重用即刪除）。"
    reasons["SPILL_TICK_S"] = "固定值（每 30 秒維護一次）。"

    ordered = {key: env[key] for key in MANAGED_KEYS}
    return {"env": ordered, "reasons": reasons, "warnings": warnings, "notes": notes, "hints": hints}


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------
def _fmt(value: float | None, unit: str = "GiB") -> str:
    return "未知" if value is None else f"{value:.1f} {unit}"


def render_report(detected: dict[str, Any], rec: dict[str, Any]) -> str:
    lines: list[str] = ["== 偵測結果 =="]
    if detected.get("gpu_name"):
        lines.append(
            f"GPU：{detected['gpu_name']}，總計 {_fmt(detected['vram_total_gib'])}，已用 {_fmt(detected['vram_used_gib'])}"
        )
    else:
        lines.append("GPU：未知")
    ram_line = f"RAM：{_fmt(detected.get('ram_gib'))}"
    if detected.get("is_wsl"):
        ram_line += f"（WSL）；Windows 主機 {_fmt(detected.get('host_ram_gib'))}"
    lines.append(ram_line)
    lines.append(
        f"磁碟：{detected.get('disk_path')} 可用 {_fmt(detected.get('disk_free_gib'))}，"
        f"既有快取 {_fmt(detected.get('cache_used_gib'))}"
    )

    lines += ["", "== 建議值 =="]
    for key, value in rec["env"].items():
        lines.append(f"{key}={value}  # {rec['reasons'].get(key, '')}")

    if rec["warnings"]:
        lines += ["", "== 警告 =="] + [f"- {w}" for w in rec["warnings"]]
    if rec["hints"]:
        lines += ["", "== WSL2 =="] + [f"- {h}" for h in rec["hints"]]
    if rec["notes"]:
        lines += ["", "== 注意 =="] + [f"- {n}" for n in rec["notes"]]

    lines += ["", "== .env 區塊 ==", "```dotenv"]
    lines += [f"{key}={value}" for key, value in rec["env"].items()]
    lines.append("```")
    return "\n".join(lines)


_KEY_LINE = re.compile(r"^[ \t]*((?:export[ \t]+)?)([A-Za-z_][A-Za-z0-9_]*)[ \t]*=")


def merge_env_text(text: str, env: dict[str, str]) -> str:
    """Replace the live `KEY=` lines of the managed keys; keep every other line; append the missing keys.

    A line written as `export KEY=value` counts as KEY; the replacement keeps its `export ` prefix.
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        match = _KEY_LINE.match(line)
        if match and match.group(2) in env:
            key = match.group(2)
            eol = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"{match.group(1)}{key}={env[key]}{eol}")
            seen.add(key)
        else:
            out.append(line)
    missing = [key for key in env if key not in seen]
    if missing:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append("# --- tools/recommend_config.py ---\n")
        out += [f"{key}={env[key]}\n" for key in missing]
    return "".join(out)


def write_env(path: Path, env: dict[str, str]) -> str:
    """Write the managed keys into `path` (created if missing; a .bak copy is made first when it exists).

    The existing file is read BOM-safe (utf-8-sig). A UTF-8 BOM is preserved if the file had one; otherwise none is added.
    """
    existed = path.exists()
    had_bom = False
    if existed:
        raw = path.read_bytes()
        had_bom = raw.startswith(codecs.BOM_UTF8)
        original = raw.decode("utf-8-sig")
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    else:
        original = ""
    updated = merge_env_text(original, env)
    data = updated.encode("utf-8")
    path.write_bytes(codecs.BOM_UTF8 + data if had_bom else data)
    return f"已寫入 {path}" + (f"（備份：{path.name}.bak）" if existed else "（新建）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spill-dir", help="path whose disk holds the SSD tier (default: env SPILL_DIR, else cwd)")
    parser.add_argument("--write", metavar="PATH", help="update or create this .env file (managed keys only)")
    parser.add_argument("--json", action="store_true", help="print JSON on stdout")
    args = parser.parse_args(argv)

    measured_spill = args.spill_dir or os.environ.get("SPILL_DIR")
    detected = detect(spill_dir=measured_spill, measure_cache=bool(measured_spill))
    rec = recommend(detected)

    if args.json:
        print(json.dumps({"detected": detected, "recommendation": rec}, ensure_ascii=False, indent=2))
    else:
        print(render_report(detected, rec))

    if args.write:
        try:
            message = write_env(Path(args.write), rec["env"])
        except (OSError, UnicodeError) as exc:
            print(f"寫入失敗：{exc}", file=sys.stderr)
            return 1
        print(message, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
