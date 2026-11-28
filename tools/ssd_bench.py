#!/usr/bin/env python3
"""SSD read-speed benchmark for the SSD tier's directory (SPILL_DIR, docs/tiered-cache.md "SSD 讀取速度與估計"). Stdlib only.

    python tools/ssd_bench.py --dir D:\\kvcache [--gib 4] [--threads 1,2,4,8] [--block-mib 32]

It writes --gib of random data into q27_ssd_bench_*.bin files in --dir (fsync'd), reads them back once for a sequential pass
(one thread, one file after another) and once per thread count (the files split across that many threads, each thread with its
own file handle, reads of --block-mib into one preallocated buffer), and prints the wall-clock GB/s of each. The files are
always deleted at the end, also on an error or Ctrl-C.

Caveats, printed as well:
  * A cached read is not a disk read. The OS cache keeps what was just written. On Linux the files are dropped from the cache
    with posix_fadvise(DONTNEED) before each pass; on Windows this script cannot drop the cache. Use a --gib larger than the
    free RAM for a fair number there; the first pass after writing may still be served from the cache.
  * The number is the file read alone: no host -> device copy, no decoding. It is the upper bound of what a restore can load.
"""

import argparse
import os
import sys
import threading
import time
import uuid

PREFIX = "q27_ssd_bench_"


def parse_threads(text):
    try:
        out = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("--threads takes a list of numbers, e.g. 1,2,4,8")
    if not out or any(t < 1 or t > 64 for t in out):
        raise argparse.ArgumentTypeError("--threads: each count must be 1..64")
    return out


def write_files(paths, size, block):
    """Random data, fsync'd: the files are on the device before anything is read."""
    for p in paths:
        with open(p, "wb") as f:
            left = size
            while left > 0:
                n = min(block, left)
                f.write(os.urandom(n))
                left -= n
            f.flush()
            os.fsync(f.fileno())


def drop_cache(paths):
    """Linux only: asks the kernel to drop the files' pages (a best effort; a no-op elsewhere)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            return False
    return True


def read_files(paths, block):
    """Reads each file front to back with its own unbuffered handle (io.FileIO.readinto: a raw read that releases the GIL)
    into one preallocated buffer. Returns the bytes read; raises if a file is shorter than it was written."""
    buf = bytearray(block)
    view = memoryview(buf)
    total = 0
    for p in paths:
        with open(p, "rb", buffering=0) as f:
            got = 0
            while True:
                n = f.readinto(view)
                if not n:
                    break
                got += n
        if got != os.path.getsize(p):
            raise RuntimeError(f"{p}: read {got} bytes, expected {os.path.getsize(p)}")
        total += got
    return total


def timed_pass(groups, block):
    """Runs read_files for each group on its own thread; returns (bytes, seconds) of the whole pass."""
    results, errors = [], []

    def work(files):
        try:
            results.append(read_files(files, block))
        except Exception as e:  # noqa: BLE001 - reported below, from the main thread
            errors.append(e)

    threads = [threading.Thread(target=work, args=(g,)) for g in groups]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    if errors:
        raise errors[0]
    return sum(results), dt


def main(argv=None):
    ap = argparse.ArgumentParser(description="SSD 讀取速度測試（q27 SSD 分層快取的目錄）。")
    ap.add_argument("--dir", required=True, help="目錄（SPILL_DIR 所在的磁碟）")
    ap.add_argument("--gib", type=float, default=4.0, help="寫入與讀取的總量（GiB，預設 4；應大於可用記憶體才公平）")
    ap.add_argument("--threads", type=parse_threads, default=[1, 2, 4, 8], help="要測的執行緒數，例如 1,2,4,8")
    ap.add_argument("--block-mib", type=int, default=32, help="每次讀取的區塊大小（MiB，預設 32，與引擎相同）")
    args = ap.parse_args(argv)
    if not args.gib > 0:
        ap.error("--gib must be positive")
    if args.block_mib < 1:
        ap.error("--block-mib must be at least 1")
    if not os.path.isdir(args.dir):
        ap.error(f"--dir is not a directory: {args.dir}")

    block = args.block_mib << 20
    n_files = max(args.threads)  # each thread gets its own files: at least as many files as the largest thread count
    per_file = max(1, int(args.gib * (1 << 30)) // n_files)
    tag = uuid.uuid4().hex[:8]
    paths = [os.path.join(args.dir, f"{PREFIX}{tag}_{i}.bin") for i in range(n_files)]
    try:
        print("注意：快取會讓讀取數字偏高。", flush=True)
        if os.name == "nt":
            print("Windows：本工具無法丟棄 OS 快取。寫入後的第一次讀取可能來自快取；請讓 --gib 大於可用 RAM。", flush=True)
        else:
            print("Linux：每次讀取前以 posix_fadvise(DONTNEED) 丟棄這些檔案的快取（盡力而為）。", flush=True)
        print(f"寫入 {n_files} 個檔案，共 {per_file * n_files / (1 << 30):.2f} GiB 到 {args.dir} ...", flush=True)
        write_files(paths, per_file, block)

        print("\n順序讀取（單一執行緒，檔案一個接一個）：", flush=True)
        drop_cache(paths)
        nbytes, dt = timed_pass([paths], block)
        print(f"  1 執行緒順序讀取: {nbytes / dt / 1e9:.2f} GB/s（{dt:.2f} 秒）", flush=True)

        print("\n多執行緒讀取（檔案平分給各執行緒，各自開啟檔案）：", flush=True)
        for t in args.threads:
            groups = [paths[i::t] for i in range(t)]  # thread k reads files k, k+t, ...
            drop_cache(paths)
            nbytes, dt = timed_pass(groups, block)
            print(f"  {t:>2} 執行緒: {nbytes / dt / 1e9:.2f} GB/s（{dt:.2f} 秒）", flush=True)
        print("\n數字為牆鐘時間的平均；只包含檔案讀取，不含 GPU 複製。", flush=True)
        return 0
    finally:
        for p in paths:  # (always: also after an error or Ctrl-C)
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"警告：無法刪除 {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
