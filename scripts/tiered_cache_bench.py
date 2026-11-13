"""Tiered KV cache timings (SPILL_RAM_GIB, SPILL_DIR; docs/tiered-cache.md): what the engine calls cost at 32K, 128K
and 256K tokens per checkpoint.

For each size, on a fresh engine (KV_POOL_TOKENS = 1.25 x size, so the next conversation evicts the checkpoint):

  evict_spill   the prefill of B evicts A's checkpoint, copying it to host RAM (reported: prefill time minus the
                same prefill with nothing to evict, an estimate; GB/s of the copy)
  promote_ram   prefix_begin of A restores the checkpoint from RAM (the restore is inside prefix_begin)
  backup_idle   spill_backup_one until False: copies VRAM checkpoints to RAM and keeps them in VRAM
  evict_backup  the prefill of C evicts A, whose copy is already in RAM (copies nothing; not timed)
  export_ssd    spill_export of the RAM entry to a file in SPILL_DIR
  -- a second engine (the restart): --
  import_ssd    spill_import of that file
  promote_ssd   prefix_begin of A restores it from the file

Engine creation and the prefills of the checkpoints are not timed. Every timed call is bracketed by
time.perf_counter(). Engines get the server's draft model (--spec, default as the server's: dspine, else dflash2, else none). The environment is passed through unchanged, so a flag the engine does not know is ignored
and the run still works. Compare variants in one command:

    python scripts/tiered_cache_bench.py --compare "SPILL_PINNED_GIB=0" "SPILL_PINNED_GIB=8"
    python scripts/tiered_cache_bench.py --compare "SPILL_PINNED_GIB=0 SPILL_ASYNC_IO=0" \\
        "SPILL_PINNED_GIB=8 SPILL_ASYNC_IO=1" "SPILL_PINNED_GIB=8 SPILL_PINNED_SLAB_MIB=256 SPILL_ASYNC_IO=1"
    (pinned entries are segmented: one entry may span several slabs; SPILL_ASYNC_IO only overlaps the CPU for pinned entries)
    docker compose --profile dev run --rm dev python scripts/tiered_cache_bench.py --spill-dir /cache/kv/bench

Each size builds two engines and prefills 4 conversations of that size, so a run takes several minutes.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "scripts"))
import native_env  # noqa: E402 - before q27lib and qwen27b (HF_HOME, Q27_LIB; see native_env.py)
DEFAULT_WEIGHTS = native_env.apply()

OPS = ("evict_spill", "promote_ram", "backup_idle", "evict_backup", "export_ssd", "import_ssd", "promote_ssd")
SUFFIX = ".q27m"  # a manifest in SPILL_DIR/entries; its chunk files go to SPILL_DIR/chunks (the engine decides)


def parse_env(text):
    """'A=1 B=2' -> {'A': '1', 'B': '2'}; an empty string is no change."""
    out = {}
    for tok in text.split():
        if "=" not in tok:
            raise SystemExit(f"--compare: '{tok}' is not NAME=VALUE")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def gbps(nbytes, ms):
    """GB/s of nbytes moved in ms (None when either is missing or zero)."""
    if not nbytes or not ms or ms <= 0:
        return None
    return nbytes / (ms / 1000.0) / 1e9


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return (time.perf_counter() - t0) * 1000.0, r


def row(size, op, ms, nbytes, note=""):
    return {"size": size, "op": op, "ms": None if ms is None else round(ms, 2), "bytes": int(nbytes or 0),
            "gbps": None if gbps(nbytes, ms) is None else round(gbps(nbytes, ms), 3), "note": note}


def ram_entry(eng, pos):
    """The RAM entry of a checkpoint with `pos` tokens (or None)."""
    for e in eng.spill_list():
        if e["pos"] == pos and e["ram_bytes"] > 0:
            return e
    return None


def ram_total(eng):
    return sum(e["ram_bytes"] for e in eng.spill_list())


def set_env(size):
    os.environ["STATE_HOST_CACHE"] = os.environ.get("STATE_HOST_CACHE", "1")
    os.environ["SPILL_RAM_GIB"] = os.environ.get("SPILL_RAM_GIB", "40")
    os.environ["KV_POOL_TOKENS"] = str(size + size // 4 + 4096)
    os.environ["CHECKPOINT_INTERVAL"] = str(-(-2 * size // 64) * 64)  # (only the explicit checkpoints; a multiple of 64)
    os.environ["VRAM_BACKUP_IDLE_S"] = "0"


def grow_pinned(eng):
    """SPILL_PINNED_GIB > 0: pins the slabs now (the frontend does this while idle), so the entries get pinned memory."""
    if float(os.environ.get("SPILL_PINNED_GIB", "0") or 0) > 0 and hasattr(eng, "spill_grow_pinned"):
        while eng.spill_grow_pinned(1):
            pass


def run_size(size, docs, spill_dir, weights, kv_dtype, rows, info, spec):
    from q27lib import Engine
    from qwen27b.paths import engine_target
    from tiered_cache_test import draft_path, make_prompt

    draft = draft_path(spec)  # (the server's draft is loaded in its engine: the same VRAM budget and pool)

    A, B, C = (make_prompt(docs, size, i) for i in (0, 1, 2))
    promote_ids = A + [5]  # the checkpoint of A is a prefix of this

    set_env(size)
    t0 = time.perf_counter()
    eng = Engine(engine_target(weights), max_tokens=size + 8192, kv_dtype=kv_dtype, draft=draft)
    grow_pinned(eng)
    info.append({"size": size, "engine_create_s": round(time.perf_counter() - t0, 2)})
    try:
        # the checkpoint of A (its prefill is not timed)
        eng.prefix_begin(A)
        eng.forward_sample(A, 0.0, 1, 1.0)
        eng.checkpoint()

        # evict_spill: B's prefill evicts A, copying it to RAM; control: the same prefill with nothing to evict
        def b_prefill():
            eng.prefix_begin(B)
            return eng.forward_sample(B, 0.0, 1, 1.0)
        ms_evict, _ = timed(b_prefill)
        e = ram_entry(eng, size)
        ms_ctrl, _ = timed(lambda: (eng.reset(), eng.forward_sample(B, 0.0, 1, 1.0)))
        if e is None:
            rows.append(row(size, "evict_spill", None, 0, "no copy in RAM: the pool did not evict the checkpoint"))
        else:
            rows.append(row(size, "evict_spill", ms_evict - ms_ctrl, e["ram_bytes"],
                            f"prefill {ms_evict:.0f} ms, control {ms_ctrl:.0f} ms"))

        # promote_ram: prefix_begin restores A from RAM
        e = ram_entry(eng, size)
        nbytes = e["ram_bytes"] if e else 0
        ms, cached = timed(lambda: eng.prefix_begin(promote_ids))
        rows.append(row(size, "promote_ram", ms, nbytes, "" if cached == size else f"restored {cached} tokens, not {size}"))

        # backup_idle: copies while idle (A stays in VRAM)
        before = ram_total(eng)
        copies = [0]

        def backup_all():
            while eng.spill_backup_one():
                copies[0] += 1
        ms, _ = timed(backup_all)
        rows.append(row(size, "backup_idle", ms, ram_total(eng) - before, f"{copies[0]} copies"))

        # evict_backup: C's prefill evicts A, whose copy is already in RAM (not timed: no copy happens)
        eng.prefix_begin(C)
        eng.forward_sample(C, 0.0, 1, 1.0)

        # export_ssd: the RAM entry of A to a file
        e = ram_entry(eng, size)
        if e is None:
            rows.append(row(size, "export_ssd", None, 0, "no RAM entry to export"))
            return
        os.makedirs(os.path.join(spill_dir, "entries"), exist_ok=True)
        path = os.path.join(spill_dir, "entries", uuid.uuid4().hex + SUFFIX)
        ms, _ = timed(lambda: eng.spill_export(e["id"], path))
        usage = eng.spill_disk_usage()  # (the files this entry took: its manifest and its chunk files; an older engine: the one file)
        fbytes = usage["physical_bytes"] if usage else os.path.getsize(path)
        rows.append(row(size, "export_ssd", ms, e["ram_bytes"], f"files {fbytes / 1e9:.2f} GB"))
    finally:
        eng.close()

    # the restart: a second engine imports the file and restores from it
    eng = Engine(engine_target(weights), max_tokens=size + 8192, kv_dtype=kv_dtype, draft=draft)
    grow_pinned(eng)
    try:
        ms, eid = timed(lambda: eng.spill_import(path))
        rows.append(row(size, "import_ssd", ms, fbytes, "" if eid else "import returned 0 (not this model / KV type)"))
        ms, cached = timed(lambda: eng.prefix_begin(promote_ids))
        rows.append(row(size, "promote_ssd", ms, fbytes, "" if cached == size else f"restored {cached} tokens, not {size}"))
    finally:
        eng.close()


def run_once(args):
    """All sizes in this process; returns {"env": ..., "rows": [...], "info": [...]}."""
    from tiered_cache_test import load_docs
    docs = load_docs(1)
    own = args.spill_dir is None
    spill_dir = args.spill_dir or tempfile.mkdtemp(prefix="tiered_cache_bench_")
    os.makedirs(spill_dir, exist_ok=True)
    rows, info = [], []
    try:
        for size in args.sizes:
            for sub in ("entries", "chunks"):
                for f in pathlib.Path(spill_dir, sub).glob("*"):
                    f.unlink()
            run_size(size, docs, spill_dir, args.weights, args.kv_dtype, rows, info, args.spec)
    finally:
        if own:
            shutil.rmtree(spill_dir, ignore_errors=True)
    env = {k: os.environ[k] for k in sorted(os.environ) if k.startswith(("SPILL_", "STATE_", "KV_", "CHECKPOINT_", "VRAM_"))}
    return {"env": env, "rows": rows, "info": info}


def print_table(title, cols):
    """cols: [(label, result)]; one row per (size, op), the ms and GB/s of each column."""
    print(f"\n== {title} ==")
    head = f"{'size':>8} {'op':<12}" + "".join(f" | {lab:>24}" for lab, _ in cols)
    print(head)
    print("-" * len(head))
    keys = []
    for _, res in cols:
        for r in res["rows"]:
            k = (r["size"], r["op"])
            if k not in keys:
                keys.append(k)
    for size, op in keys:
        cells = []
        for _, res in cols:
            r = next((x for x in res["rows"] if (x["size"], x["op"]) == (size, op)), None)
            if r is None or r["ms"] is None:
                cells.append(f"{'-':>24}")
            else:
                g = f"{r['gbps']:.2f} GB/s" if r["gbps"] is not None else "-"
                cells.append(f"{r['ms']:>10.1f} ms {g:>11}")
        print(f"{size:>8} {op:<12}" + "".join(f" | {c:>24}" for c in cells))
    for lab, res in cols:
        for r in res["rows"]:
            if r["note"]:
                print(f"  [{lab}] {r['size']} {r['op']}: {r['note']}")


def main():
    from tiered_cache_test import SPEC_CHOICES, default_spec

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="weight profile: nvidia | gdnfp4 | quasar | a model directory (default: %(default)s)")
    ap.add_argument("--kv-dtype", default="nvfp4", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--spec", choices=SPEC_CHOICES,
                    help="draft model of the engines (default: the server's: $SPEC, else dspine if downloaded, else dflash2, else none)")
    ap.add_argument("--sizes", default="32768,131072,262144", help="checkpoint sizes in tokens, comma-separated")
    ap.add_argument("--spill-dir", help="where the SSD files go (default: a temp dir; use the kvcache volume to measure it)")
    ap.add_argument("--json", help="also write the results here")
    ap.add_argument("--compare", nargs="+", metavar="ENV",
                    help='run once per variant, each a list of NAME=VALUE (e.g. "SPILL_PINNED_GIB=0" "SPILL_PINNED_GIB=8")')
    # internal: one variant in a child process
    ap.add_argument("--child-json", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.spec = args.spec or default_spec()
    print(f"draft model: {args.spec}", flush=True)
    args.sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    if args.compare:
        results = []
        tmp = tempfile.mkdtemp(prefix="tiered_bench_cmp_")
        try:
            for i, variant in enumerate(args.compare):
                env = dict(os.environ)
                env.update(parse_env(variant))
                out = os.path.join(tmp, f"v{i}.json")
                cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child-json", out,
                       "--sizes", ",".join(map(str, args.sizes)), "--weights", args.weights,
                       "--kv-dtype", args.kv_dtype, "--spec", args.spec]
                if args.spill_dir:
                    cmd += ["--spill-dir", args.spill_dir]
                print(f"\n# variant {i + 1}: {variant or '(no change)'}", flush=True)
                rc = subprocess.run(cmd, env=env).returncode
                if rc != 0:
                    print(f"variant {variant!r} exited {rc}", flush=True)
                    results.append((variant or "(no change)", {"env": {}, "rows": [], "info": [], "error": rc}))
                    continue
                results.append((variant or "(no change)", json.load(open(out))))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print_table("comparison", results)
        if args.json:
            json.dump({"variants": [{"name": n, **r} for n, r in results]}, open(args.json, "w"), indent=1)
        return

    res = run_once(args)
    if args.child_json:
        json.dump(res, open(args.child_json, "w"))
        return
    print(f"env: {res['env'] or '(defaults)'}")
    for i in res["info"]:
        print(f"  size {i['size']}: engine creation {i['engine_create_s']} s (not timed)")
    print_table("tiered cache", [("this run", res)])
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
