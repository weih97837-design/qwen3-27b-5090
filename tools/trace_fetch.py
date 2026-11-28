#!/usr/bin/env python3
"""Download the request trace of a server (GET /admin/trace, docs/request-trace.md) into a local directory, incrementally.

    python tools/trace_fetch.py http://HOST:8080 --out trace-copy [--since EPOCH_SECONDS] [--follow SECONDS]

Each trace file is copied to --out under its own name. A state file (.trace_fetch.json in --out) keeps the offset reached per file, so
a second run (or --follow, which polls every SECONDS until Ctrl-C) fetches only the new bytes (?from=<offset>), and only complete
lines are ever written. A file the server rotated away stays in --out (its state entry is dropped); a file that was replaced (the
server's copy is shorter than our offset) is fetched again from the start. Then read the copies: python tools/cache_sim.py DIR/*.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trace_io  # noqa: E402

STATE = ".trace_fetch.json"
CHUNK = 1 << 20


def load_state(out):
    try:
        s = json.loads((out / STATE).read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in s.get("files", {}).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def save_state(out, files):
    tmp = out / (STATE + ".tmp")
    tmp.write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    os.replace(tmp, out / STATE)


def listing(base):
    with trace_io.request(base.rstrip("/") + "/admin/trace") as r:
        try:
            info = json.loads(trace_io.body(r).read())
        except ValueError as e:
            raise trace_io.TraceSourceError(f"{base}: not a trace listing ({e})") from None
    if not info.get("enabled"):
        raise trace_io.TraceSourceError(f"{base}: the request trace is off on that server (REQUEST_TRACE)")
    return info


def fetch_file(base, name, path, offset, log):
    """Append the bytes of `name` from `offset` to `path`; returns the new offset (after the last complete line written)."""
    from urllib.parse import quote
    url = f"{base.rstrip('/')}/admin/trace/{quote(name)}?from={offset}"
    with trace_io.request(url) as r:
        end = offset
        try:
            with open(path, "r+b" if path.exists() else "wb") as f:
                f.seek(offset)
                f.truncate()
                src = trace_io.body(r)
                while True:
                    b = src.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
                    i = b.rfind(b"\n")
                    if i >= 0:
                        end = f.tell() - len(b) + i + 1
                f.truncate(end)  # (only complete lines: a cut transfer keeps what ended in a newline)
                f.flush()
        except (EOFError, OSError) as e:
            log(f"{name}: transfer cut at {end} ({type(e).__name__}: {e})")
            raise
    return end


def fetch_once(base, out, since=None, log=print):
    """One pass over the server's files. Returns the number of bytes written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = load_state(out)
    info = listing(base)
    listed = {f["name"]: f for f in info.get("files", [])}
    got = 0
    for name, meta in listed.items():
        if since is not None and meta.get("last_ts") is not None and meta["last_ts"] < since:
            continue
        path = out / name
        have = path.stat().st_size if path.exists() else 0
        off = min(state.get(name, 0), have)  # (the state ahead of the file: a crash between write and state; the file ahead: trim)
        if off > meta.get("complete_bytes", meta["size"]):
            off = 0  # (the server's file is shorter: it is another file with that name now)
        if off == meta.get("complete_bytes", meta["size"]) and have == off:
            state[name] = off
            continue
        try:
            new = fetch_file(base, name, path, off, log)
        except trace_io.TraceSourceError as e:
            log(f"{name}: skipped ({e})")  # (rotated away between the listing and the download)
            continue
        except (EOFError, OSError):
            state[name] = off if not path.exists() else min(path.stat().st_size, off)
            save_state(out, state)
            raise
        got += new - off
        state[name] = new
        save_state(out, state)
    for gone in [n for n in state if n not in listed]:
        del state[gone]  # (rotated away on the server: the local copy stays)
    save_state(out, state)
    return got


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_url", help="http://HOST:8080 (the server's port)")
    ap.add_argument("--out", required=True, help="local directory for the copies")
    ap.add_argument("--since", type=float, help="skip files whose last event is older than this epoch time")
    ap.add_argument("--follow", type=float, metavar="SECONDS", help="keep polling every SECONDS until Ctrl-C")
    a = ap.parse_args(argv)
    try:
        while True:
            try:
                n = fetch_once(a.base_url, a.out, a.since)
                print(f"{time.strftime('%H:%M:%S')} fetched {n:,} bytes", flush=True)
            except (trace_io.TraceSourceError, EOFError, OSError) as e:
                if not a.follow:
                    sys.exit(f"trace_fetch: {e}")
                print(f"trace_fetch: {e} (retrying)", file=sys.stderr, flush=True)
            if not a.follow:
                return 0
            time.sleep(a.follow)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
