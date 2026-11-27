"""Request trace recorder (docs/request-trace.md): data for a trace-driven simulator of cache eviction policies.

Off unless REQUEST_TRACE is set (a directory or a file name prefix; `--request-trace` in server.py sets it). It appends one JSON
line per event from the scheduler loop (worker.py): "hdr" at the start of each file, "req" when a request begins, "done" when it
ends, "snap" every SNAP_S while busy. It writes counts and hashes only: no prompt or output text, no token ids. A prompt is
described by chained hashes of its 256-token blocks (block_chain), so two prompts with the same prefix have the same first hashes.

Never raises into the scheduler: any error disables the recorder with one log line.
"""

import array
import hashlib
import json
import os
import pathlib
import time

BLOCK = 256            # tokens per hashed block (full blocks only)
SEED = b"q27-request-trace-v1"
SNAP_S = 10.0          # s between "snap" events while requests are in flight
FILE_MIB = 64          # a new file at this size
MAX_MIB = 512          # REQUEST_TRACE_MAX_MIB default: the oldest files go beyond this total
ENGINE_VERSION = "q27-phase6"  # (what GET /version reports)


def package_version():
    """The release tag of the Windows package (app\\VERSION, written by package.ps1 next to frontend/), or None in a checkout:
    the header names the build so traces of two versions can be told apart."""
    try:
        return (pathlib.Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="ascii").strip() or None
    except (OSError, ValueError):
        return None


SETTING_NAMES = ("STATE_HOST_CACHE", "CHECKPOINT_INTERVAL", "PREFILL_CHUNK", "MAX_SEQS", "KV_POOL_TOKENS", "KV_DTYPE", "MAX_TOKENS",
                 "SPEC", "WEIGHTS")


def block_chain(ids, prev=SEED, data=None):
    """Chained hashes of the full BLOCK-token blocks of ids (a sequence of ints, or int32 bytes in `data`): h_k is the first 8
    bytes (16 hex digits) of blake2b(d_{k-1} + the block's int32 ids as bytes), d_k its whole digest, d_{-1} = prev. Returns
    (list of h_k, d of the last block). A longer sequence with the same first k blocks has the same first k hashes."""
    if data is None:
        data = array.array("i", ids).tobytes()
    step = 4 * BLOCK
    out = []
    for i in range(0, len(data) - step + 1, step):
        prev = hashlib.blake2b(prev + data[i:i + step]).digest()
        out.append(prev[:8].hex())
    return out, prev


def hash_agent(agent):
    """A stable short hash of each "/" part of an agent id (REQUEST_TRACE_HASH_AGENTS=1)."""
    return "/".join(hashlib.blake2b(p.encode("utf-8", "replace"), digest_size=5).hexdigest() for p in str(agent).split("/"))


class Recorder:
    def __init__(self, target, header, hash_agents=False, max_mib=MAX_MIB, file_mib=FILE_MIB, log=print, snap_s=SNAP_S):
        t = str(target)
        if os.path.isdir(t) or t.endswith(("/", "\\")):
            t = os.path.join(t, "trace")
        self.prefix = t
        self.header, self.hash_agents, self.log, self.snap_s = header, hash_agents, log, snap_s
        self.max_bytes, self.file_bytes = int(max_mib * 1048576), int(file_mib * 1048576)
        self.f, self.size, self.path, self.dead = None, 0, None, False
        self.stem, self.k = None, 0
        self.last_snap = 0.0
        self._open()

    # -- files
    def _files(self):
        d = os.path.dirname(self.prefix) or "."
        base = os.path.basename(self.prefix) + "-"
        try:
            names = sorted(n for n in os.listdir(d) if n.startswith(base) and n.endswith(".jsonl"))
        except OSError:
            return []
        return [os.path.join(d, n) for n in names]

    def _open(self):
        d = os.path.dirname(self.prefix) or "."
        os.makedirs(d, exist_ok=True)
        stem = f"{self.prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
        k = self.k + 1 if stem == self.stem else 0
        path = stem + (f"_{k}" if k else "") + ".jsonl"
        while os.path.exists(path):  # (two files in one second: the name stays unique and sorts after)
            k += 1
            path = f"{stem}_{k}.jsonl"
        self.stem, self.k = stem, k
        if self.f is not None:
            self.f.close()
        self.f, self.path, self.size = open(path, "a", encoding="utf-8"), path, 0
        self._write({"t": "hdr", "version": 1, "block_tokens": BLOCK, "ts": time.time(), **self.header})
        files = self._files()  # oldest first (the names sort by time)
        total = 0
        for p in files:
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
        for p in files:
            if total <= self.max_bytes or p == path:
                break
            try:
                n = os.path.getsize(p)
                os.remove(p)
                total -= n
            except OSError:
                pass

    def _write(self, rec):
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        self.f.write(line)
        self.f.flush()
        self.size += len(line.encode("utf-8"))

    def emit(self, rec):
        if self.dead:
            return
        try:
            if self.size >= self.file_bytes:
                self._open()
            self._write(rec)
        except Exception as e:  # noqa: BLE001 - tracing must never hurt serving
            self.disable(e)

    def disable(self, e):
        if self.dead:
            return
        self.dead = True
        try:
            self.log(f"request trace: {type(e).__name__}: {e}; recording is off until the next start")
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.f is not None:
                self.f.close()
        except Exception:  # noqa: BLE001
            pass

    # -- events (the scheduler calls these; each is safe to call blindly)
    def agent_name(self, s):
        a = s.agent_id or s.agent
        return hash_agent(a) if self.hash_agents else a

    def request(self, s, restored_from=None):
        if self.dead:
            return
        try:
            from stats import media_of
            blocks, _ = block_chain(None, data=s.ids_bytes())  # (eng_ids: what prefix_begin keys on)
            self.emit({"t": "req", "ts": time.time(), "rid": s.rid, "agent": self.agent_name(s), "client": s.client,
                       "prompt_tokens": len(s.prompt), "media_tokens": (media_of(s.images) or {}).get("tokens", 0),
                       "max_tokens": s.max_new, "cached": s.cached, "reusable": s.reusable, "diverged_at": s.diverged_at,
                       "restored_from": restored_from, "queued_s": round(s.begun - s.arrival, 4), "blocks": blocks})
        except Exception as e:  # noqa: BLE001
            self.disable(e)

    def done(self, s, reason):
        if self.dead:
            return
        try:
            end = []
            if s.started:
                allh, _ = block_chain(s.prompt + s.out)
                end = allh[len(s.prompt) // BLOCK:]
            self.emit({"t": "done", "ts": time.time(), "rid": s.rid, "reason": reason, "output_tokens": len(s.out),
                       "ttft_s": round(s.first_token - s.arrival, 4) if s.first_token else None,
                       "prefill_s": round(s.prefill_s, 4), "computed_tokens": s.computed, "preempted": s.preempted,
                       "total_tokens": len(s.prompt) + len(s.out), "end_blocks": end})
        except Exception as e:  # noqa: BLE001
            self.disable(e)

    def snap(self, sched, now):
        """A "snap" every snap_s while requests are in flight (the scheduler's report calls this; a time check when not due)."""
        if self.dead or now - self.last_snap < self.snap_s or not (sched.alive or sched.waiting):
            return
        self.last_snap = now
        try:
            st = sched.engine.cache_stats()
            rec = {"t": "snap", "ts": now, "running": len(sched.alive), "waiting": len(sched.waiting),
                   "kv_free_pages": st["free_pages"], "kv_checkpoint_only_pages": st["checkpoint_only_pages"],
                   "kv_pool_pages": st["pool_pages"], "checkpoints": st["checkpoints"],
                   "q27:prefix_cache_evictions_total": st["evicted"]}
            if sched.tiers is not None:
                sp = sched.engine.spill_stats()
                for k in ("spilled", "restored", "dropped", "written"):
                    if k in sp:
                        rec[f"q27:tier_{k}"] = sp[k]
            self.emit(rec)
        except Exception as e:  # noqa: BLE001
            self.disable(e)


def from_env(env, header_settings, log=print):
    """A Recorder when REQUEST_TRACE is set, else None. header_settings: extra settings for the header line."""
    target = (env.get("REQUEST_TRACE") or "").strip()
    if not target or target == "0":
        return None
    try:
        settings = {k: v for k, v in sorted(env.items()) if k.startswith("SPILL_") and v is not None}
        settings.update({k: env.get(k) for k in SETTING_NAMES if env.get(k) is not None})
        settings.update(header_settings or {})
        try:
            max_mib = float(env.get("REQUEST_TRACE_MAX_MIB") or MAX_MIB)
        except ValueError:
            max_mib = MAX_MIB
        rec = Recorder(target, {"engine": ENGINE_VERSION, "build": package_version(), "settings": settings},
                       hash_agents=(env.get("REQUEST_TRACE_HASH_AGENTS") or "") not in ("", "0"), max_mib=max_mib, log=log)
        log(f"request trace: recording to {rec.path} (counts and hashes only; docs/request-trace.md)")
        return rec
    except Exception as e:  # noqa: BLE001 - an unwritable path: serve without it
        log(f"request trace: cannot start ({e})")
        return None
