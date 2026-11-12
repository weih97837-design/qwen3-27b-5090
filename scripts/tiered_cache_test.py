"""Tiered KV cache exactness (SPILL_RAM_GIB, SPILL_DIR; docs/tiered-cache.md): a conversation resumed from a
checkpoint that was spilled to host RAM, to a file, or to a file an earlier process wrote must produce exactly the
greedy tokens it produces without any cache (docs/architecture.md §9).

Each scenario runs its requests with the tiered cache and as a reference with the cache off (STATE_HOST_CACHE=0,
the same requests: prefill, then the generation token by token, as prefix_cache_test.py does), and compares the
output tokens. Every scenario runs in fresh processes (the environment is read when the engine is created) and
checks the spill counters to show that the intended path was taken.

  a  RAM promote      conversation A's checkpoint is evicted to RAM by conversation B's prefill (KV_POOL_TOKENS is
                      small); A's follow-up turn restores it from RAM (spilled, then restored, grow).
  b  SSD promote      as a, then every RAM entry is written to SPILL_DIR (engine.spill_export, as the frontend's
                      policy does): A's follow-up reads the file.
  c  Restart import   process 1 writes the spilled entries to SPILL_DIR and exits; process 2 spill_imports them and
                      resumes A's follow-up from disk.
  d  Idle backup      spill_backup_one copies A's checkpoint to RAM while it stays in VRAM; B's eviction then copies
                      nothing (spilled unchanged); A's follow-up restores the backup.
  e  Speculative      as a, with the draft model (ring checkpoints). Skipped when no draft model is available.
  f  Pinned + async   as a (then the SSD round trip of b), with SPILL_PINNED_GIB=2, SPILL_PINNED_SLAB_MIB=16 (every entry
                      is several pinned segments) and SPILL_ASYNC_IO=1; the tokens must be identical and the entries
                      must have been given pinned memory (pin_hits).
  g  Async promote    SPILL_ASYNC_PROMOTE=1 with the pinned pool and async I/O, driving the engine API directly: as a, then as
                      b (C's checkpoint through the SSD tier). Before the follow-up: spill_match finds the entry, spill_restore_begin
                      starts it (1 from RAM, 2 from the file), the entry is still matched and a prefix_begin at that moment
                      does NOT see it (a loading checkpoint is invisible), spill_restore_poll is called until it reports the
                      id as restored, another request runs before the owner (the restored checkpoint is held), and the follow-up's
                      prefix_begin must report the restored length; the tokens must be identical.

Each scenario runs a few conversations (the golden prompts, prompt-len tokens each, different starts) and one
follow-up turn per conversation that continues the generation with 300 new tokens. The driver sizes the conversations
to the engine's pool, including its block rounding (choose_prompt_len): the prompt length is raised until two
conversations exceed the usable pool, so B's prefill evicts A's checkpoint. It prints the numbers at the start.

    STATE_HOST_CACHE=1 docker compose --profile dev run --rm dev python scripts/tiered_cache_test.py
    python scripts/tiered_cache_test.py --scenarios a,c --keep-dir /cache/kv/test

Native Windows package (no Docker, no torch): test.cmd runs this with the package's Python; the weights are where
start.cmd downloads them (settings.ini HF_HOME, else the package's huggingface folder), q27.dll is app\\engine\\build\\.
Without a draft model (SPEC=none, or DFlash2 not downloaded) scenario e is skipped. The default --weights is WEIGHTS
(settings.ini), else gdnfp4, in the package; nvidia elsewhere.
"""

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine/python"))
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # (the package's Python does not add this folder)
import native_env  # noqa: E402 - before q27lib and qwen27b: they read HF_HOME and Q27_LIB when imported
DEFAULT_WEIGHTS = native_env.apply()

SCENARIOS = {
    "a": ("RAM promote", ["full"]),
    "b": ("SSD promote", ["full"]),
    "c": ("restart import", ["write", "read"]),
    "d": ("idle backup reuse", ["full"]),
    "e": ("speculative decoding (ring checkpoints)", ["full"]),
    "f": ("pinned segments + async I/O", ["full"]),
    "g": ("async promote (delay hits): RAM and SSD", ["full"]),
}
# Extra engine variables of a scenario's tiered run (the driver clears every SPILL_* variable first).
SCEN_ENV = {"f": {"SPILL_PINNED_GIB": "2", "SPILL_PINNED_SLAB_MIB": "16", "SPILL_ASYNC_IO": "1"},
            "g": {"SPILL_PINNED_GIB": "2", "SPILL_PINNED_SLAB_MIB": "16", "SPILL_ASYNC_IO": "1", "SPILL_ASYNC_PROMOTE": "1"}}
SUFFIX = ".q27kv"  # the name frontend/tiercache.py indexes at start()
FOLLOW = 300  # new tokens of each follow-up turn
KEYS = ("entries", "in_ram", "on_disk", "ram_bytes", "spilled", "restored", "written", "backed_up", "pinned_bytes",
        "pin_hits", "pin_misses", "restoring", "restored_async", "restore_failed")
# Variables the tiered cache reads at engine creation: the driver sets them per run, never inherits them.
CACHE_VARS = {"STATE_HOST_CACHE", "KV_POOL_TOKENS", "CHECKPOINT_INTERVAL", "MAX_SEQS", "VRAM_BACKUP_IDLE_S"}


# ---- pure helpers (no GPU, no engine import)

def load_docs(min_len):
    """The golden prompts of at least min_len tokens (bench/golden/prompts.jsonl)."""
    docs = [json.loads(line)["ids"] for line in open(ROOT / "bench/golden/prompts.jsonl", encoding="utf-8")]
    docs = [d for d in docs if len(d) >= min_len]
    if not docs:
        raise SystemExit(f"bench/golden/prompts.jsonl has no prompt of {min_len} tokens or more")
    return docs


def make_prompt(docs, length, idx):
    """Prompt idx: a golden document rotated by idx * 997 tokens (so the starts differ), tiled if it is short."""
    base = docs[idx % len(docs)]
    rot = (idx * 997) % len(base)
    seq = base[rot:] + base[:rot]
    while len(seq) < length:
        seq = seq + seq
    return seq[:length]


def sizes(prompt_len, tokens):
    """(KV_POOL_TOKENS, max_tokens) for the scenarios: the pool holds one conversation and its checkpoint, so the
    next conversation's prefill evicts it."""
    pool = prompt_len + 2 * tokens + 1024
    return pool, prompt_len + 2 * tokens + FOLLOW + 1024


# The engine's pool (engine/src/model.cc, the sizing after max_pages_): the pages of a fixed pool are rounded up to
# whole KV blocks, and one page is kept back. The driver mirrors it so that it can size the conversations to it.
PAGE_TOKENS = 64  # kPageTokens
KV_HEADS = 4  # kKvHeads
SPEC_BLOCK = 8  # kSpecBlock: a verify block writes up to 8 positions past the committed length
KV_ROW_BYTES = {"nvfp4": 144, "fp8": 256, "bf16": 512}  # kv_row_bytes(kv_type)
EVICT_MARGIN = 4  # pages: the two conversations must exceed the usable pool by this much, so B's prefill evicts A
AHEAD_PAGES = 0  # (ensure_pages_of evicts only for pages a sequence NEEDS; the 8 pages it maps ahead are taken only if free)


def ceil_div(a, b):
    return -(-a // b)


def kv_block(kv_dtype):
    """Pages per KV block: ceil(2 MiB / page bytes), or Q27_KV_BLOCK from the environment (as the engine reads it)."""
    try:
        v = int(os.environ.get("Q27_KV_BLOCK", "") or 0)
    except ValueError:
        v = 0
    if v > 0:
        return v
    page_bytes = PAGE_TOKENS * KV_HEADS * KV_ROW_BYTES[kv_dtype]
    return ceil_div(2 << 20, page_bytes)


def pool_numbers(max_tokens, pool_tokens, kv_dtype):
    """The engine's page counts for max_tokens and a fixed KV_POOL_TOKENS (no parking areas: MAX_SEQS unset)."""
    max_pages = ceil_div(max_tokens + SPEC_BLOCK, PAGE_TOKENS)
    pool_pages = max(max_pages, ceil_div(pool_tokens, PAGE_TOKENS))
    block = kv_block(kv_dtype)
    kv_pages = ceil_div(pool_pages + 1, block) * block
    return {"max_pages": max_pages, "pool_pages": pool_pages, "kv_block": block, "kv_pages": kv_pages,
            "usable": kv_pages - 1}


def usable_pool_pages(max_tokens, pool_tokens, kv_dtype):
    """Pages the engine can hand out of a fixed pool of pool_tokens for max_tokens (see pool_numbers)."""
    return pool_numbers(max_tokens, pool_tokens, kv_dtype)["usable"]


def conv_pages(prompt_len, tokens):
    """Pages of one conversation at its checkpoint: its prompt and first generation."""
    return ceil_div(prompt_len + tokens, PAGE_TOKENS)


def fits_eviction(prompt_len, tokens, usable):
    """True when two conversations (plus AHEAD_PAGES) exceed the usable pool by EVICT_MARGIN pages."""
    return 2 * conv_pages(prompt_len, tokens) + AHEAD_PAGES >= usable + EVICT_MARGIN


def choose_prompt_len(prompt_len, tokens, kv_dtype, limit=100000):
    """The smallest P >= prompt_len for which fits_eviction holds with sizes(P, tokens) and the engine's pool."""
    for p in range(prompt_len, prompt_len + limit):
        pool, max_tokens = sizes(p, tokens)
        if fits_eviction(p, tokens, usable_pool_pages(max_tokens, pool, kv_dtype)):
            return p
    raise SystemExit(f"no prompt length up to {prompt_len + limit} tokens evicts a checkpoint with this pool")


def describe_pool(prompt_len, tokens, kv_dtype, pool, max_tokens):
    n = pool_numbers(max_tokens, pool, kv_dtype)
    conv = conv_pages(prompt_len, tokens)
    return (f"prompt-len {prompt_len}, max_tokens {max_tokens}, KV_POOL_TOKENS {pool} ({kv_dtype}): "
            f"max_pages {n['max_pages']}, pool_pages {n['pool_pages']}, kv_block {n['kv_block']}, "
            f"kv_pages {n['kv_pages']}, usable pages {n['usable']}; pages per conversation "
            f"ceil(({prompt_len}+{tokens})/{PAGE_TOKENS}) = {conv}, two conversations + {AHEAD_PAGES} = "
            f"{2 * conv + AHEAD_PAGES} (eviction margin: needs >= {n['usable'] + EVICT_MARGIN})")


def say(*args, **kw):
    """print, flushed: the progress is read while a worker runs, and survives a crash of the worker."""
    kw.setdefault("flush", True)
    print(*args, **kw)


def fmt_rc(rc):
    """An exit code in decimal and hex (a Windows crash is 0xC0000005 or 0xFFFFFFFF)."""
    return f"{rc} (0x{rc & 0xFFFFFFFF:08X})"


def delta(before, after, key):
    return after[key] - before[key]


def stats_line(st):
    return " ".join(f"{k}={st[k]}" for k in KEYS)


# ---- the run: requests on one engine (cache on or the reference)

class Run:
    def __init__(self, eng, cache, spec, n, docs, prompt_len, phase, log):
        self.eng, self.cache, self.spec, self.n = eng, cache, spec, n
        self.docs, self.L, self.phase, self.log = docs, prompt_len, phase, log
        self.ids, self.outs, self.fed = {}, {}, {}
        self.checks, self.stats = [], []

    def prompt(self, idx):
        return make_prompt(self.docs, self.L, idx)

    def check(self, name, ok, detail=""):
        if self.cache:  # (the reference has nothing to prove)
            self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
            self.log(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail != "" else ""))

    def eviction_check(self, name, ok, detail=""):
        """A check that needs an eviction: a failure also reports the engine's pool counters, so that a pool that is
        too large (no eviction) is recognised at once."""
        if self.cache and not ok:
            cs = self.eng.cache_stats()
            hint = "no eviction happened: pool too large" if cs["evicted"] == 0 else "evictions happened"
            detail = f"{detail}; {hint}: cache_stats evicted={cs['evicted']} pool_pages={cs['pool_pages']} free_pages={cs['free_pages']}"
        self.check(name, ok, detail)

    def op(self, label, fn, *args, **kw):
        """An engine call that moves cache data or ends a phase: logged before and after, so a crash shows the last step."""
        self.log(f"  > {label}")
        ret = fn(*args, **kw)
        self.log(f"  < {label} -> {ret}")
        return ret

    def evicted(self):
        return self.eng.cache_stats()["evicted"] if self.cache else 0

    def snap(self, label):
        if not self.cache:
            return dict.fromkeys(KEYS, 0)
        st = self.eng.spill_stats()
        self.stats.append({"label": label, **st})
        return st

    def _gen(self, rest):
        """Greedy tokens: the first from the prompt rest, then one per step (speculative steps with spec)."""
        self.log(f"  > generate {len(rest)} prompt tokens, {self.n} new")
        out = self._gen_tokens(rest)
        self.log(f"  < generate -> {len(out)} tokens, pos {self.eng.pos}")
        return out

    def _gen_tokens(self, rest):
        eng, n = self.eng, self.n
        if self.spec:
            out = [eng.spec_begin(rest, 0.0, 1, 1.0, seed=0, max_new=n)]
            while len(out) < n:
                out += eng.spec_step()
            return out[:n]
        out = [eng.forward_sample(rest, 0.0, 1, 1.0)]
        while len(out) < n:
            out.append(eng.forward_sample([out[-1]], 0.0, 1, 1.0))
        return out

    def turn1(self, key, ids):
        if self.cache:
            cached = self.op("prefix_begin", self.eng.prefix_begin, ids, spec=self.spec)
            self.check(f"{key}: a new conversation is not cached", cached == 0, f"cached {cached}")
        else:
            self.eng.reset()
            cached = 0
        out = self._gen(ids[cached:])
        if self.cache:
            self.op("checkpoint", self.eng.checkpoint)  # the checkpoint at the end of the generation
        self.ids[key], self.outs[key], self.fed[key] = ids, out, self.eng.pos
        self.log(f"  {key}: {len(out)} tokens, pos {self.eng.pos}")
        return out

    def turn2(self, key, follow, pre=None):
        """The conversation continues: prompt + generation + follow. With the cache it resumes at the end of the
        generation; the reference replays the first turn (prefill, then the generation) and runs the rest."""
        p2 = self.ids[key] + self.outs[key] + follow
        if self.cache:
            if pre:
                pre(p2)  # (scenario g: the background restore of the checkpoint this turn resumes from)
            cached = self.op("prefix_begin", self.eng.prefix_begin, p2, spec=self.spec)
            self.check(f"{key}+: the follow-up resumes at the end of the first generation",
                       cached == self.fed[key], f"cached {cached}, end {self.fed[key]}")
        else:
            self.eng.reset()
            replay = self._gen(self.ids[key])
            if replay != self.outs[key] or self.eng.pos != self.fed[key]:
                raise RuntimeError(f"reference replay of {key} differs from its first turn (not deterministic)")
            cached = self.eng.pos
        out = self._gen(p2[cached:])
        self.outs[key + "+"] = out
        self.log(f"  {key}+: {len(out)} tokens, resumed at {cached}")
        return out

    def save_prior(self, path):
        json.dump({"outs": self.outs, "fed": self.fed}, open(path, "w"))

    def load_prior(self, path):
        data = json.load(open(path))
        self.outs.update(data["outs"])
        self.fed.update({k: int(v) for k, v in data["fed"].items()})
        for key, idx in (("A", 0), ("B", 1)):
            self.ids[key] = self.prompt(idx)


def export_ram(eng, spill_dir):
    """Writes every entry in host RAM to spill_dir (what the frontend's policy does) and returns how many."""
    n = 0
    for e in eng.spill_list():
        if e["ram_bytes"] > 0 and not e["on_disk"]:
            path = os.path.join(spill_dir, uuid.uuid4().hex + SUFFIX)
            say(f"  > spill_export #{n + 1} {e['id']} -> {os.path.basename(path)}")
            say(f"  < spill_export #{n + 1} -> {eng.spill_export(e['id'], path)}")
            n += 1
    return n


def import_all(eng, spill_dir):
    ids = []
    for p in sorted(pathlib.Path(spill_dir).glob("*" + SUFFIX)):
        say(f"  > spill_import {p.name}")
        ids.append(eng.spill_import(str(p)))
        say(f"  < spill_import {p.name} -> {ids[-1]}")
    return ids


# ---- scenarios

def scen_a(R, spill_dir):
    R.turn1("A", R.prompt(0))
    s0 = R.snap("A's checkpoint in VRAM")
    R.turn1("B", R.prompt(1))  # B's prefill evicts A's checkpoint
    s1 = R.snap("after B's prefill (evicts A)")
    R.eviction_check("B's prefill spilled A's checkpoint to RAM", delta(s0, s1, "spilled") > 0,
                     f"spilled +{delta(s0, s1, 'spilled')}")
    R.turn2("A", R.prompt(9)[:FOLLOW])
    s2 = R.snap("after A's follow-up")
    R.check("A's follow-up restored its checkpoint from RAM", delta(s1, s2, "restored") > 0,
            f"restored +{delta(s1, s2, 'restored')}")
    R.turn1("C", R.prompt(2))  # a third conversation, then its follow-up
    R.turn2("C", R.prompt(8)[:FOLLOW])


def scen_b(R, spill_dir):
    R.turn1("A", R.prompt(0))
    R.turn1("B", R.prompt(1))
    s1 = R.snap("after B's prefill (evicts A)")
    R.eviction_check("B's prefill spilled A's checkpoint to RAM", s1["spilled"] > 0, f"spilled {s1['spilled']}")
    n = export_ram(R.eng, spill_dir) if R.cache else 0
    s2 = R.snap("RAM entries written to SPILL_DIR")
    R.check("RAM entries written to SPILL_DIR", n > 0 and s2["on_disk"] > 0, f"{n} files, on_disk {s2['on_disk']}")
    R.turn2("A", R.prompt(9)[:FOLLOW])
    s3 = R.snap("after A's follow-up (read from the file)")
    R.check("A's follow-up restored its checkpoint from the file", delta(s2, s3, "restored") > 0,
            f"restored +{delta(s2, s3, 'restored')}")


def scen_c(R, spill_dir):
    if R.phase in ("full", "write"):
        R.turn1("A", R.prompt(0))
        R.turn1("B", R.prompt(1))
        s1 = R.snap("write: after B's prefill (evicts A)")
        R.eviction_check("B's prefill spilled A's checkpoint to RAM", s1["spilled"] > 0, f"spilled {s1['spilled']}")
        if R.phase == "write":
            n = export_ram(R.eng, spill_dir) if R.cache else 0
            R.snap("write: RAM entries written to SPILL_DIR")
            R.check("RAM entries written before the process exits", n > 0, f"{n} files")
            return
    else:  # read: a new process; its engine has no history
        ids = import_all(R.eng, spill_dir)
        s = R.snap("read: after spill_import")
        R.check("the new engine imports the files of the earlier process", len(ids) > 0 and all(ids) and s["on_disk"] > 0,
                f"{len(ids)} files, on_disk {s['on_disk']}")
    s2 = R.snap("before A's follow-up")
    R.turn2("A", R.prompt(9)[:FOLLOW])
    s3 = R.snap("after A's follow-up (from the import)")
    R.check("A's follow-up restored its checkpoint from disk", delta(s2, s3, "restored") > 0,
            f"restored +{delta(s2, s3, 'restored')}")


def scen_d(R, spill_dir):
    R.turn1("A", R.prompt(0))
    s0 = R.snap("A's checkpoint in VRAM")
    copies = 0
    while R.cache and copies < 10000:
        if not R.op(f"spill_backup_one #{copies + 1}", R.eng.spill_backup_one):
            break
        copies += 1
    s1 = R.snap("after the idle backup")
    R.check("the idle backup copied A's checkpoint to RAM (VRAM kept)",
            copies > 0 and delta(s0, s1, "backed_up") > 0, f"{copies} copies")
    ev0 = R.evicted()
    R.turn1("B", R.prompt(1))  # B's prefill evicts A's checkpoint, whose copy is already in RAM
    s2 = R.snap("after B's prefill (evicts A)")
    R.eviction_check("B's prefill evicted A's checkpoint", R.evicted() - ev0 > 0, f"evicted +{R.evicted() - ev0}")
    R.eviction_check("the eviction copied nothing (spilled unchanged)", delta(s1, s2, "spilled") == 0,
                     f"spilled +{delta(s1, s2, 'spilled')}")
    R.turn2("A", R.prompt(9)[:FOLLOW])
    s3 = R.snap("after A's follow-up")
    R.eviction_check("A's follow-up restored from the backup", delta(s2, s3, "restored") > 0,
                     f"restored +{delta(s2, s3, 'restored')}")


def scen_f(R, spill_dir):
    scen_a(R, spill_dir)
    st = R.snap("pinned pool")
    R.check("the entries were given pinned memory", st["pin_hits"] > 0 and st["pinned_bytes"] > 0,
            f"pin_hits {st['pin_hits']}, pin_misses {st['pin_misses']}, pinned {st['pinned_bytes'] >> 20} MiB")
    scen_b(R, spill_dir)  # (the same conversations again, now through the SSD tier: segments written / read in order)


RESTORE_WAIT_S = 120.0  # scenario g: how long a background restore may take before it is a failure


def async_restore(R, key, p2, on_disk):
    """Scenario g: restores key's spilled checkpoint through the delay-hit API, checking each step. The follow-up's own
    prefix_begin (turn2) then has to resume at the end of the first generation, as in scenarios a and b."""
    eng = R.eng
    m = eng.spill_match(p2, R.spec)
    R.check(f"{key}: spill_match finds the spilled checkpoint ({'file' if on_disk else 'RAM'})",
            m is not None and m["pos"] == R.fed[key] and m["on_disk"] == on_disk, m)
    if m is None:
        return
    before = eng.spill_stats()
    # A finished sequence (its end-of-generation checkpoint saved) still holds its pages until the next prefix_begin /
    # reset, and the restore counts only free pages and pages only checkpoints hold. The server releases it the same
    # way before an async restore (Scheduler.release_finished_cur).
    R.op("reset", eng.reset)
    began = R.op("spill_restore_begin", eng.spill_restore_begin, m["id"])
    want = 2 if on_disk else 1
    R.check(f"{key}: spill_restore_begin returns {want}", began == want, began)
    if not began:
        return
    again = eng.spill_match(p2, R.spec)
    R.check(f"{key}: the restoring entry is still matched, so a second request waits for it too",
            again is not None and again["id"] == m["id"], again)
    cached0 = R.op("prefix_begin (loading checkpoint)", eng.prefix_begin, p2, spec=R.spec)  # (not a lookup of the loading checkpoint: invisible until the poll says so)
    R.check(f"{key}: a loading checkpoint is invisible to prefix_begin", cached0 < R.fed[key],
            f"cached {cached0} of {R.fed[key]}")
    t0, done, polls = time.monotonic(), [], 0
    R.log(f"  > spill_restore_poll until {m['id']} is restored")
    while m["id"] not in dict(done) and time.monotonic() - t0 < RESTORE_WAIT_S:
        done += eng.spill_restore_poll()
        polls += 1
        time.sleep(0.002)
    dt = time.monotonic() - t0
    R.log(f"  < spill_restore_poll -> {dict(done).get(m['id'])} after {polls} polls, {dt * 1000:.0f} ms")
    R.check(f"{key}: spill_restore_poll reports the entry as restored", dict(done).get(m["id"]) is True,
            f"{dt * 1000:.0f} ms")
    # The restored checkpoint is held for its request: other requests that run before it (a new conversation, whose
    # prefill takes pages and a state slot) take other checkpoints first. The follow-up's prefix_begin (turn2) must
    # still resume at the end of the first generation.
    other = R.prompt(4)[:1024]
    c = R.op("prefix_begin (another request)", eng.prefix_begin, other, spec=R.spec)
    R._gen(other[c:])
    R.op("checkpoint", eng.checkpoint)
    R.log(f"  {key}: another request ran before the owner (held checkpoint must survive it)")
    after = eng.spill_stats()
    R.check(f"{key}: one async restore finished, none failed, none left",
            delta(before, after, "restored_async") == 1 and delta(before, after, "restore_failed") == 0
            and after["restoring"] == 0, f"{m['bytes'] / 1e9:.2f} GB, {m['bytes'] / max(dt, 1e-9) / 1e9:.1f} GB/s")


def scen_g(R, spill_dir):
    R.turn1("A", R.prompt(0))
    s0 = R.snap("A's checkpoint in VRAM")
    R.turn1("B", R.prompt(1))  # B's prefill evicts A's checkpoint to RAM
    s1 = R.snap("after B's prefill (evicts A)")
    R.eviction_check("B's prefill spilled A's checkpoint to RAM", delta(s0, s1, "spilled") > 0,
                     f"spilled +{delta(s0, s1, 'spilled')}")
    R.turn2("A", R.prompt(9)[:FOLLOW], pre=lambda p2: async_restore(R, "A", p2, False))
    st = R.snap("pinned pool")
    R.check("the entries were given pinned memory", st["pin_hits"] > 0 and st["pinned_bytes"] > 0,
            f"pin_hits {st['pin_hits']}, pin_misses {st['pin_misses']}")
    R.turn1("C", R.prompt(2))
    ev0 = R.evicted()
    R.turn1("D", R.prompt(3))  # evicts C's checkpoint; every RAM entry is then written to the files
    R.eviction_check("D's prefill evicted C's checkpoint", R.evicted() - ev0 > 0, f"evicted +{R.evicted() - ev0}")
    n = export_ram(R.eng, spill_dir) if R.cache else 0
    s2 = R.snap("RAM entries written to SPILL_DIR")
    R.check("RAM entries written to SPILL_DIR", n > 0 and s2["on_disk"] > 0, f"{n} files, on_disk {s2['on_disk']}")
    R.turn2("C", R.prompt(8)[:FOLLOW], pre=lambda p2: async_restore(R, "C", p2, True))


SCEN_FN = {"a": scen_a, "b": scen_b, "c": scen_c, "d": scen_d, "e": scen_a, "f": scen_f, "g": scen_g}


# ---- worker: one process, one mode (ref | tiered), one phase

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def grow_pinned(eng, tiered):
    """SPILL_PINNED_GIB > 0: pins the slabs now (the frontend does this one per idle pass), so the pinned path is used."""
    if not tiered or float(os.environ.get("SPILL_PINNED_GIB", "0") or 0) <= 0 or not hasattr(eng, "spill_grow_pinned"):
        return
    say("  > spill_grow_pinned (pins the slabs)")
    n = 0
    while eng.spill_grow_pinned(1):
        n += 1
    say(f"  < spill_grow_pinned -> {n} slabs")


def worker(a):
    from q27lib import Engine
    from qwen27b.paths import DFLASH2, engine_target, snapshot

    draft = None
    if a.spec:
        try:
            draft = snapshot(*DFLASH2)
        except Exception as e:  # noqa: BLE001 - no draft model here: the scenario is skipped
            return write_json(a.out, {"skip": f"no draft model ({e})"})
    _, max_tokens = sizes(a.prompt_len, a.tokens)
    say(f"  > Engine ({a.mode}, max_tokens {max_tokens}, KV_POOL_TOKENS {os.environ.get('KV_POOL_TOKENS')})")
    eng = Engine(engine_target(a.weights), max_tokens=max_tokens, kv_dtype=a.kv_dtype, draft=draft)
    say("  < Engine ready")
    if a.spec and not eng.has_draft:
        return write_json(a.out, {"skip": "the engine has no draft model"})
    grow_pinned(eng, a.mode == "tiered")
    docs = load_docs(a.prompt_len)
    R = Run(eng, a.mode == "tiered", a.spec, a.tokens, docs, a.prompt_len, a.phase, print)
    heads = {tuple(R.prompt(i)[:64]) for i in range(4)}
    if len(heads) != 4:
        raise SystemExit("the prompts do not differ in their first 64 tokens")
    if a.prior:
        R.load_prior(a.prior)
    SCEN_FN[a.scenario](R, a.spill_dir)
    write_json(a.out, {"outs": R.outs, "fed": R.fed, "checks": R.checks, "stats": R.stats})
    say("  > eng.close()")
    eng.close()
    say("  < eng.close()")


# ---- driver

def worker_invocation(a, scen, mode, phase, spill_dir, out, prior=None):
    """The environment variables the driver sets for one worker process, and the worker's command line."""
    pool, max_tokens = sizes(a.prompt_len, a.tokens)
    pool = a.pool_tokens or pool
    # (no interval checkpoints: only the explicit ones; the engine needs a multiple of 64)
    set_env = {"KV_POOL_TOKENS": str(pool), "CHECKPOINT_INTERVAL": str(-(-2 * max_tokens // 64) * 64)}
    if mode == "ref":
        set_env["STATE_HOST_CACHE"] = "0"
    else:
        set_env.update(STATE_HOST_CACHE="1", SPILL_RAM_GIB=str(a.ram_gib), SPILL_DIR=str(spill_dir),
                       VRAM_BACKUP_IDLE_S="0")
        set_env.update(SCEN_ENV.get(scen, {}))
    # -u and faulthandler: the output is not buffered, and a crash (e.g. a Windows access violation) dumps the stack
    cmd = [sys.executable, "-u", "-X", "faulthandler", str(pathlib.Path(__file__).resolve()), "--worker",
           "--scenario", scen, "--mode", mode, "--phase", phase, "--out", str(out), "--spill-dir", str(spill_dir or ""),
           "--tokens", str(a.tokens), "--prompt-len", str(a.prompt_len), "--kv-dtype", a.kv_dtype, "--weights", a.weights,
           "--ram-gib", str(a.ram_gib), "--pool-tokens", str(pool)]
    if scen == "e":
        cmd.append("--spec")
    if prior:
        cmd += ["--prior", str(prior)]
    return set_env, cmd


def show_invocation(title, set_env, cmd):
    say(f"# {title}")
    for k in sorted(set_env):
        say(f"  {k}={set_env[k]}")
    say("  " + shlex.join(cmd))


def run_worker(a, scen, mode, phase, spill_dir, out, prior=None):
    set_env, cmd = worker_invocation(a, scen, mode, phase, spill_dir, out, prior)
    env = dict(os.environ)
    for k in [k for k in env if k.startswith("SPILL_") or k in CACHE_VARS]:
        del env[k]
    env.update(set_env)
    env.update(PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1")
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        say(f"  worker exit code {fmt_rc(rc)}: see the worker output above for the last step reached")
        return {"error": rc}
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="weight profile: nvidia | gdnfp4 | quasar | a model directory (default: %(default)s)")
    ap.add_argument("--kv-dtype", default="nvfp4", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--tokens", type=int, default=48, help="greedy tokens per turn")
    ap.add_argument("--prompt-len", type=int, default=3000,
                    help="tokens per conversation (raised to the smallest length that evicts, unless --pool-tokens)")
    ap.add_argument("--ram-gib", type=float, default=2.0, help="SPILL_RAM_GIB of the tiered runs")
    ap.add_argument("--pool-tokens", type=int,
                    help="KV_POOL_TOKENS (default: one conversation + its checkpoint + 1K; with it, --prompt-len is kept)")
    ap.add_argument("--scenarios", default="a,b,c,d,e,f,g", help="a,b,c,d,e,f,g")
    ap.add_argument("--keep-dir", help="keep the files (the SSD tier and the results) here; default: a temp dir")
    ap.add_argument("--worker-cmd", metavar="SCENARIO",
                    help="print the exact worker command lines of a scenario (its reference and tiered runs) and exit")
    # internal: one worker process
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--scenario", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=["ref", "tiered"], help=argparse.SUPPRESS)
    ap.add_argument("--phase", default="full", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--prior", help=argparse.SUPPRESS)
    ap.add_argument("--spill-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--spec", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    chosen = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    for s in chosen:
        if s not in SCENARIOS:
            ap.error(f"unknown scenario {s} (a-g)")
    if a.pool_tokens:
        _, max_tokens = sizes(a.prompt_len, a.tokens)
        say("pool: " + describe_pool(a.prompt_len, a.tokens, a.kv_dtype, a.pool_tokens, max_tokens), flush=True)
        if not fits_eviction(a.prompt_len, a.tokens, usable_pool_pages(max_tokens, a.pool_tokens, a.kv_dtype)):
            say("WARNING: the two conversations do not exceed the usable pool: no eviction is expected, "
                  "the eviction checks will fail (use a smaller --pool-tokens or a larger --prompt-len)", flush=True)
    else:
        p0 = a.prompt_len
        a.prompt_len = choose_prompt_len(p0, a.tokens, a.kv_dtype)
        if a.prompt_len != p0:
            say(f"prompt-len {p0} -> {a.prompt_len}: the smallest length whose two conversations exceed the usable pool",
                  flush=True)
        _, max_tokens = sizes(a.prompt_len, a.tokens)
        say("pool: " + describe_pool(a.prompt_len, a.tokens, a.kv_dtype, sizes(a.prompt_len, a.tokens)[0], max_tokens),
              flush=True)
    if a.worker_cmd:
        s_ = a.worker_cmd
        if s_ not in SCENARIOS:
            ap.error(f"unknown scenario {s_} (a-g)")
        base = pathlib.Path(a.keep_dir or "KEEP_DIR")  # (a placeholder without --keep-dir)
        say(f"worker commands of scenario {s_} ({SCENARIOS[s_][0]}), as the driver runs them; the other SPILL_* and cache")
        say("variables of your environment are unset. Windows cmd: set each NAME=VALUE first. Replace KEEP_DIR if needed.")
        set_env, cmd = worker_invocation(a, s_, "ref", "full", "", base / f"ref_{s_}.json")
        show_invocation("reference (cache off)", set_env, cmd)
        prior = None
        for ph in SCENARIOS[s_][1]:
            out = base / f"tiered_{s_}_{ph}.json"
            set_env, cmd = worker_invocation(a, s_, "tiered", ph, base / f"spill_{s_}", out, prior)
            show_invocation(f"tiered, phase {ph}" + (f" (after {prior.name})" if prior else ""), set_env, cmd)
            prior = out
        sys.exit(0)
    work = pathlib.Path(a.keep_dir or tempfile.mkdtemp(prefix="tiered_cache_test_")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    fails = 0
    for s in chosen:
        name, phases = SCENARIOS[s]
        say(f"\n== {s}: {name} ==", flush=True)
        spill_dir = work / f"spill_{s}"
        shutil.rmtree(spill_dir, ignore_errors=True)
        spill_dir.mkdir(parents=True)
        ref = run_worker(a, s, "ref", "full", "", work / f"ref_{s}.json")
        if "error" in ref:
            say(f"FAIL {s}: the reference run exited {fmt_rc(ref['error'])}", flush=True)
            fails += 1
            continue
        if ref.get("skip"):
            say(f"SKIP {s}: {ref['skip']}", flush=True)
            continue
        outs, checks, stats, prior, err, skip = {}, [], [], None, None, None
        for ph in phases:
            r = run_worker(a, s, "tiered", ph, spill_dir, work / f"tiered_{s}_{ph}.json", prior)
            if "error" in r:
                err = f"the tiered run ({ph}) exited {fmt_rc(r['error'])}"
                break
            if r.get("skip"):
                skip = r["skip"]
                break
            outs.update(r["outs"])
            checks += r["checks"]
            stats += r["stats"]
            prior = work / f"tiered_{s}_{ph}.json"
        if skip:
            say(f"SKIP {s}: {skip}", flush=True)
            continue
        if err:
            say(f"FAIL {s}: {err}", flush=True)
            fails += 1
            continue
        mism = [k for k in ref["outs"] if outs.get(k) != ref["outs"][k]]
        extra = [k for k in outs if k not in ref["outs"]]
        bad = [c for c in checks if not c["ok"]]
        if not checks:
            bad.append({"name": "no checks ran"})
        for c in checks:
            say(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['name']}" + (f" ({c['detail']})" if c["detail"] else ""))
        for st in stats:
            say(f"  spill [{st['label']}] " + stats_line(st))
        if mism or extra:
            say(f"  FAIL outputs differ from the reference: {mism + extra}")
            for k in mism:
                say(f"       {k} got {outs.get(k, [])[:12]}...\n       {k} ref {ref['outs'][k][:12]}...")
        else:
            say(f"  outputs identical to the reference: {len(ref['outs'])} sequences, "
                  f"{sum(len(v) for v in ref['outs'].values())} tokens")
        ok = not (mism or extra or bad)
        fails += not ok
        say(f"{'PASS' if ok else 'FAIL'} {s}: {name}", flush=True)
    say(f"\n{'all passed' if not fails else f'{fails} scenario(s) FAILED'}; files in {work}")
    if not a.keep_dir:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
