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

  h  RAM dedup        N=4 conversations whose prompts share a long common prefix (6 chunks = 24576 tokens; the chunk is
                      the RAM tier's 4096-token dedup unit, engine/src/chunk_store.h) and differ in their tails; a fifth,
                      unrelated one evicts the last so that all four are spilled. The spilled entries must hold less than
                      0.6x their logical size in RAM (physical_bytes < 0.6 * logical_bytes) with chunk_hits > 0; each is
                      then resumed (promoted back, the chunks copied into pool pages; the last two through files written
                      from the shared chunks) and its tokens must equal the uncached reference. Needs a ~25K-token pool and 25K-token prefills (a few minutes).
  i  RAM dedup, pinned + async I/O + async promote   as h, with SPILL_PINNED_GIB=1, SPILL_PINNED_SLAB_MIB=16,
                      SPILL_ASYNC_IO=1, SPILL_ASYNC_PROMOTE=1 and SPILL_RAM_GIB=4 (the chunks are pinned segments, copied on
                      the I/O stream and shared by events; every follow-up is restored through spill_match / spill_restore_*).

  j  SSD dedup        J_N=4 conversations that share a 6-chunk (24576-token) prefix are spilled and ALL written to SPILL_DIR (engine.spill_export:
                      entries/<id>.q27m manifests + chunks/<name>.kvc files shared by content). The SSD must hold the shared chunks once
                      (spill_disk_usage physical < 0.6 * logical, fewer .kvc files than chunk references). Phase 2 is a new process (a
                      new Engine on the same SPILL_DIR): it imports the manifests (spill_import, then spill_sweep removes nothing),
                      drops one entry (the SSD frees exactly its unique_disk_bytes; the shared .kvc files stay), and restores the others
                      from the files: two by the synchronous promote inside prefix_begin, one through the delay-hit API
                      (SPILL_ASYNC_PROMOTE=1: spill_match / spill_restore_begin returns 2 / poll). Every follow-up's tokens must equal the
                      uncached reference; when the last sibling is restored no chunk or manifest file is left.

  k  Retention       (docs/tiered-cache.md "Retention"; needs the engine's set_owner / spill_meta, else skipped.) Two agents share a RAM
                      tier that holds ~3 entries (phase probe measures one entry; phase run sets SPILL_RAM_GIB to 3.4 entries): B (owner 2)
                      has ONE entry, A (owner 1) many, of which one was restored once (hits 1) and the others never. When a fourth entry
                      arrives the engine's RAM cap must take its victim from A (the agent using the most RAM) although B's entry is
                      the oldest, and from A's never-hit entries rather than the older hit one: B and the hit entry stay in RAM, A's
                      oldest never-hit entry is written to SPILL_DIR. spill_meta / spill_list report owner and hits, spill_agents the
                      bytes (A > B). A conversation whose longer continuation exists in VRAM (computed with spill_sync_lookup(False),
                      so the spilled entry stays) is reported superseded, and the eviction of that continuation subsumes it instead of
                      writing it. Every restored follow-up (B from RAM, A's from the file) must equal the uncached reference. Prompts
                      are >= 25K tokens (superseded needs more than kKeepShort = 16384).

Each scenario runs a few conversations (the golden prompts, prompt-len tokens each, different starts) and one
follow-up turn per conversation that continues the generation with 300 new tokens. The driver sizes the conversations
to the engine's pool, including its block rounding (choose_prompt_len): the prompt length is raised until two
conversations exceed the usable pool, so B's prefill evicts A's checkpoint. It prints the numbers at the start.

    STATE_HOST_CACHE=1 docker compose --profile dev run --rm dev python scripts/tiered_cache_test.py
    python scripts/tiered_cache_test.py --scenarios a,c --keep-dir /cache/kv/test

Native Windows package (no Docker, no torch): test.cmd runs this with the package's Python; the weights are where
start.cmd downloads them (settings.ini HF_HOME, else the package's huggingface folder), q27.dll is app\\engine\\build\\.
--spec {dspine,dflash2,none} picks the draft model every engine of the test is created with (as the server does:
default $SPEC, else dspine if downloaded, else dflash2 if downloaded, else none); it is printed at the start. Without a
draft model (--spec none, or none downloaded) scenario e is skipped. The default --weights is WEIGHTS
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

SPEC_CHOICES = ("dspine", "dflash2", "none")


def default_spec():
    """The draft the server starts with (frontend/launch.py default_spec): $SPEC, else dspine if downloaded, else dflash2 if
    downloaded; none when neither is (the server would then still say dspine and log how to get it, a test cannot run it)."""
    from qwen27b.paths import DFLASH2, DSPINE, snapshot

    spec = os.environ.get("SPEC")
    if spec:
        return spec if spec in SPEC_CHOICES else "none"
    for name, pin in (("dspine", DSPINE), ("dflash2", DFLASH2)):
        try:
            if (snapshot(*pin) / "config.json").is_file():
                return name
        except FileNotFoundError:
            pass
    return "none"


def draft_path_or_missing(spec):
    try:
        return draft_path(spec)
    except FileNotFoundError:
        return "not downloaded: scenario e is skipped"


def draft_path(spec):
    """The draft checkpoint folder of a --spec value (None for none); FileNotFoundError when it is not downloaded."""
    from qwen27b.paths import DRAFTS, snapshot

    return snapshot(*DRAFTS[spec]) if spec in DRAFTS else None


SCENARIOS = {
    "a": ("RAM promote", ["full"]),
    "b": ("SSD promote", ["full"]),
    "c": ("restart import", ["write", "read"]),
    "d": ("idle backup reuse", ["full"]),
    "e": ("speculative decoding (ring checkpoints)", ["full"]),
    "f": ("pinned segments + async I/O", ["full"]),
    "g": ("async promote (delay hits): RAM and SSD", ["full"]),
    "h": ("RAM dedup: shared prefix, spill, promote", ["full"]),
    "i": ("RAM dedup with pinned segments + async I/O", ["full"]),
    "j": ("SSD dedup: shared chunk files, restart, drop, restore (sync + async)", ["write", "read"]),
    "k": ("retention: per-agent fairness, never-hit first, superseded", ["probe", "run"]),
}
# Extra engine variables of a scenario's tiered run (the driver clears every SPILL_* variable first).
SCEN_ENV = {"f": {"SPILL_PINNED_GIB": "2", "SPILL_PINNED_SLAB_MIB": "16", "SPILL_ASYNC_IO": "1"},
            "g": {"SPILL_PINNED_GIB": "2", "SPILL_PINNED_SLAB_MIB": "16", "SPILL_ASYNC_IO": "1", "SPILL_ASYNC_PROMOTE": "1"},
            "j": {"SPILL_ASYNC_PROMOTE": "1"},  # (prefix_begin still promotes synchronously; the delay-hit API is used for one entry)
            "i": {"SPILL_PINNED_GIB": "1", "SPILL_PINNED_SLAB_MIB": "16", "SPILL_ASYNC_IO": "1", "SPILL_ASYNC_PROMOTE": "1",
                  "SPILL_RAM_GIB": "4"}}
SUFFIX = ".q27m"  # the manifest name frontend/tiercache.py indexes at start(), in SPILL_DIR/entries (chunk files: SPILL_DIR/chunks/*.kvc)
ENTRIES = "entries"
CHUNKS = "chunks"
J_N = 4  # scenario j: conversations that share the prefix
K_OWNER_A, K_OWNER_B = 1, 2  # scenario k: the engine's agent ids (Model::set_owner)
K_CAP_ENTRIES = 3.4  # scenario k: the RAM cap in entries (phase probe measures the size of one)
FOLLOW = 300  # new tokens of each follow-up turn
KEYS = ("entries", "in_ram", "on_disk", "ram_bytes", "spilled", "restored", "written", "backed_up", "pinned_bytes",
        "pin_hits", "pin_misses", "restoring", "restored_async", "restore_failed")
# RAM dedup counters (an older engine does not report them: shown only when present)
DEDUP_KEYS = ("logical_bytes", "physical_bytes", "chunks", "chunk_hits", "chunk_misses", "dedup_saved_bytes")
H_CHUNKS = 6  # scenario h / i: the shared prefix, in dedup chunks (engine kChunkPages * kPageTokens = 4096 tokens each)
H_PREFIX = H_CHUNKS * 4096
H_MIN_TAIL = 512  # the shortest distinct tail of a prompt of scenario h
H_N = 4  # conversations that share the prefix
# Variables the tiered cache reads at engine creation: the driver sets them per run, never inherits them.
CACHE_VARS = {"STATE_HOST_CACHE", "KV_POOL_TOKENS", "CHECKPOINT_INTERVAL", "MAX_SEQS", "VRAM_BACKUP_IDLE_S"}


# ---- pure helpers (no GPU, no engine import)

def load_docs(min_len, tile=False):
    """The golden prompts of at least min_len tokens (bench/golden/prompts.jsonl). tile: with none that long, the longest
    one (make_prompt tiles a short document)."""
    docs = [json.loads(line)["ids"] for line in open(ROOT / "bench/golden/prompts.jsonl", encoding="utf-8")]
    longest = max(len(d) for d in docs)
    docs = [d for d in docs if len(d) >= (min(min_len, longest) if tile else min_len)]
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
SPEC_BLOCK = 16  # kSpecBlock: a verify block writes up to 16 positions past the committed length
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
    return " ".join(f"{k}={st[k]}" for k in KEYS + tuple(k for k in DEDUP_KEYS if k in st))


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
            os.makedirs(os.path.join(spill_dir, ENTRIES), exist_ok=True)
            path = os.path.join(spill_dir, ENTRIES, uuid.uuid4().hex + SUFFIX)
            say(f"  > spill_export #{n + 1} {e['id']} -> {os.path.basename(path)}")
            say(f"  < spill_export #{n + 1} -> {eng.spill_export(e['id'], path)}")
            n += 1
    return n


def import_all(eng, spill_dir):
    ids = []
    for p in sorted(pathlib.Path(spill_dir, ENTRIES).glob("*" + SUFFIX)):
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


def h_prompt(R, i):
    """Conversation i of scenario h: the common prefix (H_PREFIX tokens, chunk-aligned) and a tail of its own."""
    prefix = make_prompt(R.docs, H_PREFIX, 0)
    return prefix + make_prompt(R.docs, R.L - H_PREFIX, 1 + i)


def scen_h(R, spill_dir):
    if R.L < H_PREFIX + H_MIN_TAIL:
        raise SystemExit(f"scenario h needs prompts of at least {H_PREFIX + H_MIN_TAIL} tokens (got {R.L})")
    prompts = [h_prompt(R, i) for i in range(H_N)]
    if len({tuple(p[H_PREFIX:H_PREFIX + 64]) for p in prompts}) != H_N or any(p[:H_PREFIX] != prompts[0][:H_PREFIX] for p in prompts):
        raise SystemExit("scenario h: the prompts must share the prefix and differ in their tails")
    keys = "ABCD"[:H_N]
    s0 = R.snap("before the conversations")
    for k, p in zip(keys, prompts):  # each prefill evicts the previous conversation's checkpoint (the pool holds one)
        R.turn1(k, p)
    R.turn1("E", make_prompt(R.docs, R.L, 7))  # an unrelated conversation: its prefill evicts the last of A-D
    s1 = R.snap("all of A-D spilled (RAM)")
    R.eviction_check(f"the {H_N} conversations' checkpoints were all spilled to RAM", delta(s0, s1, "spilled") >= H_N,
                     f"spilled +{delta(s0, s1, 'spilled')}")
    R.check(f"the {H_N} entries are in RAM", s1["in_ram"] >= H_N and s1["on_disk"] == 0,
            f"in_ram {s1['in_ram']}, on_disk {s1['on_disk']}")
    if R.cache:
        have = all(k in s1 for k in DEDUP_KEYS)
        R.check("the engine reports the RAM dedup counters", have, "an engine without RAM dedup?" if not have else "")
        if have:
            logical, physical = s1["logical_bytes"], s1["physical_bytes"]
            R.check("shared prefix chunks are stored once: physical < 0.6 * logical", logical > 0 and physical < 0.6 * logical,
                    f"logical {logical / 2**20:.0f} MiB, physical {physical / 2**20:.0f} MiB, ratio {physical / max(logical, 1):.2f}")
            R.check("the later entries found the earlier ones' chunks (chunk_hits > 0)", s1["chunk_hits"] > 0,
                    f"hits {s1['chunk_hits']}, misses {s1['chunk_misses']}, resident chunks {s1['chunks']}")
    async_promote = R.cache and os.environ.get("SPILL_ASYNC_PROMOTE") == "1"  # (scenario i: the delay-hit restore API)

    def follow(k, on_disk):  # k's follow-up: its checkpoint is promoted back (chunks + tail copied into pool pages)
        s_before = R.snap(f"before {k}'s follow-up")
        pre = (lambda p2: async_restore(R, k, p2, on_disk)) if async_promote else None
        R.turn2(k, make_prompt(R.docs, 1000, 9 + ord(k))[:FOLLOW], pre=pre)
        s_after = R.snap(f"after {k}'s follow-up")
        R.check(f"{k}'s follow-up restored its checkpoint from {'the file' if on_disk else 'RAM'}",
                delta(s_before, s_after, "restored") > 0, f"restored +{delta(s_before, s_after, 'restored')}")

    for k in keys[:H_N // 2]:  # from RAM
        follow(k, False)
    # The rest are written to files (spill_write puts the shared chunks and the tails back into the linear stream) and
    # restored from there.
    n = export_ram(R.eng, spill_dir) if R.cache else 0
    s2 = R.snap("RAM entries written to SPILL_DIR")
    R.check("RAM entries written to SPILL_DIR", n > 0 and s2["on_disk"] > 0, f"{n} files, on_disk {s2['on_disk']}")
    for k in keys[H_N // 2:]:
        follow(k, True)
    if R.cache and all(x in s1 for x in DEDUP_KEYS):
        sf = R.snap("end")
        R.check("physical <= logical at the end", sf["physical_bytes"] <= sf["logical_bytes"] or sf["logical_bytes"] == 0,
                f"logical {sf['logical_bytes']}, physical {sf['physical_bytes']}")


class ScenarioSkip(Exception):
    """A scenario that cannot run with this engine library (reported as SKIP, not FAIL)."""


def k_ents(R):
    """The spilled entries as the engine lists them, newest id last."""
    return sorted(R.eng.spill_list(), key=lambda e: e["id"])


def k_tier(e):
    return "ram" if e["ram_bytes"] > 0 else "disk" if e["on_disk"] else "?"


def scen_k(R, spill_dir):
    """Retention (see the scenario list at the top). Phases: full (the reference: everything), probe (one entry's size), run."""
    if R.L < H_PREFIX + H_MIN_TAIL:
        raise SystemExit(f"scenario k needs prompts of at least {H_PREFIX + H_MIN_TAIL} tokens (got {R.L})")
    if R.cache and not (hasattr(R.eng, "set_owner") and R.eng.has_retention):
        raise ScenarioSkip("the engine library has no retention calls (set_owner / spill_meta / spill_agents)")
    own = (lambda o: R.eng.set_owner(o)) if R.cache else (lambda o: None)
    follow = {k: make_prompt(R.docs, 1000, 9 + ord(k[0]))[:FOLLOW] for k in ("A1", "A2", "A3", "B")}
    if R.phase in ("full", "probe"):
        s0 = R.snap("probe start")
        own(K_OWNER_A)
        R.turn1("P0", make_prompt(R.docs, R.L, 30))
        R.turn1("P1", make_prompt(R.docs, R.L, 31))  # evicts P0's checkpoint: one entry in RAM
        s1 = R.snap("probe entry")
        R.eviction_check("the probe conversation was spilled to RAM", delta(s0, s1, "spilled") >= 1 and s1["in_ram"] == 1,
                         f"spilled +{delta(s0, s1, 'spilled')}, in_ram {s1['in_ram']}")
        if R.phase == "probe":
            return
    # ---- run: B (agent 2) has one entry, A (agent 1) several; RAM holds ~3.
    conv = {"B": (K_OWNER_B, 20), "A1": (K_OWNER_A, 21), "A2": (K_OWNER_A, 22), "A3": (K_OWNER_A, 23), "A4": (K_OWNER_A, 24),
            "A5": (K_OWNER_A, 25)}

    def turn1(k):
        own(conv[k][0])
        R.turn1(k, make_prompt(R.docs, R.L, conv[k][1]))

    s0 = R.snap("run start")
    turn1("A1")
    turn1("B")                       # evicts A1's checkpoint: RAM {A1}
    own(K_OWNER_A)
    R.turn2("A1", follow["A1"])      # restores A1 (hit); its prefill evicts B's checkpoint: RAM {B}; A1's checkpoint is in the pool, hits 1
    if R.cache:  # the worker saves the end of every turn: a deeper checkpoint of the same conversation. It must inherit the hit (the
        R.op("checkpoint (the end of A1's follow-up)", R.eng.checkpoint)  # restored one is subsumed when evicted, only this one spills)
    turn1("A2")                      # evicts it: RAM {B, A1'' (the deeper checkpoint, hits 1: inherited)}
    turn1("A3")                      # RAM {B, A1', A2}
    if R.cache:
        mid = k_ents(R)
        R.check("A1's restored checkpoint came back with hits >= 1 and A's owner",
                any(e["owner"] == K_OWNER_A and e["hits"] >= 1 and k_tier(e) == "ram" for e in mid), [(e["owner"], e["hits"], k_tier(e)) for e in mid])
        R.check("B's entry carries B's owner and no hits", any(e["owner"] == K_OWNER_B and e["hits"] == 0 and k_tier(e) == "ram" for e in mid))
    sa = R.snap("RAM holds B, A1' (hit), A2 (3 entries)")
    turn1("A4")                      # evicts A3's checkpoint: 4 entries do not fit
    sb = R.snap("RAM cap enforced")
    if R.cache:
        ents = k_ents(R)
        ram = [e for e in ents if k_tier(e) == "ram"]
        disk = [e for e in ents if k_tier(e) == "disk"]
        R.check("the RAM cap was enforced (a fourth entry did not fit)", sb["spilled"] > sa["spilled"] and len(disk) + sb["dropped"] - sa["dropped"] >= 1,
                f"ram {len(ram)}, disk {len(disk)}, dropped +{sb['dropped'] - sa['dropped']}")
        R.check("fairness: the small agent's entry is still in RAM although it is the oldest",
                any(e["owner"] == K_OWNER_B for e in ram), [(e["owner"], e["hits"], k_tier(e)) for e in ents])
        R.check("the victim belonged to the big agent", all(e["owner"] == K_OWNER_A for e in disk) and len(disk) >= 1,
                [(e["owner"], e["hits"], k_tier(e)) for e in ents])
        R.check("never-hit first: the entry that was restored once is still in RAM, the evicted one was never hit",
                any(e["owner"] == K_OWNER_A and e["hits"] >= 1 for e in ram) and all(e["hits"] == 0 for e in disk),
                [(e["owner"], e["hits"], k_tier(e)) for e in ents])
        ag = {a["owner"]: a for a in R.eng.spill_agents() or []}
        R.check("spill_agents: the big agent uses more RAM than the small one, the small one has no SSD bytes",
                ag.get(K_OWNER_A, {}).get("ram_bytes", 0) > ag.get(K_OWNER_B, {}).get("ram_bytes", 1 << 60) and ag.get(K_OWNER_B, {}).get("ssd_bytes", 1) == 0, ag)
    # ---- superseded: A3's continuation computed without promoting A3's entry (spill_sync_lookup(False)). The continuation is a deeper
    # checkpoint of the same chain in the pool, so A3's entry (it may have been written to the SSD by the cap meanwhile) is superseded;
    # evicting the continuation (A5's prefill) subsumes it: spill_checkpoint erases the shallower entry instead of keeping both.
    p3 = R.ids["A3"] + R.outs["A3"] + follow["A3"]
    own(K_OWNER_A)
    if R.cache:
        R.op("spill_sync_lookup(False)", R.eng.spill_sync_lookup, False)
        try:
            cached = R.op("prefix_begin (continuation of A3, A3's entry is not promoted)", R.eng.prefix_begin, p3, spec=R.spec)
            R.check("the continuation was computed (A3's entry was not used)", cached < R.fed["A3"], f"cached {cached}")
            R.outs["A3+"] = R._gen(p3[cached:])
            R.op("checkpoint", R.eng.checkpoint)
        finally:
            R.eng.spill_sync_lookup(True)
        after = k_ents(R)
        sup = [e for e in after if e.get("superseded")]
        R.check("the shallower entry of A3 is reported superseded by the continuation in the pool",
                len(sup) == 1 and sup[0]["owner"] == K_OWNER_A and sup[0]["hits"] == 0,
                [(e["owner"], e["hits"], e.get("superseded"), k_tier(e)) for e in after])
        se = R.snap("continuation of A3 in the pool")
        turn1("A5")  # evicts the continuation
        sf = R.snap("after A5")
        R.check("evicting the continuation replaced the superseded entry (subsumed), not kept both",
                sf["subsumed"] > se["subsumed"] and not any(e.get("superseded") for e in k_ents(R)),
                f"subsumed +{sf['subsumed'] - se['subsumed']}")
        R.check("the small agent's entry survived all of this", any(e["owner"] == K_OWNER_B and k_tier(e) == "ram" for e in k_ents(R)))
    else:
        R.turn2("A3", follow["A3"])
        turn1("A5")
    # ---- restored follow-ups: B from RAM (it survived), A2 from the file the cap wrote; their tokens equal the reference
    own(K_OWNER_B)
    sc = R.snap("before B's follow-up")
    R.turn2("B", follow["B"])
    sd = R.snap("after B's follow-up")
    R.check("B's follow-up restored its checkpoint", delta(sc, sd, "restored") > 0, f"restored +{delta(sc, sd, 'restored')}")
    own(K_OWNER_A)
    R.turn2("A2", follow["A2"])
    se = R.snap("after A2's follow-up")
    R.check("A2's follow-up restored its checkpoint (the SSD file written by the RAM cap)", delta(sd, se, "restored") > 0,
            f"restored +{delta(sd, se, 'restored')}, from_disk +{delta(sd, se, 'restored_disk') if 'restored_disk' in se else '?'}")


def chunk_files(spill_dir):
    return {p.name for p in pathlib.Path(spill_dir, CHUNKS).glob("*.kvc")}


def scen_j(R, spill_dir):
    """SSD dedup (docs/tiered-cache.md): see the scenario list at the top. Phases: full (the reference), write, read."""
    if R.L < H_PREFIX + H_MIN_TAIL:
        raise SystemExit(f"scenario j needs prompts of at least {H_PREFIX + H_MIN_TAIL} tokens (got {R.L})")
    prompts = [h_prompt(R, i) for i in range(J_N)]
    if len({tuple(p[H_PREFIX:H_PREFIX + 64]) for p in prompts}) != J_N or any(p[:H_PREFIX] != prompts[0][:H_PREFIX] for p in prompts):
        raise SystemExit("scenario j: the prompts must share the prefix and differ in their tails")
    keys = "ABCD"[:J_N]
    follow = {k: make_prompt(R.docs, 1000, 9 + ord(k))[:FOLLOW] for k in keys}
    if R.phase in ("full", "write"):
        s0 = R.snap("before the conversations")
        for k, p in zip(keys, prompts):  # each prefill evicts the previous conversation's checkpoint (the pool holds one)
            R.turn1(k, p)
        R.turn1("E", make_prompt(R.docs, R.L, 7))  # an unrelated conversation: its prefill evicts the last of A-D
        s1 = R.snap("all of A-D spilled (RAM)")
        R.eviction_check(f"the {J_N} conversations' checkpoints were all spilled to RAM", delta(s0, s1, "spilled") >= J_N,
                         f"spilled +{delta(s0, s1, 'spilled')}")
        if R.phase == "write":
            n = R.op("export every RAM entry", export_ram, R.eng, spill_dir)
            s2 = R.snap("write: RAM entries written to SPILL_DIR")
            R.check("every spilled entry was written", n >= J_N and s2["on_disk"] >= J_N, f"{n} manifests, on_disk {s2['on_disk']}")
            u = R.eng.spill_disk_usage()
            R.check("the engine has the SSD dedup API (spill_disk_usage)", u is not None, "an engine without it keeps flat files")
            if u is not None:
                logical, physical = u["logical_bytes"], u["physical_bytes"]
                R.check("the shared chunks are on the SSD once: physical < 0.6 * logical", logical > 0 and physical < 0.6 * logical,
                        f"logical {logical / 2**20:.0f} MiB, physical {physical / 2**20:.0f} MiB, ratio {physical / max(logical, 1):.2f}")
                files = chunk_files(spill_dir)
                R.check("the chunk files on disk are the ones the engine counts", len(files) == u["chunk_files"],
                        f"{len(files)} files, engine {u['chunk_files']}")
                R.check(f"fewer chunk files than chunk references ({J_N} entries share {H_CHUNKS} chunks)",
                        len(files) <= J_N * ((R.L + R.n) // 4096) - (J_N - 1) * H_CHUNKS, f"{len(files)} .kvc files")
                R.check("every entry has a manifest, and no temporary file is left",
                        u["manifests"] >= J_N and not list(pathlib.Path(spill_dir).rglob("*.tmp")), f"{u['manifests']} manifests")
            return
    else:  # read: a new process; its Engine has no history, only the files
        for k, p in zip(keys, prompts):
            R.ids[k] = p
        ids = R.op("import every manifest", import_all, R.eng, spill_dir)
        swept = R.op("spill_sweep", R.eng.spill_sweep)
        s = R.snap("read: after spill_import")
        R.check("the new engine imports the manifests of the earlier process", len(ids) >= J_N and all(ids) and s["on_disk"] >= J_N,
                f"{len(ids)} manifests, on_disk {s['on_disk']}")
        R.check("a clean write phase leaves nothing to sweep", swept == 0, f"swept {swept}")
        u = R.eng.spill_disk_usage()
        if u is not None:
            R.check("after the restart the index sees the sharing: physical < 0.6 * logical",
                    u["logical_bytes"] > 0 and u["physical_bytes"] < 0.6 * u["logical_bytes"],
                    f"logical {u['logical_bytes'] / 2**20:.0f} MiB, physical {u['physical_bytes'] / 2**20:.0f} MiB")
    # (full = the reference, read = the tiered run) drop the last conversation's entry, then restore the others from the files.
    dropped = keys[-1]
    if R.cache:
        p2 = R.ids[dropped] + R.outs[dropped] + follow[dropped]
        m = R.eng.spill_match(p2, R.spec)
        R.check(f"{dropped}'s entry is found on disk", m is not None and m["on_disk"], m)
        row = next((e for e in R.eng.spill_list() if m and e["id"] == m["id"]), None)
        u0, files0 = R.eng.spill_disk_usage(), chunk_files(spill_dir)
        if m and row and u0 is not None:
            R.op("spill_drop", R.eng.spill_drop, m["id"])
            u1, files1 = R.eng.spill_disk_usage(), chunk_files(spill_dir)
            R.check("dropping an entry frees exactly its unique_disk_bytes", u0["physical_bytes"] - u1["physical_bytes"] == row["unique_disk_bytes"],
                    f"freed {u0['physical_bytes'] - u1['physical_bytes']}, unique {row['unique_disk_bytes']}")
            R.check("the chunk files the siblings share remain", len(files1) >= H_CHUNKS and files1 <= files0,
                    f"{len(files0)} -> {len(files1)} .kvc files")
    sa = R.snap("before the follow-ups")
    for k in keys[:-1]:
        async_this = k == keys[-2]  # the last sibling goes through the delay-hit API, the others through prefix_begin
        pre = (lambda p2, k=k: async_restore(R, k, p2, True)) if async_this and R.cache else None
        R.turn2(k, follow[k], pre=pre)
        sb = R.snap(f"after {k}'s follow-up")
        R.check(f"{k}'s follow-up restored its checkpoint from the files ({'async' if async_this else 'sync'})",
                delta(sa, sb, "restored") > 0, f"restored +{delta(sa, sb, 'restored')}")
        sa = sb
    if R.cache:
        left = chunk_files(spill_dir) | {p.name for p in pathlib.Path(spill_dir, ENTRIES).glob("*" + SUFFIX)}
        R.check("when the last sibling is restored no chunk or manifest file is left", not left, f"{sorted(left)[:4]}")


SCEN_FN = {"a": scen_a, "b": scen_b, "c": scen_c, "d": scen_d, "e": scen_a, "f": scen_f, "g": scen_g, "h": scen_h, "i": scen_h, "j": scen_j, "k": scen_k}


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
    from qwen27b.paths import engine_target

    draft = None
    try:
        draft = draft_path(a.spec)  # (every engine gets the server's draft, whether or not the scenario steps with it)
    except FileNotFoundError as e:
        if a.use_spec:  # (scenario e needs it; the others run without)
            return write_json(a.out, {"skip": f"no draft model ({e})"})
        say(f"  (no {a.spec} draft: {e}; the engine runs without one)")
    _, max_tokens = sizes(a.prompt_len, a.tokens)
    say(f"  > Engine ({a.mode}, max_tokens {max_tokens}, KV_POOL_TOKENS {os.environ.get('KV_POOL_TOKENS')})")
    eng = Engine(engine_target(a.weights), max_tokens=max_tokens, kv_dtype=a.kv_dtype, draft=draft)
    say("  < Engine ready")
    if a.use_spec and not eng.has_draft:
        return write_json(a.out, {"skip": "the engine has no draft model"})
    grow_pinned(eng, a.mode == "tiered")
    docs = load_docs(a.prompt_len, tile=a.scenario in ("h", "i", "j", "k"))
    R = Run(eng, a.mode == "tiered", a.use_spec, a.tokens, docs, a.prompt_len, a.phase, print)
    heads = {tuple(R.prompt(i)[:64]) for i in range(4)}
    if len(heads) != 4 and a.scenario not in ("h", "i", "j", "k"):  # (h builds its own prompts, which share their first tokens)
        raise SystemExit("the prompts do not differ in their first 64 tokens")
    if a.prior:
        R.load_prior(a.prior)
    try:
        SCEN_FN[a.scenario](R, a.spill_dir)
    except ScenarioSkip as e:
        eng.close()
        return write_json(a.out, {"skip": str(e)})
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
        if scen == "k" and phase == "run":
            set_env["SPILL_RAM_GIB"] = k_ram_gib(prior, a.ram_gib)
    # -u and faulthandler: the output is not buffered, and a crash (e.g. a Windows access violation) dumps the stack
    cmd = [sys.executable, "-u", "-X", "faulthandler", str(pathlib.Path(__file__).resolve()), "--worker",
           "--scenario", scen, "--mode", mode, "--phase", phase, "--out", str(out), "--spill-dir", str(spill_dir or ""),
           "--tokens", str(a.tokens), "--prompt-len", str(a.prompt_len), "--kv-dtype", a.kv_dtype, "--weights", a.weights,
           "--ram-gib", str(a.ram_gib), "--pool-tokens", str(pool), "--spec", a.spec]
    if scen == "e":
        cmd.append("--use-spec")
    if prior:
        cmd += ["--prior", str(prior)]
    return set_env, cmd


def k_ram_gib(prior, default):
    """Scenario k, phase run: SPILL_RAM_GIB holding K_CAP_ENTRIES entries, from the size of the one the probe phase measured
    (its stats entry "probe entry"); the default (--ram-gib) when there is no probe result (e.g. --worker-cmd before a run)."""
    try:
        for st in json.load(open(prior))["stats"]:
            if st.get("label") == "probe entry" and st.get("ram_bytes"):
                return f"{K_CAP_ENTRIES * st['ram_bytes'] / 2**30:.4f}"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return str(default)


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


def scen_args(a, scen):
    """The arguments a scenario runs with: h and i need prompts longer than the shared prefix (and a pool that is still too
    small for two of them), so their length is raised and the pool re-sized for it (j too)."""
    if scen not in ("h", "i", "j", "k") or a.pool_tokens:
        return a
    b = argparse.Namespace(**vars(a))
    b.prompt_len = choose_prompt_len(max(a.prompt_len, H_PREFIX + H_MIN_TAIL), a.tokens, a.kv_dtype)
    return b


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
    ap.add_argument("--spec", choices=SPEC_CHOICES,
                    help="draft model of every engine (default: the server's: $SPEC, else dspine if downloaded, else dflash2, else none)")
    ap.add_argument("--scenarios", default="a,b,c,d,e,f,g,h,i,j,k", help="a,b,c,d,e,f,g,h,i,j,k")
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
    ap.add_argument("--use-spec", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    a.spec = a.spec or default_spec()
    say(f"draft model: {a.spec}" + ("" if a.spec == "none" else f" ({draft_path_or_missing(a.spec)})"), flush=True)

    chosen = [s.strip() for s in a.scenarios.split(",") if s.strip()]
    for s in chosen:
        if s not in SCENARIOS:
            ap.error(f"unknown scenario {s} (a-k)")
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
            ap.error(f"unknown scenario {s_} (a-k)")
        base = pathlib.Path(a.keep_dir or "KEEP_DIR")  # (a placeholder without --keep-dir)
        say(f"worker commands of scenario {s_} ({SCENARIOS[s_][0]}), as the driver runs them; the other SPILL_* and cache")
        say("variables of your environment are unset. Windows cmd: set each NAME=VALUE first. Replace KEEP_DIR if needed.")
        a = scen_args(a, s_)
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
        sa = scen_args(a, s)
        if sa is not a:
            say(f"prompt-len {a.prompt_len} -> {sa.prompt_len} for scenario {s} (a prefix of {H_PREFIX} shared tokens + tails)")
        ref = run_worker(sa, s, "ref", "full", "", work / f"ref_{s}.json")
        if "error" in ref:
            say(f"FAIL {s}: the reference run exited {fmt_rc(ref['error'])}", flush=True)
            fails += 1
            continue
        if ref.get("skip"):
            say(f"SKIP {s}: {ref['skip']}", flush=True)
            continue
        outs, checks, stats, prior, err, skip = {}, [], [], None, None, None
        for ph in phases:
            r = run_worker(sa, s, "tiered", ph, spill_dir, work / f"tiered_{s}_{ph}.json", prior)
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
