"""Start settings of the engine (the stats page, POST /admin/engine): the weights, KV cache type, speculative
decoding, vision tower, number of sequences, context limit and VRAM reserve an engine process starts with, and what
each setting leaves for the context before the engine starts.

The plans come from the engine's own memory planner (q27lib.Engine.plan: the pools of the engine's constructor for a
device with the given free memory, without CUDA; a plan takes 10 to 80 ms), in a helper process: it sets the
environment the engine reads (VISION, MAX_SEQS, VRAM_RESERVE_GIB, ...) for each plan, which the server's own process
must not change while it starts engines (a spawned engine copies it).
"""

import json
import math
import multiprocessing as mp
import os
import re
import sys
import threading

from qwen27b.paths import DFLASH2, DRAFTS, DSPINE, PROFILES, TARGET, downloaded_repos, engine_target, snapshot, unreadable_repos
from tokenizer import fingerprint

CHOICES = {"kv_dtype": ("nvfp4", "fp8", "bf16"), "spec": ("dspine", "dflash2", "none"), "vision": ("host", "fp8", "bf16", "off")}
MAX_SEQS = 8                # (the engine's lanes of a batched step)
MAX_CONTEXT = 262144        # max_position_embeddings of the checkpoint
DEFAULT_RESERVE_GIB = 1.5   # VRAM_RESERVE_GIB when unset (memory.cc budget_for)
# The values the page shows a plan for, besides the current one.
VARIANTS = {"max_seqs": tuple(range(1, MAX_SEQS + 1)), "max_tokens": (32768, 65536, 131072, 196608, 262144),
            "vram_reserve_gib": (None, 0.5, 1.0, 2.0, 3.0, 4.0)}  # (None: the default, 1.5)
LM_HEAD_ENV = os.environ.get("Q27_LM_HEAD_CKPT", "")  # (engine_target("quasar") sets it in this process)
GiB = 1 << 30
# Engine settings whose library default (q27lib.Engine, for the tests and scripts) is not the server's, set when
# unset: the server starts the same with or without compose.yaml, which lists them too (tests/test_parsing.py).
# MAX_TOKENS is an upper bound the engine lowers to what the VRAM budget fits (e.g. with MAX_SEQS=8); the prefix
# cache keeps 4 GiB of pinned state checkpoints. The tiered cache (docs/tiered-cache.md): a 512K-token KV pool, and
# 64 GiB of host RAM for the checkpoints it gives up, written to SPILL_DIR (the compose kvcache volume here; on
# Windows a per-user folder, see spill_dir_default). These match compose.yaml (tests/test_parsing.py); an env
# variable, a .env value or a settings.ini [env] entry of the tray app overrides them, as the setdefault does.
ENGINE_ENV_DEFAULTS = {"MAX_TOKENS_FIT": "1", "STATE_HOST_CACHE": "8", "KV_POOL_TOKENS": "524288",
                       "SPILL_RAM_GIB": "64", "SPILL_DIR": "/cache/kv"}
# The tiered cache settings (docs/tiered-cache.md): key -> (variable, default, kind). They are start settings like the
# rest: the engine process reads the engine's (SPILL_RAM_GIB, SPILL_PINNED_GIB, SPILL_ASYNC_*) and worker.TierPolicy
# the policy's (retention, SSD cap and free space, TTL, VRAM backup) from its environment when it loads, so a change
# takes effect with an engine restart. The defaults are those of compose.yaml and TierPolicy (tests/test_launch.py).
# kind: "num" (a number, 0 or more; no upper limit: the machine decides), "flag" (0 or 1) or "path".
TIER_SETTINGS = {
    "spill_ram_gib": ("SPILL_RAM_GIB", 64, "num"),                # 0: no tiered cache at all
    "spill_ram_retention_s": ("SPILL_RAM_RETENTION_S", 3600, "num"),
    "spill_ssd_gib": ("SPILL_SSD_GIB", 100, "num"),
    "spill_ssd_min_free_gib": ("SPILL_SSD_MIN_FREE_GIB", 10, "num"),
    "spill_ssd_min_tokens": ("SPILL_SSD_MIN_TOKENS", 32768, "num"),  # shorter spilled entries are never written to the SSD
    "spill_ttl_days": ("SPILL_TTL_DAYS", 7, "num"),
    "vram_backup_idle_s": ("VRAM_BACKUP_IDLE_S", 60, "num"),
    "spill_pinned_gib": ("SPILL_PINNED_GIB", 8, "num"),
    "spill_async_io": ("SPILL_ASYNC_IO", True, "flag"),
    "spill_async_promote": ("SPILL_ASYNC_PROMOTE", True, "flag"),
    "spill_fair_agents": ("SPILL_FAIR_AGENTS", True, "flag"),  # a tier's victim comes from the agent using the most of it (retention)
    "spill_dir": ("SPILL_DIR", None, "path"),                      # None: spill_dir_default(); "": no SSD tier
}
TIER_INTEGER = frozenset({"spill_ssd_min_tokens"})  # numbers that count whole things: a fraction is rounded down
TIER_ENV = frozenset(v[0] for v in TIER_SETTINGS.values())
# The variables the settings start from (server.py).
ENV_NAMES = {"weights": "WEIGHTS", "kv_dtype": "KV_DTYPE", "spec": "SPEC", "vision": "VISION", "max_seqs": "MAX_SEQS",
             "max_tokens": "MAX_TOKENS", "vram_reserve_gib": "VRAM_RESERVE_GIB",
             **{k: v[0] for k, v in TIER_SETTINGS.items()}}


def spill_dir_default(platform=os.name, env=os.environ):
    """SPILL_DIR when unset: the compose volume on Linux; on Windows (the tray app, no Docker) a folder of the user."""
    if platform != "nt":
        return ENGINE_ENV_DEFAULTS["SPILL_DIR"]
    return os.path.join(env.get("LOCALAPPDATA") or os.path.expanduser("~"), "qwen3-27b-5090", "kvcache")


def engine_env_defaults(platform=os.name, env=os.environ):
    """ENGINE_ENV_DEFAULTS with the platform's SPILL_DIR."""
    return dict(ENGINE_ENV_DEFAULTS, SPILL_DIR=spill_dir_default(platform, env))


def set_engine_env(root):
    """The environment an engine of the server starts with, besides the settings (`root`: the checkout or package).
    Called before the engine process is spawned (it copies os.environ) and before TierPolicy.from_env reads it."""
    for k, v in engine_env_defaults().items():
        os.environ.setdefault(k, v)
    vocab = root / "bench/draft-vocab/vocab-128k.txt"
    if "Q27_DRAFT_VOCAB" not in os.environ and vocab.exists():
        os.environ["Q27_DRAFT_VOCAB"] = str(vocab)


def tier_number(value):
    """A tier setting's number as stored: an int when it has no fraction, else rounded to 3 places."""
    v = round(float(value), 3)
    return int(v) if v == int(v) else v


def tier_env_value(key, value):
    """A tier setting as its variable has it ("1" / "0" for a flag)."""
    return ("1" if value else "0") if TIER_SETTINGS[key][2] == "flag" else str(value)


def atoi(text):
    """C's atoi: leading white space, an optional sign, then the digits ("1.0" is 1, "true" and "" are 0)."""
    m = re.match(r"[ \t\n\v\f\r]*([+-]?)([0-9]+)", text)
    return 0 if not m else (-1 if m.group(1) == "-" else 1) * int(m.group(2))


def parse_tier(key, text, warn=None):
    """The tier setting `key` from the text of its variable (unset or empty: the default). A text TierPolicy or the
    engine would refuse (not a number, or negative) gives the default, with a warning."""
    var, default, kind = TIER_SETTINGS[key]
    if kind == "path":
        return spill_dir_default() if text is None else text
    if text is None or not text.strip():
        return default
    if kind == "flag":
        return atoi(text) != 0  # (capi.cc: atoi(value) != 0)
    try:
        v = float(text)
        if math.isfinite(v) and v >= 0:
            return int(v) if key in TIER_INTEGER else tier_number(v)
    except ValueError:
        pass
    (warn or (lambda m: print(m, file=sys.stderr, flush=True)))(f"{var}={text!r} is not a number of 0 or more: using {default}")
    return default


def downloaded(pin):
    try:
        return (snapshot(*pin) / "config.json").is_file()
    except FileNotFoundError:
        return False


def default_spec():
    """SPEC, else dspine; dflash2 when only DFlash2 is downloaded (a checkout updated from before DSpine became the
    default, whose weights folder has DFlash2 alone: the server still starts, and logs how to get DSpine)."""
    spec = os.environ.get("SPEC")
    if spec:
        return spec
    return "dflash2" if not downloaded(DSPINE) and downloaded(DFLASH2) else "dspine"


def env_settings():
    """The start settings from the variables, by default the server's."""
    env = os.environ.get
    reserve = env("VRAM_RESERVE_GIB")
    return {**{k: parse_tier(k, env(v[0])) for k, v in TIER_SETTINGS.items()}, "weights": env("WEIGHTS", "gdnfp4"), "kv_dtype": env("KV_DTYPE", "nvfp4"), "spec": default_spec(),
            "vision": env("VISION") or "host", "max_seqs": max(1, min(int(env("MAX_SEQS", 4)), MAX_SEQS)),
            "max_tokens": int(env("MAX_TOKENS", MAX_CONTEXT)), "vram_reserve_gib": parse_reserve(reserve)}


def parse_reserve(value):
    """VRAM_RESERVE_GIB as the engine reads it (memory.cc): empty is the default (None), unparsable or negative is 0."""
    if not value:
        return None
    try:
        v = float(value)
    except ValueError:
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def auto_budget(free, total=None, reserve=None):
    """The engine's budget (memory.cc budget_for: free memory minus the reserve, `reserve` GiB or the default), in
    bytes; without the legacy VRAM_BUDGET_GIB, which can only lower it."""
    reserve = DEFAULT_RESERVE_GIB if reserve is None else reserve
    return int(max(0.0, free / GiB - reserve) * GiB)


def model_dir(weights):
    """The model directory of a weight profile (or a directory); FileNotFoundError when it is not downloaded."""
    path = engine_target(weights)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{weights}: no config.json in {path}")
    return path


class Launch:
    """The settings the next engine start uses (changed with a start, restart or save from the admin API; they last
    until the server restarts, which starts from the variables again) and their plans."""

    KEYS = ("weights", "kv_dtype", "spec", "vision", "max_seqs", "max_tokens", "vram_reserve_gib", *TIER_SETTINGS)

    def __init__(self, base, settings, tokenizer=None):
        self.base = base            # the rest of the engine's configuration (prefill chunk, scheduling, ...)
        self.tokenizer = tokenizer  # fingerprint() of the server's tokenizer: weights with another one are refused
        self.lock = threading.Lock()
        self.plans_cache = {}
        self.planner = None
        self.extra_weights = () if settings["weights"] in PROFILES else (settings["weights"],)
        self.settings = self.at_start = self.validate(settings, check_budget=False)

    # ------------------------------------------------------------ settings

    def weights_list(self):
        """The profiles, the server's own weights, and the checkpoints of the model downloaded from other repositories
        (paths.downloaded_repos: "org/name" or "org/name@commit")."""
        listed = PROFILES + self.extra_weights
        return listed + tuple(w for w in downloaded_repos() if w not in listed)

    def weights_choices(self):
        """{weights: None if they can be loaded, else why not}."""
        out = {}
        for w in self.weights_list():
            try:
                path = model_dir(w)
                out[w] = None
                if self.tokenizer and fingerprint(path) != self.tokenizer:
                    out[w] = f"{w}: another tokenizer (vocabulary) than the server's (start the server with WEIGHTS={w})"
            except (FileNotFoundError, KeyError, ValueError) as e:
                out[w] = str(e)
        return out

    def env_changes(self):
        """{variable: value} that would start the server with the settings changed since it started ("": automatic)."""
        return {ENV_NAMES[k]: tier_env_value(k, self.settings[k]) if k in TIER_SETTINGS
                else "" if self.settings[k] is None else str(self.settings[k])
                for k in self.KEYS if self.settings[k] != self.at_start[k]}

    def validate(self, changes, base=None, device=None, check_budget=True):
        """The settings with `changes` applied; ValueError names a bad one."""
        s = dict(base or getattr(self, "settings", {}), **changes)
        for k, (_, default, kind) in TIER_SETTINGS.items():  # (settings made without the tier ones: the defaults)
            s.setdefault(k, spill_dir_default() if kind == "path" else default)
        unknown = set(changes) - set(self.KEYS)
        if unknown:
            raise ValueError(f"unknown setting {sorted(unknown)[0]!r} (settings: {', '.join(self.KEYS)})")
        if s["weights"] not in self.weights_list():
            raise ValueError(f"weights: one of {', '.join(self.weights_list())}")
        if self.tokenizer and s["weights"] != getattr(self, "settings", s)["weights"]:
            why = self.weights_choices()[s["weights"]]
            if why and "tokenizer" in why:
                raise ValueError(f"weights: {why}")
        for k, values in CHOICES.items():
            if s[k] not in values:
                raise ValueError(f"{k}: one of {', '.join(values)}")
        for k, lo, hi in (("max_seqs", 1, MAX_SEQS), ("max_tokens", 2048, MAX_CONTEXT)):
            v = s[k]
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"{k}: an integer from {lo} to {hi}")
        b = s["vram_reserve_gib"]
        if b is not None:
            if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or b < 0:
                raise ValueError("vram_reserve_gib: null (the default, 1.5) or a number of GiB, 0 or more")
            s["vram_reserve_gib"] = round(float(b), 2)
            if check_budget and device is not None and s["vram_reserve_gib"] * GiB > device[1]:
                raise ValueError(f"vram_reserve_gib: at most the device's memory, {device[1] / GiB:.2f} GiB")
        for k, (_, _, kind) in TIER_SETTINGS.items():
            v = s[k]
            if kind == "num":
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise ValueError(f"{k}: a number, 0 or more")
                s[k] = tier_number(v)
                if k in TIER_INTEGER:
                    s[k] = int(s[k])  # (a count: a fraction is rounded down)
            elif kind == "flag":
                if not isinstance(v, bool):
                    raise ValueError(f"{k}: true or false")
            elif not isinstance(v, str) or "\0" in v or len(v) > 4096 or v.strip() != v:
                raise ValueError(f"{k}: a folder path, without spaces around it (empty: no SSD tier)")
            elif check_budget and v and os.path.exists(v) and not os.path.isdir(v):  # (not for the server's own start)
                raise ValueError(f"{k}: {v!r} is a file, not a folder")
        return s

    def config(self, s):
        """The engine's configuration (worker.run) for settings s; FileNotFoundError when weights are missing."""
        path = model_dir(s["weights"])
        draft = snapshot(*DRAFTS[s["spec"]]) if s["spec"] in DRAFTS else None
        eos = json.loads((path / "generation_config.json").read_text(encoding="utf-8"))["eos_token_id"]
        env = {"VISION": s["vision"], "MAX_SEQS": str(s["max_seqs"]),
               "VRAM_RESERVE_GIB": None if s["vram_reserve_gib"] is None else f"{s['vram_reserve_gib']:g}",
               # (a legacy VRAM_BUDGET_GIB inherited from settings.ini would cap the budget unseen: the reserve rules)
               "VRAM_BUDGET_GIB": None,
               # (the tiered cache: the engine and worker.TierPolicy read these when the engine process loads)
               **{TIER_SETTINGS[k][0]: tier_env_value(k, s[k]) for k in TIER_SETTINGS},
               "Q27_LM_HEAD_CKPT": str(snapshot(*TARGET)) if s["weights"] == "quasar" else LM_HEAD_ENV}
        return dict(self.base, model=str(path), weights=s["weights"], draft=str(draft) if draft else None,
                    max_tokens=s["max_tokens"], kv_dtype=s["kv_dtype"], max_seqs=s["max_seqs"],
                    eos=eos if isinstance(eos, list) else [eos], env=env, settings=dict(s))

    # ------------------------------------------------------------ plans

    def plan_many(self, settings_list, device):
        """The plans of several settings on a device with (free after the CUDA context, total) bytes: dicts of
        max_tokens (context per request), kv_tokens (the KV pool, shared by the requests), budget, device (bytes of
        the device pools), fits and pools, or {"error"}."""
        free, total = device
        out, jobs = [None] * len(settings_list), {}
        with self.lock:
            for i, s in enumerate(settings_list):
                try:
                    cfg = self.config(s)
                except FileNotFoundError as e:
                    out[i] = {"error": str(e)}
                    continue
                job = {"model": cfg["model"], "draft": cfg["draft"], "max_tokens": s["max_tokens"],
                       "prefill_chunk": cfg["prefill_chunk"], "kv_dtype": s["kv_dtype"],
                       "env": {k: v for k, v in cfg["env"].items() if k not in TIER_ENV},  # (no pool depends on them)
                       "free": free, "total": total}
                key = json.dumps(job, sort_keys=True)
                if key in self.plans_cache:
                    out[i] = self.plans_cache[key]
                else:
                    jobs.setdefault(key, (job, []))[1].append(i)
            if jobs:
                results = self._run([job for job, _ in jobs.values()])
                if len(self.plans_cache) > 4096:
                    self.plans_cache.clear()
                for (key, (_, idx)), r in zip(jobs.items(), results):
                    self.plans_cache[key] = r
                    for i in idx:
                        out[i] = r
        return out

    def plans(self, s, device):
        """The plan of settings s and, for each setting, the plans with one value changed (the page's choices)."""
        variants = {"weights": self.weights_list(), **CHOICES, **VARIANTS}
        listed = [s]
        for k, values in variants.items():
            listed += [dict(s, **{k: v}) for v in values]
        res = self.plan_many(listed, device)
        out, i = {"settings": s, "plan": res[0], "variants": {}}, 1
        for k, values in variants.items():
            out["variants"][k] = [{"value": v, "plan": r} for v, r in zip(values, res[i:i + len(values)])]
            i += len(values)
        return out

    def plan_answer(self, changes, device, source):
        """What a page or the tray app shows for changes to the next start's settings on this device: POST
        /admin/engine/plan, and frontend/plan.py without a server."""
        s = self.validate(changes, device=device, check_budget=False)
        return dict(self.plans(s, device), device={"free": device[0], "total": device[1], "source": source},
                    weights=self.weights_choices(), at_start=self.at_start)

    def _run(self, jobs):
        if self.planner is None or not self.planner[0].is_alive():
            ctx = mp.get_context("spawn")
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=planner, args=(child, self.base["engine_python"]), daemon=True, name="q27-planner")
            proc.start()
            child.close()
            self.planner = (proc, parent)
        try:
            self.planner[1].send(jobs)
            return self.planner[1].recv()
        except (EOFError, OSError) as e:
            self.planner = None
            return [{"error": f"the planner process failed: {e}"}] * len(jobs)


def planner(conn, engine_python):
    """Process entry point (multiprocessing, spawn): answers lists of plan jobs with lists of plans."""
    import sys

    sys.path.insert(0, engine_python)
    from q27lib import Engine

    devnull = os.open(os.devnull, os.O_WRONLY)  # (the engine logs every plan; errors come back in the answers)
    os.dup2(devnull, 2)

    base = dict(os.environ)
    while True:
        try:
            jobs = conn.recv()
        except EOFError:
            return
        out = []
        for job in jobs:
            os.environ.clear()
            os.environ.update(base)
            for k, v in job["env"].items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            try:
                e = Engine.plan(job["model"], job["free"], job["total"], max_tokens=job["max_tokens"],
                                prefill_chunk=job["prefill_chunk"], kv_dtype=job["kv_dtype"], draft=job["draft"])
            except RuntimeError as err:
                out.append({"error": str(err).removeprefix("q27_plan: ")})
                continue
            try:
                pools = e.memory_report()["pools"]
                c, d = e.cache_stats(), e.device_memory()
                dev = {p["name"]: p["bytes"] for p in pools if not p["host"]}
                out.append({"max_tokens": e.max_tokens, "kv_tokens": c["pool_pages"] * 64, "budget": d["budget"],
                            "device": sum(dev.values()), "fits": sum(dev.values()) <= d["budget"], "pools": dev,
                            "host": sum(p["bytes"] for p in pools if p["host"])})
            finally:
                e.close()
        conn.send(out)
