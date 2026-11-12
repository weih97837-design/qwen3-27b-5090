"""ctypes binding of libq27.so (q27.dll on Windows): the engine (Engine) and the kernel wrappers used by the tests (call)."""

import array
import ctypes
import os

# Windows: the CMake build in engine/build (cmake -S engine -B engine/build); Linux: the dev image's /build.
_DEFAULT_LIB = (os.path.join(os.path.dirname(__file__), "../build/q27.dll") if os.name == "nt"
                else "/build/engine/libq27.so")
LIB_PATH = os.environ.get("Q27_LIB", _DEFAULT_LIB)
# Draft vocabulary of the DFlash2 candidates (tools/draft_vocab.py); Q27_DRAFT_VOCAB=none: the full vocabulary.
DRAFT_VOCAB = os.path.join(os.path.dirname(__file__), "../../bench/draft-vocab/vocab-128k.txt")

P, I, I64, F, SZ = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_size_t
SIGNATURES = {
    "q27_create": ([ctypes.c_char_p, I, I, I, ctypes.c_char_p], P),
    "q27_plan": ([ctypes.c_char_p, I, I, I, ctypes.c_char_p, I64, I64], P),
    "q27_spec_begin": ([P, P, I, F, I, F, ctypes.c_uint64, F, I, F, F, F, I, P, I, I, I, I], I),
    "q27_spec_draft": ([P, P], I),
    "q27_spec_set_masked": ([P, I], I),
    "q27_spec_draft_lanes": ([P, P, I, P], I),
    "q27_mask_rows": ([P, P, P], P),
    "q27_spec_step": ([P, P], I),
    "q27_spec_step2": ([P, P, I], I),
    "q27_prefill": ([P, P, I, I, I, P], I),
    "q27_park_slots": ([P], I),
    "q27_seq_park": ([P, P], I),
    "q27_seq_unpark": ([P, I], I),
    "q27_seq_swap": ([P, I], I),
    "q27_seq_drop": ([P, I], I),
    "q27_add_checkpoint_at": ([P, I, I, P], I),
    "q27_share_mark": ([P, I], I),
    "q27_seq_suspend": ([P, I], I),
    "q27_has_draft": ([P], I),
    "q27_max_lanes": ([P], I),
    "q27_max_tokens": ([P], I),
    "q27_spec_step_lanes": ([P, P, I, P, P], I),
    "q27_spec_stats": ([P, P], I),
    "q27_prefix_begin": ([P, P, I, P, I, I, P], I),
    "q27_checkpoint": ([P], I),
    "q27_cache_stats": ([P, P], I),
    "q27_cache_grow": ([P, I, P], I),
    "q27_spill_list": ([P, ctypes.c_char_p, I], I),
    "q27_spill_list_ex": ([P, ctypes.c_char_p, I], I),
    "q27_spill_disk_usage": ([P, P, I], I),
    "q27_spill_sweep": ([P, P], I),
    "q27_spill_export": ([P, ctypes.c_uint64, ctypes.c_char_p], I),
    "q27_spill_drop": ([P, ctypes.c_uint64], I),
    "q27_spill_import": ([P, ctypes.c_char_p, P], I),
    "q27_spill_stats": ([P, P, I], I),
    "q27_spill_backup_one": ([P, P], I),
    "q27_spill_backup_next": ([P, I, P], I),
    "q27_spill_grow_pinned": ([P, I, P], I),
    "q27_spill_match": ([P, P, I, I, P, P], I),
    "q27_spill_restore_begin": ([P, ctypes.c_uint64, P], I),
    "q27_spill_restore_poll": ([P, P, P, I, P], I),
    "q27_spill_release": ([P, ctypes.c_uint64, P], I),
    "q27_spill_sync_lookup": ([P, I], I),
    "q27_memory_report": ([P, ctypes.c_char_p, I], I),
    "q27_device_memory": ([P, P], I),
    "q27_has_vision": ([P], I),
    "q27_mm_set": ([P, P, I], I),
    "q27_mm_encode": ([P, I, P], I),
    "q27_mm_encode_rgb": ([P, I, P, P], I),
    "q27_mm_embeds": ([P, I, P], I),
    "q27_mm_put": ([P, I, P], I),
    "q27_debug_kv": ([P, I, I, I, P], I),
    "q27_debug_ssm": ([P, I, P], I),
    "q27t_skinny_bf16": ([I, I, I, P, P, P, P, SZ], I),
    "q27t_gemm_bf16": ([I, I, I, P, P, P], I),
    "q27t_vision_pixels": ([P, P, P, I, I, P], I),
    "q27t_gemm_bf16_ex": ([I, I, I, P, P, F, P, F, P], I),
    "q27t_gdn_verify": ([P, I, I, P, P, P, P, P, P] + [P] * 8 + [P, P, P, P, P, P, I, P, F, P, P, F, I], I),
    "q27t_topk_rows": ([P, I, I, I, P, P], I),
    "q27_destroy": ([P], None),
    "q27_reset": ([P], I),
    "q27_pos": ([P], I),
    "q27_forward_topk": ([P, P, I, I, P, P], I),
    "q27_forward_sample": ([P, P, I, F, I, F, F, F, F, F, I, P, I, I], I),
    "q27_bias_set": ([P, P, P, I, I], I),
    "q27_logprobs": ([P, I, P, P, P, P], I),
    "q27_penalty_init": ([P, P, I], I),
    "q27_profile_dump": ([P], I),
    "q27_configure": ([P], I),
    "q27_last_error": ([], ctypes.c_char_p),
    "q27t_last_us": ([], F),
    "q27t_rmsnorm": ([P, P, I, I, F, I, P], I),
    "q27t_embed_gather": ([P, P, I, I, P], I),
    "q27t_quant_fp4": ([P, I, I, F, P, P], I),
    "q27t_quant_fp8": ([P, I64, F, P], I),
    "q27t_silu_mul": ([P, P, I64, P], I),
    "q27t_linear_bf16": ([P, P, I, I, I, P], I),
    "q27t_swizzle_sf": ([P, I, I, P], I),
    "q27t_swizzle_sf_glu": ([P, I, I, P, I], I),
    "q27t_sf_bytes": ([I64, I64], I64),
    "q27t_gemm_fp4": ([I, I, I, P, P, P, P, F, P, F, P], I),
    "q27t_gemm_fp8": ([I, I, I, P, P, F, P, F, P], I),
    "q27t_gemm_fp4_glu": ([I, I, I, P, P, P, P, F, F, F, P, P], I),
    "q27t_skinny_fp4": ([I, I, I, P, P, P, P, F, P, F, P, P, SZ], I),
    "q27t_skinny_fp8": ([I, I, I, P, P, F, P, F, P, P, SZ], I),
    "q27t_gemm_fp8_scaled": ([I, I, I, P, P, P, P, P, F, P], I),
    "q27t_skinny_fp8_scaled": ([I, I, I, P, P, P, P, P, F, P, P, SZ], I),
    "q27t_quant_fp8_rows": ([P, I64, I, I, P, P], I),
    "q27t_silu_mul_quant_fp8_rows": ([P, P, I64, I, I, P, P], I),
    "q27t_rmsnorm_quant_rows": ([P, P, I, F, P, P, P], I),
    "q27t_dequant_fp8_rows": ([P, P, I, I, P], I),
    "q27t_bench_gemm": ([I, I, I, I, I, P, P, P, P, I64, I64, I, I, P, P, SZ, P], I),
    "q27t_gdn_prep": ([P, P, P, P, P, P, P, I, P, P, P, P, P], I),
    "q27t_gdn_recurrent": ([P, P, P, P, P, I, P, P], I),
    "q27t_gdn_chunked": ([P, P, P, P, P, I, I, P, P, P], I),
    "q27t_gdn_chunk_ws_bytes": ([I], SZ),
    "q27t_gdn_gated_norm": ([P, P, P, I, F, P, P, F], I),
    "q27t_gdn_decode_prep": ([P, P, P, P, P, P, P, P, P, P, P], I),
    "q27t_gdn_decode": ([P, P, P, P, P, P, P, P, P, P, F, P, P, F], I),
    "q27t_rmsnorm_quant": ([P, P, I, F, P, I, F, P, P], I),
    "q27t_silu_mul_quant_fp4": ([P, P, I64, I, I, F, P, P], I),
    "q27t_silu_mul_quant_fp4_bf16": ([P, P, I64, I, I, F, P, P], I),
    "q27t_silu_mul_quant_fp4_glu": ([P, I, I, F, F, F, P, P], I),
    "q27t_silu_mul_quant_fp4_glu_bf16": ([P, I, I, F, F, F, P, P], I),
    "q27t_gemm_fp4_bf16": ([I, I, I, P, P, P, P, F, P], I),
    "q27t_linear_bf16_pair": ([P, P, P, I, I, I, P, P], I),
    "q27t_attn_prep": ([P, P, P, P, P, I, I, P, P, I, P, P], I),
    "q27t_attention": ([P, P, I, I, P, P, I, P, P, P, SZ], I),
    "q27t_attention_decode": ([P, P, I, I, P, P, I, P, P, P, SZ], I),
    "q27t_logsoftmax_topk": ([P, I, I, I, P, P], I),
    "q27t_sample": ([P, I, I, F, I, F, P, P, F, P, F, F], I),
}

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB_PATH)
        for name, (args, res) in SIGNATURES.items():
            try:
                fn = getattr(_lib, name)
            except AttributeError:  # an older build (Q27_LIB) without this entry point
                continue
            fn.argtypes, fn.restype = args, res
    return _lib


def _arg(x):
    if hasattr(x, "data_ptr"):
        return x.data_ptr()
    return x


def call(name, *args):
    """Calls a kernel wrapper; torch tensors are passed by data pointer. Raises on a CUDA / check error."""
    rc = getattr(lib(), name)(*map(_arg, args))
    if rc != 0:
        raise RuntimeError(f"{name}: {lib().q27_last_error().decode()}")


class KvFull(RuntimeError):
    """The KV pool cannot hold the running sequence's next positions (C API -2). Nothing ran for the failed
    part: free pages (park and suspend / drop another sequence) and retry. Engine.pos tells how far a
    prompt got."""


class Engine:
    """The engine: one running sequence, others parked (MAX_SEQS, Phase 6). Token ids are plain Python ints."""

    KV_TYPES = {"bf16": 0, "fp8": 1, "nvfp4": 2}

    def __init__(self, model_dir, max_tokens=32768, prefill_chunk=2048, kv_fp8=False, draft=None, kv_dtype=None):
        """kv_dtype: "bf16" | "fp8" | "nvfp4" (default: kv_fp8 picks fp8 or bf16)."""
        self._create("q27_create", model_dir, max_tokens, prefill_chunk, kv_fp8, draft, kv_dtype)

    @classmethod
    def plan(cls, model_dir, free, total, max_tokens=32768, prefill_chunk=2048, kv_dtype="bf16", draft=None):
        """The memory plan of Engine(...) (with the same environment) on a device with `free` of `total` bytes free
        after the CUDA context, without CUDA: nothing is allocated or loaded. Its memory_report(), max_tokens,
        cache_stats() and device_memory() answer; close() it."""
        self = cls.__new__(cls)
        self._create("q27_plan", model_dir, max_tokens, prefill_chunk, False, draft, kv_dtype, int(free), int(total))
        return self

    def _create(self, fn, model_dir, max_tokens, prefill_chunk, kv_fp8, draft, kv_dtype, *extra):
        self.max_tokens = max_tokens
        self._masks = None
        if draft and "Q27_DRAFT_VOCAB" not in os.environ and os.path.exists(DRAFT_VOCAB):
            os.environ["Q27_DRAFT_VOCAB"] = os.path.abspath(DRAFT_VOCAB)
        if os.environ.get("Q27_DRAFT_VOCAB") == "none":
            os.environ["Q27_DRAFT_VOCAB"] = ""
        kv = self.KV_TYPES[kv_dtype] if kv_dtype else int(bool(kv_fp8))
        self.h = getattr(lib(), fn)(str(model_dir).encode(), max_tokens, prefill_chunk, kv,
                                    str(draft).encode() if draft else None, *extra)
        if not self.h:
            raise RuntimeError(f"{fn}: {lib().q27_last_error().decode()}")
        self.max_tokens = lib().q27_max_tokens(self.h)  # (MAX_TOKENS_FIT=1 lowers it to what the VRAM fits)

    def close(self):
        if self.h:
            lib().q27_destroy(self.h)
            self.h = None

    def _check(self, rc, what):
        if rc == -2:
            raise KvFull(f"{what}: {lib().q27_last_error().decode()}")
        if rc != 0:
            raise RuntimeError(f"{what}: {lib().q27_last_error().decode()}")

    def reset(self):
        self._check(lib().q27_reset(self.h), "q27_reset")

    @property
    def pos(self):
        return lib().q27_pos(self.h)

    def forward_topk(self, ids, k=20):
        """Runs ids; returns per position a list of (token, logprob) for the k most likely next tokens."""
        n = len(ids)
        src = array.array("i", ids)
        out_i, out_l = array.array("i", bytes(4 * n * k)), array.array("f", bytes(4 * n * k))
        self._check(lib().q27_forward_topk(self.h, src.buffer_info()[0], n, k, out_i.buffer_info()[0],
                                           out_l.buffer_info()[0]), "q27_forward_topk")
        return [list(zip(out_i[p * k:(p + 1) * k], out_l[p * k:(p + 1) * k])) for p in range(n)]

    def configure(self, **env):
        """Sets / clears (None) Q27_* environment switches and makes the engine re-read them (drops the
        captured decode graph). Compile-time switches still need another build."""
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        self._check(lib().q27_configure(self.h), "q27_configure")

    def profile_dump(self):
        """Prints the per-op GPU time since the last dump to stderr (needs Q27_PROFILE=1)."""
        self._check(lib().q27_profile_dump(self.h), "q27_profile_dump")

    def forward_sample(self, ids, temperature=0.0, top_k=20, top_p=1.0, uniform=0.5, min_p=0.0, presence=0.0,
                       frequency=0.0, logprobs=-1, masked=False, biased=False):
        """Runs ids and returns the sampled next token (temperature <= 0: greedy). presence / frequency:
        OpenAI's penalties by the sequence's output counts (penalty_init). logprobs >= 0: the token's logprobs
        with that many top entries are kept for logprobs(). masked: only the tokens row 0 of the grammar masks
        allows (mask_row). biased: the sequence's logit bias (bias_set) is added to the logits first."""
        src = array.array("i", ids)
        out = ctypes.c_int32()
        self._check(lib().q27_forward_sample(self.h, src.buffer_info()[0], len(ids), temperature, top_k, top_p,
                                             uniform, min_p, presence, frequency, logprobs, ctypes.byref(out),
                                             int(masked), int(biased)), "q27_forward_sample")
        return out.value

    def mask_row(self, i):
        """Grammar masks (structured output): (address, bytes) of row i of the pinned bitmasks the engine reads (bit
        t % 32 of 32-bit word t / 32: token t allowed). Row 8 z + j masks verify row j of lane z of a step (z = 0
        alone); forward_sample and spec_begin use row 0. Write them before the call that reads them."""
        if self._masks is None:
            words, rows = ctypes.c_int(), ctypes.c_int()
            base = lib().q27_mask_rows(self.h, ctypes.byref(words), ctypes.byref(rows))
            self._masks = (base, 4 * words.value, rows.value)
        base, size, rows = self._masks
        assert 0 <= i < rows
        return base + i * size, size

    def logprobs(self, lane=-1):
        """Logprobs of the tokens the last forward_sample / spec_begin / spec_step returned (lane -1), or of lane
        `lane` of the last spec_step_lanes: per token (logprob, [(id, logprob)] of the most likely entries, as many
        as the sampling asked for), the raw distribution (before the temperature, penalties and cuts). [] when the
        sampling asked for none."""
        top = ctypes.c_int()
        lp, ids, tlp = (ctypes.c_float * 8)(), (ctypes.c_int32 * 160)(), (ctypes.c_float * 160)()
        n = lib().q27_logprobs(self.h, lane, ctypes.byref(top), lp, ids, tlp)
        k = top.value
        return [(lp[i], list(zip(ids[20 * i:20 * i + k], tlp[20 * i:20 * i + k]))) for i in range(n)]

    def penalty_init(self, ids):
        """Sets the running sequence's penalty counts to its output so far (ids) before its first sampled
        token; forward_sample and the speculative steps count the tokens they output."""
        src = array.array("i", ids)
        self._check(lib().q27_penalty_init(self.h, src.buffer_info()[0], len(ids)), "q27_penalty_init")

    def bias_set(self, bias, eos_before=0):
        """Sets the running sequence's logit bias: {token id: value added to its logit} (at most 300 entries; {}:
        none), and min_tokens: the end-of-sequence ids are banned for the tokens at sequence positions below
        eos_before (0: none; the prompt's length + min_tokens). Used by the calls with biased=True; a parked sequence
        keeps its own."""
        ids = array.array("i", list(bias))
        vals = array.array("f", [float(v) for v in bias.values()])
        self._check(lib().q27_bias_set(self.h, ids.buffer_info()[0] if ids else None,
                                       vals.buffer_info()[0] if vals else None, len(ids), eos_before), "q27_bias_set")

    @property
    def has_draft(self):
        return bool(lib().q27_has_draft(self.h))

    def spec_begin(self, ids, temperature=0.0, top_k=20, top_p=1.0, seed=0, uniform=0.5, max_new=0, min_p=0.0,
                   presence=0.0, frequency=0.0, logprobs=-1, masked=False, until=None, biased=False,
                   ignore_eos=False):
        """Speculative decoding: runs the prompt and returns the first token (also fills the draft context).
        masked: grammar masks (mask_row; row 0 for the first token); each step is then spec_draft + spec_step.
        until: a token that ends a step as its last one (the end of the reasoning, before a forced output or a
        grammar; spec_set_masked clears it). biased: the sequence's logit bias (bias_set) on every step. ignore_eos:
        an end-of-sequence id does not end a step (it is fed like any token)."""
        src = array.array("i", ids)
        out = ctypes.c_int32()
        self._check(lib().q27_spec_begin(self.h, src.buffer_info()[0], len(ids), temperature, top_k, top_p,
                                         seed & (2**64 - 1), uniform, max_new, min_p, presence, frequency,
                                         logprobs, ctypes.byref(out), int(masked), -1 if until is None else until,
                                         int(biased), int(ignore_eos)),
                    "q27_spec_begin")
        return out.value

    def spec_draft(self):
        """A masked sequence's step, first part: drafts it and returns its 8 verify tokens (the last token, then the
        7 drafted ones). Write mask rows 0..7 (the grammar after each row's tokens), then spec_step verifies."""
        out = (ctypes.c_int32 * 8)()
        self._check(lib().q27_spec_draft(self.h, out), "q27_spec_draft")
        return list(out)

    def spec_set_masked(self, area=None):
        """Grammar masks for a decoding sequence (None: the running one, else a parking area) from its next step on,
        as spec_begin(masked=True) would have set (a step launched ahead is dropped)."""
        self._check(lib().q27_spec_set_masked(self.h, -1 if area is None else area), "q27_spec_set_masked")

    def spec_draft_lanes(self, lanes):
        """spec_draft for a batched step with masked lanes (spec_step_lanes follows): per lane its 8 verify tokens."""
        n = len(lanes)
        src = (ctypes.c_int32 * n)(*[-1 if a is None else a for a in lanes])
        out = (ctypes.c_int32 * (8 * n))()
        self._check(lib().q27_spec_draft_lanes(self.h, src, n, out), "q27_spec_draft_lanes")
        return [list(out[8 * z:8 * z + 8]) for z in range(n)]

    def spec_step(self, ahead=True):
        """One draft + verify step; returns the new tokens (1..8). ahead=False: the next step is not launched
        before returning (the caller switches to another sequence next)."""
        out = (ctypes.c_int32 * 8)()
        n = lib().q27_spec_step2(self.h, out, int(ahead)) if not ahead else lib().q27_spec_step(self.h, out)
        if n < 0:
            self._check(n, "q27_spec_step")
        return list(out[:n])

    @property
    def max_lanes(self):
        """Sequences a batched step can run (MAX_SEQS with a draft model; 0 / 1: no batched steps)."""
        return lib().q27_max_lanes(self.h)

    def spec_step_lanes(self, lanes):
        """One batched speculative step (P6-10) of the decoding sequences `lanes` (None: the running one, else
        a parking area): the same tokens as a spec_step of each. Returns their new tokens (lists of 1..8)."""
        n = len(lanes)
        src = (ctypes.c_int32 * n)(*[-1 if a is None else a for a in lanes])
        out, counts = (ctypes.c_int32 * (8 * n))(), (ctypes.c_int32 * n)()
        self._check(lib().q27_spec_step_lanes(self.h, src, n, out, counts), "q27_spec_step_lanes")
        return [list(out[8 * z:8 * z + counts[z]]) for z in range(n)]

    def prefill(self, ids, budget, spec=False):
        """Prompt chunks of ids (the rest of the prompt) while more than one prefill chunk is left, at least
        `budget` tokens; returns the tokens run. The rest goes to forward_sample / spec_begin (spec: the latter)."""
        src = array.array("i", ids)
        out = ctypes.c_int()
        self._check(lib().q27_prefill(self.h, src.buffer_info()[0], len(ids), budget, int(spec), ctypes.byref(out)),
                    "q27_prefill")
        return out.value

    @property
    def park_slots(self):
        """Parking areas for other sequences (MAX_SEQS - 1)."""
        return lib().q27_park_slots(self.h)

    def seq_park(self):
        """Parks the running sequence; returns its area (the engine then has no sequence)."""
        out = ctypes.c_int()
        self._check(lib().q27_seq_park(self.h, ctypes.byref(out)), "q27_seq_park")
        return out.value

    def seq_unpark(self, area):
        """Drops the running sequence (if any) and resumes the one parked in `area`."""
        self._check(lib().q27_seq_unpark(self.h, area), "q27_seq_unpark")

    def seq_swap(self, area):
        """Parks the running sequence in `area` and resumes the one that was there."""
        self._check(lib().q27_seq_swap(self.h, area), "q27_seq_swap")

    def add_checkpoint_at(self, area, pos):
        """Plans one more checkpoint at pos (rounded down to 64) in a sequence that is already prefilling: the running one
        (area -1) or the one parked in `area`. False when the engine cannot (too near its position or another
        checkpoint, no prefix cache): nothing is saved there then."""
        if not hasattr(lib(), "q27_add_checkpoint_at"):  # (an older library)
            return False
        ok = ctypes.c_int()
        self._check(lib().q27_add_checkpoint_at(self.h, area, pos, ctypes.byref(ok)), "q27_add_checkpoint_at")
        return bool(ok.value)

    def share_mark(self, pos):
        """Tells the engine a waiting request shares the checkpoint at pos (rounded down to 64): it is kept from the
        covered-first eviction until that request has used it. A no-op with an older library."""
        if hasattr(lib(), "q27_share_mark"):
            self._check(lib().q27_share_mark(self.h, pos), "q27_share_mark")

    def seq_drop(self, area):
        """Gives the pages of the sequence parked in `area` back."""
        self._check(lib().q27_seq_drop(self.h, area), "q27_seq_drop")

    def seq_suspend(self, area):
        """Saves the sequence parked in `area` as a prefix cache checkpoint, then drops it."""
        self._check(lib().q27_seq_suspend(self.h, area), "q27_seq_suspend")

    def prefix_begin(self, ids, at=(), spec=False):
        """Prefix cache (STATE_HOST_CACHE > 0): resets the sequence and restores the deepest cached prefix of
        ids; returns its length. Run ids[cached:] next (forward_sample, or spec_begin when spec); that prefill
        saves checkpoints at the positions in `at` (and at CHECKPOINT_INTERVAL multiples)."""
        src, pts = array.array("i", ids), array.array("i", sorted(at) or [0])
        out = ctypes.c_int()
        self._check(lib().q27_prefix_begin(self.h, src.buffer_info()[0], len(ids), pts.buffer_info()[0], len(at),
                                           int(spec), ctypes.byref(out)), "q27_prefix_begin")
        return out.value

    def checkpoint(self):
        """Saves a checkpoint of the sequence so far (e.g. after a generation)."""
        self._check(lib().q27_checkpoint(self.h), "q27_checkpoint")

    def cache_grow(self, n=1):
        """Allocates up to n more pinned state slots of the prefix cache (not allocated at startup; about 80 ms each
        on WSL2); returns how many (0: all are)."""
        out = ctypes.c_int()
        self._check(lib().q27_cache_grow(self.h, n, ctypes.byref(out)), "q27_cache_grow")
        return out.value

    def cache_stats(self):
        out = (ctypes.c_int64 * 12)()
        self._check(lib().q27_cache_stats(self.h, out), "q27_cache_stats")
        keys = ("lookups", "hits", "prompt_tokens", "cached_tokens", "saved", "evicted", "checkpoints", "free_pages",
                "checkpoint_only_pages", "pool_pages", "park_pages", "slots_allocated")
        return dict(zip(keys, out))

    # ---- tiered cache (SPILL_RAM_GIB > 0; frontend/tiercache.py)

    @property
    def has_spill_dedup(self):
        """The library has the SSD dedup layout (manifests + chunk files): q27_spill_list_ex / q27_spill_disk_usage / q27_spill_sweep.
        An older library keeps one flat file per entry."""
        l = lib()
        return all(hasattr(l, n) for n in ("q27_spill_list_ex", "q27_spill_disk_usage", "q27_spill_sweep"))

    def spill_list(self):
        """(ram_bytes is the entry's LOGICAL size: with RAM dedup its chunks may be shared, see spill_stats for the physical use.)
        [{"id", "pos", "ram_bytes", "on_disk", "last_use", "backup", "busy", "path"}] of the checkpoints the KV pool gave
        up (backup: the entry is an idle backup of a checkpoint still in the pool; busy: it is being restored in the
        background, SPILL_ASYNC_PROMOTE, and can be neither written out nor dropped). With the SSD dedup library each entry also has
        "logical_bytes" (kv + state, wherever it is), "unique_disk_bytes" (what dropping it frees on the SSD: its manifest and the chunk
        files only it lists) and "new_disk_bytes" (what exporting it writes now)."""
        ex = self.has_spill_dedup
        fn = lib().q27_spill_list_ex if ex else lib().q27_spill_list
        n = 1 << 16
        while True:
            buf = ctypes.create_string_buffer(n)
            size = fn(self.h, buf, n)
            if size < 0:
                self._check(1, "q27_spill_list")
            if size < n:
                break
            n = size + 1
        out = []
        for line in buf.value.decode("utf-8", "replace").split("\n"):  # (not splitlines: a path may hold \r etc.)
            if not line:
                continue
            f = line.split("\t", 9 if ex else 6)
            e = {"id": int(f[0]), "pos": int(f[1]), "ram_bytes": int(f[2]), "on_disk": f[3] == "1",
                 "last_use": int(f[4]), "backup": f[5] == "1", "busy": f[5] == "2"}  # (column 5: 0 / 1 backup / 2 busy)
            if ex:
                e.update(logical_bytes=int(f[6]), unique_disk_bytes=int(f[7]), new_disk_bytes=int(f[8]),
                         path=f[9] if len(f) > 9 else "")
            else:
                e["path"] = f[6] if len(f) > 6 else ""
            out.append(e)
        return out

    def spill_disk_usage(self):
        """{"physical_bytes", "logical_bytes", "chunk_files", "manifests", "pending_files"} of the SSD tier (each chunk file once;
        pending_files: chunk files whose deletion failed and is retried, their bytes are in physical_bytes), or None with a
        library that has no SSD dedup (flat files: the caller sums the file sizes)."""
        if not self.has_spill_dedup:
            return None
        out = (ctypes.c_int64 * 5)()  # (an older library fills the first 4: pending_files stays 0)
        self._check(lib().q27_spill_disk_usage(self.h, out, 5), "q27_spill_disk_usage")
        return {"physical_bytes": out[0], "logical_bytes": out[1], "chunk_files": out[2], "manifests": out[3],
                "pending_files": out[4]}

    def spill_sweep(self):
        """Deletes leftover .tmp files and chunk files no manifest lists (call it after every manifest was imported), and retries the
        deletions that failed earlier (pending_files). Returns how many files went."""
        if not self.has_spill_dedup:
            return 0
        out = ctypes.c_int()
        self._check(lib().q27_spill_sweep(self.h, ctypes.byref(out)), "q27_spill_sweep")
        return out.value

    def spill_backup_one(self, victims=0):
        """Copies a checkpoint of the pool to host memory; it stays in the pool. victims > 0: one of the next `victims`
        the pool would evict (the idle backup; its copies also stay within half the RAM cap; an older library copies the
        least recently used one), 0: the least recently used one without a copy (the stop flush). True when one was
        copied, False when none needed it (call again until False while idle)."""
        out = ctypes.c_int()
        if victims > 0 and hasattr(lib(), "q27_spill_backup_next"):
            self._check(lib().q27_spill_backup_next(self.h, victims, ctypes.byref(out)), "q27_spill_backup_next")
        else:
            self._check(lib().q27_spill_backup_one(self.h, ctypes.byref(out)), "q27_spill_backup_one")
        return bool(out.value)

    def spill_grow_pinned(self, n=1):
        """SPILL_PINNED_GIB: pins up to n more slabs of host memory for spill entries (slow: call only while idle);
        returns how many were added, 0 when the pool is full or off."""
        out = ctypes.c_int()
        self._check(lib().q27_spill_grow_pinned(self.h, n, ctypes.byref(out)), "q27_spill_grow_pinned")
        return out.value

    def spill_export(self, entry_id, path):
        """Writes an entry that is in memory to `path` and frees the memory (the lookup still finds it)."""
        self._check(lib().q27_spill_export(self.h, entry_id, os.fsencode(path)), "q27_spill_export")

    def spill_drop(self, entry_id):
        """Forgets an entry and deletes its file."""
        self._check(lib().q27_spill_drop(self.h, entry_id), "q27_spill_drop")

    def spill_import(self, path):
        """Indexes a file an earlier run wrote; 0 when it is not one of this model and KV type."""
        out = ctypes.c_uint64()
        self._check(lib().q27_spill_import(self.h, os.fsencode(path), ctypes.byref(out)), "q27_spill_import")
        return out.value

    def spill_stats(self):
        out = (ctypes.c_int64 * 26)(*([-1] * 26))  # (an older engine writes the first 20 and leaves the rest -1)
        self._check(lib().q27_spill_stats(self.h, out, len(out)), "q27_spill_stats")
        keys = ("entries", "in_ram", "on_disk", "ram_bytes", "spilled", "restored", "written", "dropped", "subsumed",
                "failed", "backed_up", "pinned_bytes", "pinned_used", "pin_hits", "pin_misses",
                # SPILL_ASYNC_PROMOTE: restores in progress, finished / failed async restores, their total ms and bytes
                "restoring", "restored_async", "restore_failed", "restore_ms_total", "restore_bytes_total",
                # RAM dedup (docs/tiered-cache.md): the entries' logical size, what the RAM tier physically holds (ram_bytes
                # is the same number), shared chunks resident, chunk hits / misses of spilled entries, logical - physical
                "logical_bytes", "physical_bytes", "chunks", "chunk_hits", "chunk_misses", "dedup_saved_bytes")
        # An engine built before RAM dedup never writes the last six: absent, not a misleading 0 (callers use .get()).
        return {k: v for k, v in zip(keys, out) if v != -1}

    # ---- SPILL_ASYNC_PROMOTE (delay-hit scheduling, docs/tiered-cache.md): engine thread only, never block

    @property
    def has_spill_async(self):
        """The library has the background-restore calls (an older build does not). They do nothing unless the engine was
        created with SPILL_ASYNC_PROMOTE=1."""
        return hasattr(lib(), "q27_spill_restore_begin")

    def spill_match(self, ids, spec=False):
        """The best spilled entry deeper than the pool's own match for the prompt `ids`, or None:
        {"id", "pos", "bytes" (kv + state), "on_disk" (only on disk)}. Read-only. A restoring entry is returned too."""
        src, out, found = array.array("i", ids), (ctypes.c_int64 * 4)(), ctypes.c_int()
        self._check(lib().q27_spill_match(self.h, src.buffer_info()[0], len(ids), int(spec), out, ctypes.byref(found)),
                    "q27_spill_match")
        if not found.value:
            return None
        return {"id": out[0], "pos": out[1], "bytes": out[2], "on_disk": bool(out[3])}

    def spill_restore_begin(self, entry_id):
        """Starts restoring a spilled entry into the KV pool without blocking: 1 = its host -> device copy is queued (entry in
        RAM), 2 = a thread is reading its file, 0 = not started (the request then takes prefix_begin's own path)."""
        out = ctypes.c_int()
        self._check(lib().q27_spill_restore_begin(self.h, entry_id, ctypes.byref(out)), "q27_spill_restore_begin")
        return out.value

    def spill_restore_poll(self, cap=64):
        """Advances the restores; returns [(entry_id, ok)] for those that ended: ok True = restored and visible to
        prefix_begin (and held for its request until that runs or spill_release), False = failed (computed instead)."""
        ids, oks, n = (ctypes.c_int64 * cap)(), (ctypes.c_int32 * cap)(), ctypes.c_int()
        self._check(lib().q27_spill_restore_poll(self.h, ids, oks, cap, ctypes.byref(n)), "q27_spill_restore_poll")
        return [(ids[i], bool(oks[i])) for i in range(n.value)]

    def spill_release(self, entry_id):
        """The request that waited for restore `entry_id` gave up (timeout, cancel): its checkpoint is no longer held for it
        (the engine may evict it like any other). True if something was released."""
        out = ctypes.c_int()
        self._check(lib().q27_spill_release(self.h, entry_id, ctypes.byref(out)), "q27_spill_release")
        return bool(out.value)

    def spill_sync_lookup(self, on):
        """False: prefix_begin does not promote a spilled match synchronously (the scheduler chose to compute it)."""
        self._check(lib().q27_spill_sync_lookup(self.h, int(bool(on))), "q27_spill_sync_lookup")

    def memory_report(self):
        """The memory pools: {"pools": [{"name", "host", "bytes", "items": [(what, bytes)]}], "slots": state cache
        slots, "checkpoint_bytes": bytes of one}. Fixed once the engine is created."""
        cap = 1 << 20
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = lib().q27_memory_report(self.h, buf, cap)
            if n < 0:
                raise RuntimeError(f"q27_memory_report: {lib().q27_last_error().decode()}")
            if n < cap:
                break
            cap = n + 1
        pools, out = {}, {"pools": [], "slots": 0, "checkpoint_bytes": 0}
        for line in buf.value.decode().splitlines():
            f = line.split("\t")
            if f[0] == "pool":
                pools[f[1]] = {"name": f[1], "host": f[2] == "1", "bytes": int(f[3]), "items": []}
                out["pools"].append(pools[f[1]])
            elif f[0] == "item":
                pools[f[1]]["items"].append((f[2], int(f[3])))
            elif f[0] == "slots":
                out["slots"], out["checkpoint_bytes"] = int(f[1]), int(f[2])
        return out

    def device_memory(self):
        """{"free", "total"} device bytes now (cudaMemGetInfo), "free_at_start" (before the pools), "budget"."""
        out = (ctypes.c_int64 * 4)()
        self._check(lib().q27_device_memory(self.h, out), "q27_device_memory")
        return dict(zip(("free", "total", "free_at_start", "budget"), out))

    @property
    def has_vision(self):
        return bool(lib().q27_has_vision(self.h))

    def mm_set(self, spans):
        """Images of this request (after reset / prefix_begin): spans [(start, gh, gw)] with the merged grid."""
        flat = array.array("i", [v for sp in spans for v in sp] or [0])
        self._check(lib().q27_mm_set(self.h, flat.buffer_info()[0], len(spans)), "q27_mm_set")

    def mm_encode(self, i, pixel_values):
        """Runs the vision tower on image i: pixel_values [4 * gh * gw, 1536] (float32, numpy or CPU torch)."""
        import numpy as np

        pv = np.ascontiguousarray(np.asarray(pixel_values, dtype=np.float32))
        self._check(lib().q27_mm_encode(self.h, i, pv.ctypes.data), "q27_mm_encode")

    def mm_encode_rgb(self, i, rgb, norm):
        """Runs the vision tower on image i from the resized image: rgb [32 * gh][32 * gw][3] uint8 bytes, norm
        mean[3] + std[3] (the server's image processor; normalized and arranged into patches on the device)."""
        self._check(lib().q27_mm_encode_rgb(self.h, i, rgb, (ctypes.c_float * 6)(*norm)), "q27_mm_encode_rgb")

    def mm_embeds(self, i, rows):
        """Embeddings of image i as a uint16 (BF16 bits) numpy array [rows, 5120]."""
        import numpy as np

        out = np.empty((rows, 5120), dtype=np.uint16)
        self._check(lib().q27_mm_embeds(self.h, i, out.ctypes.data), "q27_mm_embeds")
        return out

    def mm_put(self, i, embeds):
        """Replaces the embeddings of image i (uint16 BF16 bits [rows, 5120]; tests with another vision tower)."""
        import numpy as np

        e = np.ascontiguousarray(np.asarray(embeds, dtype=np.uint16))
        self._check(lib().q27_mm_put(self.h, i, e.ctypes.data), "q27_mm_put")

    def spec_stats(self):
        """Counters since the last call: {"steps", "lookup_steps", "tokens"}."""
        out = (ctypes.c_int64 * 3)()
        self._check(lib().q27_spec_stats(self.h, out), "q27_spec_stats")
        return {"steps": out[0], "lookup_steps": out[1], "tokens": out[2]}

