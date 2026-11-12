"""Which AI agent a request belongs to (docs/configuration.md "Agent identity").

Contract with the clients: the HTTP header `X-Agent-Id: <id>` or the OpenAI body field `"user": "<id>"` (the header
wins). The id is a string of at most 128 characters from [A-Za-z0-9._/-]; anything else is replaced by `_` and the
rest truncated; an empty id is no id. The agent used for fairness is the part before the first `/` (`manager/task-1`
and `manager/probe` are both `manager`); the full id (session / sub-id) is kept as well.

Without an id the worker derives `auto:<hash>` from the rendered prompt's opening (auto_id). Pure functions only: the
server (stats.request_meta) and the worker (Scheduler.submit) both import this."""

import hashlib
import re

MAX_ID = 128
AUTO_PREFIX = "auto:"
_BAD = re.compile(r"[^A-Za-z0-9._/-]")


def sanitize(value):
    """The id as the contract allows it, or None (not a string, or empty after stripping surrounding whitespace)."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return _BAD.sub("_", value)[:MAX_ID]


def from_request(headers, body):
    """The client's id: the X-Agent-Id header, else the body's "user" field (None: neither gives one). An empty
    header counts as absent, so the body's field is used then."""
    got = None
    try:
        got = headers.get("X-Agent-Id")
        if got is None and isinstance(headers, dict):  # (a plain dict is case-sensitive; http.client headers are not)
            got = next((v for k, v in headers.items() if str(k).lower() == "x-agent-id"), None)
    except AttributeError:
        pass
    return sanitize(got) or sanitize(body.get("user") if isinstance(body, dict) else None)


def agent_of(agent_id):
    """The fairness key of an id: the part before the first "/" ("_" when that is empty); None for None."""
    if not agent_id:
        return None
    return agent_id.split("/", 1)[0] or "_"


def auto_end(prompt, im_start):
    """Where the opening ends: the start of the third message (the system prompt and the first user message), or of
    the second (a system prompt alone), else the whole prompt (capped at 4096 tokens)."""
    starts = [i for i, t in enumerate(prompt) if t == im_start]
    if len(starts) > 2:
        return starts[2]
    if len(starts) > 1:
        return starts[1]
    return min(len(prompt), 4096)


def auto_id(prompt, im_start):
    """`auto:` + 8 hex digits of a hash of the opening tokens. Limitation: agents that open with the same system prompt
    and first user message share an identity (and one that edits its system prompt gets a new one)."""
    import array
    h = hashlib.blake2b(array.array("i", prompt[:auto_end(prompt, im_start)]).tobytes(), digest_size=4)
    return AUTO_PREFIX + h.hexdigest()


def prefix_fingerprint(ids, pos):
    """Cheap fingerprint of the first pos tokens of ids: the first 4096 tokens, the 64 before pos and pos itself.
    Scheduler.agent_of_ids and the checkpoint notes use it, so a later eviction change can ask which agent saved a
    checkpoint (or a spilled entry) given the entry's token ids and length."""
    import array
    pos = min(pos, len(ids))
    h = hashlib.blake2b(digest_size=8)
    h.update(array.array("i", ids[:min(pos, 4096)]).tobytes())
    h.update(array.array("i", ids[max(4096, pos - 64):pos]).tobytes())
    h.update(pos.to_bytes(8, "little"))
    return h.digest()


def owner_id(agent):
    """The engine's id of an agent (Model::set_owner, docs/tiered-cache.md "Retention"): a stable 63-bit integer derived from the
    fairness key, so the tag stored with a checkpoint (and in the SSD manifest) means the same agent after a restart without any table.
    0 is "unknown" (no agent). Two keys colliding (2^-63) would share a bucket: harmless."""
    if not agent:
        return 0
    v = int.from_bytes(hashlib.blake2b(agent.encode("utf-8", "replace"), digest_size=8).digest(), "little") & ((1 << 63) - 1)
    return v or 1
