"""Which machine a request comes from (docs/configuration.md "Client counting"). Counting only: nothing here admits,
orders, limits or rejects a request.

Order: the header `X-Client-Id` (sanitized: letters of any script, digits and . _ - /, at most 64 characters), else the name CLIENT_NAMES gives
the peer address (`192.168.1.11=PC-A,192.168.1.0/24=Office`: an exact address wins, then the longest prefix), else
the address itself. Loopback peers are "local" (the stats page, the tray app, tests on the server itself).
The peer is the socket's address; X-Forwarded-For (its first entry) is used only when the immediate peer is a
loopback or private address (a reverse proxy in front of the server) and the header is present."""

import ipaddress
import re

MAX_CLIENT = 64
LOCAL = "local"
UNKNOWN = "unknown"
_BAD = re.compile(r"[^\w.\-/]")  # (letters of any script, so a name like 「電腦甲」 stays itself; digits . _ - /)
_cache = {}


def sanitize(value):
    """The id as the contract allows it, or None (not a string, or empty after stripping surrounding whitespace)."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return _BAD.sub("_", value)[:MAX_CLIENT] if value else None


def parse_names(text):
    """CLIENT_NAMES -> ([(ip_network, name)] longest prefix first). Entries that do not parse are skipped."""
    if text in _cache:
        return _cache[text]
    out = []
    for part in (text or "").split(","):
        key, sep, name = part.partition("=")
        name = sanitize(name)
        if not sep or not name:
            continue
        try:
            out.append((ipaddress.ip_network(key.strip(), strict=False), name))
        except ValueError:
            continue
    out.sort(key=lambda e: -e[0].prefixlen)
    if len(_cache) > 16:
        _cache.clear()
    _cache[text] = out
    return out


def _addr(text):
    try:
        a = ipaddress.ip_address((text or "").strip().split("%")[0])
    except ValueError:
        return None
    return a.ipv4_mapped if getattr(a, "ipv4_mapped", None) else a


def _header(headers, name):
    try:
        got = headers.get(name)
        if got is None and isinstance(headers, dict):
            got = next((v for k, v in headers.items() if str(k).lower() == name.lower()), None)
        return got
    except AttributeError:
        return None


def resolve(headers, peer, names=""):
    """The client of a request: headers (X-Client-Id, X-Forwarded-For), the socket's peer address (a string or None)
    and the CLIENT_NAMES setting text."""
    named = sanitize(_header(headers, "X-Client-Id"))
    if named:
        return named
    ip = _addr(peer)
    if ip is not None and (ip.is_loopback or ip.is_private):
        fwd = _header(headers, "X-Forwarded-For")
        if isinstance(fwd, str) and fwd.strip():
            first = _addr(fwd.split(",")[0])
            if first is not None:
                ip = first
    if ip is None:
        return sanitize(peer) or UNKNOWN
    if ip.is_loopback:
        return LOCAL
    for net, name in parse_names(names):
        if ip.version == net.version and ip in net:
            return name
    return sanitize(str(ip)) or UNKNOWN


def label(value):
    """A Prometheus label value: backslash, quote and newline escaped."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
