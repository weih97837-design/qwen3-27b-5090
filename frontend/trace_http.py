"""The request trace over HTTP (docs/request-trace.md, "Lesen über den Port"): GET /admin/trace, /admin/trace/<name>, /admin/trace/all.

The recorder (request_trace.py) runs in the engine process; this runs in the server process. They share only the trace directory:
the files are found the way the recorder names them (`<prefix>-YYYYMMDD-HHMMSS[_k].jsonl`, REQUEST_TRACE read from the environment on
every request). The newest file is being appended while it is read, so every response ends at the last newline ("cut"): a client never
sees a torn line, and `X-Trace-Offset` (= cut) is where a poller continues with `?from=`.

A file name from the URL is never joined to the directory: it must be one of the names found in the directory listing (regular files
only, symlinks are not listed and not followed). Files are streamed in CHUNK-sized reads, never loaded whole.

No authentication, like the stats page (the trace has agent ids, client names, lengths and hashes, no prompt or output text).
REQUEST_TRACE_HTTP=0 turns the three endpoints off (404).
"""

import json
import os
import re
import stat
import threading
import zlib
from urllib.parse import parse_qs, unquote

import request_trace

CHUNK = 256 * 1024          # bytes per read / per chunk of a streamed response (tests patch it)
SCAN_MAX = 1024 * 1024      # how far back from the end the last newline (or the last line) is searched; a line is far shorter

_cache = {}                 # (name) -> ((size, mtime_ns), info): first_ts / last_ts / cut of a file as of that size and mtime
_lock = threading.Lock()


class TraceError(Exception):
    def __init__(self, code, msg, kind="invalid_request_error"):
        super().__init__(msg)
        self.code, self.kind = code, kind


# ------------------------------------------------------------------ the directory

def http_enabled(env=None):
    return ((env if env is not None else os.environ).get("REQUEST_TRACE_HTTP") or "").strip() != "0"


def resolve(env=None):
    """(directory, file name prefix incl. the "-") of the trace target (request_trace.resolve_target, the recorder's own), or None when
    off. Only strings: no folder is created here."""
    t = request_trace.resolve_target(env if env is not None else os.environ)
    if t is None:
        return None
    if os.path.isdir(t) or t.endswith(("/", "\\")):
        t = os.path.join(t, "trace")
    return os.path.dirname(t) or ".", os.path.basename(t) + "-"


def max_bytes(env=None):
    try:
        mib = float((env if env is not None else os.environ).get("REQUEST_TRACE_MAX_MIB") or request_trace.MAX_MIB)
    except ValueError:
        mib = request_trace.MAX_MIB
    return int(mib * 1048576)


def names(d, base):
    """The trace files of the directory, oldest first (the names sort by time): regular files, no symlinks."""
    try:
        found = sorted(n for n in os.listdir(d) if n.startswith(base) and n.endswith(".jsonl"))
    except OSError:
        return []
    out = []
    for n in found:
        try:
            if stat.S_ISREG(os.lstat(os.path.join(d, n)).st_mode):
                out.append(n)
        except OSError:
            pass
    return out


def open_file(d, name):
    """The file as a binary object (no symlink followed), or None when it is gone or not a regular file."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(os.path.join(d, name), flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        return os.fdopen(fd, "rb")
    except OSError:
        os.close(fd)
        return None


# ------------------------------------------------------------------ cheap file facts (head and tail only)

def last_newline(f, size):
    """The offset after the last "\\n" before `size` (0: none within SCAN_MAX bytes of the end)."""
    pos, scanned = size, 0
    while pos > 0 and scanned < SCAN_MAX:
        n = min(CHUNK, pos)
        f.seek(pos - n)
        buf = f.read(n)
        i = buf.rfind(b"\n")
        if i >= 0:
            return pos - n + i + 1
        pos -= n
        scanned += n
    return 0


def _ts(line):
    try:
        v = json.loads(line).get("ts")
        return float(v) if isinstance(v, (int, float)) else None
    except (ValueError, AttributeError):
        return None


def first_line(f, cut):
    f.seek(0)
    got, buf = 0, b""
    while got < min(cut, SCAN_MAX):
        b = f.read(min(CHUNK, cut - got))
        if not b:
            break
        got += len(b)
        buf += b
        i = buf.find(b"\n")
        if i >= 0:
            return buf[:i]
    return b""


def last_line(f, cut):
    """The last complete line (before `cut`, which is after a newline), or b"" when it is longer than SCAN_MAX."""
    pos, buf = cut - 1, b""  # (skip the newline that ends it)
    while pos > 0 and len(buf) < SCAN_MAX:
        n = min(CHUNK, pos)
        f.seek(pos - n)
        buf = f.read(n) + buf
        pos -= n
        i = buf.rfind(b"\n")
        if i >= 0:
            return buf[i + 1:]
    return buf if pos <= 0 else b""


def facts(name, f):
    """(size, mtime, cut, first_ts, last_ts) of an open file, from the cache when its size and mtime did not change."""
    st = os.fstat(f.fileno())
    key = (st.st_size, st.st_mtime_ns)
    with _lock:
        hit = _cache.get(name)
    if hit and hit[0] == key:
        return st.st_size, st.st_mtime, *hit[1]
    cut = last_newline(f, st.st_size)
    first = last = None
    if cut:
        first = _ts(first_line(f, cut))
        last = _ts(last_line(f, cut))
    with _lock:
        _cache[name] = (key, (cut, first, last))
        while len(_cache) > 256:  # (names come from a directory listing: the cache stays small anyway)
            _cache.pop(next(iter(_cache)))
    return st.st_size, st.st_mtime, cut, first, last


def listing(env=None):
    """The body of GET /admin/trace."""
    if not http_enabled(env):
        raise TraceError(404, "the request trace over HTTP is off (REQUEST_TRACE_HTTP=0)", "not_found")
    r = resolve(env)
    if r is None:
        return {"enabled": False}
    d, base = r
    ns = names(d, base)
    files, total = [], 0
    for i, n in enumerate(ns):
        f = open_file(d, n)
        if f is None:
            continue
        with f:
            size, mtime, cut, first, last = facts(n, f)
        total += size
        # complete: the file is closed (not the newest, nothing is appended to it any more); the newest one still grows
        files.append({"name": n, "size": size, "mtime": mtime, "first_ts": first, "last_ts": last, "complete": i < len(ns) - 1,
                      "complete_bytes": cut})
    return {"enabled": True, "directory_name": os.path.basename(os.path.abspath(d)) or "", "files": files, "total_bytes": total,
            "max_bytes": max_bytes(env)}


# ------------------------------------------------------------------ streaming

def parse_range(header, total):
    """Range: bytes=a-b | a- | -n against `total` bytes -> (start, end inclusive); None when absent or not a single bytes range
    (then the whole is served); TraceError 416 when it cannot be satisfied."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
    if not m or (not m.group(1) and not m.group(2)):
        return None
    a, b = m.groups()
    if not a:
        n = int(b)
        if n == 0 or total == 0:
            raise TraceError(416, "range not satisfiable", "invalid_request_error")
        return max(0, total - n), total - 1
    start, end = int(a), int(b) if b else total - 1
    if start >= total or (b and end < start):
        raise TraceError(416, "range not satisfiable", "invalid_request_error")
    return start, min(end, total - 1)


def wants_gzip(header):
    for part in (header or "").split(","):
        tok, _, q = part.strip().partition(";")
        if tok.strip().lower() in ("gzip", "x-gzip"):
            try:
                return not q.strip().startswith("q=") or float(q.split("=", 1)[1]) > 0
            except ValueError:
                return False
    return False


class Chunked:
    """A response body in HTTP/1.1 chunked framing, optionally gzip-compressed on the fly (one CHUNK of input at a time)."""

    def __init__(self, wfile, gz):
        self.w, self.z = wfile, zlib.compressobj(5, zlib.DEFLATED, 31) if gz else None

    def _emit(self, data):
        if data:
            self.w.write(b"%x\r\n" % len(data) + data + b"\r\n")

    def write(self, data):
        self._emit(self.z.compress(data) if self.z else data)

    def finish(self):
        if self.z:
            self._emit(self.z.flush())
        self.w.write(b"0\r\n\r\n")


def copy(f, start, end, sink):
    """Write bytes [start, end) of f to sink in CHUNK reads; returns the bytes written (short when the file got shorter)."""
    f.seek(start)
    left = end - start
    while left > 0:
        b = f.read(min(CHUNK, left))
        if not b:
            break
        sink(b)
        left -= len(b)
    return end - start - left


def _query(path_q):
    return {k: v[-1] for k, v in parse_qs(path_q, keep_blank_values=True).items()}


def _int(q, key):
    if key not in q:
        return None
    try:
        v = float(q[key])
    except ValueError:
        v = -1
    if not v >= 0 or v == float("inf"):
        raise TraceError(400, f"{key} must be a number >= 0")
    return v


def serve(h, raw_path):
    """GET /admin/trace[/...] on handler h (server.Handler): raises TraceError for the error answers."""
    path, _, qs = raw_path.partition("?")
    path = path.rstrip("/")
    q = _query(qs)
    if path == "/admin/trace":
        return h._json(200, listing())
    if not http_enabled():
        raise TraceError(404, "the request trace over HTTP is off (REQUEST_TRACE_HTTP=0)", "not_found")
    r = resolve()
    if r is None:
        raise TraceError(404, "request trace is off", "not_found")
    d, base = r
    seg = path[len("/admin/trace/"):]
    if seg == "all":
        return serve_all(h, d, base, q)
    name = unquote(seg)
    if not name or "/" in name or "\\" in name or "\0" in name or name in (".", "..") or ".." in seg:
        raise TraceError(400, "bad trace file name")
    if name not in names(d, base):
        raise TraceError(404, "no such trace file", "not_found")
    f = open_file(d, name)
    if f is None:
        raise TraceError(404, "no such trace file", "not_found")
    with f:
        size, mtime, cut, _, _ = facts(name, f)
        start = _int(q, "from")
        rng = h.headers.get("Range")
        if start is not None and rng:
            raise TraceError(400, "use either ?from= or a Range header")
        if rng and (m := parse_range(rng, cut)) is not None:
            a, b = m
            h.send_response(206)
            h.send_header("Content-Type", "application/x-ndjson")
            h.send_header("Accept-Ranges", "bytes")
            h.send_header("Content-Range", f"bytes {a}-{b}/{cut}")
            h.send_header("Content-Length", str(b - a + 1))
            h.send_header("X-Trace-Offset", str(b + 1))
            h.end_headers()
            return _body(h, f, a, b + 1, None)
        if rng and cut == 0:
            raise TraceError(416, "range not satisfiable")
        start = int(start or 0)
        if start > cut:  # (the file is not the one the poller read: rotated away and replaced, or truncated)
            h.send_response(416)
            h.send_header("Content-Range", f"bytes */{cut}")
            h.send_header("X-Trace-Offset", str(cut))
            body = json.dumps({"error": {"message": f"from={start} is beyond the end of the file ({cut} bytes)",
                                         "type": "invalid_request_error", "code": 416}}).encode()
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return
        gz = wants_gzip(h.headers.get("Accept-Encoding"))
        h.send_response(200)
        h.send_header("Content-Type", "application/x-ndjson")
        h.send_header("Accept-Ranges", "bytes")
        h.send_header("Vary", "Accept-Encoding")
        h.send_header("X-Trace-Offset", str(cut))
        if gz and cut > start:
            h.send_header("Content-Encoding", "gzip")
            h.send_header("Transfer-Encoding", "chunked")
            h.end_headers()
            return _body(h, f, start, cut, Chunked(h.wfile, True))
        h.send_header("Content-Length", str(cut - start))
        h.end_headers()
        return _body(h, f, start, cut, None)


def _body(h, f, start, end, chunked):
    try:
        sink = chunked.write if chunked else h.wfile.write
        n = copy(f, start, end, sink)
        if chunked:
            chunked.finish()
        elif n < end - start:
            h.close_connection = True  # (the file got shorter under us: the promised length cannot be met)
    except (BrokenPipeError, ConnectionResetError, OSError):
        h.close_connection = True


def serve_all(h, d, base, q):
    since = _int(q, "since")
    h.send_response(200)
    h.send_header("Content-Type", "application/x-ndjson")
    h.send_header("Vary", "Accept-Encoding")
    gz = wants_gzip(h.headers.get("Accept-Encoding"))
    if gz:
        h.send_header("Content-Encoding", "gzip")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()
    out = Chunked(h.wfile, gz)
    try:
        for n in names(d, base):
            f = open_file(d, n)
            if f is None:
                continue  # (rotated away since the listing)
            with f:
                _, _, cut, _, last = facts(n, f)
                if since is not None and last is not None and last < since:
                    continue
                copy(f, 0, cut, out.write)
        out.finish()
    except (BrokenPipeError, ConnectionResetError, OSError):
        h.close_connection = True
