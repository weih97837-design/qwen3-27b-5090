"""Shared by trace_summary.py, cache_sim.py and trace_fetch.py: read trace lines from a file path or an http(s):// URL.

A URL such as http://HOST:8080/admin/trace/all is streamed line by line (urllib, gzip asked for and decoded on the fly): nothing is
loaded whole. A failure (404 because the trace is off, the server not reachable, a cut transfer) raises TraceSourceError with a message
that says what to do; the tools' main() turns it into a plain exit message.
"""

import contextlib
import gzip
import http.client
import io
import json
import urllib.error
import urllib.request

TIMEOUT_S = 60


class TraceSourceError(RuntimeError):
    pass


def is_url(s):
    return str(s).lower().startswith(("http://", "https://"))


def request(url, extra_headers=None, timeout=TIMEOUT_S):
    """urlopen with gzip accepted; HTTPError / URLError become TraceSourceError. The caller closes the response."""
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "q27-trace-tools", **(extra_headers or {})})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read(4096)).get("error", {}).get("message", "")
        except (ValueError, AttributeError, OSError):
            msg = ""
        hint = (" (the request trace is off on that server: REQUEST_TRACE, or REQUEST_TRACE_HTTP=0)" if e.code == 404 and not msg else "")
        raise TraceSourceError(f"{url}: HTTP {e.code} {msg or e.reason}{hint}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise TraceSourceError(f"{url}: {getattr(e, 'reason', e)}") from None


def body(resp):
    """The binary body of a response, gunzipped when the server compressed it."""
    return gzip.GzipFile(fileobj=resp) if (resp.headers.get("Content-Encoding") or "").lower() == "gzip" else resp


@contextlib.contextmanager
def open_lines(src):
    """A text file object (iterate it for lines) over a file path or a URL."""
    if not is_url(src):
        with open(src, encoding="utf-8") as f:
            yield f
        return
    resp = request(src)
    try:
        t = io.TextIOWrapper(body(resp), encoding="utf-8", errors="replace", newline="\n")
        try:
            yield t
        except (EOFError, OSError, urllib.error.URLError, http.client.HTTPException) as e:
            raise TraceSourceError(f"{src}: transfer failed ({type(e).__name__}: {e})") from None
    finally:
        resp.close()

