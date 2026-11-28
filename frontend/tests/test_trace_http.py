"""The request trace over HTTP (frontend/trace_http.py, GET /admin/trace*) against a real HTTPServer with server.Handler, a temporary
trace directory, and the tools that read it by URL (trace_summary, cache_sim, trace_fetch). No engine, no GPU."""

import gzip
import http.client
import io
import json
import os
import pathlib
import sys
import threading
import time
import tracemalloc

import pytest

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "tools"))

import cache_sim  # noqa: E402
import cache_sim_gen as gen  # noqa: E402
import server  # noqa: E402
import trace_fetch  # noqa: E402
import trace_http  # noqa: E402
import trace_io  # noqa: E402
import trace_summary  # noqa: E402


def line(ts, t="req", **kw):
    return json.dumps({"t": t, "ts": ts, **kw}, separators=(",", ":")) + "\n"


def write(d, name, rows, tail=""):
    p = d / name
    p.write_text("".join(rows) + tail)
    return p


@pytest.fixture
def tdir(tmp_path, monkeypatch):
    d = tmp_path / "traces"
    d.mkdir()
    monkeypatch.setenv("REQUEST_TRACE", str(d))
    monkeypatch.delenv("REQUEST_TRACE_HTTP", raising=False)
    monkeypatch.delenv("REQUEST_TRACE_MAX_MIB", raising=False)
    trace_http._cache.clear()
    return d


@pytest.fixture
def three(tdir):
    write(tdir, "trace-20260101-000000.jsonl", [line(100, "hdr", version=1), line(110, rid="a"), line(120, "done", rid="a")])
    write(tdir, "trace-20260101-010000.jsonl", [line(200, "hdr", version=1), line(210, rid="b"), line(220, "done", rid="b")])
    write(tdir, "trace-20260101-020000.jsonl", [line(300, "hdr", version=1), line(310, rid="c")], tail='{"t":"req","ts":3')  # torn
    return tdir


@pytest.fixture
def base():
    srv = server.HTTPServer(("127.0.0.1", 0), server.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(base, path, headers=None):
    c = http.client.HTTPConnection(base[len("http://"):], timeout=30)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, r, data


# ------------------------------------------------------------------ listing

def test_listing(three, base):
    code, r, data = get(base, "/admin/trace")
    j = json.loads(data)
    assert code == 200 and r.getheader("Content-Type") == "application/json"
    assert j["enabled"] and j["directory_name"] == "traces" and str(three) not in data.decode()
    assert [f["name"] for f in j["files"]] == ["trace-20260101-000000.jsonl", "trace-20260101-010000.jsonl", "trace-20260101-020000.jsonl"]
    f0, f2 = j["files"][0], j["files"][2]
    assert (f0["first_ts"], f0["last_ts"], f0["complete"]) == (100, 120, True)
    assert (f2["first_ts"], f2["last_ts"], f2["complete"]) == (300, 310, False)  # the newest: still appended; the torn tail is not counted
    assert f2["size"] > f2["complete_bytes"] and f0["size"] == f0["complete_bytes"]
    assert j["total_bytes"] == sum(f["size"] for f in j["files"]) and j["max_bytes"] == 512 * 1048576
    assert all(isinstance(f["mtime"], float) for f in j["files"])


def test_listing_ignores_other_files_and_symlinks(three, base, tmp_path):
    (three / "notes.txt").write_text("x")
    (three / "other-20260101-000000.jsonl").write_text(line(1))
    (three / "trace-20260101-030000.jsonl").symlink_to(tmp_path / "secret.jsonl")
    (tmp_path / "secret.jsonl").write_text(line(5, secret=1))
    (three / "trace-20260101-040000.jsonl").mkdir()
    j = json.loads(get(base, "/admin/trace")[2])
    assert len(j["files"]) == 3


def test_listing_cache_follows_size_and_mtime(three, base):
    p = three / "trace-20260101-020000.jsonl"
    assert json.loads(get(base, "/admin/trace")[2])["files"][2]["last_ts"] == 310
    with open(p, "a") as f:
        f.write('1,"rid":"x"}\n' + line(999))
    assert json.loads(get(base, "/admin/trace")[2])["files"][2]["last_ts"] == 999


def test_max_mib(three, base, monkeypatch):
    monkeypatch.setenv("REQUEST_TRACE_MAX_MIB", "2")
    assert json.loads(get(base, "/admin/trace")[2])["max_bytes"] == 2 * 1048576


def test_prefix_form(tmp_path, base, monkeypatch):
    monkeypatch.setenv("REQUEST_TRACE", str(tmp_path / "load1"))  # a name prefix, not a directory
    (tmp_path / "load1-20260101-000000.jsonl").write_text(line(1, "hdr"))
    (tmp_path / "trace-20260101-000000.jsonl").write_text(line(1, "hdr"))
    j = json.loads(get(base, "/admin/trace")[2])
    assert [f["name"] for f in j["files"]] == ["load1-20260101-000000.jsonl"] and j["directory_name"] == tmp_path.name


def test_recorder_off(monkeypatch, base):
    # (on by default: unset is the default folder, so "off" is an explicit off value)
    for value in ("0", "off", "FALSE", "no", "none", ""):
        monkeypatch.setenv("REQUEST_TRACE", value)
        code, _, data = get(base, "/admin/trace")
        assert code == 200 and json.loads(data) == {"enabled": False}, value
    for p in ("/admin/trace/all", "/admin/trace/trace-20260101-000000.jsonl"):
        code, _, data = get(base, p)
        assert code == 404 and "request trace is off" in json.loads(data)["error"]["message"]


def test_http_switch_off(three, base, monkeypatch):
    monkeypatch.setenv("REQUEST_TRACE_HTTP", "0")
    for p in ("/admin/trace", "/admin/trace/all", "/admin/trace/trace-20260101-000000.jsonl"):
        code, _, data = get(base, p)
        e = json.loads(data)["error"]
        assert code == 404 and "REQUEST_TRACE_HTTP=0" in e["message"] and e["type"] == "not_found"


def test_empty_directory(tdir, base):
    j = json.loads(get(base, "/admin/trace")[2])
    assert j["enabled"] and j["files"] == [] and j["total_bytes"] == 0
    assert get(base, "/admin/trace/all")[2] == b""


# ------------------------------------------------------------------ one file

def test_whole_file_stops_at_the_last_newline(three, base):
    code, r, data = get(base, "/admin/trace/trace-20260101-020000.jsonl")
    assert code == 200 and r.getheader("Content-Type") == "application/x-ndjson"
    full = (three / "trace-20260101-020000.jsonl").read_bytes()
    cut = full.rindex(b"\n") + 1
    assert data == full[:cut] and data.endswith(b"\n") and len(data) < len(full)
    assert int(r.getheader("X-Trace-Offset")) == cut and int(r.getheader("Content-Length")) == cut
    assert all(json.loads(x) for x in data.splitlines())


def test_from_continues_at_the_offset(three, base):
    name = "/admin/trace/trace-20260101-020000.jsonl"
    _, r, first = get(base, name)
    off = int(r.getheader("X-Trace-Offset"))
    code, r2, data = get(base, f"{name}?from={off}")
    assert code == 200 and data == b"" and int(r2.getheader("X-Trace-Offset")) == off  # nothing new
    p = three / "trace-20260101-020000.jsonl"
    with open(p, "a") as f:  # the torn line gets its end, one more line arrives and a new torn one starts
        f.write('1,"rid":"x"}\n' + line(400, rid="d") + '{"t":')
    code, r3, data = get(base, f"{name}?from={off}")
    rows = [json.loads(x) for x in data.splitlines()]
    assert code == 200 and [x["ts"] for x in rows] == [31, 400]
    assert data.endswith(b"\n") and int(r3.getheader("X-Trace-Offset")) == off + len(data)
    assert (first + data) == p.read_bytes()[:int(r3.getheader("X-Trace-Offset"))]


def test_from_beyond_the_end_is_416_with_the_offset(three, base):
    code, r, data = get(base, "/admin/trace/trace-20260101-000000.jsonl?from=999999")
    size = (three / "trace-20260101-000000.jsonl").stat().st_size
    assert code == 416 and r.getheader("Content-Range") == f"bytes */{size}" and int(r.getheader("X-Trace-Offset")) == size
    assert json.loads(data)["error"]["code"] == 416


@pytest.mark.parametrize("q", ["from=-1", "from=abc", "from=nan", "from=inf"])
def test_bad_from(three, base, q):
    assert get(base, f"/admin/trace/trace-20260101-000000.jsonl?{q}")[0] == 400


def test_range(three, base):
    name = "/admin/trace/trace-20260101-020000.jsonl"
    full = get(base, name)[2]
    n = len(full)
    code, r, data = get(base, name, {"Range": "bytes=5-20"})
    assert code == 206 and data == full[5:21] and r.getheader("Content-Range") == f"bytes 5-20/{n}"
    assert r.getheader("Content-Length") == "16" and r.getheader("X-Trace-Offset") == "21"
    code, r, data = get(base, name, {"Range": "bytes=10-"})
    assert code == 206 and data == full[10:] and r.getheader("X-Trace-Offset") == str(n)
    code, r, data = get(base, name, {"Range": "bytes=-7"})
    assert code == 206 and data == full[-7:]
    code, r, data = get(base, name, {"Range": f"bytes=0-{n + 5000}"})  # (the torn tail is beyond the end for a Range too)
    assert code == 206 and data == full and r.getheader("Content-Range") == f"bytes 0-{n - 1}/{n}"
    code, r, _ = get(base, name, {"Range": f"bytes={n}-"})
    assert code == 416
    assert get(base, name, {"Range": "bytes=9-3"})[0] == 416
    assert get(base, name, {"Range": "bytes=0-5", "Accept-Encoding": "gzip"})[1].getheader("Content-Encoding") is None  # (no gzip with Range)
    assert get(base, f"{name}?from=3", {"Range": "bytes=0-5"})[0] == 400


def test_gzip_round_trip(three, base):
    name = "/admin/trace/trace-20260101-000000.jsonl"
    plain = get(base, name)[2]
    code, r, data = get(base, name, {"Accept-Encoding": "gzip"})
    assert code == 200 and r.getheader("Content-Encoding") == "gzip" and r.getheader("Transfer-Encoding") == "chunked"
    assert gzip.decompress(data) == plain and r.getheader("X-Trace-Offset") == str(len(plain))
    assert get(base, name, {"Accept-Encoding": "gzip;q=0"})[1].getheader("Content-Encoding") is None
    assert get(base, name, {"Accept-Encoding": "br, deflate"})[1].getheader("Content-Encoding") is None
    code, r, data = get(base, f"{name}?from=10", {"Accept-Encoding": "gzip"})
    assert gzip.decompress(data) == plain[10:]


# ------------------------------------------------------------------ names that must not work

@pytest.mark.parametrize("p", [
    "/admin/trace/..%2f..%2fetc%2fpasswd", "/admin/trace/%2e%2e%2f%2e%2e%2fetc%2fpasswd", "/admin/trace/..%252f..%252fetc",
    "/admin/trace/../server.py", "/admin/trace/..", "/admin/trace/%2e%2e", "/admin/trace/%2Fetc%2Fpasswd", "/admin/trace//etc/passwd",
    "/admin/trace/trace-20260101-000000.jsonl%00", "/admin/trace/..%5c..%5cwindows", "/admin/trace/sub/trace-20260101-000000.jsonl",
    "/admin/trace/trace-20260101-000000.jsonl/", "/admin/trace/TRACE-20260101-000000.JSONL", "/admin/trace/trace-20260101-0000.jsonl",
    "/admin/trace/trace-20260101-030000.jsonl", "/admin/trace/secret.jsonl", "/admin/trace/%2Ftmp",
])
def test_bad_names(three, base, tmp_path, p):
    (tmp_path / "secret.jsonl").write_text(line(5, secret="TOPSECRET"))
    (three / "trace-20260101-030000.jsonl").symlink_to(tmp_path / "secret.jsonl")
    code, _, data = get(base, p)
    if p.endswith("trace-20260101-000000.jsonl/"):  # (a trailing slash is ignored, like on the other endpoints)
        assert code == 200
    else:
        assert code in (400, 404) and b"TOPSECRET" not in data and b"root:" not in data


def test_symlink_inside_is_not_followed(three, base, tmp_path):
    (tmp_path / "secret.jsonl").write_text(line(5, secret="TOPSECRET"))
    (three / "trace-20260101-030000.jsonl").symlink_to(tmp_path / "secret.jsonl")
    assert get(base, "/admin/trace/trace-20260101-030000.jsonl")[0] == 404
    assert b"TOPSECRET" not in get(base, "/admin/trace/all")[2]
    # swapped for a symlink between the listing and the open: open_file refuses it
    assert trace_http.open_file(str(three), "trace-20260101-030000.jsonl") is None


def test_name_logic_never_builds_a_path_from_the_url(three, monkeypatch):
    opened = []
    real = trace_http.open_file
    monkeypatch.setattr(trace_http, "open_file", lambda d, n: opened.append(n) or real(d, n))
    h = type("H", (), {"headers": {}, "close_connection": False})()
    for bad in ("/admin/trace/..%2fx", "/admin/trace/x.jsonl", "/admin/trace/%2Fetc"):
        with pytest.raises(trace_http.TraceError):
            trace_http.serve(h, bad)
    assert opened == []


# ------------------------------------------------------------------ all

def test_all_is_the_files_oldest_to_newest(three, base):
    code, r, data = get(base, "/admin/trace/all")
    assert code == 200 and r.getheader("Content-Type") == "application/x-ndjson" and r.getheader("Transfer-Encoding") == "chunked"
    rows = [json.loads(x) for x in data.splitlines()]
    assert [x["ts"] for x in rows] == [100, 110, 120, 200, 210, 220, 300, 310] and data.endswith(b"\n")


def test_all_since(three, base):
    ts = lambda q: [json.loads(x)["ts"] for x in get(base, "/admin/trace/all" + q)[2].splitlines()]
    assert ts("?since=0")[0] == 100
    assert ts("?since=120")[0] == 100      # (the file's last event is not older than since)
    assert ts("?since=121")[0] == 200      # the first file ends at 120: skipped
    assert ts("?since=221")[0] == 300
    assert ts("?since=311") == []
    assert get(base, "/admin/trace/all?since=x")[0] == 400


def test_all_gzip(three, base):
    plain = get(base, "/admin/trace/all")[2]
    code, r, data = get(base, "/admin/trace/all", {"Accept-Encoding": "gzip, deflate"})
    assert r.getheader("Content-Encoding") == "gzip" and gzip.decompress(data) == plain


def test_a_file_deleted_between_listing_and_open_is_skipped(three, base, monkeypatch):
    real = trace_http.names
    monkeypatch.setattr(trace_http, "names", lambda d, b: real(d, b) + ["trace-20260101-050000.jsonl"])
    assert get(base, "/admin/trace/all")[2].count(b"\n") == 8


# ------------------------------------------------------------------ memory

def peak_of(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def drain(base, path, headers=None):
    c = http.client.HTTPConnection(base[len("http://"):], timeout=120)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    n = 0
    while True:
        b = r.read(65536)
        if not b:
            break
        n += len(b)
    c.close()
    return r, n


def test_a_200_mb_file_streams_in_bounded_memory(tdir, base, monkeypatch):
    big = tdir / "trace-20260101-000000.jsonl"
    with open(big, "wb") as f:  # a sparse 200 MiB file: no disk is used; the head is a real line, the end a newline
        f.write(line(1, "hdr").encode())
        f.truncate(200 * 1048576)
        f.seek(0, 2)
        f.write(b"\n")
    size = big.stat().st_size
    monkeypatch.setattr(trace_http, "CHUNK", 64 * 1024)
    r, n = None, 0

    def run():
        nonlocal r, n
        r, n = drain(base, "/admin/trace/trace-20260101-000000.jsonl")
    peak = peak_of(run)
    assert n == size == int(r.getheader("X-Trace-Offset")) and peak < 4 * 1048576, peak
    # the listing of it reads only the head and the tail
    peak = peak_of(lambda: get(base, "/admin/trace"))
    assert peak < 4 * 1048576, peak


def test_gzip_and_all_stream_in_bounded_memory(tdir, base, monkeypatch):
    for i in range(3):
        with open(tdir / f"trace-20260101-00000{i}.jsonl", "wb") as f:
            f.write(line(i, "hdr").encode())
            f.truncate(20 * 1048576)
            f.seek(0, 2)
            f.write(b"\n")
    monkeypatch.setattr(trace_http, "CHUNK", 64 * 1024)
    for path in ("/admin/trace/trace-20260101-000001.jsonl", "/admin/trace/all"):
        out = []
        peak = peak_of(lambda: out.append(drain(base, path, {"Accept-Encoding": "gzip"})))
        assert peak < 4 * 1048576, (path, peak)


def test_copy_reads_in_chunks_only(monkeypatch):
    class Fake:  # 200 MB of 100-byte lines that never exist in memory
        def __init__(self): self.pos, self.reads = 0, []
        def seek(self, p): self.pos = p
        def read(self, n):
            self.reads.append(n)
            n = min(n, 200_000_000 - self.pos)
            self.pos += n
            return b"x" * n
    f, total = Fake(), [0]
    monkeypatch.setattr(trace_http, "CHUNK", 1 << 16)
    got = trace_http.copy(f, 0, 200_000_000, lambda b: total.__setitem__(0, total[0] + len(b)))
    assert got == total[0] == 200_000_000 and max(f.reads) <= 1 << 16


# ------------------------------------------------------------------ tools reading a URL

def synth(path, n=120):
    gen.main([str(path), "--workload", "mixed", "--requests", str(n), "--agents", "8", "--gap-s", "60", "--seed", "3"])


def test_tools_read_the_url(tdir, base):
    synth(tdir / "trace-20260101-000000.jsonl")
    local = [str(tdir / "trace-20260101-000000.jsonl")]
    url = [base + "/admin/trace/all"]
    assert trace_summary.summarize(*trace_summary.load(url)) == trace_summary.summarize(*trace_summary.load(local))
    assert trace_summary.summarize(*trace_summary.load([base + "/admin/trace/trace-20260101-000000.jsonl"]))["requests"] == 120
    ev_u, ev_l = cache_sim.load(url), cache_sim.load(local)
    assert [e[:2] for e in ev_u] == [e[:2] for e in ev_l]
    ru = cache_sim.simulate(ev_u, "legacy", ram_gib=6, ssd_gib=16, vram_pages=3000)
    rl = cache_sim.simulate(ev_l, "legacy", ram_gib=6, ssd_gib=16, vram_pages=3000)
    assert ru == rl


def test_tools_main_with_a_url(tdir, base, capsys):
    synth(tdir / "trace-20260101-000000.jsonl", 40)
    assert trace_summary.main([base + "/admin/trace/all", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["requests"] == 40
    assert cache_sim.main([base + "/admin/trace/all", "--policy", "legacy,smart", "--json"]) == 0
    assert [r["policy"] for r in json.loads(capsys.readouterr().out)] == ["legacy", "smart"]


def test_tools_say_why_a_url_fails(tdir, base, monkeypatch):
    for main, who in ((trace_summary.main, "trace_summary"), (cache_sim.main, "cache_sim")):
        monkeypatch.setenv("REQUEST_TRACE_HTTP", "0")
        with pytest.raises(SystemExit) as e:
            main([base + "/admin/trace/all"])
        assert who in str(e.value) and "404" in str(e.value) and "REQUEST_TRACE_HTTP=0" in str(e.value)
        monkeypatch.setenv("REQUEST_TRACE", "0")  # (unset would be the default folder: on)
        monkeypatch.delenv("REQUEST_TRACE_HTTP")
        with pytest.raises(SystemExit) as e:
            main([base + "/admin/trace/all"])
        assert "request trace is off" in str(e.value)
        monkeypatch.setenv("REQUEST_TRACE", str(tdir))
    with pytest.raises(SystemExit) as e:
        trace_summary.main(["http://127.0.0.1:9/admin/trace/all"])
    assert "trace_summary" in str(e.value)


def test_url_reader_streams_lines(tdir, base):
    synth(tdir / "trace-20260101-000000.jsonl", 30)
    with trace_io.open_lines(base + "/admin/trace/all") as f:
        first = next(iter(f))
    assert json.loads(first)["t"] == "hdr"
    with trace_io.open_lines(str(tdir / "trace-20260101-000000.jsonl")) as f:
        assert next(iter(f)) == first


# ------------------------------------------------------------------ trace_fetch

def test_fetch_is_incremental_and_resumes(three, base, tmp_path):
    out = tmp_path / "copy"
    log = []
    n = trace_fetch.fetch_once(base, out, log=log.append)
    names = sorted(p.name for p in out.glob("*.jsonl"))
    assert names == sorted(p.name for p in three.glob("*.jsonl"))
    newest = out / "trace-20260101-020000.jsonl"
    assert newest.read_bytes() == (three / newest.name).read_bytes()[:newest.stat().st_size] and newest.read_bytes().endswith(b"\n")
    assert n == sum(p.stat().st_size for p in out.glob("*.jsonl"))
    assert json.loads((out / ".trace_fetch.json").read_text())["files"][newest.name] == newest.stat().st_size
    assert trace_fetch.fetch_once(base, out) == 0  # nothing new: no download
    with open(three / newest.name, "a") as f:  # the torn line is finished, another line follows
        f.write('1,"rid":"x"}\n' + line(400, rid="d"))
    got = trace_fetch.fetch_once(base, out)
    assert got > 0 and newest.read_bytes() == (three / newest.name).read_bytes()
    assert [json.loads(x)["ts"] for x in newest.read_text().splitlines()][-1] == 400


def test_fetch_crash_between_write_and_state(three, base, tmp_path):
    out = tmp_path / "copy"
    trace_fetch.fetch_once(base, out)
    newest = out / "trace-20260101-020000.jsonl"
    good = newest.read_bytes()
    st = json.loads((out / ".trace_fetch.json").read_text())
    with open(newest, "ab") as f:
        f.write(b'{"t":"req","ts":5,"half')  # a partial write the state never recorded
    trace_fetch.fetch_once(base, out)
    assert newest.read_bytes() == good
    st["files"][newest.name] = 10 ** 9  # a state ahead of the file
    (out / ".trace_fetch.json").write_text(json.dumps(st))
    trace_fetch.fetch_once(base, out)
    assert newest.read_bytes() == good
    (out / ".trace_fetch.json").unlink()  # no state at all: refetched from the start, same content
    trace_fetch.fetch_once(base, out)
    assert newest.read_bytes() == good


def test_fetch_rotation_and_deleted_files(three, base, tmp_path):
    out = tmp_path / "copy"
    trace_fetch.fetch_once(base, out)
    (three / "trace-20260101-000000.jsonl").unlink()  # the server rotated the oldest away
    write(three, "trace-20260101-030000.jsonl", [line(500, "hdr"), line(510)])
    trace_fetch.fetch_once(base, out)
    assert (out / "trace-20260101-000000.jsonl").exists()  # (kept locally)
    assert (out / "trace-20260101-030000.jsonl").read_bytes() == (three / "trace-20260101-030000.jsonl").read_bytes()
    st = json.loads((out / ".trace_fetch.json").read_text())["files"]
    assert "trace-20260101-000000.jsonl" not in st and "trace-20260101-030000.jsonl" in st
    # a file replaced by a shorter one of the same name is fetched again from the start
    write(three, "trace-20260101-030000.jsonl", [line(600, "hdr")])
    trace_fetch.fetch_once(base, out)
    assert (out / "trace-20260101-030000.jsonl").read_bytes() == (three / "trace-20260101-030000.jsonl").read_bytes()
    # a name that vanishes between listing and download is skipped, not fatal
    real = trace_fetch.listing
    trace_fetch.listing = lambda b: dict(real(b), files=real(b)["files"] + [{"name": "trace-20260101-090000.jsonl", "size": 5, "complete_bytes": 5}])
    try:
        trace_fetch.fetch_once(base, out)
    finally:
        trace_fetch.listing = real


def test_fetch_since_and_errors(three, base, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    trace_fetch.fetch_once(base, out, since=221)
    assert [p.name for p in out.glob("*.jsonl")] == ["trace-20260101-020000.jsonl"]
    monkeypatch.setenv("REQUEST_TRACE_HTTP", "0")
    with pytest.raises(SystemExit) as e:
        trace_fetch.main([base, "--out", str(out)])
    assert "404" in str(e.value)
    monkeypatch.delenv("REQUEST_TRACE_HTTP")
    monkeypatch.setenv("REQUEST_TRACE", "0")  # (unset would be the default folder: on)
    with pytest.raises(SystemExit) as e:
        trace_fetch.main([base, "--out", str(out)])
    assert "off" in str(e.value)
    monkeypatch.setenv("REQUEST_TRACE", str(three))
    assert trace_fetch.main([base, "--out", str(out)]) == 0


def test_fetch_follow_picks_up_new_data(three, base, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    calls = []

    def sleep(_s):
        calls.append(1)
        if len(calls) == 1:
            with open(three / "trace-20260101-020000.jsonl", "a") as f:
                f.write('1,"rid":"x"}\n' + line(777))
        else:
            raise KeyboardInterrupt
    monkeypatch.setattr(trace_fetch.time, "sleep", sleep)
    assert trace_fetch.main([base, "--out", str(out), "--follow", "1"]) == 0
    assert (out / "trace-20260101-020000.jsonl").read_text().endswith(line(777))
