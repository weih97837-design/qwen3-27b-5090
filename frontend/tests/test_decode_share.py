"""DECODE_SHARE changes at runtime (POST /admin/concurrency, like kv_reserve and min_seq_tps): no GPU."""

import collections
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import server  # noqa: E402
from test_admission import FakeEngine, scheduler  # noqa: E402

UI = pathlib.Path(__file__).resolve().parents[1] / "ui" / "index.html"


def test_configure_applies_clamps_and_reports():
    sch = scheduler(FakeEngine(free=0), decode_share=0.5)
    assert sch.share == 0.5 and sch.settings()["decode_share"] == 0.5
    sch.credit = 4.0
    sch.configure({"decode_share": 1.5})
    assert sch.share == 1.5 and sch.credit == 1.5 and sch.settings()["decode_share"] == 1.5  # (owed time clamped)
    sch.credit = -0.3
    sch.configure({"decode_share": 3})
    assert sch.share == 3.0 and sch.credit == -0.3
    sch.configure({"decode_share": 99})
    assert sch.share == 10.0
    sch.configure({"decode_share": float("nan")})
    assert sch.share == 10.0
    sch.configure({"target_seqs": 2})  # (other changes leave it)
    assert sch.share == 10.0


def test_the_metrics_have_it():
    sch = scheduler(FakeEngine(free=0), decode_share=1.5)
    sch.m, sch.alive, sch.waiting = collections.Counter(), [], collections.deque()
    assert sch.metrics()["q27:decode_share"] == 1.5


def test_a_restarted_engine_keeps_it():
    assert '"min_seq_tps", "decode_share")' in pathlib.Path(server.__file__).read_text(encoding="utf-8")


class FakeEngineClient:
    def __init__(self):
        self.cfg, self.got = {}, []

    def configure(self, changes):
        self.got.append(changes)
        self.cfg.update(changes)  # (EngineClient.cfg: kept for a restarted engine)
        return dict(self.cfg)


def post(body):
    out = []
    h = object.__new__(server.Handler)
    h.srv = type("S", (), {"engine": FakeEngineClient()})()
    h._json = lambda code, obj: out.append((code, obj))
    h.concurrency(body)
    return out[0], h.srv.engine


def test_the_endpoint_accepts_it_and_persists_it():
    for v in (0, 0.5, 1.5, 3, 10):
        (code, obj), eng = post({"decode_share": v})
        assert code == 200 and eng.got == [{"decode_share": v}] and obj["decode_share"] == v and eng.cfg["decode_share"] == v
    _, eng = post({"decode_share": 3, "kv_reserve": 100})
    assert eng.got == [{"decode_share": 3, "kv_reserve": 100}]


@pytest.mark.parametrize("v", [-0.1, 10.5, float("nan"), float("inf"), float("-inf"), "1", True, None, [1]])
def test_the_endpoint_refuses_bad_values(v):
    with pytest.raises(server.ApiError) as e:
        post({"decode_share": v})
    assert e.value.code == 400 and "decode_share: bad value" in str(e.value)


def test_the_unknown_setting_message_lists_it():
    with pytest.raises(server.ApiError) as e:
        post({"nope": 1})
    assert "decode_share" in str(e.value)


def test_the_status_page_has_the_control():
    html = UI.read_text(encoding="utf-8")
    assert "預填時給解碼的時間比例（DECODE_SHARE）" in html
    assert "每預填 1 秒，讓解碼中的請求跑幾秒。越大 → 預填期間解碼越不會掉速，但長 prompt 的預填變慢" in html
    assert 'id: "decode-share"' in html and "decode_share: v" in html
