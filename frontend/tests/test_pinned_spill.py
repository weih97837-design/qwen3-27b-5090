"""SPILL_PINNED_GIB in the frontend: the idle pass pins one slab per poll until the engine says it is full, and the
tiers block reports the pinned pool. Fake engines, no GPU."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import stats  # noqa: E402
import test_vram_backup as tvb  # noqa: E402


class PinEngine(tvb.Base):
    def __init__(self, slabs):
        super().__init__()
        self.slabs, self.grow_calls = slabs, 0

    def spill_grow_pinned(self, n):
        self.grow_calls += 1
        if self.slabs == 0:
            return 0
        self.slabs -= 1
        return 1

    def spill_stats(self):
        d = super().spill_stats()
        d.update(pinned_bytes=8 << 30, pinned_used=1 << 30)
        return d


def _poll_idle(eng, polls):
    s = tvb.make_sched(eng, tvb.tiers_for(eng))
    s.cache_full = True
    for _ in range(polls):
        s.poll(True, wait=0)
    return s


def test_idle_polls_pin_one_slab_each_until_full():
    eng = PinEngine(slabs=3)
    s = _poll_idle(eng, 10)
    assert eng.grow_calls == 4  # three slabs, then the 0 that ends it
    assert s.pin_full


def test_an_engine_without_the_call_is_left_alone():
    eng = tvb.Base()
    s = _poll_idle(eng, 2)
    assert s.pin_full


def test_a_failing_pin_stops_the_idle_pass(capsys):
    class Boom(PinEngine):
        def spill_grow_pinned(self, n):
            self.grow_calls += 1
            raise RuntimeError("no memory")

    eng = Boom(slabs=5)
    s = _poll_idle(eng, 5)
    assert eng.grow_calls == 1 and s.pin_full


def test_no_tiers_means_no_pinning():
    eng = PinEngine(slabs=3)
    s = tvb.make_sched(eng, None)
    s.cache_full = True
    s.poll(True, wait=0)
    assert eng.grow_calls == 0


def test_tiers_block_reports_the_pinned_pool():
    eng = PinEngine(slabs=0)
    out = stats.tiers_block(eng, tvb.tiers_for(eng), environ={})
    assert out["pinned_bytes"] == 8 << 30 and out["pinned_used"] == 1 << 30


def test_tiers_block_without_the_keys_reports_zero():
    eng = tvb.Base()
    out = stats.tiers_block(eng, tvb.tiers_for(eng), environ={})
    assert out["pinned_bytes"] == 0 and out["pinned_used"] == 0
