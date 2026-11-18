"""Pure helpers of scripts/tiered_cache_test.py (scenario l's plan and chunk arithmetic); no GPU, no torch."""
import importlib.util
import pathlib

PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tiered_cache_test.py"
spec = importlib.util.spec_from_file_location("tiered_cache_test_under_test", PATH)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def test_l_chunks_of_the_failing_run():
    assert t.l_chunks([], 12288) == [2048] * 6
    assert t.l_chunks([8192, 12125, 12286], 12288) == [2048] * 5 + [1856, 128, 64]
    assert t.fmt_chunks(t.l_chunks([8192, 12125, 12286], 12288)) == "2048x5 + 1856 + 128 + 64"
    assert t.l_chunks([8192, 12125], 12288, 12096) == [192]
    assert not t.plan_final_chunk_ok([8192, 12125, 12286], 12288)
    assert t.plan_final_chunk_ok([8192, 12125], 12288)


def test_l_chunks_interval_checkpoints():
    # an interval checkpoint more than 64 tokens from the others is a cut; one within 64 of the end is not
    assert t.l_chunks([], 10000, iv=4096) == [2048, 2048, 2048, 2048, 1808]
    assert 4096 in {sum(t.l_chunks([], 10000, iv=4096)[:i]) for i in range(1, 6)}
    assert t.l_chunks([], 4100, iv=4096) == t.l_chunks([], 4100)  # (4096 is within 64 of the end 4100)


def test_l_plan_ok():
    assert t.l_plan_ok([8192, 12125, 12286], 12288) == ([8192, 12125], [12286])
    assert t.l_plan_ok([8192, 12125, 12286], 12288, 12096) == ([8192, 12125], [12286])
    assert t.l_plan_ok([8192, 12125], 12288) == ([8192, 12125], [])


def test_l_plan_ok_never_drops_the_resume_point():
    assert t.l_plan_ok([12100], 12160) == ([], [12100])  # (without a floor everything goes)
    plan, dropped = t.l_plan_ok([12100], 12160, 12096)
    assert plan == [12100] and dropped == []  # (still not ok: the caller reports it)
    assert not t.plan_final_chunk_ok(plan, 12160, 0, resume=12096)


def test_b_is_checked_too():
    # A from 0 is fine (the cut at 12000 leaves 288), B resumed at 12224 would end in a 64-token chunk
    assert t.plan_final_chunk_ok([12000], 12288)
    assert not t.plan_final_chunk_ok([12000], 12288, resume=12224)


def test_first_diff():
    assert t.first_diff([1, 2, 3], [1, 2, 3]) is None
    assert t.first_diff([1, 2, 3], [1, 9, 3]) == 1
    assert t.first_diff([1, 2], [1, 2, 3]) == 2
    assert "different at token 2 (got 198, ref 271" in t.diff_text([198, 1, 198], [198, 1, 271])
    assert t.diff_text([1], [1]).startswith("equal")
