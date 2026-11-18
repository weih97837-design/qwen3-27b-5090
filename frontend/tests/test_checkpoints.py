"""worker.Scheduler.checkpoints with fake token ids: the positions planned for a prompt (no GPU)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker  # noqa: E402

IM, USER, ASSISTANT, BRK, VEND = 99999, 98000, 98001, 97000, 96000
MEDIA = worker.VOCAB + 5  # (media ids are >= the vocabulary size)


def sched(vision_end=VEND):
    s = object.__new__(worker.Scheduler)  # (class defaults: no tiers; the ids below are what the server's config gives)
    s.im_start, s.user, s.breaks, s.vision_end = IM, USER, frozenset({BRK}), vision_end
    return s


def text(n, base=1000):
    return list(range(base, base + n))


def media(n, base=0):
    return [MEDIA + base + i for i in range(n)]


def prompt(system, user, assistant_turns=()):
    """system / user: token lists; the final user message is `user`, then the generation prompt."""
    p = [IM, 1] + system
    for u, a in assistant_turns:
        p += [IM, USER] + u + [IM, ASSISTANT] + a
    return p + [IM, USER] + user + [IM, ASSISTANT, 7]


def ends(p):
    starts = [i for i, t in enumerate(p) if t == IM]
    return starts


def test_text_only_prompt_has_no_media_checkpoint():
    p = prompt(text(300), text(500, 5000))
    st = ends(p)
    assert sched().checkpoints(p, 0) == sorted({st[1], st[-2], st[-1]})


def test_media_end_includes_vision_end_but_not_the_question():
    video = [5] + media(2000) + [VEND]
    p = prompt(text(300), video + text(300, 7000))
    st = ends(p)
    end = p.index(VEND, st[-2]) + 1
    at = sched().checkpoints(p, 0)
    assert end in at and at == sorted({st[1], st[-2], st[-1], end})
    assert all(t < worker.VOCAB for t in p[end:])


def test_image_followed_by_text_and_several_media_use_the_last():
    p = prompt(text(300), text(40, 3000) + media(900) + [VEND] + text(3000, 3000) + media(1500, 9000) + [VEND] + text(400, 8000))
    st = ends(p)
    last = max(i for i, t in enumerate(p) if t >= worker.VOCAB) + 2
    at = sched().checkpoints(p, 0)
    first = p.index(VEND) + 1
    assert last in at and first not in at and at == sorted({st[1], st[-1], last})  # (the first media's end is not planned)


def test_without_vision_end_id_or_token_the_position_follows_the_last_media_id():
    p = prompt(text(300), media(1000) + text(200, 7000))
    st = ends(p)
    end = max(i for i, t in enumerate(p) if t >= worker.VOCAB) + 1
    assert end in sched().checkpoints(p, 0)
    assert end in sched(vision_end=None).checkpoints(p, 0)


def test_media_in_an_earlier_message_only_plans_nothing_extra():
    earlier = [(media(2000) + [VEND] + text(30, 7000), text(50, 8000))]
    p = prompt(text(300), text(400, 5000), earlier)
    st = ends(p)
    assert sched().checkpoints(p, 0) == sorted({st[1], st[-2], st[-1]})


def test_media_in_the_system_message_only_plans_nothing_extra():
    p = prompt(text(100) + media(2000) + [VEND], text(400, 5000))
    st = ends(p)
    assert sched().checkpoints(p, 0) == sorted({st[1], st[-2], st[-1]})


def test_media_end_is_far_from_the_other_positions_rounded_down():
    p = prompt(text(300), media(1000) + [VEND] + text(500, 7000))
    end = p.index(VEND) + 1
    at = sched().checkpoints(p, 0)
    rounded = [q // 64 * 64 for q in at]
    assert end in at
    assert all(abs(a - b) > 64 for i, a in enumerate(rounded) for b in rounded[i + 1:])


def test_media_end_within_64_of_generation_start_replaces_it():
    p = prompt(text(300), media(1000) + [VEND] + text(70, 7000))  # a short question: the next block
    st = ends(p)
    end = p.index(VEND) + 1
    g = st[-1] // 64 * 64
    assert 0 < g - end // 64 * 64 <= 64
    at = sched().checkpoints(p, 0)
    assert end in at and st[-1] not in at and st[-2] in at and st[1] in at


def test_media_end_in_the_last_64_never_creates_a_chunk_of_64_before_the_generation_prompt():
    for q in range(1, 200):
        p = prompt(text(300), media(1000) + [VEND] + text(q, 7000))
        st = ends(p)
        at = sched().checkpoints(p, 0)
        r = sorted({x // 64 * 64 for x in at})
        assert all(b - a > 64 for a, b in zip(r, r[1:])), (q, at)


def test_media_end_close_to_another_position_is_dropped():
    # the media starts right after the system message: its end is far from starts[-2]; make the media tiny instead
    p = prompt(text(300), media(10) + [VEND] + text(2000, 7000))
    st = ends(p)
    at = sched().checkpoints(p, 0)
    assert (p.index(VEND) + 1) // 64 * 64 - st[-2] // 64 * 64 <= 64
    assert at == sorted({st[1], st[-2], st[-1]})


def test_reusable_keeps_the_media_end_away():
    p = prompt(text(300), media(1000) + [VEND] + text(2000, 7000))
    end = p.index(VEND) + 1
    near = end // 64 * 64 + 64
    assert end not in sched().checkpoints(p, near)
    assert near in sched().checkpoints(p, near)
    assert end in sched().checkpoints(p, end // 64 * 64 + 640)


def test_blank_line_b_after_media_far_away_both_exist():
    q = text(400, 7000) + [BRK] + text(900, 7500)
    p = prompt(text(300), media(1000) + [VEND] + q)
    end, b = p.index(VEND) + 1, p.index(BRK) + 1
    at = sched().checkpoints(p, 0)
    assert end in at and b in at


def test_blank_line_b_close_to_media_end_is_dropped_and_media_end_kept():
    p = prompt(text(300), media(1000) + [VEND] + text(20, 7000) + [BRK] + text(900, 7500))
    end, b = p.index(VEND) + 1, p.index(BRK) + 1
    at = sched().checkpoints(p, 0)
    assert end in at and b not in at


def test_blank_line_b_alone_spacing_and_replacement():
    st_p = prompt(text(300), text(2000, 5000) + [BRK] + text(900, 7500))
    b = st_p.index(BRK) + 1
    assert b in sched().checkpoints(st_p, 0)
    # near the end: replaces the generation prompt's start (same block)
    p = prompt(text(300), text(2000, 5000) + [BRK] + text(80, 7500))
    st = ends(p)
    at = sched().checkpoints(p, 0)
    b = p.index(BRK) + 1
    assert b in at and st[-1] not in at
    # within 64 of the sibling share's position: not added
    p = prompt(text(300), text(2000, 5000) + [BRK] + text(900, 7500))
    b = p.index(BRK) + 1
    assert b not in sched().checkpoints(p, b // 64 * 64 + 64)
    # b at the document start of the final message: close to starts[-2], dropped
    p = prompt(text(300), [BRK] + text(900, 7500))
    assert p.index(BRK) + 1 not in sched().checkpoints(p, 0)


def test_the_test_scripts_print_nothing_a_windows_code_page_cannot_show():
    """`tiered_cache_test.py --help` prints the module docstring: on the Windows CI runner stdout is a pipe in cp1252, and a
    character outside it ended the smoke test (and would end a run in a console with another code page)."""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "scripts"
    for name in ("tiered_cache_test.py", "tiered_cache_bench.py"):
        doc = ast.get_docstring(ast.parse((root / name).read_text(encoding="utf-8"))) or ""
        doc.encode("cp1252")  # raises UnicodeEncodeError for CJK, arrows, ...
