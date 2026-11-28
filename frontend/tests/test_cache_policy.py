"""CACHE_POLICY=smart: frontend/cache_policy.py against engine/src/retention.h (built from engine/tests/smart_policy_cli.cc with g++; the
agreement tests are skipped without a compiler). No GPU."""

import pathlib
import random
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))

import cache_policy as cp  # noqa: E402


@pytest.fixture(scope="module")
def cli(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++: the C++ side of the agreement test cannot be built")
    exe = tmp_path_factory.mktemp("cli") / "smart_policy_cli"
    subprocess.run([gxx, "-std=c++20", "-O1", "-Wall", "-Wextra", f"-I{ROOT / 'engine' / 'src'}",
                    str(ROOT / "engine" / "tests" / "smart_policy_cli.cc"), "-o", str(exe)], check=True)

    def run(text):
        return subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True).stdout.splitlines()
    return run


def test_policy_and_params_from_env():
    assert cp.policy_of(None) == "smart" and cp.policy_of("") == "smart" and cp.DEFAULT == "smart"  # (the server default)
    assert cp.policy_of(" Smart\n") == "smart" and cp.policy_of("LEGACY") == "legacy"
    with pytest.raises(ValueError, match="legacy or smart"):
        cp.policy_of("lru")
    p = cp.SmartParams.from_env({"CACHE_PREFILL_T0_US": "60", "CACHE_PREFILL_K_US": "x", "CACHE_SUPERSEDED_FACTOR": "-1",
                                 "CACHE_MIN_RESIDENT_S": "0", "CACHE_JUNCTION_FACTOR": "3.5"})
    assert (p.t0_us, p.k_us, p.superseded_factor, p.min_resident_s, p.junction_factor) == (60, 0.31, 0.25, 0, 3.5)
    d = cp.SmartParams()
    assert 16800 < 30000 / cp.prefill_seconds(d, 0, 30000) < 17300      # the measured speeds the defaults come from
    assert 13400 < 125000 / cp.prefill_seconds(d, 0, 125000) < 13800


def _rand_cands(rng, n):
    return [cp.Cand(depth=rng.choice([9000, 20000, 30000, 50000, 90000, 125000]) + rng.randint(0, 5) * 256,
                    parent_depth=rng.choice([0, 0, 9000, 20000]), bytes=rng.choice([0, 1 << 20, 300 << 20, 1 << 30, 3 << 30]),
                    hits=rng.choice([0, 0, 1, 4]), junction=rng.random() < 0.2, superseded=rng.random() < 0.3,
                    leaf=rng.random() < 0.6, young=rng.random() < 0.25, stamp=rng.choice([0.0, 0.0, 1.5, 3.25, rng.random() * 5]),
                    age_key=float(rng.randint(0, 6))) for _ in range(n)]


def test_pick_agrees_with_the_engine(cli):
    rng = random.Random(7)
    queries, expect = [], []
    for _ in range(600):
        p = cp.SmartParams(t0_us=rng.choice([54.0, 40.0]), k_us=rng.choice([0.31, 0.0, 1.2]),
                           superseded_factor=rng.choice([0.25, 0.0, 1.0]), junction_factor=rng.choice([2.0, 1.0, 4.0]),
                           hit_weight=rng.choice([1.0, 0.0, 0.5]))
        cands = _rand_cands(rng, rng.randint(0, 9))
        lines = [f"pick {p.t0_us!r} {p.k_us!r} {p.superseded_factor!r} {p.junction_factor!r} {p.hit_weight!r} {len(cands)}"]
        lines += [f"{c.depth!r} {c.parent_depth!r} {c.bytes!r} {c.hits} {int(c.junction)} {int(c.superseded)} {int(c.leaf)} "
                  f"{int(c.young)} {c.stamp!r} {c.age_key!r}" for c in cands]
        queries.append("\n".join(lines))
        expect.append(cp.smart_pick(cands, p))
    out = cli("\n".join(queries) + "\n")
    assert len(out) == len(expect)
    for line, k in zip(out, expect):
        _, idx, prio, reason, ys = line.split()
        assert (int(idx), int(reason), bool(int(ys))) == (k.idx, k.reason, k.young_skipped), (line, k)
        if k.idx >= 0:
            assert float(prio) == pytest.approx(k.priority, rel=1e-12, abs=1e-15)


def _rand_chains(rng, trial):
    ids = []
    for i in range(rng.randint(1, 10)):
        if ids and rng.random() < 0.7:
            base = rng.choice(ids)
            keep = rng.randint(0, len(base))
            ids.append(base[:keep] + [trial * 1000 + i * 37 + k for k in range(rng.choice([0, 1, 256, 300, 600]))])
        else:
            ids.append([trial * 100000 + i * 5000 + k for k in range(rng.randint(1, 900))])
    return [(v, rng.randint(0, 2), 1 << rng.randint(0, 2)) for v in ids]


def test_chain_info_agrees_with_the_engine(cli):
    rng = random.Random(11)
    cases = [_rand_chains(rng, t) for t in range(150)]
    text = ""
    for nodes in cases:
        text += f"chain {len(nodes)}\n" + "".join(f"{o} {t} {len(v)} {' '.join(map(str, v))}\n" for v, o, t in nodes)
    out = cli(text)
    k = 0
    for nodes in cases:
        for info in cp.chain_info(nodes):
            got = list(map(int, out[k].split()))
            k += 1
            want = [info["parent"], info["parent_depth"], info["desc_tiers"], int(info["branch"]), int(info["shared"])]
            assert got[1:] == want[1:], (nodes, got, want)
            # (the parent index may differ between equal copies of the parent: its depth is what counts)
    assert k == len(out)


def _py_ram_victim(ents, now, min_resident_s):
    """smart_ram_victim from the Python pieces: chain_info over the entries, the candidates, smart_choose."""
    p = cp.SmartParams(min_resident_s=min_resident_s)
    info = cp.chain_info([(e["ids"], e["owner"], cp.TIER_RAM) for e in ents])
    gd = cp.GdTier()
    cls = {0: [], 1: []}
    for i, (e, ci) in enumerate(zip(ents, info)):
        c = cp.Cand(depth=len(e["ids"]), parent_depth=ci["parent_depth"], bytes=e["freed"], hits=e["hits"],
                    junction=ci["branch"] or ci["shared"], superseded=e["sup"], leaf=not ci["desc_tiers"] & cp.TIER_RAM,
                    young=now - e["born"] < p.min_resident_s, stamp=gd.stamp_at(e["last_use"]), age_key=e["born"], state=e["state"], ref=i)
        cls[0 if e["live"] else 1].append(c)
    k, pick = cp.smart_choose(cls[0], cls[1], p)
    return -1 if k < 0 else cls[k][pick.idx].ref


def test_ram_victim_agrees_with_the_engine(cli):
    rng = random.Random(3)
    text, expect = "", []
    for t in range(200):
        nodes = _rand_chains(rng, t)
        ents = [{"ids": v, "owner": o, "live": rng.random() < 0.2, "freed": rng.choice([190 << 20, 400 << 20, 1 << 30, 2 << 30]),
                 "state": 190 << 20, "hits": rng.choice([0, 0, 2]), "sup": rng.random() < 0.3, "born": float(rng.randint(0, 1000)),
                 "last_use": rng.randint(0, 50)} for v, o, _ in nodes]
        now, mr = 1000.0, rng.choice([0.0, 300.0, 900.0])
        text += f"ram {now!r} {mr!r} {len(ents)}\n" + "".join(
            f"{int(e['live'])} {e['freed']} {e['state']} {e['hits']} {int(e['sup'])} {e['born']!r} {e['last_use']} {e['owner']} "
            f"{len(e['ids'])} {' '.join(map(str, e['ids']))}\n" for e in ents)
        expect.append(_py_ram_victim(ents, now, mr))
    out = cli(text)
    assert [int(line.split()[1]) for line in out] == expect


def test_gd_tier_matches_the_header_semantics():
    gd = cp.GdTier()
    assert gd.stamp_at(5) == 0
    gd.evicted(cp.Pick(0, 3.0, cp.LEAF), 10)
    assert gd.stamp_at(10) == 0 and gd.stamp_at(11) == 3.0
    gd.evicted(cp.Pick(0, 1.0, cp.NON_LEAF), 20)
    assert gd.L == 3.0
    gd.evicted(cp.Pick(0, 7.5, cp.YOUNG, True), 30)
    st = gd.stats()
    assert (st["evictions_leaf"], st["evictions_non_leaf"], st["evictions_young"], st["young_skipped"]) == (1, 1, 1, 1)
    assert st["victim_priority_mean"] == pytest.approx(11.5 / 3) and st["watermark"] == 7.5
