"""Tests for the duel eval, with scripted bots only (no model calls).

    uv run --project harness python -m pytest duel -q
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import duel


def args(a: str, b: str, seeds: int = 3, hands: int = 4, deadline: float = 5.0) -> argparse.Namespace:
    return argparse.Namespace(a=a, b=b, seeds=seeds, hands=hands, deadline_minutes=deadline,
                              decision_timeout=30.0, base_seed=duel.BASE_SEED, block=0)


def play(tmp_path: Path, a: str, b: str, **kw) -> dict:
    asyncio.run(duel.run(args(a, b, **kw), tmp_path))
    return duel.report(tmp_path)


def test_seed_list_is_fixed_and_distinct():
    assert duel.seeds(12) == duel.seeds(12)
    assert duel.seeds(12)[:5] == duel.seeds(5)  # a bigger run extends a smaller one
    assert len(set(duel.seeds(200))) == 200


def test_seed_blocks_never_overlap():
    """Blocks are separate streams: no two blocks share a deal, whatever seed counts the runs used."""
    blocks = [set(duel.seeds(n, block=b)) for b, n in ((0, 64), (1, 32), (2, 64), (3, 128))]
    for i, x in enumerate(blocks):
        for y in blocks[i + 1:]:
            assert not x & y
    assert duel.seeds(16, block=2) == duel.seeds(64, block=2)[:16]  # a bigger run extends a smaller one


def test_pooled_report(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(), two.mkdir()
    asyncio.run(duel.run(args("bot:nit", "bot:maniac", seeds=2), one))
    second = args("bot:nit", "bot:maniac", seeds=2)
    second.block = 1
    asyncio.run(duel.run(second, two))
    assert duel.report(one, two)["pairs"] == 4
    other = tmp_path / "other"
    other.mkdir()
    asyncio.run(duel.run(args("bot:nit", "bot:random", seeds=1), other))
    with pytest.raises(SystemExit):
        duel.report(one, other)


def test_pooling_runs_with_the_same_seeds_keeps_both(tmp_path):
    """Two runs on the same deals: both runs' pairs count (nothing is overwritten) and the overlap is flagged."""
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(), two.mkdir()
    asyncio.run(duel.run(args("bot:nit", "bot:maniac", seeds=2), one))
    asyncio.run(duel.run(args("bot:nit", "bot:maniac", seeds=2), two))
    s = duel.report(one, two)
    assert (s["pairs"], s["reused_seeds"]) == (4, 2)


def test_lineup_mirrors_seats():
    assert duel.lineup("A", "B", False) == ["A", "B", "A", "B", "A", "B"]
    assert duel.lineup("A", "B", True) == ["B", "A", "B", "A", "B", "A"]


def test_null_duel_scores_exactly_zero(tmp_path):
    """The same deterministic bot on both sides: mirrored twins play identically, so every edge is exactly 0."""
    s = play(tmp_path, "bot:random", "bot:random")
    d = s["b_minus_a"]
    assert (d["ev_bb100"], d["bb100_as_dealt"], d["elo"]) == (0, 0, 0)
    assert s["p_b_better_chips"] == 0.5
    assert s["verdict"]["overall"] == "no clear difference"


def test_twins_get_the_same_cards(tmp_path):
    asyncio.run(duel.run(args("bot:nit", "bot:maniac", seeds=1, hands=3), tmp_path))
    ab, ba = (json.loads(p.read_text()) for p in sorted(tmp_path.glob("*-*.json")))
    cards = lambda h: {n: s["hole_cards"] for n, s in h["starting_state"].items()}
    same = 0
    for h1, h2 in zip(ab["hands"], ba["hands"], strict=False):
        if set(h1["starting_state"]) != set(h2["starting_state"]):
            break  # a seat busted in one twin only: the deal shifts from here on (the mirror's known limit)
        assert cards(h1) == cards(h2)
        same += 1
    assert same >= 1
    assert ab["duel"]["models"] == ["bot:nit", "bot:maniac"] * 3
    assert ba["duel"]["models"] == ["bot:maniac", "bot:nit"] * 3


def test_report_is_antisymmetric(tmp_path):
    """Swapping A and B flips the sign of every edge."""
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(), two.mkdir()
    s1, s2 = play(one, "bot:nit", "bot:maniac"), play(two, "bot:maniac", "bot:nit")
    assert s1["b_minus_a"]["bb100_as_dealt"] == pytest.approx(-s2["b_minus_a"]["bb100_as_dealt"], abs=0.11)
    assert s1["b_minus_a"]["elo"] == pytest.approx(-s2["b_minus_a"]["elo"], abs=1)


def test_cut_matches_a_shorter_game(tmp_path):
    """Scoring a 6-hand game at hand 3 gives the same stacks and places as playing 3 hands."""
    long_dir, short_dir = tmp_path / "long", tmp_path / "short"
    long_dir.mkdir(), short_dir.mkdir()
    asyncio.run(duel.run(args("bot:nit", "bot:maniac", seeds=1, hands=6), long_dir))
    asyncio.run(duel.run(args("bot:nit", "bot:maniac", seeds=1, hands=3), short_dir))
    for name in ("ab", "ba"):
        long_log = json.loads(next(long_dir.glob(f"*-{name}.json")).read_text())
        short_log = json.loads(next(short_dir.glob(f"*-{name}.json")).read_text())
        assert len(short_log["hands"]) == 3
        assert duel.cut(long_log, 3)["net"] == duel.cut(short_log, 3)["net"]
        assert duel.cut(long_log, 3)["places"] == duel.cut(short_log, 3)["places"]


def test_pairs_are_scored_on_common_hands():
    """A pair whose games stopped at different hands is scored on the hands both completed."""
    logs = []
    for swapped, hands in ((False, 5), (True, 3)):
        log = {"session_info": {"seed": 1, "big_blind": 2, "starting_stacks": [200] * 6,
                                "player_names": [f"seat{i}" for i in range(6)]},
               "duel": {"swapped": swapped}, "eliminated_on_hand": {}, "ranking": {"mode": "elimination"},
               "hands": [{"hand_number": k + 1, "ending_stacks": {f"seat{i}": 200 + (k + 1) * (1 if i % 2 else -1) for i in range(6)}}
                         for k in range(hands)]}
        logs.append(log)
    (row,) = duel.pair_scores(logs)
    assert row["hands"] == 3


def test_elo_scale():
    assert duel.elo(0.5) == 0
    assert duel.elo(0.64) == pytest.approx(100, abs=1)
