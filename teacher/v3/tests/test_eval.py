"""eval.analyze tests: explicit made-hand cases, draw detection, board tags, and a cross-check
of the category ordering against phevaluator (independent of our rank/suit logic).

Run: PYTHONPATH=teacher uv run --with phevaluator python teacher/v3/tests/test_eval.py
"""

from __future__ import annotations

import random

from phevaluator import evaluate_cards

from v3.eval import analyze, CAT_RANK, RANK_ORDER

FULL = [r + s for r in RANK_ORDER for s in "cdhs"]


def test_straight_is_never_air():  # defect #1
    info = analyze(["9h", "8h"], ["7c", "6d", "5s", "2c", "Kd"])
    assert info.category == "straight", info.category
    assert info.has_pair_or_better
    assert info.made == "straight"


def test_wheel_straight():
    info = analyze(["Ah", "2d"], ["3c", "4s", "5h", "Kd", "Qc"])
    assert info.category == "straight", info.category


def test_categories_explicit():
    cases = {
        ("straight_flush",): (["9h", "8h"], ["7h", "6h", "5h"]),
        ("quads",):          (["9h", "9d"], ["9c", "9s", "2h"]),
        ("full_house",):     (["9h", "9d"], ["9c", "2s", "2h"]),
        ("flush",):          (["Ah", "5h"], ["2h", "9h", "Kh"]),
        ("straight",):       (["9h", "8d"], ["7c", "6s", "5h"]),
        ("trips",):          (["Ah", "9d"], ["9c", "9s", "2h"]),   # trips (one hole card)
        ("two_pair",):       (["Ah", "Kd"], ["Ac", "Ks", "2h"]),
        ("pair",):           (["Ah", "Qd"], ["Ac", "7s", "2h"]),
        ("high_card",):      (["Ah", "Qd"], ["9c", "7s", "2h"]),
    }
    for (cat,), (hole, board) in cases.items():
        got = analyze(hole, board).category
        assert got == cat, f"{hole}+{board}: expected {cat}, got {got}"


def test_set_vs_trips():
    assert analyze(["9h", "9d"], ["9c", "2s", "7h"]).made == "set"     # pocket pair hits board
    assert analyze(["Ah", "9d"], ["9c", "9s", "7h"]).made == "trips"   # board brings two


def test_pair_relations():
    assert analyze(["Ah", "Kd"], ["Ac", "7s", "2h"]).made == "top pair"
    assert analyze(["Qh", "Qd"], ["7c", "5s", "2h"]).made == "overpair"
    assert analyze(["8h", "8d"], ["Kc", "5s", "2h"]).made == "pocket pair below top"
    assert analyze(["Ah", "7d"], ["Kc", "7s", "2h"]).made == "second pair"
    assert analyze(["Ah", "Kd"], ["7c", "7s", "2h"]).made == "board pair"


def test_ace_high_is_high_card():  # defect #8: ace-high must not be "strong"
    info = analyze(["Ah", "Qd"], ["9c", "7s", "2h"])
    assert info.category == "high_card"
    assert not info.has_pair_or_better


def test_draws():
    fd = analyze(["Ah", "5h"], ["2h", "9h", "Kd"])            # four hearts, one more to come
    assert fd.flush_draw and fd.category != "flush"
    oesd = analyze(["9h", "8d"], ["7c", "6s", "2h"])          # 9-8-7-6 open-ended
    assert oesd.oesd and not oesd.gutshot
    gut = analyze(["9h", "5d"], ["7c", "6s", "2h"])           # needs an 8
    assert gut.gutshot and not gut.oesd


def test_board_tags():
    assert analyze(["Ah", "Kd"], ["2c", "7s", "Td"]).board_tag == "dry"
    assert analyze(["Ah", "Kd"], ["2c", "7c", "Td"]).board_tag == "two_tone"
    assert analyze(["Ah", "Kd"], ["2c", "7c", "Tc"]).board_tag == "monotone"
    assert analyze(["Ah", "Kd"], ["2c", "7c", "Tc", "9c"]).board_tag == "four_flush"
    assert analyze(["Ah", "Kd"], ["7s", "7c", "Td"]).board_tag == "paired"
    assert analyze(["Ah", "2d"], ["7c", "8s", "9d", "Tc"]).board_tag == "four_straight"
    for tag in ("four_flush", "paired", "four_straight", "monotone"):
        pass  # dangerous flags checked below


def test_connected_board_not_dry():  # Opus systemic #3
    assert analyze(["Ac", "2d"], ["9h", "Js", "8d"]).board_tag == "connected"   # 9-J-8 coordinated
    assert analyze(["Ac", "2d"], ["9h", "Js", "8h"]).board_tag == "wet"         # connected + two-tone
    assert analyze(["Ac", "2d"], ["2s", "7h", "Kd"]).board_tag == "dry"         # genuinely dry


def test_plays_the_board():  # Opus FAIL #11
    # 98 never plays on 7-7-7-K-T; the board is the hand
    assert analyze(["9h", "8d"], ["Kd", "7c", "Tc", "7d", "7s"]).plays_board
    # a hand whose kicker/cards improve on the board does NOT play the board
    assert not analyze(["Ah", "9d"], ["Kd", "7c", "Tc", "7d", "2s"]).plays_board


def test_dangerous_flags():
    assert analyze(["Ah", "Kd"], ["2c", "7c", "Tc", "9c"]).dangerous   # four flush
    assert analyze(["Ah", "Kd"], ["7s", "7c", "Td"]).dangerous        # paired
    assert not analyze(["Ah", "Kd"], ["2c", "7s", "Td"]).dangerous    # dry


def test_category_ordering_matches_phevaluator():
    """For random 7-card hands: a higher category must have a better (lower) phevaluator rank.
    This cross-checks our categorizer against an independent evaluator."""
    rng = random.Random(0)
    checks = 0
    for _ in range(3000):
        cards = rng.sample(FULL, 7)
        hole, board = cards[:2], cards[2:]
        info = analyze(hole, board)
        rank = evaluate_cards(*cards)  # lower = stronger
        # bucket by category and compare pairwise across a batch
        checks += 1
        # store for cross comparison
        test_category_ordering_matches_phevaluator._seen.append((CAT_RANK[info.category], rank, info.category))
    seen = test_category_ordering_matches_phevaluator._seen
    # verify: whenever two hands differ in category, the higher category has the <= rank boundary.
    # Group min/max phevaluator rank per category and assert the ranges do not invert.
    by_cat = {}
    for cat_rank, rank, cat in seen:
        lo, hi = by_cat.get(cat_rank, (10**9, -1))
        by_cat[cat_rank] = (min(lo, rank), max(hi, rank))
    ordered = sorted(by_cat.items())  # by CAT_RANK ascending (weak -> strong)
    for i in range(len(ordered) - 1):
        weaker_cat, (w_lo, w_hi) = ordered[i]
        stronger_cat, (s_lo, s_hi) = ordered[i + 1]
        # a stronger category's WORST (max) rank must be better (<=) than the weaker category's BEST (min) rank
        assert s_hi <= w_lo, (
            f"category ordering inverted: cat_rank {stronger_cat} max rank {s_hi} "
            f"> cat_rank {weaker_cat} min rank {w_lo}"
        )
    assert checks > 2000


test_category_ordering_matches_phevaluator._seen = []


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
