"""Tests for the Modern Poker Theory preflop grading (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpt


def test_hand_class():
    assert mpt.hand_class(["Kd", "As"]) == "AKo"
    assert mpt.hand_class(["7h", "8h"]) == "87s"
    assert mpt.hand_class(["Tc", "Td"]) == "TT"


def test_grid_orientation():
    """Suited hands sit above the diagonal: the BN opens K2s but folds K2o (book text: K2s+, K8o+)."""
    bn = mpt.RFI_GRIDS["BN"]
    assert (bn["K2s"], bn["K2o"], bn["K8o"], bn["K7o"]) == ("R", "F", "R", "F")
    assert mpt.BB_GRIDS["LJ"]["A9o"] == "F"  # "hands as strong as A9o always get folded" vs a LJ open
    assert len(bn) == 169


@pytest.mark.parametrize("pos", list(mpt.RFI))
def test_rfi_shares_match_the_book(pos):
    lo, hi = mpt.share(mpt.RFI_GRIDS[pos], "P" if pos == "SB" else "R")
    assert lo - 1 <= mpt.RFI[pos][1] <= hi + 1


@pytest.mark.parametrize("pos", list(mpt.BB_DEFENCE))
def test_bb_shares_match_the_book(pos):
    lo, hi = mpt.share(mpt.BB_GRIDS[pos], "D")
    assert lo - 1 <= mpt.BB_DEFENCE[pos][1] <= hi + 1


def hand(actions, cards=None, stacks=None):
    """A six-handed hand: seat i has position i (0 SB, 1 BB, 2 LJ, 3 HJ, 4 CO, 5 BN)."""
    cards = cards or {}
    return {
        "hand_number": 1,
        "starting_state": {
            f"seat{i}": {"position": i, "hole_cards": cards.get(i, ["7c", "2d"]),
                         "starting_stack": (stacks or {}).get(i, 200)}
            for i in range(6)
        },
        "actions": [
            {"player": f"seat{s}", "street": "preflop",
             "action": {"type": t, **({"amount": amt} if amt else {})}, "model_response": {}}
            for s, t, amt in actions
        ],
    }


def log(*hands):
    return {"session_info": {"big_blind": 2}, "hands": list(hands)}


def test_rfi_and_bb_spots():
    # LJ folds, HJ opens 2.5x with AKo, CO/BN/SB fold, BB calls with 72o
    h = hand([(2, "fold", None), (3, "raise", 5), (4, "fold", None), (5, "fold", None), (0, "fold", None), (1, "call", None)],
             cards={3: ["Ad", "Kc"], 1: ["7s", "2h"]})
    got = {(s["spot"], s["player"]): s for s in mpt.spots(log(h))}
    assert set(got) == {("RFI", "seat2"), ("RFI", "seat3"), ("BB vs HJ", "seat1")}
    assert mpt.verdict(got[("RFI", "seat3")]) == "agree"
    assert mpt.verdict(got[("RFI", "seat2")]) == "agree"  # 72o folded from LJ
    assert mpt.verdict(got[("BB vs HJ", "seat1")]) == "too_loose"
    assert got[("BB vs HJ", "seat1")]["open_bb"] == 2.5


def test_limp_and_big_opens_and_short_stacks():
    # a CO limp with AA is flagged; nothing after it is first-in; the BB facing a limp is not a BB-vs-open spot
    h = hand([(2, "fold", None), (3, "fold", None), (4, "call", None), (5, "fold", None), (0, "fold", None), (1, "check", None)],
             cards={4: ["Ah", "As"]})
    (s,) = [s for s in mpt.spots(log(h)) if s["player"] == "seat4"]
    assert mpt.verdict(s) == "limp"
    assert not [s for s in mpt.spots(log(h)) if s["player"] == "seat1"]
    # a 5x open is not graded for the BB
    big = hand([(2, "raise", 10), (3, "fold", None), (4, "fold", None), (5, "fold", None), (0, "fold", None), (1, "fold", None)])
    assert not [s for s in mpt.spots(log(big)) if s["spot"].startswith("BB")]
    # a short stack (30 BB) still to act stops RFI grading; one that has already folded doesn't
    short_behind = hand([(2, "fold", None), (3, "raise", 5), (4, "fold", None), (5, "fold", None), (0, "fold", None),
                         (1, "fold", None)], stacks={5: 60})
    assert [s["player"] for s in mpt.spots(log(short_behind)) if s["spot"] == "RFI"] == []
    short_folded = hand([(2, "fold", None), (3, "raise", 5), (4, "fold", None), (5, "fold", None), (0, "fold", None),
                         (1, "fold", None)], stacks={2: 60})
    assert [s["player"] for s in mpt.spots(log(short_folded)) if s["spot"] == "RFI"] == ["seat3"]


def test_forced_folds_are_skipped():
    h = hand([(2, "fold", None), (3, "fold", None)])
    h["actions"][1]["model_response"] = {"format_error": True}
    assert [s["player"] for s in mpt.spots(log(h))] == ["seat2"]


def test_mixed_cells_are_not_graded():
    s = {"spot": "RFI", "position": "BN", "chart": "M", "action": "raise"}
    assert mpt.verdict(s) is None
