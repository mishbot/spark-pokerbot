"""How strong every combo is: preflop by percentile, after the flop by made hand on this board.

Both are arrays over COMBOS, so a range (a weight per combo) can be filtered
by strength with one numpy expression.
"""

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from phevaluator import evaluate_cards

from pokerlab.datatypes.preflop_strength import PREFLOP_EQUITY
from pokerlab.poker.combos import COMBO_CLASSES, COMBO_COUNT, COMBOS, blocked, card_ids


def _preflop_percentiles() -> np.ndarray:
    """0 for the best combo, near 1 for the worst; ties (same class) share the class's top."""
    equity = np.array([PREFLOP_EQUITY[hand_class] for hand_class in COMBO_CLASSES])
    order = np.argsort(-equity, kind="stable")
    percentile = np.empty(COMBO_COUNT)
    percentile[order] = np.arange(COMBO_COUNT) / COMBO_COUNT
    class_top = {}
    for index in order:
        class_top.setdefault(COMBO_CLASSES[index], percentile[index])
    return np.array([class_top[hand_class] for hand_class in COMBO_CLASSES])


# Boards of the last few hands; every bot at the table reads the same ones.
BOARD_CACHE_SIZE = 64

# PREFLOP_PERCENTILE[i]: share of all combos stronger than combo i. "Top 12%" is < 0.12.
PREFLOP_PERCENTILE = _preflop_percentiles()


def board_strength(board: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(strength, live) for a flop, turn or river.

    strength[i] is the share of live combos that combo i's made hand beats or ties,
    so 1.0 is the nuts. live[i] is False when the combo uses a board card.
    """
    return _board_strength(tuple(sorted(board)))


def board_ranks(board: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(ranks, live): each combo's evaluator rank with this board (lower is stronger)."""
    return _board_ranks(tuple(sorted(board)))


@lru_cache(maxsize=BOARD_CACHE_SIZE)
def _board_strength(board: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    ranks, live = _board_ranks(board)
    live_ranks = np.sort(ranks[live])
    # Combos with a worse or equal rank, as a share of all live combos.
    beaten_or_tied = len(live_ranks) - np.searchsorted(live_ranks, ranks, side="left")
    strength = np.where(live, beaten_or_tied / len(live_ranks), 0.0)
    return strength, live


@lru_cache(maxsize=BOARD_CACHE_SIZE)
def _board_ranks(board: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    board_ids = card_ids(board)
    live = ~blocked(board_ids)
    ranks = np.full(COMBO_COUNT, np.iinfo(np.int32).max, dtype=np.int64)
    for index in np.flatnonzero(live):
        first, second = COMBOS[index]
        ranks[index] = evaluate_cards(*board_ids, first, second)
    return ranks, live
