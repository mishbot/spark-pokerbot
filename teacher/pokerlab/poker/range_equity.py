"""Equity against opponents holding weighted ranges (from bots/field/ranges.py).

Exact on the river against one opponent; otherwise Monte Carlo, drawing each
opponent's hand from its weights and completing the board.
"""

import random
from collections.abc import Sequence

import numpy as np
from phevaluator import evaluate_cards

from pokerlab.poker.cards import FULL_DECK
from pokerlab.poker.combos import COMBOS, blocked, card_ids
from pokerlab.poker.equity import BOARD_SIZE
from pokerlab.poker.strength import board_ranks

MAX_REDRAWS = 3  # a sample whose hands collide is redrawn this many times, then skipped


def range_equity(
    hole_cards: Sequence[str],
    board: Sequence[str],
    ranges: Sequence[np.ndarray],
    samples: int,
    rng: random.Random,
) -> float:
    """Expected share of the pot at showdown, between 0 and 1. Splits count fractionally.

    Combos that use our cards or the board are ignored, whatever weight they were given.
    """
    possible = ~blocked(card_ids((*hole_cards, *board)))
    live = [weights * possible for weights in ranges if (weights * possible).sum() > 0]
    if not live:
        return 1.0
    if len(board) == BOARD_SIZE and len(live) == 1:
        return _exact_river(hole_cards, board, live[0])
    return _sampled(card_ids(hole_cards), card_ids(board), live, samples, rng)


def _exact_river(hole_cards: Sequence[str], board: Sequence[str], weights: np.ndarray) -> float:
    ranks, _ = board_ranks(board)
    ours = evaluate_cards(*board, *hole_cards)
    total = weights.sum()
    wins = weights[ranks > ours].sum() + 0.5 * weights[ranks == ours].sum()
    return float(wins / total)


def _sampled(
    hole: list[int], board: list[int], ranges: list[np.ndarray], samples: int, rng: random.Random
) -> float:
    cumulative = [np.cumsum(weights) for weights in ranges]
    deck = [card for card in range(len(FULL_DECK)) if card not in hole and card not in board]
    missing = BOARD_SIZE - len(board)
    total, counted = 0.0, 0
    for _ in range(samples):
        hands = _draw_hands(cumulative, set(hole) | set(board), rng)
        if hands is None:
            continue
        used = {card for hand in hands for card in hand}
        runout = rng.sample([card for card in deck if card not in used], missing)
        full_board = board + runout
        ours = evaluate_cards(*full_board, *hole)
        theirs = [evaluate_cards(*full_board, *hand) for hand in hands]
        total += _pot_share(ours, theirs)
        counted += 1
    return total / counted if counted else 0.5


def _draw_hands(
    cumulative: list[np.ndarray], used: set[int], rng: random.Random
) -> list[tuple[int, int]] | None:
    """One hand per opponent with no shared cards, or None if collisions persist."""
    for _ in range(MAX_REDRAWS):
        taken = set(used)
        hands = []
        for cum in cumulative:
            index = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
            hand = COMBOS[min(index, len(COMBOS) - 1)]
            if hand[0] in taken or hand[1] in taken:
                break
            taken.update(hand)
            hands.append(hand)
        else:
            return hands
    return None


def _pot_share(ours: int, theirs: list[int]) -> float:
    best = min(theirs)
    if ours < best:
        return 1.0
    if ours > best:
        return 0.0
    return 1.0 / (theirs.count(best) + 1)
