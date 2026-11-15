"""Monte Carlo equity against opponents holding random cards.

Uses phevaluator, a C hand evaluator about 70x faster than PokerKit's, and a
random generator passed in by the caller, so results are reproducible.
"""

import random
from collections.abc import Sequence

from phevaluator import evaluate_cards

from pokerlab.poker.cards import FULL_DECK

# phevaluator's card ids: rank * 4 + suit, in the same order as FULL_DECK.
CARD_IDS: dict[str, int] = {card: index for index, card in enumerate(FULL_DECK)}

BOARD_SIZE = 5


def estimate_equity(
    hole_cards: Sequence[str],
    board: Sequence[str],
    opponent_count: int,
    samples: int,
    rng: random.Random,
) -> float:
    """Expected share of the pot at showdown, between 0 and 1.

    Each sample deals random hands to the opponents and completes the board.
    A split pot counts as a fractional win.
    """
    if opponent_count == 0:
        return 1.0

    hole_ids = [CARD_IDS[card] for card in hole_cards]
    board_ids = [CARD_IDS[card] for card in board]
    known = set(hole_ids) | set(board_ids)
    unseen = [card_id for card_id in CARD_IDS.values() if card_id not in known]
    missing_board = BOARD_SIZE - len(board_ids)

    total = 0.0
    for _ in range(samples):
        drawn = rng.sample(unseen, missing_board + 2 * opponent_count)
        full_board = board_ids + drawn[:missing_board]
        opponent_holes = _pairs(drawn[missing_board:])
        ours = evaluate_cards(*full_board, *hole_ids)
        theirs = [evaluate_cards(*full_board, *hole) for hole in opponent_holes]
        total += _pot_share(ours, theirs)
    return total / samples


def _pairs(card_ids: list[int]) -> list[list[int]]:
    return [card_ids[i : i + 2] for i in range(0, len(card_ids), 2)]


def _pot_share(our_rank: int, opponent_ranks: list[int]) -> float:
    """Our share of the pot given evaluator ranks, where lower is stronger."""
    best_opponent = min(opponent_ranks)
    if our_rank < best_opponent:
        return 1.0
    if our_rank > best_opponent:
        return 0.0
    tied_opponents = opponent_ranks.count(best_opponent)
    return 1.0 / (tied_opponents + 1)
