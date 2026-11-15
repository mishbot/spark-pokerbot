"""All 1,326 two-card starting hands, as arrays the range model can index.

Cards use phevaluator's ids (rank * 4 + suit, the order of FULL_DECK), so a
combo can go straight into the evaluator.
"""

from collections.abc import Iterable
from itertools import combinations

import numpy as np

from pokerlab.poker.cards import FULL_DECK, RANKS
from pokerlab.poker.equity import CARD_IDS

# COMBOS[i] = (higher card id, lower card id)
COMBOS: tuple[tuple[int, int], ...] = tuple((b, a) for a, b in combinations(range(len(FULL_DECK)), 2))
COMBO_COUNT = len(COMBOS)
FIRST = np.array([combo[0] for combo in COMBOS])
SECOND = np.array([combo[1] for combo in COMBOS])


def hand_class(first: int, second: int) -> str:
    """The 169-way class: "AA", "AKs", "AKo". The higher rank comes first."""
    high, low = max(first, second), min(first, second)
    high_rank, low_rank = RANKS[high // 4], RANKS[low // 4]
    if high_rank == low_rank:
        return high_rank * 2
    return high_rank + low_rank + ("s" if high % 4 == low % 4 else "o")


COMBO_CLASSES: tuple[str, ...] = tuple(hand_class(first, second) for first, second in COMBOS)


def card_ids(cards: Iterable[str]) -> list[int]:
    return [CARD_IDS[card] for card in cards]


def combo_index(cards: Iterable[str]) -> int:
    first, second = sorted(card_ids(cards), reverse=True)
    return _INDEX[(first, second)]


def blocked(cards: Iterable[int]) -> np.ndarray:
    """True for every combo that uses one of `cards` (ids), so it cannot be held."""
    dead = np.zeros(len(FULL_DECK), dtype=bool)
    dead[list(cards)] = True
    return dead[FIRST] | dead[SECOND]


_INDEX = {combo: index for index, combo in enumerate(COMBOS)}
