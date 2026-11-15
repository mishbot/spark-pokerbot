"""Cards as two-character strings ("As", "Td") and a seeded deck.

Strings keep every other module independent of any poker library: they
print readably, serialize to JSON as-is, and PokerKit accepts them directly.
"""

import random
from itertools import product

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK: tuple[str, ...] = tuple(rank + suit for rank, suit in product(RANKS, SUITS))


class SeededDeck:
    """A shuffled deck that deals from the top.

    The order depends only on (seed, hand_number), so every hand with the
    same seed and number is dealt identically, whoever is sitting at the table.
    """

    def __init__(self, seed: int, hand_number: int) -> None:
        self._cards = list(FULL_DECK)
        random.Random(f"{seed}-{hand_number}").shuffle(self._cards)

    def draw(self, count: int = 1) -> list[str]:
        drawn, self._cards = self._cards[:count], self._cards[count:]
        return drawn
