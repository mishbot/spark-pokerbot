"""Random numbers keyed to the decision, not to the order of play.

A bot that draws from one generator per game shifts every later draw as soon as
one earlier action differs, so two candidates compared on the same seed stop
sharing luck after the first divergence. Keying each draw to (game, seat, hand,
point in the hand) keeps the same spot getting the same draw in both games.
"""

import random
import zlib

from pokerlab.datatypes.observation import Observation


def seat_key(seed: int, seat: int) -> str:
    """The part of every key that is fixed for one seat in one game."""
    return f"{seed}-seat-{seat}"


def decision_key(key: str, observation: Observation) -> str:
    """The same seat, hand and number of earlier actions give the same key."""
    return f"{key}-hand-{observation.hand_number}-action-{len(observation.history)}"


def decision_rng(key: str, observation: Observation) -> random.Random:
    return random.Random(decision_key(key, observation))


def decision_seed(key: str, observation: Observation) -> int:
    """A 31-bit integer, for APIs that take a numeric seed (e.g. vLLM sampling)."""
    return zlib.crc32(decision_key(key, observation).encode()) & 0x7FFFFFFF
