"""Action-conditioned ranges: a weight per combo for every opponent, from this hand's actions.

Recomputed from the public history at each decision, so it needs no state and
replays identically. Preflop actions select percentile bands; actions after
the flop reweight combos by how strong their made hand is on that street's board.
"""

from collections.abc import Mapping

import numpy as np

from pokerlab.bots.rule.ranges import position_group
from pokerlab.datatypes import range_model as model
from pokerlab.datatypes.observation import PREFLOP, STREETS, Observation
from pokerlab.datatypes.replay import ActionEvent
from pokerlab.poker.combos import COMBO_COUNT, blocked, card_ids
from pokerlab.poker.replay import replay
from pokerlab.poker.strength import PREFLOP_PERCENTILE, board_strength

BOARD_SIZE = {"flop": 3, "turn": 4, "river": 5}


def opponent_ranges(observation: Observation, looseness: Mapping[int, float]) -> dict[int, np.ndarray]:
    """Weights for each opponent still in the hand, zero on combos our cards or the board block.

    `looseness[seat]` widens (> 1) or narrows (< 1) that seat's preflop bands; missing seats use 1.
    """
    positions = {view.seat: view.position for view in observation.seats}
    opponents = [v.seat for v in observation.seats if v.in_hand and v.seat != observation.seat]
    weights = {seat: np.ones(COMBO_COUNT) for seat in opponents}

    for event in replay(observation.history):
        if event.seat not in weights:
            continue
        if event.street == PREFLOP:
            band = _preflop_band(event, positions[event.seat], observation.big_blind)
            weights[event.seat] *= _band_weights(band, looseness.get(event.seat, 1.0))
        else:
            strength, _ = board_strength(observation.board[: BOARD_SIZE[event.street]])
            weights[event.seat] = _after_postflop_action(event, strength, weights[event.seat])

    dead = blocked(card_ids((*observation.hole_cards, *observation.board)))
    for seat_weights in weights.values():
        seat_weights[dead] = 0.0
    return weights


def _preflop_band(event: ActionEvent, position: str, big_blind: int) -> tuple[float, float] | None:
    """The band an action implies, or None when it says nothing (checks, folds)."""
    group = position_group(position)
    if event.kind == "call":
        if event.raises_before == 0:
            return model.LIMP_BAND
        if event.raises_before == 1:
            return model.BLIND_DEFEND_BAND if group == "blinds" else model.COLD_CALL_BAND
        return model.CALL_RERAISE_BAND
    if event.kind in ("bet", "raise"):
        if event.raises_before == 0:
            top = model.OPEN_BAND[group]
            if event.amount >= model.BIG_OPEN_BB * big_blind:
                top *= model.BIG_OPEN_SHRINK
            return (0.0, top)
        return model.RERAISE_BAND if event.raises_before == 1 else model.FOUR_BET_BAND
    return None


def _band_weights(band: tuple[float, float] | None, looseness: float) -> np.ndarray:
    if band is None:
        return np.ones(COMBO_COUNT)
    low, high = band[0] * looseness, min(1.0, band[1] * looseness)
    inside = (low <= PREFLOP_PERCENTILE) & (high > PREFLOP_PERCENTILE)
    return np.where(inside, 1.0, model.OUTSIDE_BAND_WEIGHT)


def _after_postflop_action(event: ActionEvent, strength: np.ndarray, weights: np.ndarray) -> np.ndarray:
    river = event.street == STREETS[-1]
    if event.kind == "bet":
        return _aggression(model.BET_LIKELIHOOD, event.pot_fraction, strength, weights, river)
    if event.kind == "raise":
        return _aggression(model.RAISE_LIKELIHOOD, event.pot_fraction, strength, weights, river)
    if event.kind == "call":
        weak = np.where(strength < model.CALL_WEAK_BELOW, model.CALL_WEAK_WEIGHT, 1.0)
        return weights * weak * np.where(strength > model.CALL_STRONG_ABOVE, model.CALL_STRONG_WEIGHT, 1.0)
    if event.kind == "check":
        return weights * np.where(strength > model.CHECK_STRONG_ABOVE, model.CHECK_STRONG_WEIGHT, 1.0)
    return weights


def _aggression(
    likelihood: model.AggressionLikelihood,
    fraction: float,
    strength: np.ndarray,
    weights: np.ndarray,
    river: bool,
) -> np.ndarray:
    """Keep value hands as they are; scale weak hands down to the bluff share (never up)."""
    threshold = np.clip(likelihood.base + likelihood.slope * fraction, likelihood.low, likelihood.high)
    value = strength >= threshold
    value_weight, weak_weight = weights[value].sum(), weights[~value].sum()
    if value_weight == 0 or weak_weight == 0:
        return weights
    share = likelihood.bluff_share * (model.RIVER_BLUFF_SHRINK if river else 1.0)
    scale = min(1.0, share / (1 - share) * value_weight / weak_weight)
    return np.where(value, weights, weights * scale)
