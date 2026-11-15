"""After the flop: score each action by chip EV, bias and stack risk, then choose.

The EV model is deliberately simple:
    EV[fold]  = 0
    EV[call]  = equity * (pot + to_call) - to_call
    EV[raise] = FE * pot + (1 - FE) * (equity_called * (pot + 2 * added) - added)
FE (fold equity) is the chance every opponent folds: each opponent's range weight
below a calling threshold that rises with the bet size. equity_called discounts
our equity when called; it is the crudest approximation here (bigger bets get
called by stronger hands), and the first thing to replace with a better model.

Each action's utility adds the persona's bias and subtracts a stack-risk penalty:
    U = EV + bias * pot - risk_aversion * (1 - equity) * risked * risked / stack
A greedy persona plays the highest U; the others sample from a softmax over U.
The frozen eval scores model answers with the same U (experiments/eval/DESIGN.md).
"""

import math
import random
from collections.abc import Callable, Sequence

import numpy as np

from pokerlab.bots.field.ranges import opponent_ranges
from pokerlab.datatypes.actions import FOLD, Action, ActionValue
from pokerlab.datatypes.observation import Observation
from pokerlab.datatypes.persona import Persona
from pokerlab.poker.legal import can_check, can_raise, check_or_call, raise_to_nearest
from pokerlab.poker.range_equity import range_equity
from pokerlab.poker.situation import current_bet, my_seat
from pokerlab.poker.strength import board_strength

RAISE_FRACTIONS = (1 / 3, 2 / 3, 1.0)
EQUITY_SAMPLES = 200
CALL_THRESHOLD_BASE, CALL_THRESHOLD_SLOPE = 0.35, 0.2  # opponents call with strength >= this + slope * f
CALL_THRESHOLD_MAX = 0.8
CALLED_DISCOUNT_BASE, CALLED_DISCOUNT_SLOPE = 0.10, 0.10

# fold_rate(seat, prior) -> fold probability; the identity on `prior` for bots that do not adapt
FoldRate = Callable[[int, float], float]


def decide_postflop(
    observation: Observation,
    persona: Persona,
    risk_aversion: float,
    looseness: dict[int, float],
    fold_rate: FoldRate,
    rng: random.Random,
) -> Action:
    equity, ranges = estimate(observation, looseness, EQUITY_SAMPLES, rng)
    values = action_values(observation, persona, risk_aversion, equity, ranges, fold_rate)
    if persona.greedy:
        return max(values, key=lambda value: value.utility).action
    pot = max(observation.pot, observation.big_blind)
    return _sample(values, persona.temperature * pot, rng)


def estimate(
    observation: Observation, looseness: dict[int, float], samples: int, rng: random.Random
) -> tuple[float, dict[int, np.ndarray]]:
    """Our equity against the opponents' action-conditioned ranges, and those ranges."""
    ranges = opponent_ranges(observation, looseness)
    equity = range_equity(observation.hole_cards, observation.board, list(ranges.values()), samples, rng)
    return equity, ranges


def action_values(
    observation: Observation,
    persona: Persona,
    risk_aversion: float,
    equity: float,
    ranges: dict[int, np.ndarray],
    fold_rate: FoldRate = lambda _, prior: prior,
    raise_targets: Sequence[int] | None = None,
) -> list[ActionValue]:
    """U for check or call, fold (when facing a bet) and each raise target (default: the persona's sizes)."""
    legal = observation.legal
    pot = observation.pot
    to_call = legal.to_call
    stack = max(my_seat(observation).stack, 1)
    penalty_pot = max(pot, observation.big_blind)

    def utility(ev: float, bias: float, risked: int) -> float:
        return ev + bias * penalty_pot - risk_aversion * (1 - equity) * risked * (risked / stack)

    values = [
        ActionValue(
            check_or_call(legal), utility(equity * (pot + to_call) - to_call, persona.call_bias, to_call)
        )
    ]
    if not can_check(legal):
        values.append(ActionValue(FOLD, 0.0))
    if not can_raise(legal):
        return values

    strength, _ = board_strength(observation.board)
    bet_now = current_bet(observation)
    mine = my_seat(observation).bet
    targets = raise_targets if raise_targets is not None else _default_targets(observation)
    seen = set()
    for target in targets:
        action = raise_to_nearest(legal, target)
        if action.amount in seen:
            continue
        seen.add(action.amount)
        added = action.amount - mine
        size = (action.amount - bet_now) / max(pot + to_call, 1)
        fold_equity = _fold_equity(ranges, strength, size, fold_rate)
        called = max(0.0, equity - (CALLED_DISCOUNT_BASE + CALLED_DISCOUNT_SLOPE * size))
        ev = fold_equity * pot + (1 - fold_equity) * (called * (pot + 2 * added) - added)
        values.append(ActionValue(action, utility(ev, persona.raise_bias, added)))
    return values


def _default_targets(observation: Observation) -> list[int]:
    pot_after_call = observation.pot + observation.legal.to_call
    bet_now = current_bet(observation)
    return [round(bet_now + fraction * pot_after_call) for fraction in RAISE_FRACTIONS] + [
        observation.legal.max_raise_to
    ]


def _fold_equity(
    ranges: dict[int, np.ndarray], strength: np.ndarray, size: float, fold_rate: FoldRate
) -> float:
    """Chance that every opponent folds to a bet of `size` pots."""
    threshold = min(CALL_THRESHOLD_MAX, CALL_THRESHOLD_BASE + CALL_THRESHOLD_SLOPE * size)
    everyone_folds = 1.0
    for seat, weights in ranges.items():
        total = weights.sum()
        modelled = float(weights[strength < threshold].sum() / total) if total > 0 else 0.0
        everyone_folds *= fold_rate(seat, modelled)
    return everyone_folds


def _sample(values: list[ActionValue], temperature: float, rng: random.Random) -> Action:
    best = max(value.utility for value in values)
    weights = [math.exp((value.utility - best) / temperature) for value in values]
    return rng.choices([value.action for value in values], weights=weights)[0]
