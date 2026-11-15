"""Preflop: a percentile lookup with soft band edges, so play is mixed near the edges.

Same bands as the range model (datatypes/range_model.py), scaled by the
persona's looseness. These bots never limp. Short stacks play push/fold.
"""

import random

from pokerlab.bots.rule.ranges import position_group
from pokerlab.datatypes.actions import FOLD, Action
from pokerlab.datatypes.observation import Observation
from pokerlab.datatypes.persona import Persona
from pokerlab.datatypes.range_model import CALL_RERAISE_BAND, FOUR_BET_BAND, OPEN_BAND, RERAISE_BAND
from pokerlab.poker.combos import combo_index
from pokerlab.poker.legal import all_in, can_check, check_or_call, check_or_fold, raise_to_nearest
from pokerlab.poker.replay import AGGRESSIVE
from pokerlab.poker.situation import current_bet, my_seat, street_history
from pokerlab.poker.strength import PREFLOP_PERCENTILE

EDGE_WIDTH = 0.03  # percentile width over which a hand goes from always to never in a band
OPEN_SIZE_BB = 2.5  # plus one big blind per limper
RERAISE_MULTIPLE = 3.0
FOUR_BET_MULTIPLE = 2.3
COLD_CALL_TOP = 0.25
BLIND_DEFEND_TOP = 0.40
COMMIT_SHARE = 0.3  # calling more than this share of the stack takes a premium hand
PREMIUM_TOP = 0.03
SHORT_STACK_BB = 15
PUSH_TOP_AT_SHORT, PUSH_TOP_AT_EMPTY = 0.10, 0.40  # shoving range at 15bb and near 0bb


def decide_preflop(observation: Observation, persona: Persona, rng: random.Random) -> Action:
    percentile = float(PREFLOP_PERCENTILE[combo_index(observation.hole_cards)])
    legal = observation.legal
    me = my_seat(observation)
    raises = sum(record.kind in AGGRESSIVE for record in street_history(observation))

    def plays(top: float) -> bool:
        return _in_band(percentile, top * persona.looseness, rng, persona.greedy)

    if me.stack + me.bet <= SHORT_STACK_BB * observation.big_blind:
        return _push_or_fold(observation, raises, plays)
    if raises > 0 and legal.to_call > COMMIT_SHARE * me.stack:
        return all_in(legal) if plays(PREMIUM_TOP) else FOLD

    if raises == 0:
        if plays(OPEN_BAND[position_group(observation.position)]):
            limpers = sum(record.kind == "call" for record in street_history(observation))
            return raise_to_nearest(legal, round((OPEN_SIZE_BB + limpers) * observation.big_blind))
        return check_or_fold(legal)
    if raises == 1:
        if plays(RERAISE_BAND[1]):
            return raise_to_nearest(legal, round(current_bet(observation) * RERAISE_MULTIPLE))
        defend = BLIND_DEFEND_TOP if position_group(observation.position) == "blinds" else COLD_CALL_TOP
        return check_or_call(legal) if plays(defend) else FOLD
    if plays(FOUR_BET_BAND[1]):
        return raise_to_nearest(legal, round(current_bet(observation) * FOUR_BET_MULTIPLE))
    return check_or_call(legal) if plays(CALL_RERAISE_BAND[1]) else FOLD


def _push_or_fold(observation: Observation, raises: int, plays) -> Action:
    """Shove the band for this stack depth; call a raise only with half of it."""
    depth = (my_seat(observation).stack + my_seat(observation).bet) / observation.big_blind
    shortness = max(0.0, min(1.0, 1 - depth / SHORT_STACK_BB))
    top = PUSH_TOP_AT_SHORT + shortness * (PUSH_TOP_AT_EMPTY - PUSH_TOP_AT_SHORT)
    if plays(top if raises == 0 else top / 2):
        return all_in(observation.legal)
    return check_or_fold(observation.legal) if can_check(observation.legal) else FOLD


def _in_band(percentile: float, top: float, rng: random.Random, greedy: bool) -> bool:
    """Always inside the band, never well outside it, mixed across the edge (unless greedy)."""
    chance = min(1.0, max(0.0, (top - percentile) / EDGE_WIDTH + 0.5))
    return chance >= 0.5 if greedy else rng.random() < chance
