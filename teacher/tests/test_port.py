"""Port-correctness tests for the vendored Poker-Lab teacher (handoff section 10).

Run with the vendored package on the path and the two external deps:

    PYTHONPATH=teacher uv run --with numpy --with phevaluator --with pytest \
        python -m pytest teacher/tests/test_port.py -q

The teacher is pokerlab.bots.field.bot.FieldBot with the greedy TEACHER persona, so
every decision is deterministic. These tests build Observations directly; the harness
adapter is a separate step and is not exercised here.
"""

import random

from pokerlab.bots.field.bot import FieldBot
from pokerlab.bots.randomness import seat_key
from pokerlab.datatypes.actions import ActionKind
from pokerlab.datatypes.field_personas import TEACHER
from pokerlab.datatypes.observation import (
    STREETS,
    ActionRecord,
    LegalActions,
    Observation,
    SeatView,
)
from pokerlab.poker.legal import can_check, is_legal

BB = 2
SB = 1
FULL_DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]


def teacher(seat: int = 0) -> FieldBot:
    return FieldBot(TEACHER, seat_key(1234, seat))


def decide(obs: Observation, seat: int = 0):
    return teacher(seat).decide(obs).action


# --------------------------------------------------------------------------------------
# Section 10.1 — fixed preflop spots
# --------------------------------------------------------------------------------------

def _preflop_unopened(hole, position, seat, my_stack_chips, folded_before=()):
    """A preflop spot folded to `seat`: blinds posted, some seats folded, nobody raised.

    Seat 0 is the small blind (posts 1), seat 1 the big blind (posts 2). `my_stack_chips`
    is this seat's stack+bet total (e.g. 200 for 100bb). to_call matches the big blind.
    """
    seats = []
    history = [ActionRecord("preflop", 0, "post", SB), ActionRecord("preflop", 1, "post", BB)]
    for s in range(6):
        if s == 0:
            stack, bet = (my_stack_chips - SB, SB) if seat == 0 else (200 - SB, SB)
        elif s == 1:
            stack, bet = 200 - BB, BB
        elif s == seat:
            stack, bet = my_stack_chips, 0
        else:
            stack, bet = 200, 0
        seats.append(SeatView(seat=s, position=_POS[s], stack=stack, bet=bet, in_hand=True, all_in=False))
    for s in folded_before:
        history.append(ActionRecord("preflop", s, "fold", None))
        v = seats[s]
        seats[s] = SeatView(s, v.position, v.stack, v.bet, in_hand=False, all_in=False)
    my_bet = seats[seat].bet
    to_call = BB - my_bet
    max_raise_to = seats[seat].stack + my_bet
    min_raise_to = min(BB * 2, max_raise_to)
    pot = sum(v.bet for v in seats)
    return Observation(
        hand_number=1, hand_count=30, seat=seat, position=position,
        hole_cards=tuple(hole), board=(), street="preflop", pot=pot, big_blind=BB,
        seats=tuple(seats), history=tuple(history),
        legal=LegalActions(can_fold=to_call > 0, to_call=to_call, min_raise_to=min_raise_to, max_raise_to=max_raise_to),
    )


_POS = {0: "SB", 1: "BB", 2: "UTG", 3: "HJ", 4: "CO", 5: "BTN"}


def test_aa_utg_unopened_raises_to_5():
    a = decide(_preflop_unopened(("As", "Ad"), "UTG", seat=2, my_stack_chips=200))
    assert a.kind == ActionKind.RAISE and a.amount == 5, a


def test_72o_utg_unopened_folds():
    a = decide(_preflop_unopened(("7c", "2d"), "UTG", seat=2, my_stack_chips=200))
    assert a.kind == ActionKind.FOLD, a


def test_t9s_cutoff_unopened_raises():
    a = decide(_preflop_unopened(("Th", "9h"), "CO", seat=4, my_stack_chips=200, folded_before=(2, 3)))
    assert a.kind == ActionKind.RAISE, a


def test_a2o_sb_10bb_unopened_folds():
    # 10bb = 20 chips total; folded to the small blind. Push band 0.17 < A2o pctile 0.41 -> fold.
    a = decide(_preflop_unopened(("Ah", "2d"), "SB", seat=0, my_stack_chips=20, folded_before=(2, 3, 4, 5)))
    assert a.kind == ActionKind.FOLD, a


# --------------------------------------------------------------------------------------
# Section 10.2 — determinism
# --------------------------------------------------------------------------------------

def test_determinism():
    obs = _preflop_unopened(("As", "Ad"), "UTG", seat=2, my_stack_chips=200)
    assert decide(obs) == decide(obs)
    obs2 = _preflop_unopened(("Th", "9h"), "CO", seat=4, my_stack_chips=200, folded_before=(2, 3))
    assert decide(obs2, 4) == decide(obs2, 4)


# --------------------------------------------------------------------------------------
# Section 10.3 — legality over many random spots
# --------------------------------------------------------------------------------------

def _random_obs(rng: random.Random) -> Observation:
    n = rng.randint(2, 6)
    street = rng.choice(STREETS)
    board_size = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]
    deck = rng.sample(FULL_DECK, 2 + board_size)
    hole, board = tuple(deck[:2]), tuple(deck[2:])
    others_in = rng.randint(1, n - 1)  # at least one opponent live (postflop equity needs one)
    in_hand_others = set(rng.sample(range(1, n), others_in))
    seats = []
    for s in range(n):
        in_hand = s == 0 or s in in_hand_others
        stack = rng.randint(1, 400)
        bet = rng.randint(0, 40) if street != "preflop" else rng.choice([0, SB, BB])
        seats.append(SeatView(s, _POS[s], stack, bet, in_hand=in_hand, all_in=False))
    my = seats[0]
    to_call = rng.choice([0, rng.randint(1, max(1, min(my.stack, 60)))])
    can_raise = my.stack > to_call
    max_raise_to = my.bet + my.stack
    min_raise_to = min(my.bet + to_call + BB, max_raise_to) if can_raise else None
    if can_raise and min_raise_to is None:
        can_raise = False
    legal = LegalActions(
        can_fold=to_call > 0,
        to_call=to_call,
        min_raise_to=min_raise_to if can_raise else None,
        max_raise_to=max_raise_to if can_raise else None,
    )
    pot = rng.randint(BB, 500)
    return Observation(
        hand_number=rng.randint(1, 30), hand_count=30, seat=0, position=_POS[0],
        hole_cards=hole, board=board, street=street, pot=pot, big_blind=BB,
        seats=tuple(seats), history=(), legal=legal,
    )


def test_legality_fuzz():
    rng = random.Random(0)
    bot = teacher(0)
    n = 10_000
    for i in range(n):
        obs = _random_obs(rng)
        action = bot.decide(obs).action
        assert action is not None, f"iter {i}: no action"
        assert is_legal(action, obs.legal), f"iter {i}: illegal {action} for {obs.legal}"
        if action.kind == ActionKind.FOLD:
            assert not can_check(obs.legal), f"iter {i}: folded when check was free"
        if action.kind == ActionKind.RAISE:
            assert obs.legal.min_raise_to <= action.amount <= obs.legal.max_raise_to, f"iter {i}: raise out of range"


if __name__ == "__main__":
    import sys
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
