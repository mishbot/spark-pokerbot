"""validate.check tests: clean traces pass, corrupted traces are caught, and the phevaluator
category buckets agree with eval.analyze (independent cross-check).

Run: PYTHONPATH=teacher uv run --with phevaluator python teacher/v3/tests/test_validate.py
"""

from __future__ import annotations

import random

from pokerlab.datatypes.observation import ActionRecord, LegalActions, Observation, SeatView

from v3.policy import decide
from v3.trace import render
from v3.validate import check, phe_category
from v3.eval import analyze, RANK_ORDER

FULL = [r + s for r in RANK_ORDER for s in "cdhs"]


def _case(street, pos, hole, board=(), *, pot, to_call, min_raise=None, max_raise=None,
          stack=200, my_bet=0, opp_all_in=False, history=(), deriv=None):
    me = SeatView(0, pos, stack, my_bet, True, False)
    opp = SeatView(1, "BB", 0 if opp_all_in else 150, 0, True, opp_all_in)
    opp2 = SeatView(2, "SB", 400, 0, True, False)   # hero not chip-leader
    legal = LegalActions(to_call > 0, to_call, min_raise, max_raise)
    obs = Observation(15, 30, 0, pos, tuple(hole), tuple(board), street, pot, 2, (me, opp, opp2),
                      tuple(history), legal)   # hand 15 -> not early -> not tight by default
    p = decide(obs)
    deriv = deriv or dict(button=0, me=0, sb_seat=1, bb_seat=2, n=3)
    trace = render(p, **deriv)
    truth = dict(hole=hole, board=board, pot=pot, to_call=to_call, street=street, position=pos,
                 min_raise=min_raise, max_raise=max_raise)
    return truth, trace, (p.action.kind.value if hasattr(p.action.kind, "value") else str(p.action.kind),
                          p.action.amount)


POSTS = (ActionRecord("preflop", 2, "post", 1), ActionRecord("preflop", 1, "post", 2))

CLEAN = [
    ("open", dict(street="preflop", pos="BTN", hole=["Ah", "2h"], pot=3, to_call=2, min_raise=4,
                  max_raise=200, history=POSTS, deriv=dict(button=0, me=0, sb_seat=1, bb_seat=2, n=3))),
    ("set-raise", dict(street="flop", pos="BTN", hole=["9h", "9d"], board=["9c", "2s", "7h"],
                       pot=48, to_call=20, min_raise=40, max_raise=200)),
    ("weak-fold", dict(street="flop", pos="BTN", hole=["Ah", "7d"], board=["Kc", "7s", "2h"],
                       pot=48, to_call=40, min_raise=80, max_raise=200)),
    ("straight-call", dict(street="turn", pos="BTN", hole=["9h", "8d"], board=["7c", "6s", "5h", "Kd"],
                           pot=48, to_call=20, min_raise=40, max_raise=200)),
    ("checked-air", dict(street="turn", pos="BTN", hole=["Ah", "Qd"], board=["9c", "2s", "7h", "Kd"],
                         pot=60, to_call=0, min_raise=2, max_raise=200)),
]


def test_clean_traces_pass():
    for name, kw in CLEAN:
        truth, trace, action = _case(**kw)
        viol = check(truth, trace, action)
        assert viol == [], f"{name} should be clean, got: {viol}\n{trace}"


def test_catches_bad_need():
    truth, trace, action = _case(**dict(CLEAN[1][1]))
    bad = trace.replace("need 29%", "need 10%")
    viol = check(truth, bad, action)
    assert any("I5" in x for x in viol), viol


def test_catches_straight_called_high_card():  # #1 guard, independently
    truth, trace, action = _case(**dict(CLEAN[3][1]))  # a made straight
    bad = trace.replace("Hand: straight", "Hand: high card")
    viol = check(truth, bad, action)
    assert any("I2" in x for x in viol), viol


def test_catches_edge_action_mismatch():
    truth, trace, action = _case(**dict(CLEAN[2][1]))  # weak fold
    bad = trace.replace("-> fold", "-> continue")
    viol = check(truth, bad, action)
    assert any("I11" in x for x in viol), viol


def test_i3b_catches_wrong_position():  # Opus re-vet FAIL (HU position inverted)
    truth, trace, action = _case(**dict(CLEAN[1][1]))  # a postflop (facing-bet) trace with a Position line
    bad = {**truth, "me_seat": 0, "live_order": [1, 0]}   # we act last -> should be "in position"
    assert any("I3b" in x for x in check(bad, trace, action)), "I3b should flag a wrong position label"
    ok = {**truth, "me_seat": 0, "live_order": [0, 1]}     # we act first -> "out of position" is correct
    assert not any("I3b" in x for x in check(ok, trace, action))


def test_catches_equity_number():
    truth, trace, action = _case(**dict(CLEAN[1][1]))
    bad = trace + "\nEquity 76%."
    assert any("equity" in x for x in check(truth, bad, action)), "should flag an equity number"


def test_phe_category_matches_eval():
    rng = random.Random(1)
    mism = 0
    for _ in range(3000):
        cards = rng.sample(FULL, 7)
        a = analyze(cards[:2], cards[2:]).category
        b = phe_category(cards)
        if a != b:
            mism += 1
    assert mism == 0, f"{mism} category disagreements between eval and phevaluator"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
