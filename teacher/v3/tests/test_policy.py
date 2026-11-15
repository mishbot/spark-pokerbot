"""policy.decide tests: preflop open/fold, all-in call (A1), postflop value/call/fold, no weak all-in.

Run: PYTHONPATH=teacher uv run --with phevaluator python teacher/v3/tests/test_policy.py
"""

from __future__ import annotations

from pokerlab.datatypes.actions import ActionKind
from pokerlab.datatypes.observation import ActionRecord, LegalActions, Observation, SeatView

from v3.policy import decide


def _obs(street, pos, hole, board=(), *, pot, to_call, min_raise=None, max_raise=None,
         stack=200, my_bet=0, opp_all_in=False, history=(), hand=15, lead=False, tied=False):
    # default: hand 15 (not early) and hero NOT chip-leader -> the "normal" (not tight) player.
    me = SeatView(seat=0, position=pos, stack=stack, bet=my_bet, in_hand=True, all_in=False)
    opp = SeatView(seat=1, position="BB", stack=stack if tied else (0 if opp_all_in else 150), bet=0, in_hand=True, all_in=opp_all_in)
    opp2 = SeatView(seat=2, position="SB", stack=stack if tied else (50 if lead else 400), bet=0, in_hand=True, all_in=False)
    legal = LegalActions(can_fold=to_call > 0, to_call=to_call, min_raise_to=min_raise, max_raise_to=max_raise)
    return Observation(hand_number=hand, hand_count=30, seat=0, position=pos, hole_cards=tuple(hole),
                       board=tuple(board), street=street, pot=pot, big_blind=2,
                       seats=(me, opp, opp2), history=tuple(history), legal=legal)


POSTS = (ActionRecord("preflop", 2, "post", 1), ActionRecord("preflop", 1, "post", 2))
RAISE = ActionRecord("preflop", 1, "raise", 6)


def test_preflop_open_in_range():
    p = decide(_obs("preflop", "BTN", ["Ah", "2h"], pot=3, to_call=2, min_raise=4, max_raise=200, history=POSTS))
    assert p.action.kind == ActionKind.RAISE and p.action.amount == 5, (p.action,)
    assert p.in_range and "2.5x2" in p.sizing_text


def test_preflop_fold_out_of_range():
    p = decide(_obs("preflop", "BTN", ["7c", "2d"], pot=3, to_call=2, min_raise=4, max_raise=200, history=POSTS))
    assert p.action.kind == ActionKind.FOLD and not p.in_range


def test_preflop_allin_fold_kq():  # A1: never call KQ
    p = decide(_obs("preflop", "SB", ["Kh", "Qd"], pot=120, to_call=98, opp_all_in=True,
                    history=(*POSTS, ActionRecord("preflop", 1, "raise", 150))))
    assert p.action.kind == ActionKind.FOLD and p.spot == "preflop facing an all-in"


def test_preflop_allin_call_aq():
    p = decide(_obs("preflop", "SB", ["Ah", "Qd"], pot=120, to_call=98, stack=98, opp_all_in=True,
                    history=(*POSTS, ActionRecord("preflop", 1, "raise", 150))))
    assert p.action.kind == ActionKind.CALL
    assert p.risk_class == "all-in" and p.override


def test_postflop_value_raise_set():
    p = decide(_obs("flop", "BTN", ["9h", "9d"], ["9c", "2s", "7h"], pot=48, to_call=20,
                    min_raise=40, max_raise=200))
    assert p.action.kind == ActionKind.RAISE and p.tier == "strong" and p.edge == "clear ahead"


def test_postflop_fold_weak_bad_price():
    p = decide(_obs("flop", "BTN", ["Ah", "7d"], ["Kc", "7s", "2h"], pot=48, to_call=40,
                    min_raise=80, max_raise=200))
    assert p.action.kind == ActionKind.FOLD and p.tier == "weak"
    assert p.edge in ("close", "clear behind")


def test_postflop_checked_value_bet_set():
    p = decide(_obs("turn", "BTN", ["9h", "9d"], ["9c", "2s", "7h", "Kd"], pot=60, to_call=0,
                    min_raise=2, max_raise=200))
    assert p.action.kind == ActionKind.RAISE and p.plan_word == "value"


def test_postflop_checked_air_checks():  # A2: never bluff air
    p = decide(_obs("turn", "BTN", ["Ah", "Qd"], ["9c", "2s", "7h", "Kd"], pot=60, to_call=0,
                    min_raise=2, max_raise=200))
    assert p.action.kind == ActionKind.CHECK and p.plan_word == "check"


def test_river_medium_is_value_not_protection():  # Opus FAIL #5/#17: no "protection" on the river
    # AK two pair on a PAIRED board (77) -> capped strong->medium; on the river it must be value, not protection
    p = decide(_obs("river", "BTN", ["Ah", "Kd"], ["Ac", "Ks", "7h", "7c", "2d"], pot=40, to_call=0,
                    min_raise=4, max_raise=200))
    assert p.tier == "medium" and p.plan_word == "value", (p.tier, p.plan_word)


def test_plays_board_checks_not_value():  # Opus FAIL #11
    p = decide(_obs("river", "BTN", ["9h", "8d"], ["Kd", "7c", "Tc", "7d", "7s"], pot=40, to_call=0,
                    min_raise=4, max_raise=200))
    assert p.action.kind == ActionKind.CHECK and p.tier == "air", (p.action, p.tier)


def test_no_min_bet_semibluff():  # Opus #5: don't min-bet a non-value hand
    # flop air + open-ended, but the only legal bet is below half pot -> check instead of min-bet
    p = decide(_obs("flop", "BB", ["9h", "8d"], ["7c", "6s", "Kd"], pot=4, to_call=0,
                    min_raise=4, max_raise=200))
    assert p.action.kind == ActionKind.CHECK, (p.action, p.plan_word)


FLOP_BET_RAISE = (ActionRecord("flop", 1, "bet", 10), ActionRecord("flop", 2, "raise", 30))


def test_facing_raise_folds_medium():  # Opus re-vet #4: don't stack off a non-nut hand into a raise
    p = decide(_obs("flop", "BTN", ["Ah", "Kd"], ["Ac", "7s", "2h"], pot=80, to_call=30,
                    min_raise=60, max_raise=200, history=FLOP_BET_RAISE))
    assert p.action.kind == ActionKind.FOLD, (p.action, p.tier)


def test_facing_raise_flats_the_nuts():  # the nuts call (flat), never re-jam
    p = decide(_obs("flop", "BTN", ["9h", "9d"], ["9c", "7s", "2h"], pot=80, to_call=30,
                    min_raise=60, max_raise=200, history=FLOP_BET_RAISE))
    assert p.action.kind == ActionKind.CALL and "flat" in p.sizing_text, (p.action, p.sizing_text)


def test_pot_raise_to_amount():  # Opus re-vet #2: pot-sized raise-to = my_bet + to_call + pot
    # set facing a single bet; my_bet=3, to_call=11, pot=32 -> pot raise to 46
    p = decide(_obs("flop", "BTN", ["9h", "9d"], ["9c", "7s", "2h"], pot=32, to_call=11,
                    min_raise=20, max_raise=400, my_bet=3))
    assert p.action.kind == ActionKind.RAISE and p.action.amount == 46, p.action


def test_made_flush_does_not_overfold():  # flush-tier leak: a real flush getting a price must not fold
    p = decide(_obs("turn", "BTN", ["9s", "8s"], ["2c", "4s", "2s", "6s"], pot=40, to_call=20,
                    min_raise=40, max_raise=200))
    assert p.action.kind in (ActionKind.CALL, ActionKind.RAISE) and p.tier == "strong", (p.action, p.tier)


def test_low_flush_on_four_flush_board_checks():  # flush-tier leak: junk flush is not value-bet
    p = decide(_obs("turn", "BTN", ["3c", "6s"], ["8c", "9c", "7c", "Jc"], pot=40, to_call=0,
                    min_raise=4, max_raise=200))
    assert p.action.kind == ActionKind.CHECK and p.tier == "weak", (p.action, p.tier)


def test_no_weak_all_in():  # #3: a non-strong tier never bets all-in
    # top pair, checked to, but the only legal raise is all-in (min_raise == max == stack)
    p = decide(_obs("flop", "BTN", ["Ah", "Kd"], ["Ac", "7s", "2h"], pot=60, to_call=0,
                    min_raise=200, max_raise=200, stack=200))
    assert p.action.kind == ActionKind.CHECK, (p.action, p.tier)


def test_tight_early_opens_tighter():  # core player: tight for the first 10 hands
    # K7s: in the normal BTN open (K5s+) but NOT the tight BTN open (K9s+)
    loose = decide(_obs("preflop", "BTN", ["Kh", "7h"], pot=3, to_call=2, min_raise=4, max_raise=200,
                        history=POSTS, hand=15, lead=False))
    assert loose.action.kind == ActionKind.RAISE and not loose.tight
    tight = decide(_obs("preflop", "BTN", ["Kh", "7h"], pot=3, to_call=2, min_raise=4, max_raise=200,
                        history=POSTS, hand=3))
    assert tight.action.kind == ActionKind.FOLD and tight.tight, (tight.action, tight.tight)


def test_tied_stacks_not_leading():  # tied stacks are NOT a chip lead (no false "chip-leading")
    p = decide(_obs("flop", "CO", ["Ah", "Kd"], ["Ac", "7s", "2h"], pot=20, to_call=0,
                    min_raise=4, max_raise=200, hand=15, tied=True))
    assert not p.tight and "chip-lead" not in p.tight_reason, (p.tight, p.tight_reason)


def test_tight_when_leading_allin_folds_aq():  # tight when chip-leading: all-in call is TT+/AK only
    p = decide(_obs("preflop", "SB", ["Ah", "Qd"], pot=120, to_call=98, stack=300, opp_all_in=True,
                    lead=True, hand=15, history=(*POSTS, ActionRecord("preflop", 1, "raise", 300))))
    assert p.action.kind == ActionKind.FOLD and p.tight, (p.action, p.tight)


def test_allin_need_excludes_uncallable_overshove():  # Opus: price must exclude uncallable chips
    me = SeatView(0, "BB", 174, 0, True, False)
    villain = SeatView(1, "SB", 0, 1022, True, True)   # shoved 1022 into our 174 stack
    legal = LegalActions(can_fold=True, to_call=174, min_raise_to=None, max_raise_to=None)
    obs = Observation(15, 30, 0, "BB", ("7h", "2d"), (), "preflop", 1026, 2, (me, villain),
                      (ActionRecord("preflop", 1, "raise", 1022),), legal)
    p = decide(obs)
    assert 0.45 <= p.need <= 0.52, f"true price ~49%, got {p.need} (naive off the full pot ~14%)"


def test_small_shove_calls_wide():  # price-aware all-in: a cheap shove is called wide
    p = decide(_obs("preflop", "BTN", ["Kh", "Jd"], pot=30, to_call=10, stack=200, opp_all_in=True,
                    hand=15, history=(*POSTS, ActionRecord("preflop", 1, "raise", 12))))
    assert p.action.kind == ActionKind.CALL, (p.action, p.need)


def test_tight_checks_medium_no_thin_bet():  # tight: value only, no thin protection bets
    p = decide(_obs("flop", "SB", ["Ah", "Kd"], ["Ac", "7s", "2h"], pot=20, to_call=0,
                    min_raise=4, max_raise=200, hand=3))
    assert p.action.kind == ActionKind.CHECK and p.tight, (p.action, p.tier)


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
