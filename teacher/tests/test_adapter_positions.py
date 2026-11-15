"""Adapter position/blind tests, focused on the heads-up fix (defect #5).

The harness labels position by index ("0 = small blind"), but pokerkit posts the small blind on
the BUTTON in heads-up. The adapter must label and post by the real blinds so the teacher plays
HU correctly. These tests pin that, and cross-check against a real HandRunner hand.

Run: PYTHONPATH=teacher:harness/src uv run --with pokerkit --with phevaluator --with numpy \
        python teacher/tests/test_adapter_positions.py
"""

from __future__ import annotations

from types import SimpleNamespace

from adapter import sb_bb_seats, seat_positions
from poker_harness.engine.table import HandRunner


def _ctx(button: int, live_seats: list[int]):
    seats = [SimpleNamespace(seat=s, status="") for s in live_seats]
    return SimpleNamespace(button_seat=button, seats=seats)


def test_sb_bb_seats_heads_up_button_is_sb():
    # order = [non-button, button]; HU small blind sits on the BUTTON.
    assert sb_bb_seats([1, 0], button=0) == (0, 1)   # seat 0 button -> SB, seat 1 -> BB
    assert sb_bb_seats([0, 1], button=1) == (1, 0)   # seat 1 button -> SB, seat 0 -> BB


def test_sb_bb_seats_six_handed_sb_after_button():
    order = [3, 4, 5, 0, 1, 2]  # button = 2, so first after button (3) is the SB
    assert sb_bb_seats(order, button=2) == (3, 4)


def test_seat_positions_heads_up_labels_button_sb():
    for button, other in ((0, 1), (1, 0)):
        pos, order = seat_positions(_ctx(button, [0, 1]))
        assert pos[button] == "SB", f"HU button {button} must be SB, got {pos[button]}"
        assert pos[other] == "BB", f"HU non-button {other} must be BB, got {pos[other]}"


def test_seat_positions_six_handed_unchanged():
    pos, order = seat_positions(_ctx(2, [0, 1, 2, 3, 4, 5]))
    assert pos[3] == "SB" and pos[4] == "BB" and pos[2] == "BTN"


def test_matches_real_handrunner_blinds_heads_up():
    """Ground truth: the seat the adapter calls SB is the seat pokerkit charges the SMALL blind."""
    class Dummy:
        def __init__(self, name): self.name = name

    for button in (0, 1):
        players = {0: Dummy("s0"), 1: Dummy("s1")}
        hr = HandRunner(players=players, stacks={0: 200, 1: 200}, button_seat=button,
                        hand_number=1, total_hands=30, seed=7, small_blind=1, big_blind=2)
        hr._deal()
        posted = {hr.order[i]: hr.state.bets[i] for i in range(len(hr.order))}
        sb_seat, bb_seat = sb_bb_seats(hr.order, button)
        assert posted[sb_seat] == 1, f"adapter SB seat {sb_seat} posted {posted[sb_seat]}, not the small blind"
        assert posted[bb_seat] == 2, f"adapter BB seat {bb_seat} posted {posted[bb_seat]}, not the big blind"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
    raise SystemExit(1 if failed else 0)
