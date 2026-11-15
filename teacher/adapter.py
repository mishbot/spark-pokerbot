"""Adapter between the event harness and the vendored pokerlab teacher.

`observation_from_ctx` turns a harness `DecisionContext` into a pokerlab `Observation`
(what `FieldBot.decide` reads). `to_harness_decision` maps the teacher's `Action` back to a
harness `Decision`. Kept small and pure so it is easy to test and reuse for labeling traces.
"""

from __future__ import annotations

from poker_harness.engine import Decision as HarnessDecision
from poker_harness.engine import DecisionContext

from pokerlab.datatypes.actions import Action, ActionKind
from pokerlab.datatypes.observation import ActionRecord
from pokerlab.datatypes.observation import LegalActions as PLLegal
from pokerlab.datatypes.observation import Observation
from pokerlab.datatypes.observation import SeatView as PLSeat


def normalize_card(card: str) -> str:
    """Harness `codes()` already matches pokerlab's `FULL_DECK` ('As','Td'); this is a safety net."""
    c = card.strip()
    rank, suit = c[:-1], c[-1]
    if rank == "10":
        rank = "T"
    return rank.upper() + suit.lower()


def _betting_order(ctx: DecisionContext) -> list[int]:
    """Seats still in the game, in action order (small blind first, button last) — as HandRunner.order."""
    live = sorted(v.seat for v in ctx.seats if v.status != "out")
    b = ctx.button_seat
    return [s for s in live if s > b] + [s for s in live if s <= b]


def position_words(n: int) -> list[str]:
    """Position word per action-order index, for `n` live seats (handoff section 4).

    0=SB, 1=BB; then from the button: BTN, CO (late), HJ (middle); the rest 'early' (UTG...).
    Only the group (blinds/late/middle/early, via pokerlab position_group) affects play.
    NOTE: this index->word map is correct only for n>=3; heads-up is fixed in `seat_positions`.
    """
    words = ["early"] * n
    if n >= 1:
        words[0] = "SB"
    if n >= 2:
        words[1] = "BB"
    for idx, word in ((n - 1, "BTN"), (n - 2, "CO"), (n - 3, "HJ")):
        if idx >= 0 and words[idx] == "early":
            words[idx] = word
    return words


def sb_bb_seats(order: list[int], button: int) -> tuple[int | None, int | None]:
    """(small-blind seat, big-blind seat). HEADS-UP exception: pokerkit posts the small blind on
    the BUTTON (verified against the engine). `order` is [non-button, button], so SB=button and
    BB=non-button. For n>=3 the small blind is the first seat after the button (order[0]).

    This is the ground truth for HU: the harness labels position by index with "0 = small blind",
    but in HU index 0 is the non-button, which actually posts the BIG blind. We label and post by
    the real blinds so the teacher plays HU correctly."""
    if len(order) == 2:
        non_btn = order[0] if order[1] == button else order[1]
        return button, non_btn
    sb = order[0] if order else None
    bb = order[1] if len(order) >= 2 else None
    return sb, bb


def seat_positions(ctx: DecisionContext) -> tuple[dict[int, str], list[int]]:
    order = _betting_order(ctx)
    words = position_words(len(order))
    mapping = {seat: words[i] for i, seat in enumerate(order)}
    if len(order) == 2:  # heads-up: relabel so the button is SB (see sb_bb_seats)
        sb, bb = sb_bb_seats(order, ctx.button_seat)
        mapping[sb], mapping[bb] = "SB", "BB"
    return mapping, order


def observation_from_ctx(ctx: DecisionContext, big_blind: int, small_blind: int) -> Observation:
    pos, order = seat_positions(ctx)

    seats = tuple(
        PLSeat(
            seat=v.seat,
            position=pos[v.seat],
            stack=v.stack,
            bet=v.bet,
            in_hand=v.status in ("", "all-in"),
            all_in=v.status == "all-in",
        )
        for v in ctx.seats
        if v.status != "out"
    )

    # Synthesize the blind posts the harness applies automatically (pokerlab's replay needs them
    # to track the pot and the preflop raise count); then map each public action.
    records: list[ActionRecord] = []
    sb_seat, bb_seat = sb_bb_seats(order, ctx.button_seat)  # HU: button posts the small blind
    if sb_seat is not None:
        records.append(ActionRecord("preflop", sb_seat, "post", small_blind))
    if bb_seat is not None:
        records.append(ActionRecord("preflop", bb_seat, "post", big_blind))
    for a in ctx.hand_actions:
        kind = a["action"]["type"]
        if kind == "call":
            amount = -int(a["chip_change"])  # chips actually added (handles all-in for less)
        elif kind in ("bet", "raise"):
            amount = int(a["action"]["amount"])  # the "to" total on this street
        else:
            amount = 0
        records.append(ActionRecord(a["street"], a["seat"], kind, amount))

    lg = ctx.legal
    legal = PLLegal(
        can_fold=lg.can_fold,
        to_call=0 if lg.can_check else lg.call_amount,
        min_raise_to=lg.min_raise_to if lg.raise_type else None,
        max_raise_to=lg.max_raise_to if lg.raise_type else None,
    )

    return Observation(
        hand_number=ctx.hand_number,
        hand_count=ctx.total_hands,
        seat=ctx.seat,
        position=pos[ctx.seat],
        hole_cards=tuple(normalize_card(c) for c in ctx.hole_cards),
        board=tuple(normalize_card(c) for c in ctx.board),
        street=ctx.street,
        pot=ctx.pot,
        big_blind=big_blind,
        seats=seats,
        history=tuple(records),
        legal=legal,
    )


def to_harness_decision(action: Action, ctx: DecisionContext) -> HarnessDecision:
    """Map a pokerlab Action to a harness Decision. A RAISE takes the legal name ('bet'/'raise')
    so the harness applies it with no fallback."""
    if action.kind == ActionKind.FOLD:
        return HarnessDecision("fold")
    if action.kind == ActionKind.CHECK:
        return HarnessDecision("check")
    if action.kind == ActionKind.CALL:
        return HarnessDecision("call")
    return HarnessDecision(ctx.legal.raise_type or "raise", action.amount)
