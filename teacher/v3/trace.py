"""Render a policy.Plan into the locked V3 slot text (the assistant <think> content).

Pure formatting of values the policy already computed (one source of truth). The Seat line derives
position from the button and the posted blinds — facts visible in the prompt — not from the harness
"0 = small blind" hint (see scratch_pad/ALPHA.md: the hint is wrong heads-up).
"""

from __future__ import annotations

import math

from v3.policy import Plan

CONTINUE_EDGES = {"clear ahead", "draw-priced"}


def pct(x: float | None) -> int:
    return int(math.floor((x or 0.0) * 100 + 0.5))


def _verb(kind: str, spot: str) -> str:
    if kind == "raise":
        if "unopened" in spot:
            return "open"
        if "facing a raise" in spot:
            return "reraise"
        return "raise"
    return kind  # call / fold / check


def seat_line(button: int, me: int, position: str, sb_seat: int, bb_seat: int, n: int) -> str:
    if me == sb_seat:
        extra = " (I am on the button; heads-up the button posts the small blind)" if n == 2 else ""
        return f"I post the small blind{extra}, so I am SB"
    if me == bb_seat:
        return f"I post the big blind, so I am BB"
    return f"the button is seat {button}; I am seat {me}, which is {position}"


def _risk_line(p: Plan, verb_chips: str) -> str:
    s = f"{verb_chips} {p.risked} of {p.stack} = {p.risk_class}"
    if p.override and p.rank is not None:
        s += f"; rank {p.rank}, survival matters"
    return s


def _action_line(p: Plan) -> str:
    if p.sizing_text:
        return p.sizing_text
    return p.action.kind.value if hasattr(p.action.kind, "value") else str(p.action.kind)


def _hand_tags(p: Plan) -> str:
    bits = [p.made]
    if p.draw_text:
        bits.append(p.draw_text)
    tag = f"; board {p.board_tag}" if p.board_tag and p.board_tag not in ("preflop",) else ""
    return ", ".join(bits) + tag


def render(p: Plan, *, button: int, me: int, sb_seat: int, bb_seat: int, n: int,
           in_position: bool = False) -> str:
    cards = " ".join(p.cards)
    seat = seat_line(button, me, p.position, sb_seat, bb_seat, n)
    if p.street == "preflop":
        situation = p.spot.replace("preflop ", "")
        verb_chips = "post" if p.action.kind == "fold" or p.action.kind == "check" else (
            "call" if p.action.kind == "call" else "invest")
        inout = "in" if p.in_range else "out"
        rule = f"{p.rule_text}; {p.hole_class} is {inout} -> {_verb(p.action.kind, p.spot)}"
        lines = [
            f"Spot: preflop, {situation}.",
            f"Seat: {seat}.",
            f"Hand: {cards}, {p.hole_class}.",
            f"Rule: {rule}.",
        ]
        if p.risk_class != "small" or p.override:
            lines.append(f"Risk: {_risk_line(p, verb_chips)}.")
        lines.append(f"Action: {_action_line(p)}.")
        return "\n".join(lines)

    # postflop — in/out of position is who acts last (passed in), not the SB/BB word (HU fix)
    pos_io = "in position" if in_position else "out of position"
    tier_note = (" (plays the board)" if p.plays_board else
                 " (capped on a dangerous board)" if p.capped else "")
    if p.to_call > 0:  # facing a bet
        cont = "continue" if p.edge in CONTINUE_EDGES else "fold"
        lines = [
            f"Spot: {p.street}, facing a bet.",
            f"Position: {pos_io}.",
            f"Hand: {_hand_tags(p)}; tier {p.tier}, pays up to {pct(p.price)}%" + tier_note + ".",
            f"Odds: call {p.to_call} into {p.pot} -> need {pct(p.need)}% "
            f"( = {p.to_call}/({p.pot}+{p.to_call}) ).",
            f"Edge: need {pct(p.need)}% vs price {pct(p.price)}% -> {p.edge} -> {cont}.",
        ]
        if p.risk_class != "small" or p.override:
            lines.append(f"Risk: {_risk_line(p, 'commit' if p.action.kind == 'raise' else 'call')}.")
        lines.append(f"Action: {_action_line(p)}.")
        return "\n".join(lines)

    # not facing a bet: we open the street, or it was checked to us
    spot_label = "first to act" if p.first_to_act else "checked to me"
    if p.plan_word == "check":
        behind = pos_io == "in position" and not p.first_to_act  # "check behind" needs to be last to act
        plan_line = f"Plan: check{' behind' if behind else ''}."
    else:
        clause = {"value": "bet to get called by worse", "protection": "bet to deny a free card",
                  "semi-bluff": "bet with the draw"}.get(p.plan_word, "")
        plan_line = f"Plan: {p.plan_word}. {clause}."
    lines = [
        f"Spot: {p.street}, {spot_label}.",
        f"Position: {pos_io}.",
        f"Hand: {_hand_tags(p)}; tier {p.tier}" + tier_note + ".",
        plan_line,
    ]
    if p.risk_class != "small" or p.override:
        lines.append(f"Risk: {_risk_line(p, 'commit')}.")
    lines.append(f"Action: {_action_line(p)}.")
    return "\n".join(lines)
