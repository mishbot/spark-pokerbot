"""The rule-based V3 teacher: the decision procedure IS the trace.

`decide(obs)` returns a `Plan` that holds the chosen action AND every slot value the trace shows
(spot, position, rule, tier, price, need, edge, risk, sizing, plan word). The trace renderer just
formats the Plan; the validator re-derives it independently. So the trace is logically correct and
implies the action by construction.

Decisions use only things the student can reproduce: a range-table lookup (preflop), a tier lookup
and pot-odds arithmetic (postflop). No Monte-Carlo equity picks an action. Weak all-ins are refused
(a non-strong tier never bets all-in); the dangerous-board cap lives in the tier.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerlab.datatypes.actions import Action, ActionKind
from pokerlab.poker.legal import can_check, can_raise, check_or_call, check_or_fold, raise_to_nearest

from v3.eval import analyze, HandInfo
from v3.ranges import hole_facts, open_range, continue_range, threebet_range, allin_call
from v3.tier import tier_and_price

EDGE_MARGIN = 0.08   # gap between price and need to call a spot "clear"; inside -> "close"
EARLY_HANDS = 10     # the core player is tight for the first 10 hands and whenever chip-leading


@dataclass
class Plan:
    action: Action
    street: str
    spot: str                       # routing label, e.g. "preflop unopened"
    position: str
    cards: tuple = ()
    board: tuple = ()
    # preflop
    hole_class: str | None = None
    rule_text: str | None = None
    in_range: bool | None = None
    # postflop
    made: str | None = None
    board_tag: str | None = None
    tier: str | None = None
    price: float | None = None
    need: float | None = None
    to_call: int = 0
    pot: int = 0
    edge: str | None = None         # clear ahead | draw-priced | close | clear behind
    capped: bool = False
    plays_board: bool = False
    draw_text: str = ""
    plan_word: str | None = None    # value | protection | semi-bluff | check (checked-to spots)
    first_to_act: bool = False      # we open the street (no one has acted) vs it was checked to us
    tight: bool = False             # the core player is tightening (early or chip-leading)
    tight_reason: str = ""
    # risk + sizing (all spots)
    risked: int = 0
    stack: int = 0
    risk_class: str = "small"       # small | large | all-in
    rank: int | None = None
    override: bool = False
    sizing_text: str = ""


def _chips(v) -> int:
    return v.stack + v.bet


def _leader_or_safe(obs) -> bool:
    me = _chips(next(v for v in obs.seats if v.seat == obs.seat))
    return all(_chips(v) <= me for v in obs.seats if v.seat != obs.seat)


def _rank(obs) -> int:
    mine = _chips(next(v for v in obs.seats if v.seat == obs.seat))
    return 1 + sum(_chips(v) > mine for v in obs.seats if v.seat != obs.seat)


def _tight(obs) -> tuple[bool, str]:
    """The core player tightens in the first 10 hands and whenever it holds the chip lead.
    Both facts are visible in the prompt (hand number, stacks), so the reason is a copy, not a guess."""
    early = obs.hand_number <= EARLY_HANDS
    leading = _leader_or_safe(obs)
    if early and leading:
        return True, f"early (hand {obs.hand_number}) and chip-leading"
    if early:
        return True, f"early (hand {obs.hand_number} of {obs.hand_count})"
    if leading:
        return True, "chip-leading"
    return False, ""


def _opened_preflop(obs) -> int:
    """Number of preflop raises so far (0 = unopened; 1 = a single raise; >=2 = 3bet+)."""
    return sum(1 for r in obs.history if r.street == "preflop" and r.kind in ("raise", "bet"))


def _facing_all_in(obs) -> bool:
    return obs.legal.to_call > 0 and any(v.all_in for v in obs.seats if v.seat != obs.seat)


def _risk(obs, risked: int) -> tuple[str, int]:
    me = next(v for v in obs.seats if v.seat == obs.seat)
    stack = max(me.stack, 1)
    ratio = risked / stack
    cls = "all-in" if risked >= me.stack else ("large" if ratio >= 0.5 else "small")
    return cls, stack


def _size(obs, target: int, formula: str, verb: str) -> tuple[Action, str, bool]:
    """Return (action, sizing_text, is_all_in). `verb` is the harness action type word ("bet" opens
    postflop; "raise" otherwise, including every preflop open). Keeps the text equal to the amount (I6)."""
    legal = obs.legal
    if not can_raise(legal):
        return check_or_call(legal), "", False
    lo, hi = legal.min_raise_to, legal.max_raise_to
    if target >= hi:
        return Action(ActionKind.RAISE, hi), f"all-in to {hi}", True
    if target <= lo:
        return Action(ActionKind.RAISE, lo), f"{verb} to {lo} (min)", lo >= hi
    return Action(ActionKind.RAISE, target), f"{verb} to {target} ({formula})", False


def decide(obs) -> Plan:
    pos = obs.position
    tight, reason = _tight(obs)
    if obs.street == "preflop":
        return _preflop(obs, pos, tight, reason)
    return _postflop(obs, pos, tight, reason)


def _preflop(obs, pos, tight, reason) -> Plan:
    h = hole_facts(obs.hole_cards)
    bb = obs.big_blind
    legal = obs.legal
    base = dict(street="preflop", position=pos, hole_class=h.classname, cards=tuple(obs.hole_cards),
                tight=tight, tight_reason=reason)
    pre = f"{reason}, so tighten: " if tight else ""

    if _facing_all_in(obs):
        need = legal.to_call / (obs.pot + legal.to_call) if (obs.pot + legal.to_call) > 0 else 0.0
        call, rule = allin_call(h, tight, need)
        action = Action(ActionKind.CALL) if call else Action(ActionKind.FOLD)
        risked = legal.to_call if call else 0
        cls, stack = _risk(obs, risked)
        return Plan(action=action, spot="preflop facing an all-in", rule_text=rule, in_range=call,
                    need=need, risked=risked, stack=stack, risk_class=cls, rank=_rank(obs),
                    override=cls in ("large", "all-in"),
                    sizing_text=("call all-in" if call else ""), **base)

    opens = _opened_preflop(obs)
    if opens == 0:  # unopened
        spec, summ = open_range(pos, tight)
        if spec.contains(h):
            target = round(2.5 * bb)
            action, text, _ = _size(obs, target, f"2.5x{bb}", "raise")
            return Plan(action=action, spot="preflop unopened", rule_text=f"{pre}{pos} opens {summ}",
                        in_range=True, sizing_text=text, **base)
        action = check_or_fold(legal)
        return Plan(action=action, spot="preflop unopened", rule_text=f"{pre}{pos} opens {summ}",
                    in_range=False, sizing_text="", **base)

    # facing a raise (or 3bet+)
    tb_spec, tb_summ = threebet_range()
    co_spec, co_summ = continue_range(tight)
    to_match = legal.to_call + next(v for v in obs.seats if v.seat == obs.seat).bet
    if tb_spec.contains(h):
        target = round(3 * to_match)
        action, text, _ = _size(obs, target, "3x the raise", "raise")
        return Plan(action=action, spot="preflop facing a raise", rule_text=f"{pre}reraise {tb_summ}",
                    in_range=True, sizing_text=text, **base)
    if opens == 1 and co_spec.contains(h):
        cls, stack = _risk(obs, legal.to_call)
        return Plan(action=Action(ActionKind.CALL), spot="preflop facing a raise",
                    rule_text=f"{pre}call {co_summ}", in_range=True, risked=legal.to_call, stack=stack,
                    risk_class=cls, sizing_text="call", **base)
    return Plan(action=check_or_fold(legal), spot="preflop facing a raise",
                rule_text=f"{pre}reraise {tb_summ}, else call {co_summ}, else fold", in_range=False, **base)


def _postflop(obs, pos, tight, reason) -> Plan:
    info: HandInfo = analyze(list(obs.hole_cards), list(obs.board))
    tier, price, capped = tier_and_price(info)
    legal = obs.legal
    pot, to_call = obs.pot, legal.to_call
    base = dict(street=obs.street, position=pos, made=info.made, board_tag=info.board_tag,
                tier=tier, price=price, capped=capped, plays_board=info.plays_board,
                draw_text=info.draw_text, cards=tuple(obs.hole_cards), board=tuple(obs.board),
                to_call=to_call, pot=pot, tight=tight, tight_reason=reason)
    strong_draw = info.flush_draw or info.oesd

    if to_call > 0:  # facing a bet
        need = to_call / (pot + to_call)
        gap = price - need
        if gap >= EDGE_MARGIN:
            edge = "clear ahead"
        elif gap <= -EDGE_MARGIN:
            edge = "clear behind"
        else:
            edge = "close"
        spot = f"{obs.street} facing a bet"
        aggr = sum(1 for r in obs.history if r.street == obs.street and r.kind in ("bet", "raise"))
        facing_raise = aggr >= 2   # a bet AND a raise this street = a raise war
        # tight player does not chase draws
        drawing_ok = strong_draw and need <= 0.33 and edge != "clear ahead" and not tight
        nut = tier == "strong" and not capped and not info.plays_board

        def _call(edge_word, note="call"):
            cls, stack = _risk(obs, to_call)
            return Plan(action=Action(ActionKind.CALL), spot=spot, need=need, edge=edge_word,
                        risked=to_call, stack=stack, risk_class=cls, rank=_rank(obs),
                        override=cls in ("large", "all-in"), sizing_text=note, **base)

        if facing_raise:
            # the overcommit leak: do NOT stack off a non-nut hand into a raise. Flat the nuts, fold the rest.
            if nut and edge == "clear ahead":
                return _call("clear ahead", "call (flat vs a raise)")
            if drawing_ok:
                return _call("draw-priced")
            return Plan(action=Action(ActionKind.FOLD), spot=spot, need=need, edge=edge, sizing_text="", **base)

        # facing a single bet: value-raise the nuts (true pot-sized raise-to), else continue if ahead/priced
        if nut and can_raise(legal):
            target = _mybet(obs) + to_call + pot    # pot-sized raise-to total
            action, text, all_in = _size(obs, target, "pot raise", "raise")
            risked = (action.amount or 0) - _mybet(obs)
            cls, stack = _risk(obs, risked)
            return Plan(action=action, spot=spot, need=need, edge="clear ahead", risked=risked,
                        stack=stack, risk_class=cls, rank=_rank(obs), override=cls in ("large", "all-in"),
                        sizing_text=text, **base)
        if edge == "clear ahead" or drawing_ok:
            return _call("draw-priced" if drawing_ok else "clear ahead")
        return Plan(action=Action(ActionKind.FOLD), spot=spot, need=need, edge=edge, sizing_text="", **base)

    # not facing a bet: we open the street, or it was checked to us
    acted = any(r.street == obs.street and r.kind in ("check", "bet", "call", "raise", "fold")
                for r in obs.history)
    first = not acted
    spot = f"{obs.street} {'first to act' if first else 'checked to me'}"

    def _check():
        return Plan(action=check_or_fold(legal), spot=spot, plan_word="check",
                    first_to_act=first, sizing_text="", **base)

    if tier == "strong":
        plan_word = "value"
        target = round(pot * (1.0 if info.dangerous else 0.66))
        formula = "pot" if info.dangerous else "two-thirds pot"
    elif tight:
        return _check()   # tight player: value only, no thin protection bets or semi-bluffs
    elif tier == "medium":
        # protection needs a future card to deny; on the river (none left) it is thin value
        plan_word = "protection" if obs.street in ("flop", "turn") else "value"
        target, formula = round(pot * 0.5), "half pot"
    elif strong_draw:  # strong_draw is False on the river (no card to come)
        plan_word, target, formula = "semi-bluff", round(pot * 0.5), "half pot"
    else:
        return _check()

    lo = legal.min_raise_to
    # don't min-bet a non-value hand (a min semi-bluff folds nothing); just check
    if plan_word != "value" and lo is not None and target <= lo:
        return _check()
    action, text, all_in = _size(obs, target, formula, "bet")
    if all_in and tier != "strong":   # refuse a weak all-in (#3)
        return _check()
    risked = (action.amount or 0) - _mybet(obs)
    cls, stack = _risk(obs, risked)
    return Plan(action=action, spot=spot, plan_word=plan_word, first_to_act=first, risked=risked,
                stack=stack, risk_class=cls, rank=_rank(obs), override=cls in ("large", "all-in"),
                sizing_text=text, **base)


def _mybet(obs) -> int:
    return next(v for v in obs.seats if v.seat == obs.seat).bet
