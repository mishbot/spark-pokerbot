"""v2 decision wrappers (THOR_ALPHA A1/A2/A3) + the FINAL reasoning trace + a shover villain.

All three are WRAPPERS over the vendored FieldBot(TEACHER) decision; the vendored teacher is not edited.
- A1 shove-call: facing a 20+ BB preflop all-in, decide by HAND TYPE, not the random-hand percentile.
- A2 anti-bluff: on the turn/river, do not bet/raise pure air (no pair, no flush draw).
- A3 value-widen: in CO/BTN/SB, open folded-to VALUE hands (pairs, suited aces, broadways), not air.

The FINAL trace drops the equity number (strength tier instead), derives the position, states the
situation first, shows the survival penalty on thin calls, and uses plain words. <= ~80 tokens.
"""

from __future__ import annotations

import copy
import json
import time

from adapter import observation_from_ctx, to_harness_decision
from poker_harness.engine import Decision as HDecision
from poker_harness.seat import Completion, ModelSeat

from pokerlab.bots.field.bot import FieldBot
from pokerlab.bots.field.postflop import estimate
from pokerlab.bots.field.standing import chips, lead_is_safe
from pokerlab.bots.randomness import decision_rng, seat_key
from pokerlab.bots.rule.ranges import position_group
from pokerlab.datatypes.actions import Action, ActionKind
from pokerlab.datatypes.field_personas import TEACHER
from pokerlab.poker.combos import combo_index
from pokerlab.poker.legal import raise_to_nearest
from pokerlab.poker.replay import AGGRESSIVE
from pokerlab.poker.situation import my_seat, street_history
from pokerlab.poker.strength import PREFLOP_PERCENTILE

RANKS = "23456789TJQKA"
LATE = {"CO", "BTN", "SB"}
WIDEN_TOP = 0.50          # A3: value-widen ceiling for late opens (value hands only, so safe to push)
SHOVE_BB = 20            # A1: "deep" preflop all-in threshold


def _facts(hole):
    a, b = hole
    return {"pair": a[0] == b[0], "pair_idx": RANKS.index(a[0]) if a[0] == b[0] else -1,
            "suited": a[1] == b[1], "lo": min(RANKS.index(a[0]), RANKS.index(b[0])),
            "has_ace": "A" in (a[0], b[0]), "ranks": {a[0], b[0]}}


def facing_preflop_shove(obs) -> bool:
    return (obs.street == "preflop" and obs.legal.to_call >= SHOVE_BB * obs.big_blind
            and any(v.all_in for v in obs.seats if v.seat != obs.seat))


def _leader_or_safe(obs) -> bool:
    me = chips(obs, obs.seat)
    leader = all(chips(obs, v.seat) <= me for v in obs.seats if v.seat != obs.seat)
    return leader or lead_is_safe(obs)


def shove_call_wanted(obs) -> bool:
    """A1: True = call the deep shove, False = fold. Never KQ/KJ/KT/QJ/QT/suited-connectors."""
    f = _facts(obs.hole_cards)
    pairs_ge = lambda n: f["pair"] and f["pair_idx"] >= RANKS.index(n)
    is_ak = f["ranks"] == {"A", "K"}
    is_aq = f["ranks"] == {"A", "Q"}
    if _leader_or_safe(obs):                      # leading -> tighter: TT+ / AK only
        return pairs_ge("T") or is_ak
    premium = pairs_ge("8") or is_ak or is_aq     # 88+, AQ+
    marginal = (f["pair"] and f["pair_idx"] in (RANKS.index("6"), RANKS.index("7"))) \
        or (f["has_ace"] and ("T" in f["ranks"] or "J" in f["ranks"]))   # 66-77, AT-AJ
    return premium or marginal


def value_widen_open(obs, teacher_action):
    """A3: open a folded-to value hand in CO/BTN/SB. Returns a new Action or None."""
    if obs.street != "preflop" or teacher_action.kind != ActionKind.FOLD:
        return None
    if obs.position not in LATE or obs.legal.to_call > obs.big_blind:   # obs.position is the WORD (CO/BTN/SB)
        return None
    if any(r.kind in AGGRESSIVE for r in street_history(obs)):   # unopened only
        return None
    f = _facts(obs.hole_cards)
    value = f["pair"] or (f["has_ace"] and f["suited"]) or f["lo"] >= RANKS.index("T")  # pair / suited ace / broadway
    pct = float(PREFLOP_PERCENTILE[combo_index(obs.hole_cards)])
    if value and pct <= WIDEN_TOP and obs.legal.min_raise_to is not None:
        return raise_to_nearest(obs.legal, round(2.5 * obs.big_blind))
    return None


def _has_pair(hole, board) -> bool:
    br = {c[0] for c in board}
    return hole[0][0] == hole[1][0] or hole[0][0] in br or hole[1][0] in br


def _flush_draw(hole, board) -> bool:
    for s in "cdhs":
        if sum(c[1] == s for c in list(hole) + list(board)) >= 4 and any(c[1] == s for c in hole):
            return True
    return False


def anti_bluff(obs, teacher_action):
    """A2: on the turn/river, do not bet/raise pure air. Returns a new Action or None.
    (Conservative: only suppresses no-pair, no-flush-draw aggression, which is the measured spew.)"""
    if obs.street not in ("turn", "river") or teacher_action.kind != ActionKind.RAISE:
        return None
    air = not _has_pair(obs.hole_cards, obs.board) and not _flush_draw(obs.hole_cards, obs.board)
    if not air:
        return None
    return Action(ActionKind.CHECK) if obs.legal.to_call == 0 else (
        Action(ActionKind.FOLD) if obs.legal.can_fold else Action(ActionKind.CALL))


def apply_v2(obs, teacher_action):
    """Apply the three wrappers in order; return (action, wrapper_tag)."""
    if facing_preflop_shove(obs):
        return (Action(ActionKind.CALL) if shove_call_wanted(obs) else Action(ActionKind.FOLD)), "a1_shove"
    vw = value_widen_open(obs, teacher_action)
    if vw is not None:
        return vw, "a3_widen"
    ab = anti_bluff(obs, teacher_action)
    if ab is not None:
        return ab, "a2_antibluff"
    return teacher_action, ""


# --- FINAL reasoning trace ----------------------------------------------------

def _pre_tier(pct):
    return ("premium" if pct <= 0.03 else "strong" if pct <= 0.10 else "good" if pct <= 0.25
            else "playable" if pct <= 0.45 else "weak")


def _post_tier(eq):
    return "strong" if eq >= 0.62 else "good" if eq >= 0.52 else "marginal" if eq >= 0.44 else "weak"


def _act(d) -> str:
    return d.type + (f" to {d.amount}" if d.amount is not None else "")


def v2_reasoning_line(obs, decision, equity=None, tag="") -> str:
    pos, grp = obs.position, position_group(obs.position)
    posline = f"Seat {obs.seat} acts as {pos} ({grp}); the seat number is not the position."
    if obs.street == "preflop":
        pct = float(PREFLOP_PERCENTILE[combo_index(obs.hole_cards)])
        tier = _pre_tier(pct)
        if facing_preflop_shove(obs):  # A1: lead with the situation
            verdict = ("outside the call range (88+/AQ+), so fold" if decision.type == "fold"
                       else "inside the call range (88+/AQ+), so call")
            return (f"Facing an all-in: use the call range, not the open range. {posline} "
                    f"Hand {tier} (top {pct * 100:.0f}% vs random, but an ace-heavy shove dominates high cards); "
                    f"{verdict}. {_act(decision)}.")
        if tag == "a3_widen":  # A3: lead with the override reason
            return (f"Value hand; widen the late open against callers (they do not fold, so open for value). "
                    f"{posline} Hand {tier} (top {pct * 100:.0f}%). {_act(decision)}.")
        return f"{posline} Hand {tier} (top {pct * 100:.0f}%). {_act(decision)}."
    pot, to_call = obs.pot, obs.legal.to_call
    need = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    tier = _post_tier(equity if equity is not None else 0.5)
    me = my_seat(obs)
    stack = max(me.stack, 1)
    risked = to_call if decision.type == "call" else (decision.amount - me.bet if decision.amount else 0)
    surv = ""
    if risked / stack > 0.5:
        mc = chips(obs, obs.seat)
        rank = 1 + sum(chips(obs, v.seat) > mc for v in obs.seats if v.seat != obs.seat)
        surv = (f" Risk {risked / stack * 100:.0f}% of stack, rank {rank} of {len(obs.seats)}; "
                f"survival matters, so take the low-risk line.")
    needline = (f" Need {need * 100:.0f}% ({'cheap' if need < 0.3 else 'steep'})." if to_call > 0 else " No bet to me.")
    if tag == "a2_antibluff":  # A2: lead with the override reason
        return f"No pair here; do not barrel into a calling field. {posline} Hand {tier}.{needline}{surv} {_act(decision)}."
    return f"{posline} Hand {tier}.{needline}{surv} {_act(decision)}."


# --- shover villain (for A1 coverage) -----------------------------------------

class ShoverPlayer:
    """Shoves all-in preflop with any pair or any ace; otherwise checks or folds. Produces the deep
    facing-all-in spots A1 must handle. Implements the harness Player protocol."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def decide(self, ctx, history) -> HDecision:
        hole = ctx.hole_cards
        pair = hole[0][0] == hole[1][0]
        ace = "A" in (hole[0][0], hole[1][0])
        if ctx.street == "preflop" and (pair or ace) and ctx.legal.raise_type:
            return HDecision("raise", ctx.legal.max_raise_to)
        if ctx.legal.can_check:
            return HDecision("check")
        return HDecision("fold")


# --- v2 teacher seat ----------------------------------------------------------

class TeacherSeatV2(ModelSeat):
    """Like teacher/seat.TeacherModelSeat but applies the A1/A2/A3 wrappers and writes the FINAL trace."""

    async def decide(self, ctx, history):
        self.turn += 1
        self._next_messages(ctx)
        obs = observation_from_ctx(ctx, self._big_blind, self._small_blind)
        persona = self._persona_fn(ctx.hand_number, ctx.total_hands) if self._persona_fn else TEACHER
        self._bot.persona = persona
        samples = self._equity_samples
        t0 = time.perf_counter()
        teacher_action = self._bot.decide(obs).action
        action, tag = apply_v2(obs, teacher_action)
        decision = to_harness_decision(action, ctx)
        equity = None
        if obs.street != "preflop":
            equity, _ = estimate(obs, {}, samples, decision_rng(self._key, obs))
        self._last_equity = equity
        line = v2_reasoning_line(obs, decision, equity, tag)
        decide_ms = (time.perf_counter() - t0) * 1000.0
        args = {"action": decision.type}
        if decision.amount is not None:
            args["amount"] = int(decision.amount)
        call_id = f"call-{self.name}-{self.turn}"
        tool_call = {"id": call_id, "type": "function",
                     "function": {"name": "submit_action", "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": f"<think>\n{line}\n</think>", "tool_calls": [tool_call]})
        self._pending_calls = [tool_call]
        self._records.append({"messages": copy.deepcopy(self.messages),
                              "meta": self._meta(ctx, obs, decision, decide_ms, tag)})
        decision.model_response = {
            "model_name": "teacher-v2", "provider": "scripted-teacher", "latency_seconds": 0.0,
            "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0,
            "reasoning": line, "fallback_used": False, "error": None, "format_error": False,
        }
        return decision

    def _meta(self, ctx, obs, decision, decide_ms, tag):
        me = next(v for v in ctx.seats if v.seat == ctx.seat)
        facing_all_in = ctx.legal.call_amount > 0 and any(v.status == "all-in" for v in ctx.seats)
        big_pot = me.stack > 0 and ctx.pot / me.stack > 0.5
        bust = facing_all_in or big_pot
        trivial_fold = ctx.street == "preflop" and decision.type == "fold" and not facing_all_in
        need = obs.legal.to_call / (obs.pot + obs.legal.to_call) if (obs.pot + obs.legal.to_call) > 0 else 0.0
        eq = getattr(self, "_last_equity", None)
        near_tie = bool(obs.street != "preflop" and eq is not None and abs(eq - need) < 0.03)
        hole = list(obs.hole_cards)
        cls = (hole[0][0] + hole[1][0]) if hole[0][0] == hole[1][0] else (
            "".join(sorted([hole[0][0], hole[1][0]], key=lambda r: -RANKS.index(r)))
            + ("s" if hole[0][1] == hole[1][1] else "o"))
        return {
            "seed": self._seed, "hand": ctx.hand_number, "total_hands": ctx.total_hands,
            "seat": self.name, "street": ctx.street, "action": decision.type, "amount": decision.amount,
            "pos_group": position_group(obs.position), "pos": obs.position, "stack_bb": me.stack / self._big_blind,
            "players_left": sum(v.status in ("", "all-in") for v in ctx.seats),
            "can_check": ctx.legal.can_check,
            "min_raise_to": ctx.legal.min_raise_to if ctx.legal.raise_type else None,
            "max_raise_to": ctx.legal.max_raise_to if ctx.legal.raise_type else None,
            "facing_all_in": facing_all_in, "trivial_fold": trivial_fold, "bust": bust,
            "decide_ms": decide_ms, "wrapper": tag, "hole_class": cls, "deep_shove": facing_preflop_shove(obs),
            "near_tie": near_tie, "kind": "decision",
        }

    def _scratchpad_template(self, summary) -> str:
        """A compact, fixed template computed from the game log + running tallies. < ~80 tokens."""
        t = getattr(self, "_tallies", None)
        if t is None:
            t = self._tallies = {}
        for a in summary.get("actions", []):
            p = a.get("player")
            if not p or p == self.name:
                continue
            d = t.setdefault(p, {"raise": 0, "call": 0, "fold": 0, "n": 0})
            kind = a.get("action", {}).get("type", "")
            if kind in d:
                d[kind] += 1
            d["n"] += 1
        stacks = summary.get("stacks", {})
        total = self.rules.total_hands
        hand = summary.get("hand_number", 0)
        bb = max(self._big_blind, 1)
        mine = stacks.get(self.name, 0)
        live = [(n, s) for n, s in stacks.items() if s > 0]
        rank = 1 + sum(1 for n, s in live if n != self.name and s > mine)
        parts = [f"H{hand}/{total}", f"me rank{rank}/{len(live)} {mine // bb}bb"]
        for p, d in sorted(t.items()):
            if stacks.get(p, 0) <= 0:
                continue
            tag = ("aggro" if d["raise"] >= 2 else "loose" if d["call"] >= 3
                   else "tight" if d["n"] and d["fold"] >= d["n"] - 1 else "?")
            parts.append(f"{p}:{tag}")
        parts.append(f"left {max(total - hand, 0)}")
        return " | ".join(parts)[:4900]

    async def cleanup(self, summary):
        """Write a template scratchpad via update_scratchpad (so later hands see it) AND record the
        cleanup turn as a labeled task (system + the exact cleanup prompt + the update_scratchpad call)."""
        tmpl = self._scratchpad_template(summary)

        async def _reply(messages):
            tc = {"id": f"cu-{self.name}-{summary.get('hand_number', 'x')}", "type": "function",
                  "function": {"name": "update_scratchpad", "arguments": json.dumps({"contents": tmpl})}}
            return Completion(text="<think>\nUpdate notes: opponent styles, stacks, my rank, hands left.\n</think>",
                              tool_calls=[tc], tool_arguments=json.dumps({"contents": tmpl}))

        self.cleanup_complete = _reply
        result = await super().cleanup(summary)
        # Robust snapshot (super().cleanup may reset self.messages for a fresh/walk seat): find the last
        # assistant turn that called update_scratchpad, and the user (cleanup) prompt just before it.
        ai = next((i for i in range(len(self.messages) - 1, -1, -1)
                   if self.messages[i].get("role") == "assistant"
                   and any(tc["function"]["name"] == "update_scratchpad"
                           for tc in (self.messages[i].get("tool_calls") or []))), None)
        if ai is not None:
            ui = next((j for j in range(ai - 1, -1, -1) if self.messages[j].get("role") == "user"), None)
            if ui is not None:
                cleanup_msgs = [self.messages[0], self.messages[ui], self.messages[ai]]
                self._records.append({"messages": copy.deepcopy(cleanup_msgs),
                                      "meta": {"kind": "cleanup", "seed": self._seed,
                                               "hand": summary.get("hand_number"), "seat": self.name}})
        return result


def make_v2_seat(name, seat_index, rules, big_blind, small_blind, seed, records,
                 persona_fn=None, equity_samples=1000):
    async def _unused(messages):
        raise RuntimeError("teacher seat does not call a model")
    seat = TeacherSeatV2(name=name, complete=_unused, rules=rules, model="teacher-v2")
    seat._bot = FieldBot(TEACHER, seat_key(seed, seat_index))
    seat._key = seat_key(seed, seat_index)
    seat._records = records
    seat._big_blind = big_blind
    seat._small_blind = small_blind
    seat._seed = seed
    seat._persona_fn = persona_fn
    seat._equity_samples = equity_samples
    seat._tallies = {}
    seat._last_equity = None
    return seat
