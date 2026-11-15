"""A harness seat that decides with the ported teacher and records one SFT example per decision.

`TeacherModelSeat` subclasses the harness `ModelSeat`, so it reuses the exact event prompt
machinery (`_next_messages`: system prompt, scratchpad block, per-turn tool results). Instead of
calling a model it runs `FieldBot(TEACHER)` on the adapted `Observation`, writes a short reasoning
line plus a `submit_action` call, and snapshots the conversation as a training example.

Phase 1 does not train the scratchpad: `cleanup` is a no-op, so games run with an empty scratchpad.
"""

from __future__ import annotations

import copy
import json
import time

from adapter import observation_from_ctx, to_harness_decision

from poker_harness.seat import ModelSeat

from pokerlab.bots.field.bot import FieldBot
from pokerlab.bots.field.postflop import estimate
from pokerlab.bots.field.standing import chips, lead_is_safe
from pokerlab.bots.randomness import decision_rng, seat_key
from pokerlab.bots.rule.ranges import position_group
from pokerlab.datatypes.field_personas import TEACHER
from pokerlab.datatypes.range_model import OPEN_BAND
from pokerlab.poker.combos import combo_index
from pokerlab.poker.situation import my_seat
from pokerlab.poker.strength import PREFLOP_PERCENTILE

LINE_EQUITY_SAMPLES = 200  # for the reasoning line in this sample run; the real builder uses 5000


def _act(decision) -> str:
    return decision.type + (f" to {decision.amount}" if decision.amount is not None else "")


def reasoning_line(obs, decision, key: str, persona=TEACHER, samples: int = LINE_EQUITY_SAMPLES) -> str:
    """The short label line (handoff section 8). Preflop uses the percentile/band (cheap, exact);
    postflop uses one equity estimate for the line. Pass the ACTING persona and the SAME `samples` as
    postflop.EQUITY_SAMPLES so the line's equity equals what the teacher decided with (decision_rng is
    deterministic in (key, obs), so identical inputs reproduce the exact estimate)."""
    if obs.street == "preflop":
        pct = float(PREFLOP_PERCENTILE[combo_index(obs.hole_cards)]) * 100
        open_top = OPEN_BAND[position_group(obs.position)] * persona.looseness * 100
        return f"Hand top {pct:.0f}% | open here top {open_top:.0f}% | {_act(decision)}"
    equity, _ = estimate(obs, {}, samples, decision_rng(key, obs))
    pot, to_call = obs.pot, obs.legal.to_call
    need = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
    me = my_seat(obs)
    stack = max(me.stack, 1)
    risked = to_call if decision.type == "call" else (decision.amount - me.bet if decision.amount else 0)
    my_chips = chips(obs, obs.seat)
    rank = 1 + sum(chips(obs, v.seat) > my_chips for v in obs.seats if v.seat != obs.seat)
    safe = "yes" if lead_is_safe(obs) else "no"
    return (
        f"Equity {equity * 100:.0f}% | Need {need * 100:.0f}% | Risk {risked / stack * 100:.0f}% of stack "
        f"| {rank} of {len(obs.seats)} in chips, safe lead: {safe} | {_act(decision)}"
    )


class TeacherModelSeat(ModelSeat):
    async def decide(self, ctx, history):
        self.turn += 1
        self._next_messages(ctx)  # build the event-format conversation; sets _seen, clears _pending_calls
        obs = observation_from_ctx(ctx, self._big_blind, self._small_blind)
        persona_fn = getattr(self, "_persona_fn", None)
        persona = persona_fn(ctx.hand_number, ctx.total_hands) if persona_fn else TEACHER
        self._bot.persona = persona  # tight-early picks a tightened persona by hand number
        samples = getattr(self, "_equity_samples", LINE_EQUITY_SAMPLES)
        t0 = time.perf_counter()
        decision = to_harness_decision(self._bot.decide(obs).action, ctx)
        line = reasoning_line(obs, decision, self._key, persona=persona, samples=samples)
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
                              "meta": self._meta(ctx, obs, decision, decide_ms)})
        if self._first is None:
            self._first = (ctx, (decision.type, decision.amount))
        decision.model_response = {
            "model_name": "teacher", "provider": "scripted-teacher", "latency_seconds": 0.0,
            "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0,
            "reasoning": line, "fallback_used": False, "error": None, "format_error": False,
        }
        return decision

    def _meta(self, ctx, obs, decision, decide_ms):
        """Facts used later for composition, the distribution report and quality checks."""
        me = next(v for v in ctx.seats if v.seat == ctx.seat)
        facing_all_in = ctx.legal.call_amount > 0 and any(v.status == "all-in" for v in ctx.seats)
        # Bust-deciding = genuinely stack-threatening. The blanket "river" flag over-counts once
        # calling-stations drive many hands to a small-pot showdown, so require a large pot instead.
        big_pot = me.stack > 0 and ctx.pot / me.stack > 0.5
        bust = facing_all_in or big_pot
        trivial_fold = ctx.street == "preflop" and decision.type == "fold" and not facing_all_in
        return {
            "seed": getattr(self, "_seed", None), "hand": ctx.hand_number, "total_hands": ctx.total_hands,
            "seat": self.name, "street": ctx.street, "action": decision.type, "amount": decision.amount,
            "pos_group": position_group(obs.position), "stack_bb": me.stack / self._big_blind,
            "players_left": sum(v.status in ("", "all-in") for v in ctx.seats),
            "can_check": ctx.legal.can_check,
            "min_raise_to": ctx.legal.min_raise_to if ctx.legal.raise_type else None,
            "max_raise_to": ctx.legal.max_raise_to if ctx.legal.raise_type else None,
            "facing_all_in": facing_all_in, "trivial_fold": trivial_fold, "bust": bust,
            "decide_ms": decide_ms,
        }

    async def cleanup(self, summary):
        return {"result": "unchanged", "error": None}  # Phase 1: no scratchpad training


def make_teacher_seat(name, seat_index, rules, big_blind, small_blind, seed, records,
                      persona_fn=None, equity_samples=LINE_EQUITY_SAMPLES):
    async def _unused(messages):
        raise RuntimeError("teacher seat does not call a model")

    seat = TeacherModelSeat(name=name, complete=_unused, rules=rules, model="teacher")
    seat._bot = FieldBot(TEACHER, seat_key(seed, seat_index))
    seat._key = seat_key(seed, seat_index)
    seat._records = records
    seat._big_blind = big_blind
    seat._small_blind = small_blind
    seat._seed = seed
    seat._persona_fn = persona_fn            # persona_fn(hand_number, total_hands) -> Persona, or None
    seat._equity_samples = equity_samples    # must match postflop.EQUITY_SAMPLES
    seat._first = None
    return seat


# --- SFT example conversion ---------------------------------------------------

from ac2.runtime import FunctionCall, FunctionCallOutput, LabeledTask, Message  # noqa: E402
from ac2.runtime.base.types import GraderOutput  # noqa: E402


def chat_to_items(messages):
    """OpenAI-chat messages -> ac2.runtime items for a LabeledTask `input`."""
    items = []
    for m in messages:
        if m["role"] == "tool":
            items.append(FunctionCallOutput(call_id=m["tool_call_id"], output=m["content"]))
            continue
        if m.get("content"):
            items.append(Message(role=m["role"], content=m["content"]))
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            items.append(FunctionCall(call_id=tc["id"], name=fn["name"], arguments=fn["arguments"]))
    return items


def records_to_labeled_tasks(records, tools):
    """One LabeledTask per recorded decision snapshot (full conversation up to that assistant turn).

    Accepts either a bare message list (the sample generator) or a {"messages", "meta"} dict (the
    dataset generator)."""
    tasks = []
    for r in records:
        messages = r["messages"] if isinstance(r, dict) else r
        tasks.append(
            LabeledTask(
                input=chat_to_items(messages),
                output=GraderOutput(score=0.0, reasoning="teacher sft label"),
                task_types=["labeled"],
                env_params={"tools": tools},
            )
        )
    return tasks
