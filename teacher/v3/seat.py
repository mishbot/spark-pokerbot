"""A harness seat that decides with the V3 rule-based teacher and records one SFT example per
decision, with the V3 trace as the assistant <think> content and a submit_action call.

Each record also carries `trace`, `truth`, and `action` so the independent validator can gate it.
Phase 1: cleanup is a no-op (empty scratchpad).
"""

from __future__ import annotations

import copy
import json

from adapter import observation_from_ctx, seat_positions, sb_bb_seats, to_harness_decision

from poker_harness.seat import ModelSeat

from v3.policy import decide
from v3.trace import render


class V3Seat(ModelSeat):
    async def decide(self, ctx, history):
        self.turn += 1
        self._next_messages(ctx)
        obs = observation_from_ctx(ctx, self._big_blind, self._small_blind)
        plan = decide(obs)
        decision = to_harness_decision(plan.action, ctx)

        pos_map, order = seat_positions(ctx)
        sb_seat, bb_seat = sb_bb_seats(order, ctx.button_seat)
        # postflop "in position" = we are the last live seat to act (the button side), NOT the SB/BB word
        status = {v.seat: v.status for v in ctx.seats}
        live_order = [s for s in order if status.get(s) in ("", "all-in")]
        in_position = bool(live_order) and ctx.seat == live_order[-1]
        trace = render(plan, button=ctx.button_seat, me=ctx.seat, sb_seat=sb_seat,
                       bb_seat=bb_seat, n=len(order), in_position=in_position)

        args = {"action": decision.type}
        if decision.amount is not None:
            args["amount"] = int(decision.amount)
        call_id = f"call-{self.name}-{self.turn}"
        tool_call = {"id": call_id, "type": "function",
                     "function": {"name": "submit_action", "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": f"<think>\n{trace}\n</think>",
                              "tool_calls": [tool_call]})
        self._pending_calls = [tool_call]

        truth = dict(hole=list(obs.hole_cards), board=list(obs.board), pot=obs.pot,
                     to_call=obs.legal.to_call, street=obs.street, position=obs.position,
                     min_raise=obs.legal.min_raise_to, max_raise=obs.legal.max_raise_to,
                     me_seat=ctx.seat, live_order=live_order)
        self._records.append({
            "messages": copy.deepcopy(self.messages),
            "trace": trace,
            "truth": truth,
            "action": [decision.type, decision.amount],
            "meta": {"seed": self._seed, "hand": ctx.hand_number, "street": obs.street,
                     "position": obs.position, "spot": plan.spot, "action": decision.type},
        })
        decision.model_response = {
            "model_name": "v3-teacher", "provider": "scripted-teacher", "latency_seconds": 0.0,
            "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0,
            "reasoning": trace, "fallback_used": False, "error": None, "format_error": False,
        }
        return decision

    async def cleanup(self, summary):
        return {"result": "unchanged", "error": None}


def make_v3_seat(name, seat_index, rules, big_blind, small_blind, seed, records):
    async def _unused(messages):
        raise RuntimeError("v3 seat does not call a model")

    seat = V3Seat(name=name, complete=_unused, rules=rules, model="v3-teacher")
    seat._records = records
    seat._big_blind = big_blind
    seat._small_blind = small_blind
    seat._seed = seed
    return seat
