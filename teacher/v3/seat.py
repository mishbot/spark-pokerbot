"""A harness seat that decides with the V3 rule-based teacher and records ONE SFT example per hand:
the hand's conversation through the hero's final decision (V3 trace + submit_action) followed by the
after-hand cleanup turn that calls update_scratchpad(""). So every sample contains BOTH a decision
and its empty-scratchpad cleanup (keeps the pad empty, no context pollution). The record carries the
final decision's `trace`/`truth`/`action` for the validator; `meta.street` is that decision's street.
"""

from __future__ import annotations

import copy
import json

from adapter import observation_from_ctx, seat_positions, sb_bb_seats, to_harness_decision

from poker_harness.seat import Completion, ModelSeat

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
                     me_seat=ctx.seat, live_order=live_order,
                     chips={v.seat: v.stack + v.bet for v in obs.seats})
        # collect this decision; the hand's ONE record is finalized at cleanup (with the full
        # conversation + the empty-scratchpad cleanup turn). The last decision collected is the
        # hero's latest-street action, which is what the sample is labeled by.
        self._hand_decisions.append({
            "trace": trace, "truth": truth, "action": [decision.type, decision.amount],
            "meta": {"seed": self._seed, "hand": ctx.hand_number, "street": obs.street,
                     "position": obs.position, "spot": plan.spot, "action": decision.type},
        })
        decision.model_response = {
            "model_name": "v3-teacher", "provider": "scripted-teacher", "latency_seconds": 0.0,
            "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0,
            "reasoning": trace, "fallback_used": False, "error": None, "format_error": False,
        }
        return decision

    async def _empty_update(self, messages):
        """The teacher's cleanup 'reply': one update_scratchpad call with empty contents (A-minimal)."""
        call = {"id": f"cu-{self.name}-{self.turn}", "type": "function",
                "function": {"name": "update_scratchpad", "arguments": json.dumps({"contents": ""})}}
        return Completion(text="", tool_calls=[call])

    async def cleanup(self, summary):
        # Run the real after-hand turn (builds the arena cleanup prompt + the tool result for the
        # hero's last action), answer it with an EMPTY update, then record ONE sample for the hand:
        # the full conversation (ending with that empty cleanup) + the final decision's labels.
        self.cleanup_complete = self._empty_update
        result = await super().cleanup(summary)
        if self._hand_decisions:
            last = self._hand_decisions[-1]   # the hero's latest-street (final) action this hand
            self._records.append({
                "messages": copy.deepcopy(self.messages),   # decision(s) + the empty cleanup turn
                "trace": last["trace"], "truth": last["truth"],
                "action": last["action"], "meta": last["meta"],
            })
        self._hand_decisions = []
        return result


def make_v3_seat(name, seat_index, rules, big_blind, small_blind, seed, records):
    async def _unused(messages):
        raise RuntimeError("v3 seat does not call a model")

    seat = V3Seat(name=name, complete=_unused, rules=rules, model="v3-teacher")
    seat._records = records
    seat._hand_decisions = []
    seat._big_blind = big_blind
    seat._small_blind = small_blind
    seat._seed = seed
    return seat
