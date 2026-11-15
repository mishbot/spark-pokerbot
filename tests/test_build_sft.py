"""sim/build_sft.py: decisions from a simulated game become SFT tasks that are kept, relabeled to folds, or dropped."""

import json
import random
import sys
from pathlib import Path

from ac2.runtime import FunctionCall, Reasoning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import build_sft  # noqa: E402

TOOLS = [
    {"type": "function", "function": {k: v for k, v in t.items() if k != "type"}}
    for t in json.loads((Path(__file__).resolve().parents[1] / "opsd" / "tools.json").read_text())
]


def action(street, kind, call, stack_after, put_in, pot, error=False):
    return {
        "player": "seat0",
        "street": street,
        "action": {"type": kind},
        "call_amount": call,
        "stack_after": stack_after,
        "put_in": put_in,
        "pot_before": pot,
        "model_response": {"format_error": error, "fallback_used": error},
    }


def hand(n, hole, *acts):
    return {"hand_number": n, "starting_state": {"seat0": {"hole_cards": hole}}, "actions": list(acts)}


def call(kind="decision", submit=True):
    calls = [{"id": "c1", "type": "function", "function": {"name": "submit_action", "arguments": '{"action": "call"}'}}]
    return {
        "kind": kind,
        "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "your turn"}],
        "reply": {"text": "", "reasoning": "thinking", "tool_calls": calls if submit else [], "out_tokens": 100},
    }


def test_decisions_are_kept_relabeled_or_dropped() -> None:
    log = {
        "game_id": "sim-000-1",
        "eliminated_on_hand": {},
        "hands": [
            hand(1, ["7c", "2d"], action("preflop", "call", 198, 0, 198, 210)),  # junk all-in call: relabel
            hand(2, ["Ac", "Ad"], action("preflop", "raise", 2, 190, 6, 3)),  # keep
            hand(3, ["9c", "4d"], action("preflop", "call", 10, 180, 10, 15)),  # junk call of a raise: relabel
            hand(
                4,
                ["Kc", "Qd"],
                action("preflop", "check", 0, 180, 0, 4, error=True),  # failed: drop
                action("flop", "check", 0, 180, 0, 4),  # same hand after a dropped one: drop
            ),
            hand(5, ["8c", "7d"], action("flop", "call", 50, 0, 50, 300)),  # all-in, but cheap next to the pot: keep
        ],
    }
    transcript = {
        "tools": TOOLS,
        "calls": [call(), call(), call("cleanup"), call(), call(submit=False), call(), call()],
    }
    tasks, stats = build_sft.game_tasks(log, transcript, copies=2, rng=random.Random(0))

    assert stats["kept"] == 2 and stats["relabeled: call_off"] == 1 and stats["relabeled: defend"] == 1
    assert stats["dropped: failed decision"] == 1 and stats["dropped: after a dropped decision"] == 1
    assert len(tasks) == 2 + 2 * 2  # relabeled folds are repeated
    folds = [t for t in tasks if t.env_params["label"].startswith("relabeled")]
    for t in folds:
        assert isinstance(t.input[-2], Reasoning) and isinstance(t.input[-1], FunctionCall)
        assert json.loads(t.input[-1].arguments) == {"action": "fold"}
    assert "72o" in folds[0].input[-2].content or "94o" in folds[0].input[-2].content
    assert all(t.env_params["tools"][0]["name"] == "submit_action" for t in tasks)


def test_the_bust_hand_is_dropped_and_mismatched_transcripts_refused() -> None:
    log = {
        "game_id": "g",
        "eliminated_on_hand": {"seat0": 1},
        "hands": [hand(1, ["Ac", "Ad"], action("preflop", "call", 198, 0, 198, 210))],
    }
    tasks, stats = build_sft.game_tasks(log, {"tools": TOOLS, "calls": [call()]}, 2, random.Random(0))
    assert not tasks and stats["dropped: bust hand"] == 1
    try:
        build_sft.game_tasks(log, {"tools": TOOLS, "calls": []}, 2, random.Random(0))
    except ValueError as e:
        assert "recorded decisions" in str(e)
    else:
        raise AssertionError("a transcript that doesn't match the log must be refused")
