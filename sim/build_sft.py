"""Build an SFT dataset from local simulations: seat 0's own decisions, filtered, with its worst leak relabeled.

Rejection-sampling SFT ("expert iteration"): play many games with sim/simulate.py, keep the decisions worth
imitating, fix the ones we know are wrong, fine-tune, then play the same seeds again and compare.

Each game directory holds sim-*.json (the game log) and transcript-*.json (every request seat 0 sent and the reply it
got, exactly as sent). Decisions are matched to the log's actions in order, and each becomes one `LabeledTask`:
`input` is the exact prompt followed by the reply (reasoning, text and the `submit_action` call), and
`env_params.tools` lists the tools the model saw (training refuses tool calls without them).

What happens to each decision:

    dropped     failed decisions: no single submit_action, cut off at the token cap, or an illegal move the harness
                had to change; every decision in a hand where seat 0 busted; and later decisions in a hand after an
                earlier one was dropped (they'd carry the dropped reply in their prompt)
    relabeled   calling off (nearly) the whole stack with a hand outside CALL_OFF, unless the price was small next to
                the pot; and preflop calls of a raise with a hand outside DEFEND. The reply is replaced by a short
                reasoning and a fold, and repeated --relabel-copies times so the fix carries weight
    kept        everything else, as the model wrote it

After-hand scratchpad turns are not used: the model's notes have no opponent reads worth imitating.

    uv run python sim/build_sft.py harness/runs/sim/ravi-gen-* --dataset pokerbots-sft-ravi-sim --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ac2.runtime import FunctionCall, FunctionCallOutput, GraderOutput, LabeledTask, Message, Reasoning, Tool

RANKS = "23456789TJQKA"
SUBMIT = "submit_action"
BIG_BLIND = 2
MAX_TOKENS = 5_000


def hand_class(hole: list[str]) -> str:
    """ "AA", "AKs" or "AKo"."""
    a, b = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    if a == b:
        return RANKS[a] * 2
    return RANKS[a] + RANKS[b] + ("s" if hole[0][1] == hole[1][1] else "o")


def _pairs(low: str) -> set[str]:
    return {r * 2 for r in RANKS[RANKS.index(low) :]}


# Hands worth risking the whole stack for, facing an all-in.
CALL_OFF = _pairs("T") | {"AKs", "AKo", "AQs"}
# Hands worth calling a preflop raise with (anything else folds).
DEFEND = (
    _pairs("2")
    | {f"A{r}s" for r in "23456789TJQK"}
    | {"AKo", "AQo", "AJo", "ATo", "KQs", "KJs", "KTs", "KQo", "KJo", "QJs", "QTs", "JTs", "T9s", "98s", "87s", "76s"}
)
# Calling off is fine with any hand when the price is this small a share of the pot after the call.
CHEAP_SHARE = 0.25

FOLD_REASONS = (
    (
        "I have {hand}. Calling costs {call} chips, which is {share} of my {stack}-chip stack. Someone who puts this much "
        "in usually has a strong hand, and {hand} is behind most of those hands. Busting ends my game and ranks me below "
        "everyone still in, so this is not worth it. Fold."
    ),
    (
        "{hand} is not strong enough to call {call} chips from a {stack}-chip stack. An all-in or a big raise here is "
        "usually a pair or a big ace, and I'd be a clear underdog. Survival matters more than this pot. Fold."
    ),
    (
        "Facing a bet of {call} with {hand}. Against the hands that bet this big I'm well behind, and losing would cost "
        "{share} of my stack. I'll wait for a better spot. Fold."
    ),
)
DEFEND_REASONS = (
    (
        "{hand} facing a raise. This hand plays poorly against a raising range and often ends up dominated after the "
        "flop. Calling {call} just to see a flop loses chips over time. Fold."
    ),
    (
        "Someone raised and I hold {hand}. That's too weak to call {call} here. I'd rather fold and pick better spots. "
        "Fold."
    ),
)


def items(messages: list[dict[str, Any]]) -> list:
    """Chat messages (system, user, assistant with tool calls, tool results) as AC2 transcript items."""
    out: list = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append(FunctionCallOutput(call_id=m["tool_call_id"], output=m["content"]))
            continue
        if m.get("content"):
            out.append(Message(role=role, content=m["content"]))
        for call in m.get("tool_calls") or []:
            fn = call["function"]
            out.append(FunctionCall(call_id=call["id"], name=fn["name"], arguments=fn["arguments"]))
    return out


def reply_items(reply: dict[str, Any]) -> list:
    out: list = []
    if reply.get("reasoning", "").strip():
        out.append(Reasoning(content=reply["reasoning"]))
    if reply.get("text", "").strip():
        out.append(Message(role="assistant", content=reply["text"]))
    for call in reply.get("tool_calls") or []:
        fn = call["function"]
        out.append(FunctionCall(call_id=call["id"], name=fn["name"], arguments=fn["arguments"]))
    return out


def fold_items(template: str, **facts: Any) -> list:
    return [
        Reasoning(content=template.format(**facts)),
        FunctionCall(call_id="call_fold", name=SUBMIT, arguments=json.dumps({"action": "fold"})),
    ]


def flat_tools(chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-completions tool schemas in the flat shape AC2 tasks list (as opsd/tools.json)."""
    return [Tool.model_validate({"type": "function", **t["function"]}).model_dump() for t in chat_tools]


def verdict(action: dict[str, Any], hole: list[str]) -> tuple[str, str | None]:
    """("keep" | "relabel", which rule) for one logged action of seat 0."""
    kind = action["action"]["type"]
    call = action.get("call_amount") or 0
    stack = action["stack_after"] + action.get("put_in", 0)
    hand = hand_class(hole)
    expensive = call / (action["pot_before"] + call) > CHEAP_SHARE if call else False
    if kind == "call" and call >= 0.9 * stack and hand not in CALL_OFF and expensive:
        return "relabel", "call_off"
    if action["street"] == "preflop" and kind == "call" and call > BIG_BLIND and hand not in DEFEND:
        return "relabel", "defend"
    return "keep", None


def game_tasks(log: dict[str, Any], transcript: dict[str, Any], copies: int, rng: random.Random):
    tools = flat_tools(transcript["tools"])
    decisions = [c for c in transcript["calls"] if c["kind"] == "decision"]
    actions = [(h, a) for h in log["hands"] for a in h["actions"] if a["player"] == "seat0"]
    if len(decisions) != len(actions):
        raise ValueError(f"{log['game_id']}: {len(decisions)} recorded decisions, {len(actions)} logged actions")
    busted_on = log["eliminated_on_hand"].get("seat0")
    stats: Counter = Counter()
    tasks: list[LabeledTask] = []
    tainted: set[int] = set()  # hands with a dropped decision: later prompts in them hold the dropped reply
    for call, (hand, action) in zip(decisions, actions):
        stats["decisions"] += 1
        n = hand["hand_number"]
        reply = call.get("reply") or {}
        submits = [c for c in reply.get("tool_calls") or [] if c["function"]["name"] == SUBMIT]
        response = action["model_response"]
        failed = (
            "error" in call
            or len(submits) != 1
            or reply.get("truncated")
            or (reply.get("out_tokens") or 0) >= MAX_TOKENS
            or response.get("format_error")
            or response.get("fallback_used")
        )
        hole = hand["starting_state"]["seat0"]["hole_cards"]
        decided, rule = verdict(action, hole)
        if failed:
            stats["dropped: failed decision"] += 1
            tainted.add(n)
            continue
        if n in tainted:
            stats["dropped: after a dropped decision"] += 1
            continue
        prompt = items(call["messages"])
        meta = {"tools": tools, "source": log["game_id"], "hand": n}
        if decided == "relabel":
            stack = action["stack_after"] + action.get("put_in", 0)
            facts = {
                "hand": hand_class(hole),
                "call": action["call_amount"],
                "stack": stack,
                "share": f"{min(1.0, action['call_amount'] / max(stack, 1)):.0%}",
            }
            templates = FOLD_REASONS if rule == "call_off" else DEFEND_REASONS
            for _ in range(copies):
                tasks.append(
                    LabeledTask(
                        input=prompt + fold_items(rng.choice(templates), **facts),
                        env_params=meta | {"label": f"relabeled:{rule}"},
                        output=GraderOutput(score=1.0, reasoning=f"relabeled {rule}: {facts['hand']} -> fold"),
                    )
                )
            stats[f"relabeled: {rule}"] += 1
            tainted.add(n)  # the logged hand went on with the original call
            continue
        if n == busted_on:
            stats["dropped: bust hand"] += 1
            tainted.add(n)
            continue
        tasks.append(
            LabeledTask(
                input=prompt + reply_items(reply),
                env_params=meta | {"label": "kept"},
                output=GraderOutput(score=1.0, reasoning="kept"),
            )
        )
        stats["kept"] += 1
    return tasks, stats


def load_games(dirs: list[Path]):
    for d in dirs:
        for path in sorted(d.glob("sim-*.json")):
            transcript = path.with_name(path.name.replace("sim-", "transcript-"))
            if transcript.exists():
                yield json.loads(path.read_text()), json.loads(transcript.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="simulate.py --out directories")
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--dataset", required=True, help="pokerbots-sft-<owner>-<name>")
    parser.add_argument("--relabel-copies", type=int, default=3, help="copies of each relabeled fold")
    parser.add_argument("--examples", type=int, default=0, help="print this many relabeled examples")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(0)
    tasks: list[LabeledTask] = []
    totals: Counter = Counter()
    games = 0
    for log, transcript in load_games(args.dirs):
        built, stats = game_tasks(log, transcript, args.relabel_copies, rng)
        tasks.extend(built)
        totals.update(stats)
        games += 1
    rng.shuffle(tasks)
    print(json.dumps({"games": games, **dict(sorted(totals.items())), "tasks": len(tasks)}, indent=1))
    relabeled = [t for t in tasks if t.env_params["label"].startswith("relabeled")]
    for t in relabeled[: args.examples]:
        print(f"\n{t.output.reasoning}\n  {t.input[-2].content}")
    if args.dry_run or not tasks:
        return

    from ac2.sdk import Client

    client = Client(project=args.project)
    client.datasets.create(
        args.dataset, description=f"SFT from {games} local sim games", required_types=["labeled"], exist_ok=True
    )
    for start in range(0, len(tasks), 500):
        client.datasets.add_tasks(args.dataset, tasks=tasks[start : start + 500])
    print(f"uploaded {len(tasks)} tasks to {args.dataset}")


if __name__ == "__main__":
    main()
