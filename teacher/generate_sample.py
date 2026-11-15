"""Small VERIFIED sample of teacher SFT data (Phase 1): play a few full event games with the
teacher in 2 seats vs scripted villains, record one LabeledTask per teacher decision.

Not the real builder: fixed 3 games, no composition weighting, no injection, no upload.

    PYTHONPATH=teacher:harness/src uv run --with numpy --with phevaluator --with pokerkit \
        --with fastapi --with httpx --with pydantic python teacher/generate_sample.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from adapter import observation_from_ctx, to_harness_decision
from seat import make_teacher_seat, records_to_labeled_tasks

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session

TEACHER_SEATS = (0, 3)
VILLAINS = ("nit", "maniac", "calling_station", "random")
TOOLS = json.loads((Path(__file__).resolve().parents[1] / "opsd" / "tools.json").read_text())


async def main() -> None:
    records: list = []
    teacher_seats = []
    games = 0
    legality_failures = 0
    for g in range(3):
        seed = 20000 + g
        params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
        rules = table_rules(params)
        players = []
        for i in range(rules.seats):
            if i in TEACHER_SEATS:
                seat = make_teacher_seat(f"seat{i}", i, rules, rules.big_blind, rules.small_blind, seed, records)
                players.append(seat)
                teacher_seats.append(seat)
            else:
                players.append(ScriptedPlayer(f"seat{i}", VILLAINS[i % len(VILLAINS)], seed))
        log = await play_session(f"gen-{seed}", seed, players, rules)
        games += 1
        names = {f"seat{i}" for i in TEACHER_SEATS}
        for hand in log["hands"]:
            for a in hand["actions"]:
                if a["player"] in names:
                    r = a["model_response"]
                    if r.get("fallback_used") or r.get("error") or r.get("format_error"):
                        legality_failures += 1

    # Determinism: re-run the adapter + teacher on the first captured decision.
    det = "n/a"
    for seat in teacher_seats:
        if seat._first:
            ctx, first_action = seat._first
            obs = observation_from_ctx(ctx, seat._big_blind, seat._small_blind)
            again = to_harness_decision(seat._bot.decide(obs).action, ctx)
            det = "ok" if (again.type, again.amount) == first_action else f"MISMATCH {(again.type, again.amount)} vs {first_action}"
            break

    tasks = records_to_labeled_tasks(records, TOOLS)
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    path = out / "sample.jsonl"
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t.model_dump(mode="json")) + "\n")

    print(f"games={games} teacher_decisions={len(records)} labeled_tasks={len(tasks)}")
    print(f"teacher legality failures in-game: {legality_failures}")
    print(f"determinism: {det}")
    print(f"wrote {path}")
    # Show one example (system + last user + the assistant think/call).
    if tasks:
        sample = tasks[min(5, len(tasks) - 1)]
        items = sample.model_dump(mode="json")["input"]
        msgs = [i for i in items if i.get("type") == "message"]
        print("\n--- sample example (abbreviated) ---")
        print("SYSTEM:", msgs[0]["content"][:200], "...")
        print("\nLAST USER:\n", msgs[-2]["content"][-600:] if len(msgs) >= 2 else "(none)")
        asst = [i for i in items if i.get("type") == "message" and i["role"] == "assistant"][-1]
        call = [i for i in items if i.get("type") == "function_call"][-1]
        print("\nASSISTANT:", asst["content"])
        print("CALL:", call["name"], call["arguments"])


if __name__ == "__main__":
    asyncio.run(main())
