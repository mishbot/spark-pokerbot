"""Local simulations: several event-rules games on this machine, with your deployment in seat 0 and an opponent lineup
you choose, then a scorecard.

    uv run --project harness python sim/simulate.py --games 5                              # base_mihail vs the mixed bots
    uv run --project harness python sim/simulate.py --lineup shover,shover,shover,tag,nit  # stress-test all-in calls
    uv run --project harness python sim/simulate.py --lineup model,model,model,model,model --games 3  # vs base Qwen
    uv run --project harness python sim/simulate.py --lineup house,house,tag,tag,nit       # two house-model seats
    uv run --project harness python sim/simulate.py --report harness/runs/sim               # rescore saved logs

Each game uses the harness's own engine and "event" preset (6 seats, 30 hands, carried stacks, elimination ranking,
scratchpad, talk, 5,000/10,000 token caps), so places come from the same ranking the arena uses. Seat 0 calls
`--model` through the AC2 inference API with your `AC2_API_KEY`; no AC2 job runs. Lineup entries:

    random, maniac, nit, calling_station   the harness's scripted seats (no model)
    shover, tag                            our bots in sim/bots.py (no model)
    house                                  vals-hackathon/ac-vals-model (slow: ~23 s a decision)
    model                                  --opponent-model, base_mihail unless you say otherwise
    <project>/<deployment>                 any other deployment

The lineup rotates one seat per game so no bot always sits right after seat 0. Bots answer instantly, so an all-bot
lineup runs as fast as seat 0 thinks. Logs go to --out (harness/runs/sim, git-ignored): one JSON per game, plus
summary.json. Games already saved in --out (same seed) are reused, so a stopped run picks up where it left off: rerun
the same command, with a different --parallel if you like.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
from poker_harness.completions import (
    GATEWAY_URL,
    cleanup_params,
    opponent_params,
    relay_completion,
    table_rules,
    with_preset,
)
from poker_harness.game import play_session
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bots import BOTS, hand_class, make_bot  # noqa: E402

HOUSE_MODEL = "vals-hackathon/ac-vals-model"
BASE_MODEL = "vals-hackathon/base_mihail"
DEFAULT_LINEUP = "tag,shover,maniac,nit,calling_station"
POLICY = "seat0"


def lineup_for(lineup: list[str], game: int) -> list[str]:
    """Seats 1-5 for game `game`: the lineup rotated by one seat per game."""
    k = game % len(lineup)
    return lineup[k:] + lineup[:k]


def model_for(entry: str, opponent_model: str) -> str | None:
    """The deployment a lineup entry calls, or None for a bot."""
    if entry in BOTS:
        return None
    if entry == "house":
        return HOUSE_MODEL
    if entry == "model":
        return opponent_model
    if "/" in entry:
        return entry
    raise ValueError(f"unknown lineup entry {entry!r}; use a bot ({', '.join(BOTS)}), house, model or <project>/<name>")


def event_params(seed: int, hands: int | None) -> dict[str, Any]:
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    if hands:
        params["total_hands"] = hands
    if table_rules(params).table_talk:  # every seat's tool gains `say`, as on the harness server
        params["submit_fields"] = talk_fields(params["submit_fields"])
    return params


async def play_game(
    client: httpx.AsyncClient, key: str, args: argparse.Namespace, game: int, seed: int, seats: list[str]
) -> dict[str, Any]:
    params = event_params(seed, args.hands)
    rules = table_rules(params)

    def model_seat(name: str, model: str, seat_params: dict[str, Any]) -> ModelSeat:
        return ModelSeat(
            name,
            relay_completion(client, GATEWAY_URL, key, seat_params),
            rules,
            model=model,
            cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(seat_params)),
        )

    players: list[Any] = [model_seat(POLICY, args.model, params | {"model_label": args.model})]
    for i, entry in enumerate(seats, start=1):
        model = model_for(entry, args.opponent_model)
        if model is None:
            players.append(make_bot(entry, f"seat{i}", seed))
        else:
            players.append(model_seat(f"seat{i}", model, opponent_params(params | {"opponent_model": model})))

    log = await play_session(f"sim-{game:03d}-{seed}", seed, players, rules)
    log["sim"] = {
        "game": game,
        "model": args.model,
        "lineup": dict(zip([f"seat{i}" for i in range(1, 6)], seats)),
    }
    return log


def all_in_spots(log: dict[str, Any], player: str = POLICY) -> list[dict[str, Any]]:
    """`player`'s decisions where calling would put it all-in: what it held and whether it called."""
    spots = []
    for hand in log["hands"]:
        hole = hand["starting_state"].get(player, {}).get("hole_cards")
        for a in hand["actions"]:
            if a["player"] != player or not a.get("call_amount"):
                continue
            stack_before = a["stack_after"] + a.get("put_in", 0)
            if a["call_amount"] >= stack_before:
                spots.append({"hand": hand["hand_number"], "hole": hand_class(hole), "action": a["action"]["type"]})
    return spots


def chips_by_opponent(log: dict[str, Any], player: str = POLICY) -> dict[str, float]:
    """`player`'s net chips per hand, credited to the opponents it was playing against: those still in at the end,
    or, when everyone folded to `player`, those who folded after putting chips in. Keyed by lineup entry."""
    lineup = log["sim"]["lineup"]
    out: dict[str, float] = defaultdict(float)
    for hand in log["hands"]:
        if player not in hand["stacks_before"]:
            continue
        net = hand["ending_stacks"][player] - hand["stacks_before"][player]
        folded = {a["player"] for a in hand["actions"] if a["action"]["type"] == "fold"}
        dealt = [p for p in hand["stacks_before"] if p != player and p in lineup]
        rivals = [p for p in dealt if p not in folded]
        if not rivals:
            paid = {a["player"] for a in hand["actions"] if a.get("put_in")}
            rivals = [p for p in dealt if p in paid] or dealt
        for p in rivals:
            out[lineup[p]] += net / len(rivals)
    return dict(out)


def game_row(log: dict[str, Any]) -> dict[str, Any]:
    mine = [a for h in log["hands"] for a in h["actions"] if a["player"] == POLICY]
    failed = Counter(a["model_response"].get("error") or "?" for a in mine if a["model_response"].get("format_error"))
    big_blind = log["session_info"]["big_blind"]
    start = log["session_info"]["starting_stacks"][0]
    return {
        "game": log["sim"]["game"],
        "game_id": log["game_id"],
        "place": log["ranking"]["places"][POLICY],
        "busted_on": log["eliminated_on_hand"].get(POLICY),
        "hands": len(log["hands"]),
        "net_bb": (log["final_stacks"][POLICY] - start) / big_blind,
        "decisions": len(mine),
        "failed": sum(failed.values()),
        "errors": dict(failed),
        "all_in_spots": all_in_spots(log),
        "chips_by_opponent": chips_by_opponent(log),
        "lineup": log["sim"]["lineup"],
        "places": log["ranking"]["places"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    places = [r["place"] for r in rows]
    by_entry: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        for seat, entry in r["lineup"].items():
            by_entry[entry].append(r["places"][seat])
    spots = [s for r in rows for s in r["all_in_spots"]]
    versus: dict[str, float] = defaultdict(float)
    for r in rows:
        for entry, chips in r["chips_by_opponent"].items():
            versus[entry] += chips
    return {
        "games": len(rows),
        "mean_place": statistics.mean(places),
        "place_counts": dict(sorted(Counter(places).items())),
        "busted": sum(r["busted_on"] is not None for r in rows),
        "mean_net_bb": statistics.mean(r["net_bb"] for r in rows),
        "decisions": sum(r["decisions"] for r in rows),
        "failed": sum(r["failed"] for r in rows),
        "opponent_mean_place": {k: statistics.mean(v) for k, v in sorted(by_entry.items())},
        "all_in_called": [s["hole"] for s in spots if s["action"] != "fold"],
        "all_in_folded": [s["hole"] for s in spots if s["action"] == "fold"],
        "net_bb_vs": {k: v / 2 for k, v in sorted(versus.items())},  # big blind 2
    }


def report(rows: list[dict[str, Any]], summary: dict[str, Any], model: str) -> None:
    print(f"\n{'game':>4} {'place':>5} {'bust':>5} {'net BB':>7} {'failed':>7}  lineup (seats 1-5)")
    for r in sorted(rows, key=lambda r: r["game"]):
        bust = f"h{r['busted_on']}" if r["busted_on"] else "-"
        lineup = ",".join(r["lineup"].values())
        print(
            f"{r['game']:>4} {r['place']:>5} {bust:>5} {r['net_bb']:>+7.1f} {r['failed']:>3}/{r['decisions']:<3}  {lineup}"
        )
    s = summary
    print(f"\n{model} over {s['games']} games")
    print(f"  mean place {s['mean_place']:.2f} (1 best, 6 worst; 3.5 is an even table)  places {s['place_counts']}")
    print(f"  busted {s['busted']}/{s['games']}   mean net {s['mean_net_bb']:+.1f} BB")
    print(f"  failed decisions {s['failed']}/{s['decisions']}")
    print(f"  facing all-in: called {len(s['all_in_called'])} {s['all_in_called']}")
    print(f"                 folded {len(s['all_in_folded'])} {s['all_in_folded']}")
    print("  seat 0 net BB vs each: " + ", ".join(f"{k} {v:+.1f}" for k, v in s["net_bb_vs"].items()))
    print("  opponents' mean place: " + ", ".join(f"{k} {v:.2f}" for k, v in s["opponent_mean_place"].items()))


async def run(args: argparse.Namespace, lineup: list[str]) -> list[dict[str, Any]]:
    key = os.environ.get("AC2_API_KEY")
    if not key:
        raise SystemExit("export AC2_API_KEY first")
    rng = random.Random(args.seed)
    seeds = [rng.getrandbits(48) for _ in range(args.games)]
    args.out.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.parallel)
    rows: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:

        async def one(game: int) -> None:
            saved = args.out / f"sim-{game:03d}-{seeds[game]}.json"
            if saved.exists():  # same seed, same deal: a log from an earlier run of this command
                log = json.loads(saved.read_text())
                if log["sim"]["model"] != args.model:
                    raise SystemExit(
                        f"{saved} was played by {log['sim']['model']}, not {args.model}; use another --out"
                    )
                rows.append(game_row(log))
                print(f"game {game}: reusing {saved.name}", flush=True)
                return
            async with gate:
                seats = lineup_for(lineup, game)
                log = await play_game(client, key, args, game, seeds[game], seats)
                (args.out / f"{log['game_id']}.json").write_text(json.dumps(log, indent=1))
                row = game_row(log)
                rows.append(row)
                bust = f"busted hand {row['busted_on']}" if row["busted_on"] else "survived"
                print(f"game {game}: place {row['place']}, {bust}, {row['net_bb']:+.1f} BB", flush=True)

        await asyncio.gather(*(one(g) for g in range(args.games)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=BASE_MODEL, help="seat 0's deployment, <project>/<name>")
    parser.add_argument("--lineup", default=DEFAULT_LINEUP, help="five comma-separated entries for seats 1-5")
    parser.add_argument("--opponent-model", default=BASE_MODEL, help="what `model` in the lineup calls")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--hands", type=int, help="hands per game (default: the event's 30)")
    parser.add_argument("--seed", type=int, default=1, help="seeds the games' deals; same seed, same deals")
    parser.add_argument("--parallel", type=int, default=2, help="games played at once")
    parser.add_argument("--out", type=Path, default=Path("harness/runs/sim"))
    parser.add_argument("--report", type=Path, metavar="DIR", help="only rescore the game logs already in DIR")
    args = parser.parse_args()

    if args.report:
        logs = [json.loads(p.read_text()) for p in sorted(args.report.glob("sim-*.json"))]
        if not logs:
            parser.error(f"no sim-*.json logs in {args.report}")
        rows = [game_row(log) for log in logs]
        report(rows, summarize(rows), logs[0]["sim"]["model"])
        return

    lineup = [e.strip() for e in args.lineup.split(",") if e.strip()]
    if len(lineup) != 5:
        parser.error(f"--lineup needs 5 entries (seats 1-5), got {len(lineup)}")
    for entry in lineup:
        try:
            model_for(entry, args.opponent_model)
        except ValueError as e:
            parser.error(str(e))

    rows = asyncio.run(run(args, lineup))
    summary = summarize(rows)
    (args.out / "summary.json").write_text(
        json.dumps({"model": args.model, "summary": summary, "games": rows}, indent=1)
    )
    report(rows, summary, args.model)
    print(f"\nlogs: {args.out}")


if __name__ == "__main__":
    main()
