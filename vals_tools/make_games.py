"""Make the full-game eval datasets: 30-hand event games, one AC2 task per game, exactly as the arena plays them.

Writes <prefix>-games (60 games, the policy's starting seat cycling through all six) and <prefix>-games-smoke (2
games). Unlike the single-hand splits, stacks carry over between hands, seats bust, and the scratchpad written after
each hand is read in the next, so these measure what the arena ranks: survival and final place.

    uv run python -m vals_tools.make_games --dry-run
    uv run python -m vals_tools.make_games
    uv run python -m vals_tools.eval --dataset pokerbots-games-smoke --deployment vals-hackathon/<deployment>
    uv run python -m vals_tools.eval --dataset pokerbots-games --deployment vals-hackathon/<deployment>
    ac2 traces download <eval id> -o traces/ && uv run python analysis/game_stats.py traces/traces-<eval id>.jsonl

The sample grader sums the seat's EV over the game and clips it to +-5 BB, and any format error zeroes the game, so
`vals_tools.score` says little here; `analysis/game_stats.py` reads bust rate and place from the traces instead.
"""

from __future__ import annotations

import argparse

from ac2.sdk import Client

from vals_tools.tasks import SEATS, Opponents, game_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="vals-hackathon")
    parser.add_argument("--prefix", default="pokerbots")
    parser.add_argument("--opponent-model", default="vals-hackathon/ac-vals-model")
    parser.add_argument("--opponent-api", choices=["responses", "chat"], default="chat")
    parser.add_argument("--opponent-effort", default="none", help="reasoning effort for the opponents, e.g. medium")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    effort = None if args.opponent_effort == "none" else args.opponent_effort
    built = game_datasets(args.prefix, Opponents(args.opponent_model, args.opponent_api, effort), SEATS)
    if args.dry_run:
        for name, tasks in built.items():
            seats = [t.env_params["policy_seats"][0] for t in tasks]
            print(name, len(tasks), "games; policy seats", {s: seats.count(s) for s in sorted(set(seats))})
            print("  first task:", tasks[0].env_params)
        return
    client = Client(project=args.project)
    for name, tasks in built.items():
        client.datasets.create(name, description=f"{len(tasks)} 30-hand {SEATS}-max event games.", exist_ok=True)
        client.datasets.add_tasks(name, tasks=tasks)
        print(f"{name}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
