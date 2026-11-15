"""Duel: is model B better than model A? A fixed, mirrored head-to-head eval under the event's rules that finishes
inside a wall-clock budget (default 28 minutes).

    uv run --project harness python duel/duel.py --a vals-hackathon/old --b vals-hackathon/new --dry-run
    uv run --project harness python duel/duel.py --a vals-hackathon/old --b vals-hackathon/new
    uv run --project harness python duel/duel.py --report duel/runs/<run>      # rescore saved logs
    uv run --project harness python duel/duel.py --a bot:nit --b bot:calling_station --seeds 4   # no model calls

Design (why each piece is there):

- Event rules, unmodified: the harness's "event" preset (6 seats, 200 chips, blinds 1/2, carried stacks, busting,
  scratchpad with its cleanup turn, talk, 5,000/10,000 token caps, thinking on). The models are told the game is 30
  hands, exactly as in the arena; the runner stops it after `--hands` hands, so play matches the arena's early and
  middle game. A full 30-hand game takes ~50 minutes of sequential model calls, so it can't fit the budget.
- Head-to-head 3 v 3: A sits in seats 0, 2, 4 and B in 1, 3, 5, so every hand pits the two models against each other
  and the table is zero-sum between them: every chip one team wins the other loses.
- Mirrored seeds: every seed is played twice, the second time with A and B swapped. The deck depends only on the
  seed, so each model holds the other's cards in the same seats. Card luck cancels within a pair; what is left is
  mostly the difference in play. The pair, not the game, is the unit of every statistic.
- Fixed seed list (`--base-seed`, never used by any training split): every run of this eval deals the same cards, so a
  new model is measured on the same deals as the last one.
- All-in luck removed: each hand's result also gets an EV label (`poker_harness.ev`, opponents keep their real
  cards, the runout is averaged), so a pre-river all-in counts at its equity. This is the primary metric.
- Hard deadline: every table plays in parallel; at `--deadline-minutes` unfinished tables stop, and each pair is
  scored on the hands both of its games completed.

Outputs, with B measured against A (positive = B better):
- `chips`: B's edge in BB per 100 hands per seat, EV-adjusted (primary) and as dealt, with a 95% interval over pairs.
- `elo`: from finishing places at the cut (the arena ranks by place), every A seat against every B seat.
- `p_b_better_*` (summary.json only): share of bootstrap resamples with B ahead. Not for decisions: it read 0.98 in a
  null test of one model against itself.
- per model: decisions, failed decisions (afk), reply-cap hits, median latency and output tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from poker_harness.completions import GATEWAY_URL, cleanup_params, relay_completion, table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.ev import label_ev
from poker_harness.game import placements, play_session
from poker_harness.prompts import talk_fields
from poker_harness.seat import ModelSeat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpt

HERE = Path(__file__).resolve().parent
BASE_SEED = 6_710_011  # its own base seed; the training/eval splits use 6_610_001 ... 6_690_009
A_SEATS = (0, 2, 4)
B_SEATS = (1, 3, 5)
SEATS = 6
BOOTSTRAP = 4000
MAX_INFRA_FAILED = 0.02


def seeds(count: int, base_seed: int = BASE_SEED, block: int = 0) -> list[int]:
    """The first `count` seeds of block `block`. Block 0 is the standard set; every other block is its own stream,
    so blocks never share deals whatever `--seeds` each run used, and their results pool (`--report r1 r2 ...`)."""
    rng = random.Random(base_seed if block == 0 else f"{base_seed}/block{block}")
    return [rng.getrandbits(48) for _ in range(count)]


def lineup(a: str, b: str, swapped: bool) -> list[str]:
    """The model in each seat: A in 0, 2, 4 and B in 1, 3, 5, or the reverse in the swapped game."""
    first, second = (b, a) if swapped else (a, b)
    return [first if s in A_SEATS else second for s in range(SEATS)]


def event_params(seed: int, timeout: float) -> dict[str, Any]:
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    params["decision_timeout_seconds"] = timeout
    if table_rules(params).table_talk:  # every seat's tool gains `say`, as on the harness server
        params["submit_fields"] = talk_fields(params["submit_fields"])
    return params


class StopGame(Exception):
    """Raised from the per-hand callback once the table has played its hands."""


@dataclass
class Game:
    seed: int
    swapped: bool
    log: dict[str, Any] | None = None
    seconds: float = 0.0
    stopped_early: bool = False
    error: str | None = None

    @property
    def name(self) -> str:
        return f"{self.seed}-{'ba' if self.swapped else 'ab'}"


def make_seat(name: str, entry: str, seed: int, params: dict[str, Any], client: httpx.AsyncClient, key: str, rules):
    if entry.startswith("bot:"):
        return ScriptedPlayer(name, entry[4:], seed)
    seat_params = params | {"model_label": entry}
    return ModelSeat(
        name,
        relay_completion(client, GATEWAY_URL, key, seat_params),
        rules,
        model=entry,
        timeout_seconds=params["decision_timeout_seconds"],
        cleanup_complete=relay_completion(client, GATEWAY_URL, key, cleanup_params(seat_params)),
    )


async def play_game(game: Game, args: argparse.Namespace, client: httpx.AsyncClient, key: str, out: Path) -> Game:
    params = event_params(game.seed, args.decision_timeout)
    rules = table_rules(params)  # total_hands stays 30: the models see the arena's game length
    models = lineup(args.a, args.b, game.swapped)
    players = [make_seat(f"seat{i}", m, game.seed, params, client, key, rules) for i, m in enumerate(models)]
    latest: dict[str, Any] = {}
    hand_end: list[float] = []  # seconds from the table's start to each hand's end, cleanup included
    started = time.perf_counter()

    def on_hand(log: dict[str, Any]) -> None:
        hand_end.append(round(time.perf_counter() - started, 1))
        latest["log"] = log
        (out / f"{game.name}.partial.json").write_text(json.dumps(log))
        print(f"  {game.name}: hand {len(log['hands'])}", flush=True)
        if len(log["hands"]) >= args.hands:
            raise StopGame

    try:
        # both twins share the id: it seeds the EV runouts, so identical play gets identical EV
        game.log = await play_session(f"duel-{game.seed}", game.seed, players, rules, on_hand=on_hand)
    except StopGame:
        game.log = latest["log"]
    except asyncio.CancelledError:
        game.log, game.stopped_early = latest.get("log"), True
    except Exception as exc:  # noqa: BLE001 - one broken table must not sink the run
        game.log, game.error = latest.get("log"), f"{type(exc).__name__}: {str(exc)[:300]}"
    game.seconds = time.perf_counter() - started
    if game.log is not None:
        game.log["duel"] = {"a": args.a, "b": args.b, "swapped": game.swapped, "models": models,
                            "stopped_early": game.stopped_early, "error": game.error, "seconds": round(game.seconds),
                            "hand_end_seconds": hand_end}
        label_ev(game.log, {f"seat{i}" for i in range(SEATS)}, runouts=64)
        (out / f"{game.name}.json").write_text(json.dumps(game.log, indent=1))
    (out / f"{game.name}.partial.json").unlink(missing_ok=True)
    return game


INFRA_ERRORS = ("RelayError", "TimeoutError", "ReadTimeout", "ConnectError", "RemoteProtocolError", "HTTPStatusError")


async def preflight(client: httpx.AsyncClient, key: str, models: set[str]) -> None:
    """One tiny request per deployment before any table starts, so a stopped deployment fails the run in seconds
    instead of producing 30 minutes of forced folds."""
    for model in sorted(m for m in models if not m.startswith("bot:")):
        body = {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = await client.post(f"{GATEWAY_URL}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
        if not r.is_success:
            raise SystemExit(f"preflight: {model} answered HTTP {r.status_code}: {r.text[:200]}")
        print(f"preflight: {model} OK", flush=True)


async def run(args: argparse.Namespace, out: Path) -> list[Game]:
    key = os.environ.get("AC2_API_KEY", "")
    if not key and not (args.a.startswith("bot:") and args.b.startswith("bot:")):
        raise SystemExit("export AC2_API_KEY first")
    games = [Game(seed, swapped) for seed in seeds(args.seeds, args.base_seed, args.block) for swapped in (False, True)]
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.decision_timeout + 30, connect=30.0), limits=limits) as client:
        await preflight(client, key, {args.a, args.b})
        tasks = [asyncio.create_task(play_game(g, args, client, key, out)) for g in games]
        _, pending = await asyncio.wait(tasks, timeout=args.deadline_minutes * 60)
        if pending:
            print(f"deadline: stopping {len(pending)} unfinished tables", flush=True)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    return games


# ---------------------------------------------------------------------------------------------------------------
# Scoring


def cut(log: dict[str, Any], hands: int) -> dict[str, Any]:
    """The game as it stood after `hands` hands: stacks, busts and places."""
    names = log["session_info"]["player_names"]
    start = log["session_info"]["starting_stacks"][0]
    played = log["hands"][:hands]
    stacks = dict.fromkeys(names, start)
    ev_net = dict.fromkeys(names, 0.0)
    for hand in played:
        for name in names:
            stacks[name] = hand["ending_stacks"].get(name, stacks[name])
            ev = (hand.get("ev") or {}).get(name)
            if ev:  # EV minus as-dealt, in chips: what all-in luck added or took
                ev_net[name] += (ev["ev_bb"] - ev["realized_bb"]) * log["session_info"]["big_blind"]
    busted = {n: h for n, h in log.get("eliminated_on_hand", {}).items() if h <= hands}
    view = {"session_info": log["session_info"], "final_stacks": stacks, "eliminated_on_hand": busted, "hands": played}
    return {
        "net": {n: stacks[n] - start for n in names},
        "ev_net": {n: stacks[n] - start + ev_net[n] for n in names},
        "places": placements(view, log.get("ranking", {}).get("mode", "elimination")),
    }


def team(log: dict[str, Any], side: str) -> list[str]:
    """Seat names of side "A" or "B" in this game; by seat, so a model against itself (a null test) still works."""
    first = B_SEATS if log["duel"]["swapped"] else A_SEATS
    seats = first if side == "A" else tuple(s for s in range(SEATS) if s not in first)
    return [f"seat{i}" for i in seats]


def pair_scores(logs: list[dict[str, Any]], at: int | None = None) -> list[dict[str, Any]]:
    """One row per seed whose two games both exist: B's edge on the hands both games completed (at most `at`)."""
    by_seed: dict[tuple[str, int], dict[bool, dict[str, Any]]] = {}
    for log in logs:
        key = (log.get("_run", ""), log["session_info"]["seed"])
        by_seed.setdefault(key, {})[log["duel"]["swapped"]] = log
    rows = []
    for (_, seed), twins in sorted(by_seed.items()):
        if len(twins) != 2:
            continue
        hands = min(len(t["hands"]) for t in twins.values())
        if at is not None:
            if hands < at:
                continue  # scoring at a fixed length: a pair that didn't get there is left out
            hands = at
        if hands == 0:
            continue
        net = ev = wins = 0.0
        single = []
        for log in twins.values():
            state = cut(log, hands)
            bs, as_ = team(log, "B"), team(log, "A")
            net += sum(state["net"][s] for s in bs)
            # The harness samples each seat's all-in runouts on its own, so seats' EV labels don't sum to zero;
            # half of B minus A is zero-sum by construction and exactly 0 when both sides play the same.
            team_ev = (sum(state["ev_net"][s] for s in bs) - sum(state["ev_net"][s] for s in as_)) / 2
            ev += team_ev
            single.append(team_ev)
            places = state["places"]
            wins += sum(1.0 if places[x] < places[y] else 0.5 if places[x] == places[y] else 0.0 for x in bs for y in as_)
        bb = twins[False]["session_info"]["big_blind"]
        per = 2 * 3 * hands / 100  # seat-hands per game pair, in hundreds
        rows.append({
            "seed": seed, "hands": hands,
            "b_bb100": net / bb / per, "b_ev_bb100": ev / bb / per,
            "b_place_wins": wins / 18,  # 2 games x 3 B seats x 3 A seats
            "single_game_ev_bb100": [x / bb / (per / 2) for x in single],
        })
    return rows


def mirror_gain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How much the mirroring helps: the spread of one game's score against the spread of a mirrored pair's.
    Without mirroring, a pair's two games would be independent and its spread would be single / sqrt(2)."""
    single = [x for r in rows for x in r["single_game_ev_bb100"]]
    if len(rows) < 3:
        return {}
    sd_single, sd_pair = statistics.stdev(single), statistics.stdev(r["b_ev_bb100"] for r in rows)
    unpaired = sd_single / math.sqrt(2)
    return {
        "sd_single_game": round(sd_single, 1), "sd_pair": round(sd_pair, 1), "sd_pair_if_unmirrored": round(unpaired, 1),
        "games_saved_factor": round((unpaired / sd_pair) ** 2, 1) if sd_pair else float("inf"),
    }


def t975(df: int) -> float:
    """Student t 97.5% quantile (Cornish-Fisher expansion; within 0.01 of the exact value for df >= 5)."""
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% Student-t half-width over pairs. (A percentile bootstrap ran narrow at 64 pairs: in a null test
    of one model against itself it excluded zero, while this interval did not.)"""
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, float("nan")
    return m, t975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))


def elo(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return 400 * math.log10(p / (1 - p))


def bootstrap(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    ev_means, elos = [], []
    for _ in range(BOOTSTRAP):
        sample = [rng.choice(rows) for _ in rows]
        ev_means.append(statistics.fmean(r["b_ev_bb100"] for r in sample))
        elos.append(elo(statistics.fmean(r["b_place_wins"] for r in sample)))

    def p_ahead(values: list[float]) -> float:  # ties (a null test) count half
        return round(sum(1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in values) / BOOTSTRAP, 3)

    ev, ev_half = mean_ci([r["b_ev_bb100"] for r in rows])
    rate, rate_half = mean_ci([r["b_place_wins"] for r in rows])
    if math.isnan(ev_half):
        ev_half = rate_half = 0.0
    return {
        "p_b_better_chips": p_ahead(ev_means),
        "p_b_better_places": p_ahead(elos),
        # the verdict's intervals: Student t over pairs; places on the win-rate scale, then shown as Elo
        "elo_ci95": [round(elo(rate - rate_half)), round(elo(rate + rate_half))],
        "ev_bb100_ci95": [round(ev - ev_half, 1), round(ev + ev_half, 1)],
    }


def model_stats(logs: list[dict[str, Any]], side: str) -> dict[str, Any]:
    decisions = [a for log in logs for h in log["hands"] for a in h["actions"] if a["player"] in team(log, side)]
    responses = [a.get("model_response") or {} for a in decisions]
    lat = [r["latency_seconds"] for r in responses if r.get("latency_seconds") is not None]
    tok = [r["out_tokens"] for r in responses if r.get("out_tokens") is not None]
    errors = Counter(str(r.get("error"))[:60] for r in responses if r.get("format_error"))
    infra = sum(1 for r in responses if r.get("format_error") and str(r.get("error")).startswith(INFRA_ERRORS))
    actions = Counter(a["action"]["type"] for a in decisions)
    cleanups = Counter(
        rec.get("cleanup", {}).get("result")
        for log in logs for seat in team(log, side) for rec in (log.get("private", {}).get(seat, {}).get("hands") or {}).values()
    )
    return {
        "decisions": len(decisions),
        "failed": sum(errors.values()),
        "failed_rate": round(sum(errors.values()) / max(len(decisions), 1), 4),
        "infra_failed": infra,  # the deployment or network failed, not the model: these make a run invalid
        "infra_failed_rate": round(infra / max(len(decisions), 1), 4),
        "errors": dict(errors.most_common(4)),
        "latency_s_median": round(statistics.median(lat), 1) if lat else None,
        "latency_s_p90": round(sorted(lat)[int(0.9 * len(lat))], 1) if lat else None,
        "out_tokens_median": round(statistics.median(tok)) if tok else None,
        "actions": dict(actions),
        "scratchpad": {k: v for k, v in cleanups.items() if k},
    }


def load_logs(*runs: Path) -> list[dict[str, Any]]:
    """The finished game logs of one or more runs (`<seed>-ab.json`, `<seed>-ba.json`)."""
    logs = []
    for out in runs:
        for p in sorted(out.glob("*.json")):
            if p.name[0].isdigit() and "partial" not in p.name:
                log = json.loads(p.read_text())
                log["_run"] = str(out)  # pairs are matched within a run, never across runs
                logs.append(log)
    return logs


def reused_seeds(logs: list[dict[str, Any]]) -> int:
    """Seeds played in more than one of the pooled runs. Such pairs share their cards, so they are not independent
    and the pooled interval is a little too narrow; pool runs from different `--block`s instead."""
    runs_of: dict[int, set[str]] = {}
    for log in logs:
        runs_of.setdefault(log["session_info"]["seed"], set()).add(log.get("_run", ""))
    return sum(len(r) > 1 for r in runs_of.values())


def report(*runs: Path, at: int | None = None) -> dict[str, Any]:
    """Score one run, or pool several runs of the same A and B (each pair counts once, by seed)."""
    logs = load_logs(*runs)
    if not logs:
        raise SystemExit(f"no game logs in {', '.join(map(str, runs))}")
    a, b = logs[0]["duel"]["a"], logs[0]["duel"]["b"]
    if any((log["duel"]["a"], log["duel"]["b"]) != (a, b) for log in logs):
        raise SystemExit("these runs don't share the same A and B; pool only runs of one match-up")
    rows = pair_scores(logs, at)
    if not rows:
        raise SystemExit("no seed has both of its games: nothing to compare")
    ev, ev_ci = mean_ci([r["b_ev_bb100"] for r in rows])
    net, net_ci = mean_ci([r["b_bb100"] for r in rows])
    p = statistics.fmean(r["b_place_wins"] for r in rows)
    summary = {
        "a": a, "b": b,
        "pairs": len(rows), "games": len(logs), "runs": len(runs), "reused_seeds": reused_seeds(logs),
        "hands_per_game": {"min": min(r["hands"] for r in rows), "median": statistics.median(r["hands"] for r in rows)},
        "b_minus_a": {
            "ev_bb100": round(ev, 1), "ev_bb100_ci95": round(ev_ci, 1),
            "bb100_as_dealt": round(net, 1), "bb100_as_dealt_ci95": round(net_ci, 1),
            "place_win_rate": round(p, 3), "elo": round(elo(p)),
        },
        **bootstrap(rows),
        "models": {side: {"name": name, **model_stats(logs, side),
                          "preflop_vs_book": mpt.grade(logs, lambda log, s=side: team(log, s))}
                   for side, name in (("A", a), ("B", b))},
        "mirror": mirror_gain(rows),
        "per_pair": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "single_game_ev_bb100"} for r in rows],
    }
    def call(ci: list[float]) -> str:
        return "B better" if ci[0] > 0 else "B worse" if ci[1] < 0 else "no clear difference"

    # Chips are the sharper signal; places are what the arena rates. They can disagree (a high-variance style wins
    # chips but busts more), so both are reported and the verdict is only clean when they agree.
    chips, places = call(summary["ev_bb100_ci95"]), call(summary["elo_ci95"])
    summary["verdict"] = {"chips": chips, "places": places,
                          "overall": chips if chips == places else f"mixed: chips say {chips}, places say {places}"}
    infra = max(summary["models"][t]["infra_failed_rate"] for t in ("A", "B"))
    if infra > MAX_INFRA_FAILED:  # forced folds from a broken deployment measure the deployment, not the model
        summary["verdict"]["overall"] = f"INVALID: {100 * infra:.1f}% of one side's decisions failed on the deployment or network"
    if len(runs) == 1:
        (runs[0] / ("summary.json" if at is None else f"summary-at-{at}.json")).write_text(json.dumps(summary, indent=1))
    return summary


def print_summary(s: dict[str, Any]) -> None:
    d = s["b_minus_a"]
    print(f"\nB = {s['b']}  vs  A = {s['a']}")
    print(f"{s['pairs']} mirrored pairs ({s['games']} games), hands per game: median {s['hands_per_game']['median']}, min {s['hands_per_game']['min']}")
    if s.get("reused_seeds"):
        print(f"  warning: {s['reused_seeds']} seeds appear in more than one pooled run; their pairs share cards, so the "
              "interval is a little too narrow (pool different --block runs instead)")
    print(f"  B edge, EV-adjusted : {d['ev_bb100']:+.1f} bb/100 per seat  (95% CI {s['ev_bb100_ci95'][0]:+.1f} .. {s['ev_bb100_ci95'][1]:+.1f})")
    print(f"  B edge, as dealt    : {d['bb100_as_dealt']:+.1f} bb/100  (+/- {d['bb100_as_dealt_ci95']:.1f})")
    print(f"  B place win rate    : {d['place_win_rate']:.3f}  ->  Elo {d['elo']:+d}  (95% CI {s['elo_ci95'][0]:+d} .. {s['elo_ci95'][1]:+d})")
    print(f"  verdict             : {s['verdict']['overall']}")
    if s.get("mirror"):
        m = s["mirror"]
        print(f"  mirroring           : pair sd {m['sd_pair']} vs {m['sd_pair_if_unmirrored']} unmirrored "
              f"(worth {m['games_saved_factor']}x as many unmirrored games)")
    for tag in ("A", "B"):
        m = s["models"][tag]
        print(f"  {tag} {m['name']}: {m['decisions']} decisions, {m['failed']} failed ({100 * m['failed_rate']:.1f}%), "
              f"latency med {m['latency_s_median']} s / p90 {m['latency_s_p90']} s, out tokens med {m['out_tokens_median']}")
        if m["errors"]:
            print(f"      errors: {m['errors']}")
    print("  preflop vs Modern Poker Theory (agree %, graded decisions; clear hands only):")
    seen = s["models"]["A"]["preflop_vs_book"].keys() | s["models"]["B"]["preflop_vs_book"].keys()
    for key in [k for k in mpt.SPOT_ORDER + ["all"] if k in seen]:
        cells = []
        for tag in ("A", "B"):
            v = s["models"][tag]["preflop_vs_book"].get(key)
            cells.append(f"{100 * v['agree_rate']:5.1f}% of {v['graded']:4d}" if v and v.get("agree_rate") is not None
                         else "          n/a")
        print(f"    {key:9s}  A {cells[0]}   B {cells[1]}")
    print("  (per-spot loose/tight/limp counts and open sizes: summary.json, or duel/mpt.py <run>)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", help="baseline: <project>/<deployment>, or bot:<archetype> for a test")
    parser.add_argument("--b", help="challenger: <project>/<deployment>, or bot:<archetype>")
    parser.add_argument("--seeds", type=int, default=64, help="mirrored seeds; tables in parallel = 2 x seeds "
                        "(64 = 128 tables: the most that fit 10 hands in 28 minutes; see README)")
    parser.add_argument("--hands", type=int, default=10, help="hands per game before the cut (the models are told 30)")
    parser.add_argument("--block", type=int, default=0, help="seed block: 0 = the standard deals; 1, 2, ... = fresh "
                        "deals for extra runs to pool with --report")
    parser.add_argument("--deadline-minutes", type=float, default=28.0)
    parser.add_argument("--decision-timeout", type=float, default=300.0, help="a slower reply checks or folds")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED, help="keep the default so runs stay comparable")
    parser.add_argument("--out", type=Path, help="default duel/runs/<utc time>")
    parser.add_argument("--report", type=Path, nargs="+", help="rescore saved run directories (several pool) and exit")
    parser.add_argument("--at", type=int, help="with --report: score every pair at this many hands")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.report:
        print_summary(report(*args.report, at=args.at))
        return
    if not (args.a and args.b):
        parser.error("--a and --b are required")
    out = args.out or HERE / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    plan = {
        "a": args.a, "b": args.b, "block": args.block, "seeds": seeds(args.seeds, args.base_seed, args.block),
        "tables": 2 * args.seeds,
        "hands": args.hands, "deadline_minutes": args.deadline_minutes, "out": str(out),
        "seats": {"A": list(A_SEATS), "B": list(B_SEATS), "swapped game": "reversed"},
    }
    print(json.dumps(plan, indent=1))
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    started = time.perf_counter()
    games = asyncio.run(run(args, out))
    print(f"\nwall time {((time.perf_counter() - started) / 60):.1f} min; "
          f"{sum(g.stopped_early for g in games)} tables stopped at the deadline, {sum(bool(g.error) for g in games)} errored")
    for g in games:
        if g.error:
            print(f"  {g.name}: {g.error}", file=sys.stderr)
    print_summary(report(out))


if __name__ == "__main__":
    main()
