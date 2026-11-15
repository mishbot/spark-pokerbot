"""Generate V3 decision samples from real games and gate EVERY sample through the independent
validator. Per the correctness contract: no logically-incorrect sample ships; the run ABORTS if
any sample has a violation (nothing is written).

Usage:
    PYTHONPATH=teacher:harness/src uv run --with pokerkit --with phevaluator --with numpy \
        --with fastapi --with httpx --with pydantic python teacher/v3/generate.py --n 25 --out <path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.engine.table import Decision
from poker_harness.game import play_session

from v3.seat import make_v3_seat
from v3.validate import check

try:
    from v2_policy import ShoverPlayer  # a shover creates deep all-in spots (A1 coverage)
except Exception:  # pragma: no cover
    ShoverPlayer = None

TEACHER_SEAT = 0


import random as _random

BET_SIZES = (0.33, 0.5, 0.75, 1.0, 1.5)   # vary the size so the model learns the arithmetic, not a constant


class LeadBettor:
    """Calls to see flops, then LEADS postflop when first to act, with a VARIED bet size; calls when
    facing a bet. Stays in pots so the teacher faces bets (fixes RC5). The size menu avoids a single-size
    fingerprint that would make the hero's need% a memorizable constant (Fable pitfall)."""

    def __init__(self, name: str, seed: int = 0) -> None:
        self.name = name
        self.rng = _random.Random(f"{name}:{seed}")

    def _choose(self, ctx):
        L = ctx.legal
        if ctx.street == "preflop":
            return Decision("check") if L.can_check else Decision("call")
        if L.can_check:
            if L.raise_type:
                frac = self.rng.choice(BET_SIZES)
                amt = min(L.max_raise_to, max(L.min_raise_to, round(ctx.pot * frac)))
                return Decision(L.raise_type, amt)
            return Decision("check")
        return Decision("call")

    async def decide(self, ctx, history):
        d = self._choose(ctx)
        d.model_response = {"model_name": self.name, "provider": "scripted", "latency_seconds": 0.0,
                            "in_tokens": 0, "out_tokens": 0, "reasoning_tokens": 0, "cost": 0.0,
                            "reasoning": "", "fallback_used": False, "error": None}
        return d


# Field with real aggression: lead-bettors create facing-a-bet spots, a maniac adds raises, a
# station adds passive calls, a shover adds deep all-ins (A1 coverage).
def _villains(seed, rng):
    plan = {1: "bettor", 2: "bettor", 3: "maniac", 4: "calling_station", 5: "shover"}
    seats = {}
    for i, arch in plan.items():
        if arch == "bettor":
            seats[i] = LeadBettor(f"seat{i}", seed)
        elif arch == "shover" and ShoverPlayer is not None:
            seats[i] = ShoverPlayer(f"seat{i}")
        else:
            seats[i] = ScriptedPlayer(f"seat{i}", arch if arch != "shover" else "maniac", seed)
    return seats


async def _play(seed, records):
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    rules = table_rules(params)
    villains = _villains(seed, random.Random(seed + 1))
    players = []
    for i in range(rules.seats):
        if i == TEACHER_SEAT:
            players.append(make_v3_seat(f"seat{i}", i, rules, rules.big_blind, rules.small_blind, seed, records))
        else:
            players.append(villains[i])
    await play_session(f"v3gen-{seed}", seed, players, rules)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="teacher/v3/out/samples.jsonl")
    ap.add_argument("--max-games", type=int, default=200)
    args = ap.parse_args()

    records: list[dict] = []
    seed = 1000
    games = 0
    while len(records) < args.n and games < args.max_games:
        await _play(seed, records)
        seed += 1
        games += 1

    records = records[: args.n]

    # GATE: validate every sample. Abort on any violation.
    all_viol = []
    for idx, r in enumerate(records):
        viol = check(r["truth"], r["trace"], tuple(r["action"]))
        if viol:
            all_viol.append((idx, r["meta"], viol))
    if all_viol:
        print(f"ABORT: {len(all_viol)}/{len(records)} samples have violations (nothing written):\n")
        for idx, meta, viol in all_viol[:20]:
            print(f"  [{idx}] {meta['spot']} {meta['position']} {meta['street']}: {viol}")
        raise SystemExit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    spots = Counter(r["meta"]["spot"] for r in records)
    streets = Counter(r["meta"]["street"] for r in records)
    actions = Counter(r["meta"]["action"] for r in records)
    print(f"OK: {len(records)} samples, all pass the validator (0 violations). Wrote {out}")
    print(f"  games played: {games}")
    print(f"  spots:   {dict(spots)}")
    print(f"  streets: {dict(streets)}")
    print(f"  actions: {dict(actions)}")


if __name__ == "__main__":
    asyncio.run(main())
