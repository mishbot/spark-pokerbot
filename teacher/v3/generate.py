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
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def prompt_hash(messages) -> str:
    """Exact-dedup key: a hash of the PROMPT the model sees (system + user + tool turns), ignoring
    the assistant label and volatile tool-call ids. Identical spots dedup regardless of the action."""
    parts = [m.get("content", "") for m in messages if m["role"] in ("system", "user", "tool")]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()

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
STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
# per-batch street quota (B-balanced 35/25/22/18, scaled to 300)
STREET_TARGET = {"preflop": 105, "flop": 75, "turn": 66, "river": 54}


def select_quota(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """`pool` holds one sample per hand (its latest street). Fill each street to target, SCARCE street
    first (river -> turn -> flop -> preflop) so rare late-street hands aren't lost, then backfill to n."""
    from collections import defaultdict
    by_street = defaultdict(list)
    for r in pool:
        by_street[r["meta"]["street"]].append(r)
    for s in by_street:
        rng.shuffle(by_street[s])
    chosen = []
    for s in ("river", "turn", "flop", "preflop"):
        tgt = STREET_TARGET[s]
        chosen += by_street[s][:tgt]
        by_street[s] = by_street[s][tgt:]
    if len(chosen) < n:   # backfill from leftovers (preflop/flop are the deep buckets)
        leftover = [r for s in ("preflop", "flop", "turn", "river") for r in by_street[s]]
        chosen += leftover[: n - len(chosen)]
    rng.shuffle(chosen)
    return chosen[:n]


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
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="teacher/v3/out/batch.jsonl")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--seen", default="")           # file of prompt hashes to exclude (cross-batch dedup)
    ap.add_argument("--verify-n", type=int, default=30)
    ap.add_argument("--verify-out", default="")      # a random subset for the Opus reviewer
    ap.add_argument("--max-games", type=int, default=1000)
    args = ap.parse_args()

    seen: set[str] = set()
    if args.seen and Path(args.seen).exists():
        seen = set(Path(args.seen).read_text().split())

    # Collect a POOL of one-per-hand samples (each hand's LATEST-street decision), deduped across
    # batches, until the pool is large enough to quota-select from (or we run out of games).
    pool: list[dict] = []          # decision samples (one per hand, latest street)
    cleanup_pool: list[dict] = []  # after-hand update_scratchpad="" samples
    pool_hashes: set[str] = set()
    seed, games, dup = args.seed_start, 0, 0
    pool_target = 5 * args.n
    while len(pool) < pool_target and games < args.max_games:
        recs: list[dict] = []
        await _play(seed, recs)
        seed += 1
        games += 1
        by_hand: dict = {}
        for r in recs:
            h = prompt_hash(r["messages"])
            if r["meta"]["street"] == "cleanup":
                if h in seen or h in pool_hashes:
                    dup += 1
                    continue
                cleanup_pool.append(r)
                pool_hashes.add(h)
            else:
                by_hand.setdefault(r["meta"]["hand"], []).append(r)
        for vlist in by_hand.values():
            r = max(vlist, key=lambda x: STREET_ORDER[x["meta"]["street"]])
            h = prompt_hash(r["messages"])
            if h in seen or h in pool_hashes:
                dup += 1
                continue
            pool.append(r)
            pool_hashes.add(h)

    rng = random.Random(args.seed_start)
    cleanup_n = max(1, round(0.05 * args.n))   # ~5% cleanup rows (Fable)
    batch = select_quota(pool, args.n - cleanup_n, rng)
    rng.shuffle(cleanup_pool)
    batch += cleanup_pool[:cleanup_n]
    rng.shuffle(batch)
    batch_hashes = {prompt_hash(r["messages"]) for r in batch}

    # GATE: validate every sample. Abort on any violation (nothing written).
    all_viol = []
    for idx, r in enumerate(batch):
        viol = check(r["truth"], r["trace"], tuple(r["action"]))
        if viol:
            all_viol.append((idx, r["meta"], viol))
    if all_viol:
        print(f"ABORT: {len(all_viol)}/{len(batch)} samples have violations (nothing written):\n")
        for idx, meta, viol in all_viol[:20]:
            print(f"  [{idx}] {meta['spot']} {meta['position']} {meta['street']}: {viol}")
        raise SystemExit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in batch:
            f.write(json.dumps(r) + "\n")

    if args.seen:
        with Path(args.seen).open("a") as f:
            f.write("\n".join(batch_hashes) + "\n")

    if args.verify_out:
        sample = random.Random(args.seed_start).sample(batch, min(args.verify_n, len(batch)))
        with Path(args.verify_out).open("w") as f:
            for r in sample:
                f.write(json.dumps(r) + "\n")

    spots = Counter(r["meta"]["spot"] for r in batch)
    streets = Counter(r["meta"]["street"] for r in batch)
    actions = Counter(r["meta"]["action"] for r in batch)
    print(f"BATCH OK: {len(batch)} unique samples, 0 violations. seeds {args.seed_start}..{seed-1} "
          f"({games} games, {dup} dups skipped). Wrote {out}"
          + (f" + verify {args.verify_out}" if args.verify_out else ""))
    print(f"  streets: {dict(streets)}")
    print(f"  actions: {dict(actions)}")


if __name__ == "__main__":
    asyncio.run(main())
