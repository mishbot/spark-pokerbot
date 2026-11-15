"""v2 SFT generator — one differentiated 5k set per run (core / coverage / leaks), for 3-way parallel.

Adds over generate_v2.py's verified core:
- cleanup/scratchpad tasks (TeacherSeatV2.cleanup now writes a template + records the turn),
- near-tie drop (postflop |equity-need| < 0.03 spots teach 50/50; dropped),
- profile-specific field mix + stratified pick-weight (Set-2 middle/early coverage, Set-3 leak spots),
- v2 gate (no stale short-stack minimum; v2 is deliberately deep, THOR_ALPHA).

    PYTHONPATH=teacher:harness/src uv run --with numpy --with phevaluator --with pokerkit \
        --with fastapi --with httpx --with pydantic python teacher/generate_v2_sets.py \
        --set core --out teacher/out/v2core --dataset-name pokerbots-sft-v2-core \
        --equity-samples 5000 --target 5000 --time-budget-s 540 --seed-base 900000 --upload
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from pathlib import Path

import pokerlab.bots.field.postflop as postflop

from generate_dataset import TOOLS, compose, distribution, rebalance, _stack_bucket
from generate_v2 import FIELDS_V2, play_game, v2_quality, a1_verify, a2_verify, a3_verify
from seat import records_to_labeled_tasks

# Per-game field distribution over FIELDS_V2 keys, per set.
PROFILE_FIELDS = {
    "core":     {"mixed": 0.40, "shovers": 0.25, "stations": 0.15, "nits": 0.10, "maniacs": 0.10},
    "coverage": {"stations": 0.40, "nits": 0.35, "mixed": 0.25},   # full tables -> more EP/HJ/middle spots
    "leaks":    {"shovers": 0.70, "mixed": 0.20, "maniacs": 0.10},  # facing-all-in + barrel heavy
}


def base_weight(r) -> float:
    m = r["meta"]
    if m["street"] != "preflop":
        return 4.0
    if m["stack_bb"] < 15:
        return 4.0
    if m["trivial_fold"]:
        return 0.5
    if m["players_left"] >= 4:
        return 2.0
    return 1.0


def profile_weight(r, profile: str) -> float:
    m = r["meta"]
    w = base_weight(r)
    if profile == "coverage":
        if m["pos_group"] in ("early", "middle"):
            w *= 3.0            # fix thor's HJ/middle under-play
        if m["street"] == "preflop" and m["action"] != "fold":
            w *= 1.5            # the opening range (incl. value-widen)
    elif profile == "leaks":
        if m.get("deep_shove"):
            w *= 6.0            # A1 facing-shove
        if m.get("wrapper") == "a2_antibluff":
            w *= 4.0            # A2 barrel spots
        if m["bust"]:
            w *= 3.0
    return w


def sample_one_per_hand(decisions, rng, profile):
    groups = {}
    for r in decisions:
        m = r["meta"]
        groups.setdefault((m["seed"], m["hand"], m["seat"]), []).append(r)
    return [rng.choices(g, weights=[profile_weight(x, profile) for x in g])[0] for g in groups.values()]


def build_examples(records, profile):
    decisions = [r for r in records if r["meta"].get("kind") == "decision" and not r["meta"].get("near_tie")]
    sampled = sample_one_per_hand(decisions, random.Random(1), profile)
    dec_ex = rebalance(compose(sampled, random.Random(2)), random.Random(4))
    cleanups = [r for r in records if r["meta"].get("kind") == "cleanup"]
    n_clean = min(len(cleanups), round(0.08 * len(dec_ex) / 0.92))
    clean_ex = random.Random(5).sample(cleanups, n_clean) if n_clean else []
    return dec_ex, clean_ex


V2_GATE = {"postflop_min": 0.35, "fold_max": 0.45, "full_ring_min": 0.25,
           "bust_lo": 0.12, "bust_hi": 0.20, "late_min": 0.10}  # no short_stack_min (v2 is deep)


def v2_gate(dec_ex):
    n = max(len(dec_ex), 1)
    postflop = sum(r["meta"]["street"] != "preflop" for r in dec_ex) / n
    fold = sum(r["meta"]["action"] == "fold" for r in dec_ex) / n
    full_ring = sum(r["meta"]["players_left"] >= 4 for r in dec_ex) / n
    bust = sum(r["meta"]["bust"] for r in dec_ex) / n
    late = sum(r["meta"]["hand"] >= 23 for r in dec_ex) / n
    checks = {
        "postflop": (postflop, postflop >= V2_GATE["postflop_min"], f">= {V2_GATE['postflop_min']}"),
        "fold": (fold, fold <= V2_GATE["fold_max"], f"<= {V2_GATE['fold_max']}"),
        "full_ring(>=4)": (full_ring, full_ring >= V2_GATE["full_ring_min"], f">= {V2_GATE['full_ring_min']}"),
        "bust_deciding": (bust, V2_GATE["bust_lo"] <= bust <= V2_GATE["bust_hi"], f"{V2_GATE['bust_lo']}-{V2_GATE['bust_hi']}"),
        "late(hand>=23)": (late, late >= V2_GATE["late_min"], f">= {V2_GATE['late_min']}"),
    }
    return {"pass": all(ok for _, ok, _ in checks.values()), "checks": checks}


def cleanup_quality(clean_ex):
    fails = 0
    for r in clean_ex:
        tcs = r["messages"][-1].get("tool_calls") or []
        if len(tcs) != 1 or tcs[0]["function"]["name"] != "update_scratchpad":
            fails += 1; continue
        try:
            c = json.loads(tcs[0]["function"]["arguments"])["contents"]
            if not isinstance(c, str) or len(c) >= 5000:
                fails += 1
        except Exception:
            fails += 1
    return {"total": len(clean_ex), "failed": fails}


def tool_choice(dec_ex, clean_ex):
    dec_ok = sum((r["messages"][-1].get("tool_calls") or [{}])[0].get("function", {}).get("name") == "submit_action"
                 for r in dec_ex)
    cln_ok = sum((r["messages"][-1].get("tool_calls") or [{}])[0].get("function", {}).get("name") == "update_scratchpad"
                 for r in clean_ex)
    return dec_ok, len(dec_ex), cln_ok, len(clean_ex)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["core", "coverage", "leaks"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--equity-samples", type=int, default=5000)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--max-games", type=int, default=4000)
    ap.add_argument("--time-budget-s", type=float, default=540.0)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    postflop.EQUITY_SAMPLES = args.equity_samples
    fields = PROFILE_FIELDS[args.set]
    field_rng, villain_rng = random.Random(hash(args.set) & 0xffff), random.Random(43)
    records, field_by_seed, played = [], {}, []
    legality = 0
    start = time.perf_counter()
    for g in range(args.max_games):
        seed = args.seed_base + g
        field = field_rng.choices(list(fields), weights=list(fields.values()))[0]
        field_by_seed[seed] = field
        legality += await play_game(seed, field, records, args.equity_samples, villain_rng)
        played.append(seed)
        dec_ex, _ = build_examples(records, args.set)
        if (g + 1) % 20 == 0:
            print(f"[{args.set}] game {g+1} dec_ex={len(dec_ex)} t={time.perf_counter()-start:.0f}s", flush=True)
        if len(dec_ex) >= args.target or time.perf_counter() - start > args.time_budget_s:
            break

    dec_ex, clean_ex = build_examples(records, args.set)
    val_seeds = set(random.Random(3).sample(played, max(1, round(len(played) * 0.05))))
    train_dec = [r for r in dec_ex if r["meta"]["seed"] not in val_seeds]
    val_dec = [r for r in dec_ex if r["meta"]["seed"] in val_seeds]
    train = train_dec + clean_ex  # cleanup tasks go to train
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val_dec)):
        tasks = records_to_labeled_tasks(rows, TOOLS)
        with (out / f"{name}.jsonl").open("w") as f:
            for t in tasks:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")

    qc = v2_quality(train_dec)
    cq = cleanup_quality(clean_ex)
    gate = v2_gate(train_dec)
    dist = distribution(train_dec, field_by_seed)
    d_ok, d_n, c_ok, c_n = tool_choice(train_dec, clean_ex)
    dec_records = [r for r in records if r["meta"].get("kind") == "decision"]
    a1_rows, _ = a1_verify(dec_records)
    a3_vpip, a3_w = a3_verify(dec_records)
    a2_fired, a2_tr, a2_bets = a2_verify(dec_records)

    print(f"\n===== SET {args.set} =====")
    print(f"games={len(played)} decision_train={len(train_dec)} decision_val={len(val_dec)} cleanup={len(clean_ex)} "
          f"legality={legality} wall={time.perf_counter()-start:.0f}s")
    print(f"decision quality: {qc}  cleanup quality: {cq}")
    print(f"tool choice: decisions {d_ok}/{d_n} submit_action, cleanups {c_ok}/{c_n} update_scratchpad")
    print(f"GATE: {'PASS' if gate['pass'] else 'FAIL'}")
    for k, (v, ok, crit) in gate["checks"].items():
        print(f"  [{'ok' if ok else 'XX'}] {k}: {v*100:.0f}% (need {crit})")
    print("A1 call-rate by cat:", {c: f"{calls}/{tot}" for c, (calls, tot) in sorted(a1_rows.items())})
    print("A3 late VPIP:", {p: f"{v}/{o}" for p, (v, o) in sorted(a3_vpip.items())}, "widened", a3_w)
    print(f"A2 fired {a2_fired}; turn/river bet {a2_bets}/{a2_tr}")
    print("distribution:", {k: v for k, v in dist.items()})

    if args.upload:
        if not gate["pass"]:
            print(f"NOT UPLOADING {args.dataset_name}: gate failed.")
            return
        from ac2.sdk import Client
        client = Client(project="vals-hackathon")
        client.datasets.create(args.dataset_name, required_types=["labeled"],
                               description=f"v2 SFT {args.set}: {len(train)} tasks (A1/A2/A3 + scratchpad).",
                               exist_ok=True)
        tasks = records_to_labeled_tasks(train, TOOLS)
        for i in range(0, len(tasks), 500):
            client.datasets.add_tasks(args.dataset_name, tasks=tasks[i:i + 500])
        print(f"UPLOADED {len(tasks)} tasks to {args.dataset_name}")


if __name__ == "__main__":
    asyncio.run(main())
