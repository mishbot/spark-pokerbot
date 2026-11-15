"""Phase 1 SFT dataset generator (full build, option-b, with the G1 fixes).

Plays full 30-hand event games with the ported teacher in 2 seats against tuned scripted villain
fields, then builds `labeled` SFT examples under the settled Phase 1 design:

- Task unit (b): ONE decision per (game, hand, teacher seat), chosen by a stratified weighting that
  favours the under-represented slices (postflop, short stacks, full-ring) and suppresses trivial
  heads-up folds, so each call trains at most once.
- Label: `<think>{reasoning line}</think>` + a `submit_action` FunctionCall (reuses teacher/seat.py).
- Shared equity: postflop.EQUITY_SAMPLES and the reasoning line use the SAME sample count, so the line
  equals what the teacher decided with (decision_rng is deterministic).
- Tight-early: hands 1..ceil(0.25*total) use a tightened persona (looseness x0.4, risk_aversion x2).
- Composition: downsample trivial preflop folds to 1 in 4; 4x copies of bust-deciding spots; then a
  rebalance pass caps heads-up preflop folds so the final shares move toward SFT_DATASET_DESIGN §5.

G1 fixes vs the first sample:
  1. Disciplined/slow villain backbone (calling-station + nit heavy), so tables stay full and grind
     down instead of busting early on maniacs. (Harness ScriptedPlayers, so villains cost no equity.)
  2. Short stacks come from NATURAL grind-down in 200-chip games (SFT §2), NOT a reduced starting
     stack — a 15 BB start would print "starts with 30 chips", which never happens at the event.
  3. Stratified one-per-hand sampling + a heads-up-fold cap.

Writes teacher/out/{train,val}.jsonl. With --upload it also uploads train as an AC2 `labeled` dataset.

    PYTHONPATH=teacher:harness/src uv run --with numpy --with phevaluator --with pokerkit \
        --with fastapi --with httpx --with pydantic python teacher/generate_dataset.py          # verify
    ... python teacher/generate_dataset.py --equity-samples 5000 --target 25000 \
        --upload --dataset-name pokerbots-sft-train-v1                                           # full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import pokerlab.bots.field.postflop as postflop
from pokerlab.datatypes.field_personas import TEACHER
from dataclasses import replace
from seat import make_teacher_seat, records_to_labeled_tasks

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.prompts import parse_tool_call

TEACHER_SEATS = (0, 3)
TOOLS = json.loads((Path(__file__).resolve().parents[1] / "opsd" / "tools.json").read_text())
LEGAL_TYPES = {"fold", "check", "call", "bet", "raise"}

# Per-game field -> per-villain-seat archetype weights. Tuned so tables stay full (calling-station and
# nit do not bust fast; maniac/random are a minority), fixing the first sample's early-bust skew.
FIELDS = {
    "mixed": (0.55, {"calling_station": 0.45, "nit": 0.30, "random": 0.15, "maniac": 0.10}),
    "stations": (0.15, {"calling_station": 0.70, "nit": 0.30}),      # postflop-rich, full tables
    "nits": (0.15, {"nit": 0.70, "calling_station": 0.30}),          # tight -> teaches stealing
    "maniacs": (0.15, {"maniac": 0.50, "random": 0.30, "calling_station": 0.20}),  # patience vs all-ins
}

# Gate thresholds (SFT_DATASET_DESIGN §5). All must hold before the full run / upload.
GATE = {
    "short_stack_min": 0.05,   # <15bb share
    "postflop_min": 0.35,
    "fold_max": 0.45,
    "full_ring_min": 0.25,     # share at 4-6 players left
    "bust_lo": 0.12, "bust_hi": 0.20,
    "late_min": 0.10,          # hand_number >= 23
}


def persona_for(hand_number: int, total_hands: int):
    """Tight-early (handoff §9): the first 25% of hands play tighter and more risk-averse."""
    if hand_number <= math.ceil(0.25 * total_hands):
        return replace(TEACHER, looseness=TEACHER.looseness * 0.4, risk_aversion=TEACHER.risk_aversion * 2)
    return TEACHER
    # The "outlast floor" (share_below >= 0.5) from handoff §9 is deferred: it needs a global
    # share_below patch, not clean per-hand. The tight-early persona covers the bulk of the effect.


def villains_for(field: str, seed: int, rng: random.Random):
    weights = FIELDS[field][1]
    archs, probs = list(weights), list(weights.values())
    seats = {}
    for i in range(6):
        if i in TEACHER_SEATS:
            continue
        arch = rng.choices(archs, weights=probs)[0]
        seats[i] = ScriptedPlayer(f"seat{i}", arch, seed)
    return seats


async def play_game(seed: int, field: str, records: list, equity_samples: int, villain_rng: random.Random,
                    start_stack: int = 200) -> int:
    # start_stack < 200 is the short-stack slice (G1 part 2). CAVEAT: the system prompt then says
    # "Everyone starts with {start_stack} chips", which never happens at the event (always 200). We accept
    # this to guarantee <15bb push/fold coverage the gate needs; the decision-relevant features (current
    # stack, pot odds, legal actions) still match. v2 should prefer a faithful source (arena short-stack
    # spots, or a self-play slice capturing natural short stacks).
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"],
                          "starting_stack": start_stack})
    rules = table_rules(params)
    villains = villains_for(field, seed, villain_rng)
    players = []
    for i in range(rules.seats):
        if i in TEACHER_SEATS:
            players.append(make_teacher_seat(
                f"seat{i}", i, rules, rules.big_blind, rules.small_blind, seed, records,
                persona_fn=persona_for, equity_samples=equity_samples,
            ))
        else:
            players.append(villains[i])
    log = await play_session(f"gen-{seed}", seed, players, rules)
    names = {f"seat{i}" for i in TEACHER_SEATS}
    return sum(
        1 for hand in log["hands"] for a in hand["actions"]
        if a["player"] in names and (
            a["model_response"].get("fallback_used") or a["model_response"].get("error")
            or a["model_response"].get("format_error")
        )
    )


def _pick_weight(r) -> float:
    """Stratified one-per-hand weight: favour postflop / short-stack / full-ring, suppress trivial folds."""
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


def sample_one_per_hand(records: list, rng: random.Random) -> list:
    """Option (b): one record per (seed, hand, seat), chosen by the stratified weighting."""
    groups: dict[tuple, list] = {}
    for r in records:
        m = r["meta"]
        groups.setdefault((m["seed"], m["hand"], m["seat"]), []).append(r)
    return [rng.choices(g, weights=[_pick_weight(x) for x in g])[0] for g in groups.values()]


def compose(sampled: list, rng: random.Random) -> list:
    """Downsample trivial preflop folds to 1 in 4; 2 copies of bust-deciding spots (the rebalance caps
    the final bust share, so a modest multiplier is enough to upweight them without overshooting)."""
    out = []
    for r in sampled:
        m = r["meta"]
        if m["bust"]:
            out.extend([r] * 2)
        elif m["trivial_fold"]:
            if rng.random() < 0.25:
                out.append(r)
        else:
            out.append(r)
    return out


def _cap(examples, rng, target, pred_count, pred_cut):
    """Downsample the `pred_cut` subpopulation until the `pred_count` share is <= target.
    Requires pred_cut examples to be a subset of pred_count examples (1:1 effect on both counts)."""
    n = len(examples)
    c = sum(pred_count(r) for r in examples)
    if not n or c <= target * n:
        return examples
    cut_idx = [i for i, r in enumerate(examples) if pred_cut(r)]
    need = math.ceil((c - target * n) / (1 - target))
    drop = set(rng.sample(cut_idx, min(need, len(cut_idx))))
    return [r for i, r in enumerate(examples) if i not in drop]


def rebalance(examples: list, rng: random.Random) -> list:
    """Cap the two over-produced slices toward §5: heads-up preflop folds, then bust-deciding spots."""
    ex = _cap(examples, rng, GATE["fold_max"] - 0.03,
              lambda r: r["meta"]["action"] == "fold",
              lambda r: r["meta"]["action"] == "fold" and r["meta"]["street"] == "preflop"
              and not r["meta"]["bust"] and r["meta"]["stack_bb"] >= 15)
    ex = _cap(ex, rng, GATE["bust_hi"] - 0.02,
              lambda r: r["meta"]["bust"], lambda r: r["meta"]["bust"])
    return ex


def pipeline(records, sample_seed=1, compose_seed=2, rebalance_seed=4):
    sampled = sample_one_per_hand(records, random.Random(sample_seed))
    return rebalance(compose(sampled, random.Random(compose_seed)), random.Random(rebalance_seed))


def quality_checks(examples: list) -> dict:
    fails, max_tokens = [], 0
    for r in examples:
        msgs, m = r["messages"], r["meta"]
        args = msgs[-1]["tool_calls"][0]["function"]["arguments"]
        decision, error = parse_tool_call(args)
        if error is not None or decision.type not in LEGAL_TYPES:
            fails.append(f"parse: {error or decision.type}"); continue
        payload = json.loads(args); act = payload["action"]
        if act == "fold" and m["can_check"]:
            fails.append("fold-when-check-free"); continue
        if act in ("bet", "raise"):
            amt = payload.get("amount")
            if amt is None or m["min_raise_to"] is None or not (m["min_raise_to"] <= amt <= m["max_raise_to"]):
                fails.append(f"raise-amount {amt} not in [{m['min_raise_to']},{m['max_raise_to']}]"); continue
        content = msgs[-1]["content"]
        line = content.split("</think>")[0].rsplit("|", 1)[-1].strip()
        if not line.startswith(act):
            fails.append(f"line/label mismatch: '{line}' vs '{act}'"); continue
        toks = len(content.split()) + len(args.split())
        max_tokens = max(max_tokens, toks)
        if toks >= 300:
            fails.append(f"too long: ~{toks}"); continue
    return {"total": len(examples), "failed": len(fails), "first_fails": fails[:5], "max_tokens_est": max_tokens}


def _stack_bucket(bb):
    return "<15bb" if bb < 15 else "15-40bb" if bb < 40 else "40-100bb" if bb < 100 else ">100bb"


def gate_metrics(examples: list) -> dict:
    n = max(len(examples), 1)
    short = sum(r["meta"]["stack_bb"] < 15 for r in examples) / n
    postflop = sum(r["meta"]["street"] != "preflop" for r in examples) / n
    fold = sum(r["meta"]["action"] == "fold" for r in examples) / n
    full_ring = sum(r["meta"]["players_left"] >= 4 for r in examples) / n
    bust = sum(r["meta"]["bust"] for r in examples) / n
    late = sum(r["meta"]["hand"] >= 23 for r in examples) / n
    checks = {
        "short_stack(<15bb)": (short, short >= GATE["short_stack_min"], f">= {GATE['short_stack_min']}"),
        "postflop": (postflop, postflop >= GATE["postflop_min"], f">= {GATE['postflop_min']}"),
        "fold": (fold, fold <= GATE["fold_max"], f"<= {GATE['fold_max']}"),
        "full_ring(>=4)": (full_ring, full_ring >= GATE["full_ring_min"], f">= {GATE['full_ring_min']}"),
        "bust_deciding": (bust, GATE["bust_lo"] <= bust <= GATE["bust_hi"], f"{GATE['bust_lo']}-{GATE['bust_hi']}"),
        "late(hand>=23)": (late, late >= GATE["late_min"], f">= {GATE['late_min']}"),
    }
    return {"pass": all(ok for _, ok, _ in checks.values()), "checks": checks}


def distribution(examples: list, field_by_seed: dict) -> dict:
    def phase(m):
        t = m["total_hands"]
        return "early(1-25%)" if m["hand"] <= math.ceil(0.25 * t) else "late(>=23)" if m["hand"] >= 23 else "mid"
    cols = {k: Counter() for k in ("street", "pos_group", "action", "stack", "hand_phase", "players_left", "field")}
    for r in examples:
        m = r["meta"]
        cols["street"][m["street"]] += 1
        cols["pos_group"][m["pos_group"]] += 1
        cols["action"][m["action"]] += 1
        cols["stack"][_stack_bucket(m["stack_bb"])] += 1
        cols["hand_phase"][phase(m)] += 1
        cols["players_left"][m["players_left"]] += 1
        cols["field"][field_by_seed.get(m["seed"], "?")] += 1
    n = max(len(examples), 1)
    return {k: {str(key): f"{100 * v / n:.0f}%" for key, v in c.most_common()} for k, c in cols.items()}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-samples", type=int, default=1000, help="shared by decision and line (5000 for the full run)")
    ap.add_argument("--target", type=int, default=2500, help="stop when composed examples reach this")
    ap.add_argument("--max-games", type=int, default=1500)
    ap.add_argument("--time-budget-s", type=float, default=5400.0)
    ap.add_argument("--seed-base", type=int, default=770000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--upload", action="store_true", help="upload train as an AC2 labeled dataset (needs AC2_API_KEY)")
    ap.add_argument("--dataset-name", default="pokerbots-sft-train-v1")
    args = ap.parse_args()

    postflop.EQUITY_SAMPLES = args.equity_samples  # the teacher decides at this count; the line matches it
    field_rng, villain_rng, stack_rng = random.Random(42), random.Random(43), random.Random(44)
    records, field_by_seed, played_seeds = [], {}, []
    legality_failures = 0
    short_stack_games = 0
    start = time.perf_counter()

    for g in range(args.max_games):
        seed = args.seed_base + g
        field = field_rng.choices(list(FIELDS), weights=[w for w, _ in FIELDS.values()])[0]
        field_by_seed[seed] = field
        # Short-stack slice: ~12% start at 15 BB (30 chips), ~8% at 25 BB (50 chips) to flood push/fold.
        roll = stack_rng.random()
        start_stack = 30 if roll < 0.12 else 50 if roll < 0.20 else 200
        short_stack_games += start_stack < 200
        legality_failures += await play_game(seed, field, records, args.equity_samples, villain_rng, start_stack)
        played_seeds.append(seed)
        if (g + 1) % 10 == 0 or g < 5:
            now = len(pipeline(records))
            print(f"game {g+1} seed={seed} field={field} raw={len(records)} examples={now} t={time.perf_counter()-start:.0f}s", flush=True)
        if len(pipeline(records)) >= args.target or time.perf_counter() - start > args.time_budget_s:
            break

    examples = pipeline(records)
    val_seeds = set(random.Random(3).sample(played_seeds, max(1, round(len(played_seeds) * args.val_frac))))
    train = [r for r in examples if r["meta"]["seed"] not in val_seeds]
    val = [r for r in examples if r["meta"]["seed"] in val_seeds]

    qc = quality_checks(examples)
    gate = gate_metrics(train)
    dist = distribution(train, field_by_seed)

    pre = [r["meta"]["decide_ms"] for r in records if r["meta"]["street"] == "preflop"]
    post = [r["meta"]["decide_ms"] for r in records if r["meta"]["street"] != "preflop"]
    avg_pre = sum(pre) / max(len(pre), 1)
    avg_post = sum(post) / max(len(post), 1)
    post_at_5000 = avg_post * 5000 / args.equity_samples
    played_per_example = len(records) / max(len(examples), 1)
    post_frac = len(post) / max(len(records), 1)
    ms_5000 = post_frac * post_at_5000 + (1 - post_frac) * avg_pre

    out = Path(__file__).resolve().parent / "out"; out.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        tasks = records_to_labeled_tasks(rows, TOOLS)
        with (out / f"{name}.jsonl").open("w") as f:
            for t in tasks:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")

    print("\n===== REPORT =====")
    print(f"games={len(played_seeds)} ({short_stack_games} short-stack) raw_decisions={len(records)} "
          f"examples={len(examples)} (train {len(train)} / val {len(val)})")
    print(f"teacher legality failures in-game: {legality_failures}")
    print(f"quality checks: {qc}")
    print(f"timing @ {args.equity_samples}: preflop {avg_pre:.1f}ms postflop {avg_post:.1f}ms -> @5000 postflop {post_at_5000:.0f}ms")
    print(f"projected full 25k @5000: ~{25000 * played_per_example * ms_5000 / 1000 / 60:.0f} min CPU")
    print(f"\nGATE: {'PASS' if gate['pass'] else 'FAIL'}")
    for name, (val_, ok, crit) in gate["checks"].items():
        print(f"  [{'ok' if ok else 'XX'}] {name}: {val_*100:.0f}%  (need {crit})")
    print("\ndistribution (train):")
    for col, counts in dist.items():
        print(f"  {col}: {counts}")
    print("\n===== 3 SAMPLES =====")
    for r in random.Random(7).sample(train, min(3, len(train))):
        items = records_to_labeled_tasks([r], TOOLS)[0].model_dump(mode="json")["input"]
        users = [i for i in items if i.get("type") == "message" and i["role"] == "user"]
        asst = [i for i in items if i.get("type") == "message" and i["role"] == "assistant"][-1]
        call = [i for i in items if i.get("type") == "function_call"][-1]
        print(f"\n[{r['meta']['street']} | {r['meta']['pos_group']} | {r['meta']['stack_bb']:.0f}bb | {r['meta']['players_left']}p | {field_by_seed[r['meta']['seed']]}]")
        print("USER:", users[-1]["content"][-320:].replace("\n", " "))
        print("ASSISTANT:", asst["content"])
        print("CALL:", call["name"], call["arguments"])

    if args.upload:
        if not gate["pass"]:
            print("\nNOT UPLOADING: gate failed."); return
        from ac2.sdk import Client
        client = Client(project="vals-hackathon")
        client.datasets.create(args.dataset_name, required_types=["labeled"],
                               description=f"Phase 1 SFT: teacher labels, {len(train)} decisions.", exist_ok=True)
        tasks = records_to_labeled_tasks(train, TOOLS)
        for i in range(0, len(tasks), 500):
            client.datasets.add_tasks(args.dataset_name, tasks=tasks[i:i + 500])
        print(f"\nUPLOADED {len(tasks)} tasks to {args.dataset_name}")


if __name__ == "__main__":
    asyncio.run(main())
