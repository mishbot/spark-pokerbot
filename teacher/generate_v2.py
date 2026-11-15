"""v2 SFT generator: A1/A2/A3 wrappers + the FINAL trace. Produces a verified ~2k sample; no upload.

Reuses the v1 pipeline/gate/distribution (generate_dataset.py) but swaps in the v2 seat (v2_policy.py),
adds a shover-heavy field so A1 facing-shove spots appear, and runs deep (200-chip) games only (THOR_ALPHA:
the arena is deep; drop the forced short-stack hack). Cleanup/scratchpad tasks and the near-tie filter are
DEFERRED for this sample (see the report).

    PYTHONPATH=teacher:harness/src uv run --with numpy --with phevaluator --with pokerkit \
        --with fastapi --with httpx --with pydantic python teacher/generate_v2.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import pokerlab.bots.field.postflop as postflop

from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session
from poker_harness.prompts import parse_tool_call

from generate_dataset import TEACHER_SEATS, TOOLS, LEGAL_TYPES, persona_for, pipeline, gate_metrics, distribution
from seat import records_to_labeled_tasks
from v2_policy import RANKS, ShoverPlayer, make_v2_seat

# Deep arena field; a shover slice creates the deep facing-all-in spots A1 must handle.
FIELDS_V2 = {
    "mixed":    (0.40, {"calling_station": 0.45, "nit": 0.30, "random": 0.15, "maniac": 0.10}),
    "shovers":  (0.25, {"shover": 0.60, "calling_station": 0.25, "nit": 0.15}),
    "stations": (0.15, {"calling_station": 0.70, "nit": 0.30}),
    "nits":     (0.10, {"nit": 0.70, "calling_station": 0.30}),
    "maniacs":  (0.10, {"maniac": 0.50, "random": 0.30, "calling_station": 0.20}),
}


def villains_for(field, seed, rng):
    weights = FIELDS_V2[field][1]
    archs, probs = list(weights), list(weights.values())
    seats = {}
    for i in range(6):
        if i in TEACHER_SEATS:
            continue
        arch = rng.choices(archs, weights=probs)[0]
        seats[i] = ShoverPlayer(f"seat{i}") if arch == "shover" else ScriptedPlayer(f"seat{i}", arch, seed)
    return seats


async def play_game(seed, field, records, equity_samples, villain_rng):
    params = with_preset({"preset": "event", "seed": seed, "submit_fields": ["action", "amount"]})
    rules = table_rules(params)
    villains = villains_for(field, seed, villain_rng)
    players = []
    for i in range(rules.seats):
        if i in TEACHER_SEATS:
            players.append(make_v2_seat(f"seat{i}", i, rules, rules.big_blind, rules.small_blind, seed,
                                        records, persona_fn=persona_for, equity_samples=equity_samples))
        else:
            players.append(villains[i])
    log = await play_session(f"gen2-{seed}", seed, players, rules)
    names = {f"seat{i}" for i in TEACHER_SEATS}
    return sum(1 for h in log["hands"] for a in h["actions"] if a["player"] in names and (
        a["model_response"].get("fallback_used") or a["model_response"].get("error")
        or a["model_response"].get("format_error")))


def v2_quality(examples):
    fails, mx = [], 0
    for r in examples:
        msgs, m = r["messages"], r["meta"]
        args = msgs[-1]["tool_calls"][0]["function"]["arguments"]
        decision, error = parse_tool_call(args)
        if error is not None or decision.type not in LEGAL_TYPES:
            fails.append(f"parse:{error or decision.type}"); continue
        payload = json.loads(args); act = payload["action"]
        if act == "fold" and m["can_check"]:
            fails.append("fold-when-check-free"); continue
        if act in ("bet", "raise"):
            amt = payload.get("amount")
            if amt is None or m["min_raise_to"] is None or not (m["min_raise_to"] <= amt <= m["max_raise_to"]):
                fails.append(f"raise-amt {amt} not in [{m['min_raise_to']},{m['max_raise_to']}]"); continue
        content = msgs[-1]["content"]
        if act not in content:
            fails.append(f"act '{act}' absent from line"); continue
        toks = len(content.split()) + len(args.split()); mx = max(mx, toks)
        if toks >= 120:
            fails.append(f"too long ~{toks}"); continue
    return {"total": len(examples), "failed": len(fails), "first_fails": fails[:5], "max_tokens_est": mx}


def _cat(cls):
    if len(cls) == 2 and cls[0] == cls[1]:
        idx = RANKS.index(cls[0])
        if idx >= RANKS.index("8"):
            return "premium 88+"
        if idx in (RANKS.index("6"), RANKS.index("7")):
            return "marginal 66-77"
        return "small pair"
    ranks = {cls[0], cls[1]}
    if ranks in ({"A", "K"}, {"A", "Q"}):
        return "premium AQ+"
    if "A" in ranks and (ranks & {"T", "J"}):
        return "marginal AT-AJ"
    if "A" not in ranks and ranks <= {"K", "Q", "J", "T"}:
        return "never KQ/QJ/KT.."
    return "other"


def a1_verify(records):
    rows = defaultdict(lambda: [0, 0])  # category -> [calls, total]
    examples = []
    for r in records:
        m = r["meta"]
        if not m["deep_shove"]:
            continue
        c = _cat(m["hole_class"])
        rows[c][1] += 1
        if m["action"] == "call":
            rows[c][0] += 1
        if len(examples) < 6:
            examples.append((m["hole_class"], m["action"]))
    return rows, examples


def a3_verify(records):
    vpip = defaultdict(lambda: [0, 0])  # pos -> [voluntary, opportunities]  (preflop, not facing a raise)
    widened = 0
    for r in records:
        m = r["meta"]
        if m["street"] != "preflop" or m["facing_all_in"]:
            continue
        if m["wrapper"] == "a3_widen":
            widened += 1
        if m.get("pos") in ("CO", "BTN", "SB"):
            vpip[m["pos_group"]][1] += 1
            if m["action"] != "fold":
                vpip[m["pos_group"]][0] += 1
    return vpip, widened


def a2_verify(records):
    fired = sum(r["meta"]["wrapper"] == "a2_antibluff" for r in records)
    tr = [r for r in records if r["meta"]["street"] in ("turn", "river")]
    bets = sum(r["meta"]["action"] in ("bet", "raise") for r in tr)
    return fired, len(tr), bets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity-samples", type=int, default=300)
    ap.add_argument("--target", type=int, default=1800)
    ap.add_argument("--max-games", type=int, default=150)
    ap.add_argument("--time-budget-s", type=float, default=2400.0)
    ap.add_argument("--seed-base", type=int, default=880000)
    args = ap.parse_args()

    postflop.EQUITY_SAMPLES = args.equity_samples
    field_rng, villain_rng = random.Random(42), random.Random(43)
    records, field_by_seed, played = [], {}, []
    legality = 0
    start = time.perf_counter()
    for g in range(args.max_games):
        seed = args.seed_base + g
        field = field_rng.choices(list(FIELDS_V2), weights=[w for w, _ in FIELDS_V2.values()])[0]
        field_by_seed[seed] = field
        legality += await play_game(seed, field, records, args.equity_samples, villain_rng)
        played.append(seed)
        if (g + 1) % 10 == 0 or g < 3:
            print(f"game {g+1} seed={seed} field={field} raw={len(records)} ex={len(pipeline(records))} "
                  f"t={time.perf_counter()-start:.0f}s", flush=True)
        if len(pipeline(records)) >= args.target or time.perf_counter() - start > args.time_budget_s:
            break

    examples = pipeline(records)
    val_seeds = set(random.Random(3).sample(played, max(1, round(len(played) * 0.05))))
    train = [r for r in examples if r["meta"]["seed"] not in val_seeds]
    val = [r for r in examples if r["meta"]["seed"] in val_seeds]

    out = Path(__file__).resolve().parent / "out" / "v2"; out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        tasks = records_to_labeled_tasks(rows, TOOLS)
        with (out / f"{name}.jsonl").open("w") as f:
            for t in tasks:
                f.write(json.dumps(t.model_dump(mode="json")) + "\n")

    qc = v2_quality(examples)
    gate = gate_metrics(train)
    dist = distribution(train, field_by_seed)
    a1_rows, a1_ex = a1_verify(records)
    a3_vpip, a3_widened = a3_verify(records)
    a2_fired, a2_tr, a2_bets = a2_verify(records)

    print("\n===== v2 REPORT =====")
    print(f"games={len(played)} raw_decisions={len(records)} examples={len(examples)} "
          f"(train {len(train)} / val {len(val)})  legality_failures={legality}")
    print(f"quality: {qc}")
    print(f"\nGATE: {'PASS' if gate['pass'] else 'FAIL'}")
    for name, (v, ok, crit) in gate["checks"].items():
        print(f"  [{'ok' if ok else 'XX'}] {name}: {v*100:.0f}% (need {crit})")
    print("\n--- A1 shove-call (deep facing-all-in spots), call-rate by hand category ---")
    for c, (calls, tot) in sorted(a1_rows.items()):
        print(f"  {c:18s} call {calls}/{tot} = {100*calls/max(tot,1):.0f}%")
    print(f"  (NEVER category should be ~0% call; PREMIUM high. examples: {a1_ex})")
    print("\n--- A3 value-widen: late-position preflop VPIP (not facing a raise) ---")
    for pos, (v, opp) in sorted(a3_vpip.items()):
        print(f"  {pos}: VPIP {v}/{opp} = {100*v/max(opp,1):.0f}%   (thor arena: CO 21, BTN 25, SB 16)")
    print(f"  a3_widen wrapper fired {a3_widened} times")
    print(f"\n--- A2 anti-bluff: fired {a2_fired} times; turn/river bet-or-raise {a2_bets}/{a2_tr} "
          f"= {100*a2_bets/max(a2_tr,1):.0f}% (base thor turn high-card bet ~57%) ---")
    print("\ndistribution (train):")
    for col, counts in dist.items():
        print(f"  {col}: {counts}")
    print("\n===== 5 SAMPLE TRACES =====")
    want = ["a1_shove", "a3_widen", "a2_antibluff"]
    picks = []
    for tag in want:
        c = [r for r in train if r["meta"]["wrapper"] == tag]
        if c:
            picks.append(random.Random(1).choice(c))
    post = [r for r in train if r["meta"]["street"] != "preflop" and r["meta"]["wrapper"] == ""]
    picks += random.Random(2).sample(post, min(2, len(post)))
    for r in picks:
        items = records_to_labeled_tasks([r], TOOLS)[0].model_dump(mode="json")["input"]
        asst = [i for i in items if i.get("type") == "message" and i["role"] == "assistant"][-1]
        call = [i for i in items if i.get("type") == "function_call"][-1]
        m = r["meta"]
        print(f"\n[{m['wrapper'] or 'teacher'} | {m['street']} | {m['pos_group']} | {m['hole_class']} | {m['stack_bb']:.0f}bb]")
        print("THINK:", asst["content"].replace("\n", " "))
        print("CALL :", call["name"], call["arguments"])


if __name__ == "__main__":
    asyncio.run(main())
