"""Scorecard for an eval's traces: reliability, token use and playing style of the policy seat.

    ac2 traces download <eval_id> -o traces/
    uv run python analysis/trace_stats.py traces/traces-<eval_id>.jsonl [--examples N]

One trace = one hand. Decision turns are the policy's `get_completion` spans; the cleanup turn (update_scratchpad) is
reported separately.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict

POSITIONS = {0: "SB", 1: "BB", 2: "UTG", 3: "HJ", 4: "CO", 5: "BTN"}  # 6-max, position 0 = small blind
CAP = 5000


def last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def all_user(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str))


def parse_output(raw: str) -> tuple[list[dict], str]:
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        return [], ""
    calls = [i for i in items if i.get("type") == "function_call"]
    reasoning = "".join(i.get("content") or "" for i in items if i.get("type") == "reasoning")
    return calls, reasoning


def hand_class(cards: list[str]) -> str:
    """AKs / T9o / 77 style."""
    order = "23456789TJQKA"
    a, b = sorted(cards, key=lambda c: order.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "   n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--examples", type=int, default=3, help="print this many worst-scoring hands")
    args = ap.parse_args()

    with open(args.path) as f:
        traces = [json.loads(line) for line in f if line.strip()]

    hands = []
    for t in traces:
        if not t.get("scores"):
            continue
        art = json.loads(t["scores"][0]["artifacts"] or "{}")
        comps = sorted((s for s in t["spans"] if s["name"] == "get_completion"), key=lambda s: s["start_time"])
        decisions, cleanups = [], []
        for s in comps:
            msgs = json.loads(s["input"])
            prompt = last_user(msgs)
            calls, reasoning = parse_output(s["output"])
            attrs = json.loads(s["attributes"] or "{}")
            rec = {
                "tokens_out": s["tokens_out"],
                "reasoning_tokens": s["reasoning_tokens"],
                "truncated": bool(attrs.get("truncated")),
                "calls": calls,
                "reasoning": reasoning,
                "prompt": prompt,
                "history": all_user(msgs),
            }
            (cleanups if "update_scratchpad exactly once" in prompt else decisions).append(rec)
        first = decisions[0]["history"] if decisions else ""
        cards = re.search(r"Your hole cards: (\S+) (\S+)", first)
        pos = re.search(r"\(position (\d)", first)
        hands.append(
            {
                "trace": t["trace_id"],
                "score": t["scores"][0]["score"],
                "summary": t["scores"][0]["reasoning"],
                "ev_bb": art.get("ev_bb"),
                "realized_bb": art.get("realized_bb"),
                "format_errors": art.get("policy_format_errors", 0),
                "folded": art.get("folded"),
                "decisions": decisions,
                "cleanups": cleanups,
                "cards": [cards.group(1), cards.group(2)] if cards else None,
                "pos": POSITIONS.get(int(pos.group(1))) if pos else "?",
            }
        )

    n = len(hands)
    decs = [d for h in hands for d in h["decisions"]]
    cleans = [c for h in hands for c in h["cleanups"]]
    evs = [h["ev_bb"] for h in hands if h["ev_bb"] is not None]
    print(f"== {n} hands, {len(decs)} decisions, {len(cleans)} cleanup turns ==\n")

    # --- results ---
    mean = statistics.fmean(evs) if evs else 0
    ci = 1.96 * statistics.stdev(evs) / len(evs) ** 0.5 if len(evs) > 1 else float("nan")
    print("RESULTS")
    print(f"  EV BB/hand          {mean:+.3f}  (95% CI ±{ci:.3f})   = {mean * 100:+.1f} bb/100")
    print(
        f"  realized BB/hand    {statistics.fmean(h['realized_bb'] for h in hands if h['realized_bb'] is not None):+.3f}"
    )
    print(f"  mean grader score   {statistics.fmean(h['score'] for h in hands):.2f}  (6 = break-even, 0 = failure)")

    # --- reliability ---
    fe_hands = sum(1 for h in hands if h["format_errors"])
    trunc = sum(d["truncated"] for d in decs)
    no_call = sum(1 for d in decs if not d["truncated"] and not any(c["name"] == "submit_action" for c in d["calls"]))
    multi = sum(1 for d in decs if sum(c["name"] == "submit_action" for c in d["calls"]) > 1)
    print("\nRELIABILITY")
    print(f"  hands with a format error   {fe_hands:5d}  {pct(fe_hands, n)}")
    print(f"  decisions cut at cap        {trunc:5d}  {pct(trunc, len(decs))}")
    print(f"  decisions w/o submit_action {no_call:5d}  {pct(no_call, len(decs))}")
    print(f"  decisions with >1 call      {multi:5d}  {pct(multi, len(decs))}")
    c_ok = sum(1 for c in cleans if any(x["name"] == "update_scratchpad" for x in c["calls"]) and not c["truncated"])
    print(f"  cleanup turns that updated  {c_ok:5d}  {pct(c_ok, len(cleans))}")

    # --- tokens ---
    toks = sorted(d["tokens_out"] for d in decs)
    if toks:

        def q(p: float) -> int:
            return toks[min(len(toks) - 1, int(p * len(toks)))]

        print("\nTOKENS PER DECISION (cap 5000, thinking included)")
        print(f"  p50 {q(0.5)}   p90 {q(0.9)}   p99 {q(0.99)}   max {toks[-1]}")
        print(
            f"  over 3000: {pct(sum(t > 3000 for t in toks), len(toks))}   over 4000: {pct(sum(t > 4000 for t in toks), len(toks))}"
        )
        by_street = defaultdict(list)
        for d in decs:
            st = re.search(r"Street: (\w+)", d["prompt"])
            by_street[st.group(1) if st else "?"].append(d["tokens_out"])
        print("  median by street: " + ", ".join(f"{k} {int(statistics.median(v))}" for k, v in by_street.items()))
    ctoks = sorted(c["tokens_out"] for c in cleans)
    if ctoks:
        print(f"  cleanup turn: p50 {ctoks[len(ctoks) // 2]}  max {ctoks[-1]}  (cap 10000)")

    # --- style ---
    acts = Counter()
    vpip = pfr = pre_n = 0
    allin = 0
    for h in hands:
        pre = [d for d in h["decisions"] if "Street: preflop" in d["prompt"]]
        moves = []
        for d in h["decisions"]:
            for c in d["calls"]:
                if c["name"] == "submit_action":
                    try:
                        a = json.loads(c["arguments"])
                    except ValueError:
                        continue
                    acts[a.get("action")] += 1
                    legal = re.search(r"-(\d+)\nCall submit_action", d["prompt"])
                    if a.get("amount") and legal and int(a["amount"]) >= int(legal.group(1)):
                        allin += 1
                    if d in pre:
                        moves.append(a.get("action"))
        if pre:
            pre_n += 1
            vpip += any(m in ("call", "raise", "bet", "all_in") for m in moves)
            pfr += any(m in ("raise", "bet") for m in moves)
    tot = sum(acts.values())
    print("\nSTYLE")
    print(f"  VPIP {pct(vpip, pre_n)}  (voluntarily played preflop; solid 6-max ~ 22-28%)")
    print(f"  PFR  {pct(pfr, pre_n)}  (raised preflop; solid ~ 17-22%)")
    print("  actions: " + ", ".join(f"{k} {pct(v, tot).strip()}" for k, v in acts.most_common()))
    print(f"  all-in shoves: {allin}")

    by_pos = defaultdict(list)
    for h in hands:
        if h["ev_bb"] is not None:
            by_pos[h["pos"]].append(h["ev_bb"])
    print("\nEV BB/HAND BY POSITION")
    for p in ["UTG", "HJ", "CO", "BTN", "SB", "BB"]:
        if by_pos[p]:
            print(f"  {p:4} {statistics.fmean(by_pos[p]):+7.2f}  (n={len(by_pos[p])})")

    # --- worst hands ---
    if args.examples:
        print(f"\nWORST {args.examples} HANDS")
        for h in sorted(hands, key=lambda h: h["ev_bb"] if h["ev_bb"] is not None else 0)[: args.examples]:
            print(
                f"\n--- {h['pos']} {hand_class(h['cards']) if h['cards'] else '?'}: {h['summary']}  trace {h['trace']}"
            )
            for d in h["decisions"]:
                st = re.search(r"Street: (\w+)", d["prompt"])
                board = re.search(r"Board: (.*)", d["prompt"])
                legal = re.search(r"Legal actions: (.*)", d["prompt"])
                call = next((c["arguments"] for c in d["calls"] if c["name"] == "submit_action"), "NO CALL")
                print(
                    f"  {st.group(1) if st else '?':8} board [{board.group(1) if board else ''}]  legal: "
                    f"{legal.group(1) if legal else ''}  -> {call}  ({d['tokens_out']} tok)"
                )


if __name__ == "__main__":
    main()
