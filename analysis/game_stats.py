"""Scorecard for full-game evals (e.g. pokerbots-games): bust rate and final place, as the arena ranks them.

    ac2 traces download <eval_id> -o traces/
    uv run python analysis/game_stats.py traces/traces-<eval_id>.jsonl

One trace = one game. The policy seat's cleanup prompt after every hand it was dealt into ends with
`Stacks now: seat0 197, seat1 0 (out), ...`, which is enough to rebuild its chip trail, the hand it busted on and its
place under the event's elimination ranking: survivors rank above busted seats by chips; busted seats rank by when they
busted (later is better), and seats busting on the same hand by their stack before it. The sample grader's score is
not used: it clips a whole game's EV to +-5 BB and ignores rank.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

START = 200


def parse_stacks(text: str) -> dict[str, int]:
    found = re.findall(r"Stacks now: (.*)", text)
    return {s: int(v) for s, v in re.findall(r"(seat\d) (\d+)", found[-1])} if found else {}


def place(seat: str, trail: list[tuple[int, dict[str, int]]]) -> tuple[int, int | None]:
    """(place 1-6, hand busted on or None) from the seat's per-hand stacks."""
    last_hand, last = trail[-1]
    if last.get(seat, 0) > 0:  # survived: rank among survivors by chips (ties share the better place)
        return 1 + sum(v > last[seat] for v in last.values()), None
    before = trail[-2][1] if len(trail) > 1 else dict.fromkeys(last, START)
    alive_after = sum(v > 0 for v in last.values())  # all of them bust later or survive: they rank above us
    same_hand_better = sum(
        1 for s, v in last.items() if s != seat and v == 0 and before.get(s, 0) > 0 and before[s] > before.get(seat, 0)
    )
    return 1 + alive_after + same_hand_better, last_hand


def game(trace: dict) -> dict | None:
    comps = sorted((s for s in trace["spans"] if s["name"] == "get_completion"), key=lambda s: s["start_time"])
    seat, trail, decisions, cut, no_call, cleanups, updated, pad_lens = None, [], 0, 0, 0, 0, 0, []
    for s in comps:
        msgs = json.loads(s["input"])
        users = [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
        prompt = users[-1] if users else ""
        if seat is None and (m := re.search(r"Your seat: (\d)", "\n".join(users))):
            seat = f"seat{m.group(1)}"
        out = json.loads(s["output"] or "[]")
        calls = [o for o in out if o.get("type") == "function_call"]
        truncated = bool(json.loads(s["attributes"] or "{}").get("truncated"))
        if "update_scratchpad exactly once" in prompt:
            cleanups += 1
            hand = re.search(r"## Hand (\d+) is over", prompt)
            stacks = parse_stacks(prompt)
            if hand and stacks:
                trail.append((int(hand.group(1)), stacks))
            ups = [c for c in calls if c.get("name") == "update_scratchpad"]
            if ups and not truncated:
                updated += 1
                try:
                    pad_lens.append(len(json.loads(ups[0]["arguments"]).get("contents", "")))
                except (ValueError, AttributeError):
                    pass
        else:
            decisions += 1
            cut += truncated
            no_call += not truncated and not any(c.get("name") == "submit_action" for c in calls)
    if seat is None or not trail:
        return None
    trail.sort(key=lambda x: x[0])
    where, bust = place(seat, trail)
    art = json.loads(trace["scores"][0]["artifacts"] or "{}") if trace.get("scores") else {}
    return {
        "trace": trace["trace_id"],
        "seat": seat,
        "place": where,
        "bust": bust,
        "final": trail[-1][1].get(seat, 0),
        "hands": trail[-1][0],
        "chips": [st.get(seat, 0) for _, st in trail],
        "decisions": decisions,
        "cut": cut,
        "no_call": no_call,
        "cleanups": cleanups,
        "updated": updated,
        "pad_len": statistics.median(pad_lens) if pad_lens else 0,
        "ev_bb": art.get("ev_bb"),
        "format_errors": art.get("policy_format_errors"),
        "scored": bool(trace.get("scores")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--trails", action="store_true", help="print every game's chip trail")
    args = ap.parse_args()
    with open(args.path) as f:
        games = [g for line in f if line.strip() and (g := game(json.loads(line)))]
    done = [g for g in games if g["scored"]]
    print(f"== {len(done)} finished games ({len(games) - len(done)} more in progress) ==\n")
    if not done:
        return
    n = len(done)
    busts = [g for g in done if g["bust"]]
    places = Counter(g["place"] for g in done)
    print("RANK (what the arena rates)")
    print(f"  bust rate        {len(busts)}/{n} = {100 * len(busts) / n:.0f}%")
    print(f"  mean place       {statistics.fmean(g['place'] for g in done):.2f}  (1 best, 6 worst; 3.5 = average)")
    print("  places           " + "  ".join(f"#{p}: {places[p]}" for p in range(1, 7)))
    print(f"  top-2 finishes   {sum(g['place'] <= 2 for g in done)}/{n}")
    if busts:
        print("  busted on hands  " + " ".join(str(g["bust"]) for g in sorted(busts, key=lambda g: g["bust"])))
    survivors = [g for g in done if not g["bust"]]
    if survivors:
        print(f"  survivors' chips median {statistics.median(g['final'] for g in survivors)}  (start {START})")
    evs = [g["ev_bb"] for g in done if g["ev_bb"] is not None]
    if len(evs) > 1:
        ci = 1.96 * statistics.stdev(evs) / len(evs) ** 0.5
        print(f"  EV BB per game   {statistics.fmean(evs):+.1f} (95% CI +-{ci:.1f})")

    decs = sum(g["decisions"] for g in done)
    print("\nRELIABILITY")
    print(
        f"  decisions {decs}, cut at the cap {sum(g['cut'] for g in done)}, no submit_action {sum(g['no_call'] for g in done)}"
    )
    print(
        f"  games with any format error {sum(bool(g['format_errors']) for g in done)}/{n}  (the sample grader scores these 0)"
    )
    print(f"  cleanup turns updated {sum(g['updated'] for g in done)}/{sum(g['cleanups'] for g in done)}")
    print(f"  median scratchpad length {statistics.median(g['pad_len'] for g in done):.0f} chars (limit 5000)")

    by_seat = {}
    for g in done:
        by_seat.setdefault(g["seat"], []).append(g)
    print("\nBY STARTING SEAT  (seat0 posts the small blind on hand 1)")
    for s in sorted(by_seat):
        gs = by_seat[s]
        print(
            f"  {s}: mean place {statistics.fmean(g['place'] for g in gs):.2f}, busts {sum(bool(g['bust']) for g in gs)}/{len(gs)}"
        )

    if args.trails:
        print("\nGAMES")
        for g in sorted(done, key=lambda g: g["place"]):
            result = f"busted hand {g['bust']}" if g["bust"] else f"{g['final']} chips"
            print(f"  #{g['place']} {g['seat']} {result:16} [{' '.join(map(str, g['chips']))}]  trace {g['trace']}")


if __name__ == "__main__":
    main()
