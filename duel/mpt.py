"""Preflop grading against Modern Poker Theory (Acevedo, 2019), ch. 5: 6-max cash, 100 BB equilibrium ranges.

    uv run --project harness python duel/mpt.py duel/runs/<run>           # grade a duel run, per side
    uv run --project harness python duel/mpt.py --check                     # chart shares vs the book's numbers

Two spots, both read straight from game logs, so grading costs no model calls:

- RFI (raise first in): everyone before the hero folded. LJ/HJ/CO/BN are raise-or-fold at 2.5x (Hand Ranges 47,
  42, 38, 35); the SB plays (raise 3x or limp) or folds (Hand Range 32).
- BB defence: one player opened, everyone else folded, the hero is the big blind. Defend (call or 3-bet) or fold
  (Hand Ranges 76, 78, 80, 82, 84). Only opens of at most 4 BB are graded; the charts face 2.5x (SB 3x).

Each chart is a 13 x 13 grid transcribed from the book's images (rows/columns A..2; suited above the diagonal,
offsuit below). Cells the book splits between actions are `M` and not graded: only clear hands count. Graded only at
full six-handed tables with the hero and every seat behind it holding at least 50 BB, and only decisions the model
actually made (forced check/folds after a failed turn are left out).

Caveats: the book assumes 5% rake and equilibrium opponents. The arena has no rake and plays LLMs, so correct
ranges there are somewhat wider: read "too tight" with that in mind. Agreement is a style diagnostic, not a score.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RANKS = "AKQJT98765432"
POSITIONS = {0: "SB", 1: "BB", 2: "LJ", 3: "HJ", 4: "CO", 5: "BN"}
MIN_STACK_BB = 50
MAX_OPEN_BB = 4.0
SPOT_ORDER = [f"RFI {p}" for p in ("LJ", "HJ", "CO", "BN", "SB")] + [f"BB vs {p}" for p in ("LJ", "HJ", "CO", "BN", "SB")]

# R raise, P play (raise or limp, SB only), D defend (call or 3-bet), F fold, M mixed (not graded)
RFI = {
    "LJ": ("RRRRRRRRRRRRF RRRRRRMMMMFFF RRRRRRFFFFFFF RRMRRMFFFFFFF RMFFRMFFFFFFF FFFFFRFFFFFFF FFFFFFRMFFFFF "
           "FFFFFFFRFFFFF FFFFFFFFMMFFF FFFFFFFFFMMFF FFFFFFFFFFMFF FFFFFFFFFFFMF FFFFFFFFFFFFM", 17.1),
    "HJ": ("RRRRRRRRRRRRR RRRRRRRRMMFFF RRRRRRMFFFFFF RRRRRRFFFFFFF RRMMRRMFFFFFF MFFFFRMFFFFFF FFFFFFRMFFFFF "
           "FFFFFFFRMFFFF FFFFFFFFRMFFF FFFFFFFFFRMFF FFFFFFFFFFMFF FFFFFFFFFFFMF FFFFFFFFFFFFM", 21.4),
    "CO": ("RRRRRRRRRRRRR RRRRRRRRRRRMF RRRRRRRRRFFFF RRRRRRRMFFFFF RRRRRRRMFFFFF MFFFMRRRFFFFF MFFFFFRRFFFFF "
           "FFFFFFFRMFFFF FFFFFFFFRMFFF MFFFFFFFFRMFF FFFFFFFFFFRFF FFFFFFFFFFFMF FFFFFFFFFFFFM", 27.8),
    "BN": ("RRRRRRRRRRRRR RRRRRRRRRRRRR RRRRRRRRRRRRM RRRRRRRRRRRFF RRRRRRRRRFFFF RRRRRRRRRFFFF RRFMMRRRRRFFF "
           "RFFFFFMRRRMFF RFFFFFFFRRRFF RFFFFFFFFRRRF MFFFFFFFFFRFF FFFFFFFFFFFRF FFFFFFFFFFFFR", 43.4),
    "SB": ("PPPPPPPPPPPPP PPPPPPPPPPPPP PPPPPPPPPPPPP PPPPPPPPPPPPP PPPPPPPPPPPPM PPPPPPPPPPPMF PPPPPPPPPPPFF "
           "PPPPPPPPPPPFF PPMFFMMPPPPPF PPMFFFFFMPPPP PMFFFFFFFFPPM PFFFFFFFFFFPF PFFFFFFFFFFFP", 62.1),
}
BB_DEFENCE = {  # keyed by the opener's position
    "LJ": ("DDDDDDDDDDDDD DDDDDDDDDDDDM DDDDDDDMMMFFF DDMDDDDMFFFFF DMMMDDDDMFFFF FFFFMDDDDMFFF FFFFFFDDDDFFF "
           "FFFFFFFDDDDFF FFFFFFFFDDDMF FFFFFFFFFDDDM FFFFFFFFFFDDM FFFFFFFFFFFDM FFFFFFFFFFFFD", 28.6),
    "HJ": ("DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDMFF DDDDDDDMFFFFF DDMMDDDDMFFFF MFFFMDDDDMFFF FFFFFMDDDDFFF "
           "FFFFFFFDDDDFF FFFFFFFFDDDMF MFFFFFFFFDDDM FFFFFFFFFFDDM FFFFFFFFFFFDM FFFFFFFFFFFFD", 31.5),
    "CO": ("DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDM DDDDDDDDDMFFF DDDDDDDDMFFFF DFFMMDDDDMFFF DFFFFMDDDDFFF "
           "FFFFFFFDDDDFF FFFFFFFFDDDFF DFFFFFFFFDDDM FFFFFFFFFFDDM FFFFFFFFFFFDF FFFFFFFFFFFFD", 35.4),
    "BN": ("DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDMF DDDDDDDDDDDFF "
           "DDFFFMDDDDDDF DMFFFFFDDDDDM DFFFFFFFMDDDD DFFFFFFFFMDDD DFFFFFFFFFFDD MFFFFFFFFFFFD", 56.8),
    "SB": ("DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDDD DDDDDDDDDDDMF "
           "DDMMDDDDDDDDM DDMFFFMDDDDDD DDFFFFFMDDDDD DMFFFFFFFDDDD DFFFFFFFFFFDD DFFFFFFFFFFFD", 64.6),
}


def grid(spec: str) -> dict[str, str]:
    """{"AKs": "R", "AKo": "R", "AA": "R", ...} from 13 rows of 13 cells."""
    rows = spec.split()
    assert len(rows) == 13 and all(len(r) == 13 for r in rows), spec
    out = {}
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            hi, lo = RANKS[min(i, j)], RANKS[max(i, j)]
            out[hi + lo + ("" if i == j else "s" if j > i else "o")] = cell
    return out


RFI_GRIDS = {pos: grid(spec) for pos, (spec, _) in RFI.items()}
BB_GRIDS = {pos: grid(spec) for pos, (spec, _) in BB_DEFENCE.items()}


def hand_class(cards: list[str]) -> str:
    """["Kd", "As"] -> "AKo"."""
    a, b = sorted(cards, key=lambda c: RANKS.index(c[0]))
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def combos(cls: str) -> int:
    return 6 if len(cls) == 2 else 4 if cls.endswith("s") else 12


def share(g: dict[str, str], play: str) -> tuple[float, float]:
    """% of the 1,326 starting hands the chart plays: clear cells only, and with mixed cells counted in full."""
    clear = sum(combos(c) for c, v in g.items() if v == play) / 1326 * 100
    return clear, clear + sum(combos(c) for c, v in g.items() if v == "M") / 1326 * 100


def check() -> None:
    print("chart          book %   clear-only %   clear+mixed %   (the book's number should sit in between)")
    for name, charts, grids, play in (("RFI", RFI, RFI_GRIDS, None), ("BB vs", BB_DEFENCE, BB_GRIDS, "D")):
        for pos, (_, book) in charts.items():
            lo, hi = share(grids[pos], play or ("P" if pos == "SB" else "R"))
            ok = "ok" if lo - 1 <= book <= hi + 1 else "CHECK"
            print(f"{name} {pos:3s}       {book:5.1f}      {lo:5.1f}          {hi:5.1f}        {ok}")


# ---------------------------------------------------------------------------------------------------------------
# Spots from logs


def kind(action: dict[str, Any]) -> str:
    t = action["action"]["type"]
    return "raise" if t in ("raise", "bet") else t


def spots(log: dict[str, Any]) -> list[dict[str, Any]]:
    """Every gradable RFI and BB-defence decision in one game log."""
    out = []
    bb = log["session_info"]["big_blind"]
    for hand in log["hands"]:
        state = hand["starting_state"]
        if len(state) != 6:
            continue  # the charts are six-handed
        pos = {name: s["position"] for name, s in state.items()}
        stack_bb = {name: s["starting_stack"] / bb for name, s in state.items()}
        pre = [a for a in hand["actions"] if a["street"] == "preflop"]
        for i, a in enumerate(pre):
            hero, before = a["player"], pre[:i]
            acted = {b["player"] for b in before}
            if hero in acted:
                break  # every first decision has been seen; nothing after this is first-in or BB-vs-open
            if (a.get("model_response") or {}).get("format_error"):
                continue  # a forced check/fold, not the model's choice
            behind = [n for n in state if n != hero and n not in acted]  # still to act
            deep = stack_bb[hero] >= MIN_STACK_BB and all(stack_bb[n] >= MIN_STACK_BB for n in behind)
            cls = hand_class(state[hero]["hole_cards"])
            p = POSITIONS[pos[hero]]
            row = {"hand": hand["hand_number"], "player": hero, "position": p, "hand_class": cls,
                   "action": kind(a), "amount_bb": (a["action"].get("amount") or 0) / bb}
            if p != "BB" and all(kind(b) == "fold" for b in before):
                if deep:
                    out.append(row | {"spot": "RFI", "chart": RFI_GRIDS[p][cls]})
            elif p == "BB":
                raises = [b for b in before if kind(b) == "raise"]
                if len(raises) == 1 and all(kind(b) in ("fold", "raise") for b in before):
                    opener = raises[0]
                    size = (opener["action"].get("amount") or 0) / bb
                    o = POSITIONS[pos[opener["player"]]]
                    if size <= MAX_OPEN_BB and deep and stack_bb[opener["player"]] >= MIN_STACK_BB:
                        out.append(row | {"spot": f"BB vs {o}", "chart": BB_GRIDS[o][cls], "open_bb": size})
    return out


def verdict(spot: dict[str, Any]) -> str | None:
    """agree / too_loose / too_tight / limp (a non-SB limp, which the book never does), or None if not graded."""
    chart, act = spot["chart"], spot["action"]
    if chart == "M":
        return None
    if spot["spot"] == "RFI":
        if spot["position"] == "SB":
            plays = act in ("raise", "call")
            return "agree" if plays == (chart == "P") else "too_loose" if plays else "too_tight"
        if act == "call":
            return "limp" if chart == "R" else "too_loose"
        plays = act == "raise"
        return "agree" if plays == (chart == "R") else "too_loose" if plays else "too_tight"
    defends = act in ("call", "raise")
    return "agree" if defends == (chart == "D") else "too_loose" if defends else "too_tight"


def grade(logs: list[dict[str, Any]], seats_of) -> dict[str, Any]:
    """Per spot: graded decisions, agreement with the book, and how the misses lean. `seats_of(log)` names the
    seats to grade in each log."""
    by_spot: dict[str, Counter] = defaultdict(Counter)
    opens: dict[str, list[float]] = defaultdict(list)
    for log in logs:
        mine = set(seats_of(log))
        for s in spots(log):
            if s["player"] not in mine:
                continue
            v = verdict(s)
            key = s["spot"] if s["spot"] != "RFI" else f"RFI {s['position']}"
            by_spot[key]["seen"] += 1
            if v:
                by_spot[key]["graded"] += 1
                by_spot[key][v] += 1
            if s["spot"] == "RFI" and s["action"] == "raise":
                opens[s["position"]].append(s["amount_bb"])
    out = {}
    for key in SPOT_ORDER:
        c = by_spot.get(key)
        if not c:
            continue
        g = c["graded"]
        out[key] = {"seen": c["seen"], "graded": g, "agree": c["agree"],
                    "agree_rate": round(c["agree"] / g, 3) if g else None,
                    "too_loose": c["too_loose"], "too_tight": c["too_tight"], "limp": c["limp"]}
        pos = key.split()[-1]
        if key.startswith("RFI") and opens.get(pos):
            out[key]["open_bb_median"] = round(statistics.median(opens[pos]), 2)
    total = Counter()
    for key, v in out.items():
        total["graded"] += v["graded"]
        total["agree"] += v["agree"]
    if total["graded"]:
        out["all"] = {"graded": total["graded"], "agree": total["agree"],
                      "agree_rate": round(total["agree"] / total["graded"], 3)}
    return out


def print_grades(name: str, g: dict[str, Any]) -> None:
    print(f"\n{name}: preflop vs Modern Poker Theory ch. 5 (clear hands only)")
    print("  spot        graded  agree   too loose  too tight  limp   open size (BB, book 2.5 / SB 3)")
    for key, v in g.items():
        if key == "all":
            continue
        rate = f"{100 * v['agree_rate']:5.1f}%" if v["agree_rate"] is not None else "   n/a"
        print(f"  {key:10s}  {v['graded']:6d}  {rate}  {v['too_loose']:9d}  {v['too_tight']:9d}  {v['limp']:4d}"
              f"   {v.get('open_bb_median', '')}")
    if "all" in g:
        print(f"  all         {g['all']['graded']:6d}  {100 * g['all']['agree_rate']:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="a duel run directory")
    parser.add_argument("--check", action="store_true", help="compare each chart's share with the book's number")
    args = parser.parse_args()
    if args.check or not args.run:
        check()
        if not args.run:
            return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from duel import load_logs, team

    logs = load_logs(args.run)
    names = logs[0]["duel"]
    for side in ("A", "B"):
        print_grades(f"{side} = {names[side.lower()]}", grade(logs, lambda log, s=side: team(log, s)))


if __name__ == "__main__":
    main()
