"""Independent validator: re-derive every fact from the raw game state and check the rendered trace
and the action against it. Imports NONE of the generator modules (eval/tier/ranges/policy); the
hand category is re-derived with phevaluator directly, so a shared bug cannot pass both.

`check(truth, trace_text, action)` returns a list of violation strings (empty == logically correct).
`truth` is a dict: hole, board, pot, to_call, street, position, min_raise, max_raise, and the
derived seat facts (button, me, sb_seat, bb_seat, n). `action` is (kind, amount).
"""

from __future__ import annotations

import math
import re

from phevaluator import evaluate_cards

# phevaluator rank (1 = best .. 7462 = worst), standard 5-card category boundaries.
_CUTOFFS = [(10, "straight_flush"), (166, "quads"), (322, "full_house"), (1599, "flush"),
            (1609, "straight"), (2467, "trips"), (3325, "two_pair"), (6185, "pair"), (7462, "high_card")]

# trace made-label -> category bucket
_LABEL_CAT = {
    "straight flush": "straight_flush", "quads": "quads", "full house": "full_house",
    "flush": "flush", "straight": "straight", "set": "trips", "trips": "trips",
    "two pair": "two_pair", "overpair": "pair", "top pair": "pair", "second pair": "pair",
    "weak pair": "pair", "pocket pair below top": "pair", "board pair": "pair", "high card": "high_card",
}
_CONTINUE = {"clear ahead", "draw-priced"}


def phe_category(cards: list[str]) -> str:
    rank = evaluate_cards(*cards)
    for hi, name in _CUTOFFS:
        if rank <= hi:
            return name
    return "high_card"


def _need_pct(to_call: int, pot: int) -> int:
    return int(math.floor(to_call / (pot + to_call) * 100 + 0.5)) if (pot + to_call) > 0 else 0


def _find(pattern: str, text: str):
    m = re.search(pattern, text)
    return m


def check(truth: dict, trace: str, action: tuple[str, int | None]) -> list[str]:
    v: list[str] = []
    kind, amount = action

    # I1 hole cards (preflop only: the postflop Hand line states the made hand, not the cards;
    # postflop hole correctness is covered by I2, which uses the real cards via phevaluator).
    if truth["street"] == "preflop":
        m = _find(r"Hand:\s*([^,;]+)", trace)
        if m:
            shown = set(m.group(1).split())
            if shown != set(truth["hole"]):
                v.append(f"I1 hole mismatch: trace {shown} vs truth {set(truth['hole'])}")
        else:
            v.append("I1 no Hand line")

    # I2 made hand vs phevaluator (incl. straight-never-air)
    if truth["street"] != "preflop":
        cat = phe_category(list(truth["hole"]) + list(truth["board"]))
        mlabel = _find(r"Hand:\s*([a-z ]+?)[,;]", trace)
        if mlabel:
            claimed = _LABEL_CAT.get(mlabel.group(1).strip())
            if claimed is None:
                v.append(f"I2 unknown made label '{mlabel.group(1).strip()}'")
            elif claimed != cat:
                v.append(f"I2 made-hand mismatch: trace '{mlabel.group(1).strip()}'({claimed}) vs phevaluator {cat}")

    # I3 position (truth already encodes the HU fix)
    pos = None
    m = _find(r"which is (\w+)", trace) or _find(r"so I am (SB|BB)", trace)
    if m:
        pos = m.group(1)
        if pos != truth["position"]:
            v.append(f"I3 position mismatch: trace {pos} vs truth {truth['position']}")

    # I3b in/out of position (postflop): we are "in position" iff we are the last live seat to act
    if truth["street"] != "preflop" and truth.get("live_order"):
        m = _find(r"Position:\s*(in position|out of position)", trace)
        if m:
            expect = "in position" if truth["me_seat"] == truth["live_order"][-1] else "out of position"
            if m.group(1) != expect:
                v.append(f"I3b position: trace '{m.group(1)}' but we are {expect} (last live acts in position)")

    # I5 need% arithmetic (facing a bet only)
    if truth["to_call"] > 0 and truth["street"] != "preflop":
        want = _need_pct(truth["to_call"], truth["pot"])
        for got in re.findall(r"need (\d+)%", trace):
            if abs(int(got) - want) > 1:
                v.append(f"I5 need% wrong: trace {got}% vs computed {want}%")

    # I6 sizing legality
    if kind == "raise":
        lo, hi = truth.get("min_raise"), truth.get("max_raise")
        if amount is None or lo is None or hi is None or not (lo <= amount <= hi):
            v.append(f"I6 illegal raise amount {amount} (legal [{lo},{hi}])")

    # I11 edge -> action word, and the word -> the real action
    m = _find(r"->\s*(clear ahead|clear behind|draw-priced|close)\s*->\s*(continue|fold)", trace)
    if m:
        edge, word = m.group(1), m.group(2)
        expect = "continue" if edge in _CONTINUE else "fold"
        if word != expect:
            v.append(f"I11 edge/word mismatch: edge '{edge}' -> word '{word}' (expected {expect})")
        if word == "continue" and kind not in ("call", "raise"):
            v.append(f"I11 'continue' but action is {kind}")
        if word == "fold" and kind != "fold":
            v.append(f"I11 'fold' but action is {kind}")

    # I14 Action line verb consistent with the submitted action type (harness: bet opens, raise re-raises)
    m = _find(r"Action:\s*([\w-]+)", trace)
    if m:
        verb = m.group(1)
        allowed = {"bet": {"bet", "all-in"}, "raise": {"raise", "all-in", "open", "reraise"},
                   "call": {"call"}, "fold": {"fold"}, "check": {"check"}}
        if verb not in allowed.get(kind, set()):
            v.append(f"I14 action-line verb '{verb}' inconsistent with submitted {kind}")

    # no fabricated equity NUMBER (the word "equity" in strategy prose is fine; a percent is not)
    if re.search(r"equity\D{0,8}\d+%|\d+%\s*equity", trace, re.I):
        v.append("no-equity: the trace states an equity number")

    return v
