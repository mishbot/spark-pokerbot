"""Postflop strength TIER from the made hand (never from equity), with a fixed price per tier and
the dangerous-board cap built in.

The tier is what the trace shows and what the postflop rule decides with. It is a function of the
made hand + board only, so it is stable across streets for the same made hand (fixes #7), ace-high
is never above air (fixes #8), and it is independent of any Monte-Carlo equity (fixes the #6 word).

Price = the highest pot-odds `need%` at which this tier can continue for value. The postflop rule
compares the computed need% to this price (clear if the gap is large, else close).

Dangerous-board cap (RC7 / cap-in-teacher): on a board whose nut hand has changed (four-flush,
monotone, paired, four-straight), a made hand that does NOT use that feature is lowered one tier,
so the trace's tier and the action agree and we stop over-committing one pair on scary boards.
"""

from __future__ import annotations

from v3.eval import HandInfo, CAT_RANK

# tier -> price (max need% to continue for value), strongest first
TIERS = ("strong", "medium", "weak", "air")
PRICE = {"strong": 0.55, "medium": 0.30, "weak": 0.15, "air": 0.0}
_ORDER = {t: i for i, t in enumerate(TIERS)}


def _base_tier(info: HandInfo) -> str:
    cat = info.category
    if CAT_RANK[cat] >= CAT_RANK["straight"]:      # straight, flush, boat, quads, SF
        return "strong"
    if cat == "trips":                              # set or trips
        return "strong"
    if cat == "two_pair":
        return "strong"
    if cat == "pair":
        if info.made in ("overpair", "top pair"):
            return "medium"
        return "weak"                               # second pair, weak pair, pocket below top, board pair
    return "air"                                    # high card (incl. ace-high)


def _lower(tier: str) -> str:
    return TIERS[min(_ORDER[tier] + 1, len(TIERS) - 1)]


def tier_and_price(info: HandInfo) -> tuple[str, float, bool]:
    """(tier, price, capped). Cap when the board threatens a hand our made hand does not beat:
    our category must be >= the board's max threat (flush on a 3-flush board, straight on a
    four-straight board, full house on a paired board), else we lower one tier."""
    if info.plays_board:   # the hole cards do not improve on the board -> can only chop, never value
        return "air", PRICE["air"], False
    base = _base_tier(info)
    threatened = info.max_threat >= 0 and CAT_RANK[info.category] < info.max_threat
    if threatened and base != "air":
        return _lower(base), PRICE[_lower(base)], True
    return base, PRICE[base], False
