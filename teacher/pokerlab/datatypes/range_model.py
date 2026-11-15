"""The numbers behind the action-conditioned range model (bots/field/ranges.py).

Ranges are bands of preflop percentile: (0, 0.12) is the top 12% of combos.
Combos outside a band keep a small weight so an unexpected line never
empties a range. All values are starting heuristics, not solver output.
"""

from dataclasses import dataclass

# Preflop bands by action, before scaling by the seat's looseness.
OPEN_BAND = {"early": 0.12, "middle": 0.20, "late": 0.40, "blinds": 0.30}
BIG_OPEN_BB = 5.0  # an open this large or larger is narrower
BIG_OPEN_SHRINK = 0.7
LIMP_BAND = (0.08, 0.50)  # limpers rarely hold premiums
COLD_CALL_BAND = (0.04, 0.25)
BLIND_DEFEND_BAND = (0.04, 0.40)
CALL_RERAISE_BAND = (0.02, 0.12)
RERAISE_BAND = (0.0, 0.05)
FOUR_BET_BAND = (0.0, 0.025)
OUTSIDE_BAND_WEIGHT = 0.15


@dataclass(frozen=True)
class AggressionLikelihood:
    """A bet or raise of pot fraction f is "value" for strength >= clip(base + slope * f, low, high).

    Weaker combos (bluffs and draws) are scaled down until they make up `bluff_share`
    of the range. A fixed per-combo multiplier would not work: weak combos are most
    of any range, so even a 0.1 multiplier leaves them about half of it.
    """

    base: float
    slope: float
    low: float
    high: float
    bluff_share: float


BET_LIKELIHOOD = AggressionLikelihood(base=0.45, slope=0.25, low=0.50, high=0.90, bluff_share=0.25)
RAISE_LIKELIHOOD = AggressionLikelihood(base=0.60, slope=0.25, low=0.65, high=0.95, bluff_share=0.10)
RIVER_BLUFF_SHRINK = 0.5  # no draws left on the river, so fewer weak hands bet

# A call removes some of the weakest and some of the strongest combos (which would raise).
CALL_WEAK_BELOW, CALL_WEAK_WEIGHT = 0.35, 0.30
CALL_STRONG_ABOVE, CALL_STRONG_WEIGHT = 0.90, 0.50
# A check removes some strong combos (which would bet).
CHECK_STRONG_ABOVE, CHECK_STRONG_WEIGHT = 0.70, 0.40

# Adaptive bots scale a seat's bands by its observed VPIP, shrunk toward this prior.
PRIOR_VPIP = 0.25
PRIOR_HANDS = 20
