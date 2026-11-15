"""Per-seat statistics from earlier hands, for the adaptive bots.

Every estimate is shrunk toward a prior worth PRIOR_HANDS hands, so a seat
seen for three hands is not judged on three hands.
"""

from collections import defaultdict

from pokerlab.datatypes.observation import PREFLOP, HandSummary
from pokerlab.datatypes.range_model import PRIOR_HANDS, PRIOR_VPIP
from pokerlab.datatypes.tracking import SeatCounts
from pokerlab.poker.replay import replay


class Tracker:
    def __init__(self) -> None:
        self.counts: dict[int, SeatCounts] = defaultdict(SeatCounts)

    def record(self, summary: HandSummary) -> None:
        voluntary, raised = set(), set()
        for event in replay(summary.history):
            counts = self.counts[event.seat]
            if event.street == PREFLOP:
                if event.kind in ("call", "bet", "raise"):
                    voluntary.add(event.seat)
                if event.kind in ("bet", "raise"):
                    raised.add(event.seat)
            elif event.to_call > 0:
                counts.faced_bet += 1
                counts.folded_to_bet += event.kind == "fold"
        for seat in summary.seats:
            counts = self.counts[seat]
            counts.hands += 1
            counts.voluntary += seat in voluntary
            counts.preflop_raises += seat in raised

    def looseness(self, seat: int) -> float:
        """Observed VPIP relative to the prior: 2.0 plays twice as many hands as a standard player."""
        counts = self.counts[seat]
        vpip = (counts.voluntary + PRIOR_VPIP * PRIOR_HANDS) / (counts.hands + PRIOR_HANDS)
        return vpip / PRIOR_VPIP

    def fold_rate(self, seat: int, prior: float) -> float:
        """How often the seat folds to a bet after the flop, shrunk toward `prior`."""
        counts = self.counts[seat]
        return (counts.folded_to_bet + prior * PRIOR_HANDS) / (counts.faced_bet + PRIOR_HANDS)
