"""Where a seat stands at the table: what busting would cost, and whether its lead is safe."""

from pokerlab.datatypes.observation import Observation
from pokerlab.poker.situation import my_seat

BLINDS_PER_ORBIT_BB = 1.5


def chips(observation: Observation, seat: int) -> int:
    """Stack plus this street's bet; earlier streets' chips are in the pot and not counted."""
    view = next(view for view in observation.seats if view.seat == seat)
    return view.stack + view.bet


def share_below(observation: Observation) -> float:
    """Share of the other players still in the game with fewer chips: what busting would give up."""
    mine = chips(observation, observation.seat)
    others = [chips(observation, view.seat) for view in observation.seats if view.seat != observation.seat]
    return sum(other < mine for other in others) / max(len(others), 1)


def lead_is_safe(observation: Observation) -> bool:
    """True when the gap to the next stack below is larger than the blinds left to pay.

    Folding every hand then keeps this seat ahead of that player, whatever happens elsewhere.
    """
    mine = chips(observation, observation.seat)
    below = [
        chips(observation, view.seat)
        for view in observation.seats
        if view.seat != observation.seat and chips(observation, view.seat) < mine
    ]
    if not below:
        return False
    hands_left = observation.hand_count - observation.hand_number + 1
    blinds_left = BLINDS_PER_ORBIT_BB * observation.big_blind * hands_left / len(observation.seats)
    return mine - max(below) > blinds_left and my_seat(observation).stack > 0
