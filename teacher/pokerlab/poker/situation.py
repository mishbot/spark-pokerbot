"""Facts read from an Observation that bots need often."""

from pokerlab.datatypes.observation import ActionRecord, Observation, SeatView


def my_seat(observation: Observation) -> SeatView:
    return next(view for view in observation.seats if view.seat == observation.seat)


def opponents_in_hand(observation: Observation) -> int:
    return sum(view.in_hand for view in observation.seats if view.seat != observation.seat)


def current_bet(observation: Observation) -> int:
    """The bet to match on this street."""
    return my_seat(observation).bet + observation.legal.to_call


def street_history(observation: Observation) -> list[ActionRecord]:
    return [record for record in observation.history if record.street == observation.street]
