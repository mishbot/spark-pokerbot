"""Replay a hand's public history into ActionEvents, tracking the pot and street bets."""

from collections.abc import Iterable

from pokerlab.datatypes.observation import ActionRecord
from pokerlab.datatypes.replay import ActionEvent

BLIND_POST = "post"
AGGRESSIVE = ("bet", "raise")


def replay(history: Iterable[ActionRecord]) -> list[ActionEvent]:
    events = []
    pot = 0
    street = None
    street_bets: dict[int, int] = {}
    raises = 0
    for record in history:
        if record.street != street:
            street, street_bets, raises = record.street, {}, 0
        current = max(street_bets.values(), default=0)
        mine = street_bets.get(record.seat, 0)
        to_call = current - mine

        if record.kind == BLIND_POST:
            street_bets[record.seat] = mine + record.amount
            pot += record.amount
            continue

        fraction = 0.0
        if record.kind == "call":
            street_bets[record.seat] = mine + record.amount
            pot_change = record.amount
        elif record.kind in AGGRESSIVE:
            fraction = (record.amount - current) / max(pot + to_call, 1)
            pot_change = record.amount - mine
            street_bets[record.seat] = record.amount
        else:
            pot_change = 0

        amount = record.amount or 0
        events.append(
            ActionEvent(record.street, record.seat, record.kind, fraction, raises, amount, to_call, pot)
        )
        pot += pot_change
        raises += record.kind in AGGRESSIVE
    return events
