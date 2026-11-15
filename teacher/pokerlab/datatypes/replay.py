"""One public action with the context needed to read it: how big it was and what it faced."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionEvent:
    """A history entry after replaying the hand up to it.

    `kind` is "fold", "check", "call", "bet" or "raise" (blind posts are skipped).
    `pot_fraction` is the size of a bet or raise increment relative to the pot it
    faced, after calling; 0 for other actions.
    `raises_before` counts bets and raises earlier on the same street.
    """

    street: str
    seat: int
    kind: str
    pot_fraction: float
    raises_before: int
    amount: int  # as logged: chips added by a call, the raise-to total for a bet or raise, else 0
    to_call: int  # chips the seat had to add to call, before acting
    pot_before: int  # chips in the middle before the action
