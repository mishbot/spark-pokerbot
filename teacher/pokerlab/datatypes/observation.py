"""What one player sees when it is their turn to act.

An Observation is plain structured data. Bots read it directly, and the
model seat turns it into prompt text with one separate function, so the
prompt format can change without touching the table.
"""

from dataclasses import dataclass

STREETS = ("preflop", "flop", "turn", "river")
PREFLOP = STREETS[0]

# How each kind of history entry reads in text. "post" is a blind and "bet"
# is the first raise on a street; the rest are ActionKind values.
HISTORY_VERBS = {
    "post": "posts",
    "fold": "folds",
    "check": "checks",
    "call": "calls",
    "bet": "bets",
    "raise": "raises to",
}


@dataclass(frozen=True)
class SeatView:
    """One seat as everyone at the table can see it.

    There is deliberately no player name: bot names reveal their style
    ("loose-aggressive"), which no real opponent would announce.
    """

    seat: int
    position: str
    stack: int
    bet: int  # chips put in on the current street
    in_hand: bool  # has not folded
    all_in: bool


@dataclass(frozen=True)
class ActionRecord:
    """One entry in the hand's public action history.

    `kind` is a key of HISTORY_VERBS.
    `amount` is the blind posted, the chips added by a call, or the raise-to total.
    """

    street: str
    seat: int
    kind: str
    amount: int | None = None


@dataclass(frozen=True)
class LegalActions:
    """The actions available right now. Helpers that use it live in poker/legal.py."""

    can_fold: bool
    to_call: int  # 0 means checking is allowed
    min_raise_to: int | None  # None means raising is not allowed
    max_raise_to: int | None


@dataclass(frozen=True)
class Observation:
    """The state of the hand from the point of view of the seat to act.

    Helpers that read it live in poker/situation.py.
    """

    hand_number: int
    hand_count: int
    seat: int
    position: str
    hole_cards: tuple[str, ...]
    board: tuple[str, ...]
    street: str
    pot: int  # every chip in the middle, including bets on this street
    big_blind: int
    seats: tuple[SeatView, ...]
    history: tuple[ActionRecord, ...]
    legal: LegalActions


@dataclass(frozen=True)
class HandSummary:
    """What every player at the table learns when a hand ends.

    Only public information: the action, the board, the cards shown at
    showdown, and who won what. Players that adapt to their opponents
    read this; folded hands stay hidden.
    """

    hand_number: int
    seats: tuple[int, ...]  # dealing order, button last
    positions: dict[int, str]
    board: tuple[str, ...]
    history: tuple[ActionRecord, ...]
    shown_cards: dict[int, tuple[str, ...]]  # seats that reached showdown
    payoffs: dict[int, int]  # chips won or lost this hand
