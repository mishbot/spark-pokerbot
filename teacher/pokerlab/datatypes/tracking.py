"""What an adaptive bot remembers about each opponent across hands."""

from dataclasses import dataclass


@dataclass
class SeatCounts:
    """Running counts for one seat. Mutable: the tracker adds to it after every hand."""

    hands: int = 0
    voluntary: int = 0  # put chips in preflop without being forced (VPIP)
    preflop_raises: int = 0  # PFR
    faced_bet: int = 0  # times the seat faced a bet or raise after the flop
    folded_to_bet: int = 0
