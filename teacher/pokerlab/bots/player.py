"""The interface every seat implements."""

from abc import ABC, abstractmethod

from pokerlab.datatypes.actions import Decision
from pokerlab.datatypes.observation import HandSummary, Observation


class Player(ABC):
    """A seat at the table: a scripted bot, an adaptive bot, or a model.

    A Player object lives for one game, so it can remember earlier hands.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def decide(self, observation: Observation) -> Decision:
        """Choose an action for the seat that is to act."""

    def hand_finished(self, summary: HandSummary) -> None:  # noqa: B027 (optional hook, not abstract)
        """Called on every seat dealt into the hand once it ends. Most players ignore it."""
