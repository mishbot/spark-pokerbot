"""A field bot: the shared range engine, played by one persona."""

from pokerlab.bots.field.postflop import decide_postflop
from pokerlab.bots.field.preflop import decide_preflop
from pokerlab.bots.field.risk import preflop_persona, risk_aversion
from pokerlab.bots.field.tracker import Tracker
from pokerlab.bots.player import Player
from pokerlab.bots.randomness import decision_rng
from pokerlab.datatypes.actions import Decision
from pokerlab.datatypes.observation import PREFLOP, HandSummary, Observation
from pokerlab.datatypes.persona import Persona


class FieldBot(Player):
    def __init__(self, persona: Persona, key: str) -> None:
        """`key` comes from bots.randomness.seat_key."""
        super().__init__(persona.name)
        self.persona = persona
        self._key = key
        self._tracker = Tracker()

    def hand_finished(self, summary: HandSummary) -> None:
        if self.persona.adaptive:
            self._tracker.record(summary)

    def decide(self, observation: Observation) -> Decision:
        persona = self.persona
        rng = decision_rng(self._key, observation)
        if observation.street == PREFLOP:
            return Decision(decide_preflop(observation, preflop_persona(persona, observation), rng))
        risk = risk_aversion(persona, observation)
        looseness = self._looseness(observation)
        return Decision(decide_postflop(observation, persona, risk, looseness, self._fold_rate, rng))

    def _looseness(self, observation: Observation) -> dict[int, float]:
        if not self.persona.adaptive:
            return {}
        return {view.seat: self._tracker.looseness(view.seat) for view in observation.seats}

    def _fold_rate(self, seat: int, prior: float) -> float:
        return self._tracker.fold_rate(seat, prior) if self.persona.adaptive else prior
