"""How much a persona fears losing its stack in this spot.

Survival personas scale risk aversion by the share of players below them: busting
gives up those places. With a safe lead late in the game they become very
risk-averse rather than folding everything, so they still bet a near-certain
winner (the penalty scales with 1 - equity) and fold anything marginal.
The frozen eval scores answers with this same function.
"""

from dataclasses import replace

from pokerlab.bots.field.standing import lead_is_safe, share_below
from pokerlab.datatypes.observation import Observation
from pokerlab.datatypes.persona import Persona

SAFE_LEAD_RISK_MULTIPLIER = 20.0
SAFE_LEAD_LOOSENESS = 0.15  # preflop: only premium hands


def risk_aversion(persona: Persona, observation: Observation) -> float:
    if not persona.survival:
        return persona.risk_aversion
    scaled = persona.risk_aversion * share_below(observation)
    return scaled * SAFE_LEAD_RISK_MULTIPLIER if lead_is_safe(observation) else scaled


def preflop_persona(persona: Persona, observation: Observation) -> Persona:
    """With a safe lead, a survival persona plays only premium hands preflop."""
    if persona.survival and lead_is_safe(observation):
        return replace(persona, looseness=persona.looseness * SAFE_LEAD_LOOSENESS)
    return persona
