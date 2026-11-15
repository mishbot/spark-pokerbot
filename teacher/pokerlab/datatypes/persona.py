"""How a field bot plays: one shared engine (bots/field/), different personas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """Knobs of the field engine.

    After the flop a bot scores each action by its chip EV plus a bias, minus a
    penalty for risking its stack, and samples from a softmax of the scores:
        U[a] = EV[a] + bias[a] * pot - risk_aversion * P(lose) * risked * (risked / stack)
        P(a) ∝ exp(U[a] / (temperature * pot))
    """

    name: str
    looseness: float  # scales every preflop band; 1 plays the standard ranges
    call_bias: float  # added to the call score, in pots
    raise_bias: float  # added to every raise score, in pots
    risk_aversion: float  # λ above; survival bots scale it by how many players they would fall behind
    temperature: float  # κ above; higher plays more mixed
    survival: bool = False  # scales risk aversion by rank and folds out a safe lead late in the game
    adaptive: bool = False  # reads opponents from earlier hands (looseness and fold rates)
    greedy: bool = False  # always plays its best action: no mixing (the teacher, for clean labels)
