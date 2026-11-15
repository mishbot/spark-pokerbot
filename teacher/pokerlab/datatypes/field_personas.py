"""The field personas. Sampled personas draw each knob from a range, per (game, seat)."""

from pokerlab.datatypes.persona import Persona

SURVIVAL = Persona(
    name="field-survival",
    looseness=0.85,
    call_bias=0.0,
    raise_bias=0.0,
    risk_aversion=1.5,
    temperature=0.03,
    survival=True,
)

EXPLOITER = Persona(
    name="field-exploiter",
    looseness=1.0,
    call_bias=0.0,
    raise_bias=0.0,
    risk_aversion=0.3,
    temperature=0.05,
    adaptive=True,
)

PRESSURE = Persona(
    name="field-pressure",
    looseness=1.3,
    call_bias=0.0,
    raise_bias=0.08,
    risk_aversion=0.2,
    temperature=0.06,
    adaptive=True,
)

# The SFT teacher and the eval's scorer: the survival persona, always playing its best action.
TEACHER = Persona(
    name="teacher",
    looseness=SURVIVAL.looseness,
    call_bias=SURVIVAL.call_bias,
    raise_bias=SURVIVAL.raise_bias,
    risk_aversion=SURVIVAL.risk_aversion,
    temperature=SURVIVAL.temperature,
    survival=True,
    greedy=True,
)

SAMPLED_NAME = "field-sampled"
# (low, high) for each knob of a sampled persona, drawn uniformly.
SAMPLED_RANGES = {
    "looseness": (0.7, 1.5),
    "call_bias": (-0.05, 0.10),
    "raise_bias": (-0.05, 0.15),
    "risk_aversion": (0.0, 0.5),
    "temperature": (0.03, 0.10),
}

FIXED_PERSONAS = {persona.name: persona for persona in (SURVIVAL, EXPLOITER, PRESSURE, TEACHER)}
