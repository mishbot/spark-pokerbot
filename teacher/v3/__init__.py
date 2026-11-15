"""V3 rule-based teacher + trace renderer + independent validator.

The teacher's decision procedure IS the V3 trace: preflop class-range tables and postflop
tier -> price -> need -> edge -> plan. So the trace is logically correct AND implies the action
by construction. `validate.py` re-derives every field independently to prove it.
"""
