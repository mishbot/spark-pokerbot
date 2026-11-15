"""Preflop position groups.

Trimmed vendor copy: only `position_group` and its table are used by the field
teacher. The upstream file also parses hand-range notation with pokerkit
(`in_range`/`_parse`); that is outside the teacher's decision path and is left
out so the vendored teacher has no pokerkit dependency.
"""

# Which position names belong to which group, for choosing an opening range.
POSITION_GROUPS = {
    "SB": "blinds",
    "BB": "blinds",
    "BTN": "late",
    "CO": "late",
    "HJ": "middle",
    "LJ": "middle",
}


def position_group(position: str) -> str:
    """The group for an opening range. UTG, UTG+n and MP are "early"."""
    return POSITION_GROUPS.get(position, "early")
