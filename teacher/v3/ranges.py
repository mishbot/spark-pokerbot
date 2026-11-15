"""Preflop class-range tables: the SINGLE source of truth for both the decision and the trace.

A range is built from family thresholds (pairs, suited/offsuit aces, kings, queens, broadways,
suited connectors). The SAME spec produces (a) the membership test and (b) the human summary
string the trace prints ("22+, A2s+, A9o+, KTs+, KJo+, QTs+, JTs, T9s, 98s"). So the rule the
trace states is exactly the rule the decision used (fixes #4). Ranges are standard-ish 6-max and
are tunable later via the gameplay A/B; the architecture, not the exact cutoffs, is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

RANKS_DESC = "AKQJT98765432"
RVAL = {r: 12 - i for i, r in enumerate(RANKS_DESC)}  # A=12 ... 2=0


@dataclass(frozen=True)
class Hole:
    hi: int          # higher rank value (0..12)
    lo: int          # lower rank value
    suited: bool
    pair: bool

    @property
    def classname(self) -> str:
        hr, lr = RANKS_DESC[12 - self.hi], RANKS_DESC[12 - self.lo]
        if self.pair:
            return hr * 2
        return hr + lr + ("s" if self.suited else "o")


def hole_facts(cards) -> Hole:
    a, b = cards
    ra, rb = RVAL[a[0]], RVAL[b[0]]
    hi, lo = max(ra, rb), min(ra, rb)
    return Hole(hi=hi, lo=lo, suited=a[1] == b[1] and ra != rb, pair=ra == rb)


# A range spec: for each family, the MINIMUM low-card rank value to include (None = family excluded).
# Families key off the higher card: 'Axs' = suited ace (low card >= value), 'Axo' = offsuit ace, etc.
@dataclass(frozen=True)
class RangeSpec:
    pair_min: int | None            # include pairs with rank >= this (0 = 22+)
    axs_min: int | None             # suited ace, kicker >= this
    axo_min: int | None             # offsuit ace, kicker >= this
    kxs_min: int | None
    kxo_min: int | None
    qxs_min: int | None
    qxo_min: int | None
    jxs_min: int | None             # suited jack (JTs etc.)
    suited_conn_min: int | None     # suited connectors (gap 1), include if low card >= this

    def contains(self, h: Hole) -> bool:
        if h.pair:
            return self.pair_min is not None and h.hi >= self.pair_min
        hi, lo, suited = h.hi, h.lo, h.suited
        if hi == RVAL["A"]:
            m = self.axs_min if suited else self.axo_min
            return m is not None and lo >= m
        if hi == RVAL["K"]:
            m = self.kxs_min if suited else self.kxo_min
            return m is not None and lo >= m
        if hi == RVAL["Q"]:
            m = self.qxs_min if suited else self.qxo_min
            return m is not None and lo >= m
        if hi == RVAL["J"] and suited:
            return self.jxs_min is not None and lo >= self.jxs_min
        # suited connectors (gap of exactly 1 rank), e.g. T9s 98s 87s
        if suited and hi - lo == 1 and self.suited_conn_min is not None:
            return lo >= self.suited_conn_min
        return False


def _r(v: int) -> str:
    return RANKS_DESC[12 - v]


def summary(spec: RangeSpec) -> str:
    parts: list[str] = []
    if spec.pair_min is not None:
        parts.append(f"{_r(spec.pair_min)}{_r(spec.pair_min)}+")

    def ace_king(letter, s_min, o_min):
        if s_min is not None:
            parts.append(f"{letter}{_r(s_min)}s+")
        if o_min is not None:
            parts.append(f"{letter}{_r(o_min)}o+")
    ace_king("A", spec.axs_min, spec.axo_min)
    ace_king("K", spec.kxs_min, spec.kxo_min)
    ace_king("Q", spec.qxs_min, spec.qxo_min)
    if spec.jxs_min is not None:
        parts.append(f"J{_r(spec.jxs_min)}s+")
    if spec.suited_conn_min is not None:
        parts.append(f"suited connectors {_r(spec.suited_conn_min)}x+")
    return ", ".join(parts)


# Standard-ish 6-max OPEN ranges (unopened pot), by position word.
_EP = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["2"], axo_min=RVAL["J"], kxs_min=RVAL["T"],
                kxo_min=None, qxs_min=RVAL["J"], qxo_min=None, jxs_min=RVAL["T"], suited_conn_min=RVAL["7"])
_MID = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["2"], axo_min=RVAL["T"], kxs_min=RVAL["9"],
                 kxo_min=RVAL["J"], qxs_min=RVAL["9"], qxo_min=None, jxs_min=RVAL["9"], suited_conn_min=RVAL["6"])
_LATE = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["2"], axo_min=RVAL["8"], kxs_min=RVAL["5"],
                  kxo_min=RVAL["T"], qxs_min=RVAL["7"], qxo_min=RVAL["J"], jxs_min=RVAL["8"], suited_conn_min=RVAL["5"])

OPEN: dict[str, RangeSpec] = {
    "EP": _EP, "HJ": _MID, "CO": _LATE, "BTN": _LATE, "SB": _LATE, "BB": _MID,
}

# TIGHT opens (used in the first 10 hands and whenever we hold the chip lead): drop the loosest
# families from each position. The core player is this tight player.
_T_EP = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["T"], axo_min=RVAL["J"], kxs_min=RVAL["T"],
                  kxo_min=RVAL["Q"], qxs_min=RVAL["T"], qxo_min=None, jxs_min=RVAL["T"], suited_conn_min=None)
_T_MID = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["8"], axo_min=RVAL["T"], kxs_min=RVAL["T"],
                   kxo_min=RVAL["J"], qxs_min=RVAL["T"], qxo_min=None, jxs_min=RVAL["T"], suited_conn_min=RVAL["8"])
_T_LATE = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["2"], axo_min=RVAL["9"], kxs_min=RVAL["9"],
                    kxo_min=RVAL["T"], qxs_min=RVAL["9"], qxo_min=RVAL["J"], jxs_min=RVAL["9"], suited_conn_min=RVAL["6"])
TIGHT_OPEN: dict[str, RangeSpec] = {
    "EP": _T_EP, "HJ": _T_MID, "CO": _T_LATE, "BTN": _T_LATE, "SB": _T_LATE, "BB": _T_MID,
}
_TIGHT_CONTINUE = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["J"], axo_min=RVAL["K"], kxs_min=RVAL["J"],
                            kxo_min=None, qxs_min=RVAL["J"], qxo_min=None, jxs_min=RVAL["T"], suited_conn_min=None)

# Continue (call) vs a single raise; 3bet/raise with the premium set.
_CONTINUE = RangeSpec(pair_min=RVAL["2"], axs_min=RVAL["T"], axo_min=RVAL["Q"], kxs_min=RVAL["T"],
                      kxo_min=None, qxs_min=RVAL["T"], qxo_min=None, jxs_min=RVAL["T"], suited_conn_min=RVAL["7"])
_THREEBET = RangeSpec(pair_min=RVAL["T"], axs_min=RVAL["Q"], axo_min=RVAL["K"], kxs_min=None,
                      kxo_min=None, qxs_min=None, qxo_min=None, jxs_min=None, suited_conn_min=None)


def open_range(position: str, tight: bool = False) -> tuple[RangeSpec, str]:
    spec = (TIGHT_OPEN if tight else OPEN).get(position, _T_MID if tight else _MID)
    return spec, summary(spec)


def continue_range(tight: bool = False) -> tuple[RangeSpec, str]:
    spec = _TIGHT_CONTINUE if tight else _CONTINUE
    return spec, summary(spec)


def threebet_range() -> tuple[RangeSpec, str]:
    return _THREEBET, summary(_THREEBET)


SMALL_SHOVE_NEED = 0.33  # a shove this cheap (good pot odds) is called wide, regardless of tightness


# --- all-in call range (A1), the single source for decision and trace --------
def allin_call(h: Hole, tight: bool, need: float) -> tuple[bool, str]:
    """Decide a preflop all-in CALL. A small shove (good price) is called wide; a deep shove is
    called by HAND TYPE and NEVER KQ/KJ/KT/QJ/QT (the field is ace-heavy and dominates high cards).
    When tight (early or chip-leading) the deep-shove range is only TT+/AK. Returns (call, rule_text)."""
    pairs_ge = lambda letter: h.pair and h.hi >= RVAL[letter]
    is_ak = (not h.pair) and h.hi == RVAL["A"] and h.lo == RVAL["K"]
    is_aq = (not h.pair) and h.hi == RVAL["A"] and h.lo == RVAL["Q"]
    both_broadway = (not h.pair) and h.lo >= RVAL["T"]

    if need <= SMALL_SHOVE_NEED:   # cheap shove: the price justifies a wide call
        rule = f"the shove is small (need {round(need * 100)}%), so call a wide range: any pair, any ace, any two broadway"
        return (h.pair or h.hi == RVAL["A"] or both_broadway), rule

    if tight:
        return (pairs_ge("T") or is_ak), "early or leading, so call only TT+/AK"
    rule = "call 88+/AQ+ and 66-77/ATs-AJs; never KQ/KJ/KT/QJ/QT"
    premium = pairs_ge("8") or is_ak or is_aq
    marginal = (h.pair and RVAL["6"] <= h.hi <= RVAL["7"]) or \
               (h.hi == RVAL["A"] and h.lo in (RVAL["T"], RVAL["J"]) and h.suited)
    return (premium or marginal), rule
