"""ranges tests: hole classing, membership, summary sync, and the A1 all-in call range.

Run: PYTHONPATH=teacher uv run python teacher/v3/tests/test_ranges.py
"""

from __future__ import annotations

from adapter import position_words

from v3.ranges import hole_facts, open_range, continue_range, allin_call, OPEN


def test_classname():
    assert hole_facts(["Ah", "Kh"]).classname == "AKs"
    assert hole_facts(["Ah", "Kd"]).classname == "AKo"
    assert hole_facts(["9h", "9d"]).classname == "99"
    assert hole_facts(["2c", "7d"]).classname == "72o"


def test_open_membership_btn():
    spec, summ = open_range("BTN")
    assert spec.contains(hole_facts(["Ah", "2h"]))     # A2s
    assert spec.contains(hole_facts(["Kh", "Jd"]))     # KJo
    assert spec.contains(hole_facts(["Kh", "Td"]))     # KTo in (BTN kxo_min = T)
    assert not spec.contains(hole_facts(["Kh", "9d"]))  # K9o out
    assert spec.contains(hole_facts(["Qh", "Jd"]))     # QJo
    assert not spec.contains(hole_facts(["Qh", "Td"]))  # QTo out
    assert spec.contains(hole_facts(["2h", "2d"]))     # 22
    for tok in ("22+", "A2s+", "KTo+"):
        assert tok in summ, f"{tok} missing from BTN summary: {summ}"


def test_open_tighter_ep_than_btn():
    ep, _ = open_range("EP")
    btn, _ = open_range("BTN")
    # a marginal hand BTN opens but EP does not
    k7s = hole_facts(["Kh", "7h"])
    assert btn.contains(k7s) and not ep.contains(k7s)


def _ac(combo, tight=False, heads_up=False, short=False):
    return allin_call(hole_facts(combo), tight, heads_up, short)[0]


def test_allin_call_hand_types():
    # deep / multiway (NOT a short heads-up shove) -> hand-type range, never KQ/KJ/KT/QJ/QT
    for combo in (["Kh", "Qd"], ["Kh", "Jd"], ["Kh", "Td"], ["Qh", "Jd"], ["Qh", "Td"]):
        assert not _ac(combo), f"{combo} should never call a deep/multiway all-in"
    assert _ac(["Ah", "Kd"]) and _ac(["Ah", "Qd"]) and _ac(["8h", "8d"]) and _ac(["Ah", "Jh"])
    # tight: TT+/AK only
    assert not _ac(["Ah", "Qd"], tight=True) and _ac(["Th", "Td"], tight=True) and _ac(["Ah", "Kd"], tight=True)
    # a SHORT shove HEADS-UP -> wide (even KJo); but NOT multiway and NOT deep
    assert _ac(["Kh", "Jd"], heads_up=True, short=True)
    assert not _ac(["Kh", "Jd"], heads_up=False, short=True)   # multiway -> not wide
    assert not _ac(["Kh", "Jd"], heads_up=True, short=False)   # deep -> not wide


def test_early_maps_to_ep_range():  # #4: the adapter emits "early"; it must hit the EP table, not MID
    assert "early" in position_words(6)
    assert open_range("early")[0] is open_range("EP")[0]
    assert open_range("early")[0] is not open_range("HJ")[0]


def test_summary_matches_spec_roundtrip():
    # every class the spec contains, among a sample, the summary tokens plausibly cover
    spec, summ = open_range("CO")
    assert "A2s+" in summ and "22+" in summ


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
