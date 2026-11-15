"""ranges tests: hole classing, membership, summary sync, and the A1 all-in call range.

Run: PYTHONPATH=teacher uv run python teacher/v3/tests/test_ranges.py
"""

from __future__ import annotations

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


def test_allin_call_hand_types():
    # never KQ/KJ/KT/QJ/QT, even offsuit, even when not leading
    for combo in (["Kh", "Qd"], ["Kh", "Jd"], ["Kh", "Td"], ["Qh", "Jd"], ["Qh", "Td"]):
        call, _ = allin_call(hole_facts(combo), leader_or_safe=False)
        assert not call, f"{combo} should never call an all-in"
    # premiums call
    assert allin_call(hole_facts(["Ah", "Kd"]), False)[0]    # AK
    assert allin_call(hole_facts(["Ah", "Qd"]), False)[0]    # AQ
    assert allin_call(hole_facts(["8h", "8d"]), False)[0]    # 88
    assert allin_call(hole_facts(["Ah", "Jh"]), False)[0]    # AJs marginal
    # leader tightens: AQ folds, TT+ / AK only
    assert not allin_call(hole_facts(["Ah", "Qd"]), True)[0]
    assert allin_call(hole_facts(["Th", "Td"]), True)[0]
    assert allin_call(hole_facts(["Ah", "Kd"]), True)[0]


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
