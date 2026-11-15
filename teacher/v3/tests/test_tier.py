"""tier.tier_and_price tests: base tiers, ace-high floor (#8), stability (#7), dangerous cap (RC7).

Run: PYTHONPATH=teacher uv run --with phevaluator python teacher/v3/tests/test_tier.py
"""

from __future__ import annotations

from v3.eval import analyze
from v3.tier import tier_and_price, PRICE


def t(hole, board):
    return tier_and_price(analyze(hole, board))


def test_base_tiers():
    assert t(["9h", "9d"], ["9c", "2s", "7h"])[0] == "strong"   # set
    assert t(["Ah", "Kd"], ["Ac", "Ks", "2h"])[0] == "strong"  # two pair
    assert t(["Ah", "Kd"], ["Ac", "7s", "2h"])[0] == "medium"  # top pair
    assert t(["Qh", "Qd"], ["7c", "5s", "2h"])[0] == "medium"  # overpair
    assert t(["Ah", "7d"], ["Kc", "7s", "2h"])[0] == "weak"    # second pair
    assert t(["Ah", "Qd"], ["9c", "7s", "2h"])[0] == "air"     # ace-high


def test_ace_high_is_air():  # #8
    tier, price, _ = t(["Ah", "Qd"], ["9c", "7s", "2h"])
    assert tier == "air" and price == 0.0


def test_tier_stable_across_streets():  # #7 — same made hand, later streets, same tier
    flop = t(["Ah", "Kd"], ["Ac", "7s", "2h"])          # top pair
    turn = t(["Ah", "Kd"], ["Ac", "7s", "2h", "3d"])    # still top pair (blank)
    assert flop[0] == turn[0] == "medium"


def test_dangerous_cap_lowers_non_feature_hand():  # RC7
    # two pair on a FOUR-FLUSH board (we hold no club) -> capped strong->medium
    tier, price, capped = t(["Ah", "Kd"], ["Ac", "Kc", "2c", "7c"])
    assert capped and tier == "medium", (tier, capped)
    # top pair on a FOUR-FLUSH board (we hold no club) -> capped medium->weak
    tier2, _, capped2 = t(["Ah", "Kd"], ["Ac", "7c", "2c", "9c"])
    assert capped2 and tier2 == "weak", (tier2, capped2)
    # two pair on a PAIRED board (vulnerable to a boat) -> capped strong->medium
    tier3, _, capped3 = t(["Ah", "Kd"], ["Ac", "Ks", "7h", "7c"])
    assert capped3 and tier3 == "medium", (tier3, capped3)


def test_plays_the_board_is_air():  # Opus FAIL #11
    tier, price, _ = t(["9h", "8d"], ["Kd", "7c", "Tc", "7d", "7s"])
    assert tier == "air" and price == 0.0


def test_nut_hand_not_capped_on_dangerous_board():
    # we HAVE the flush on a four-flush board -> strong, not capped
    tier, _, capped = t(["Ah", "Qc"], ["Ac", "Kc", "2c", "7c"])
    assert tier == "strong" and not capped
    # boat on a paired board -> strong, not capped
    tier2, _, capped2 = t(["7h", "7d"], ["Ac", "7s", "Ah"])
    assert tier2 == "strong" and not capped2


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
