"""sim/: our bots play legally and to their rules, and simulate.py runs games end to end against a fake gateway.

uv run --project harness pytest sim
"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from poker_harness.completions import table_rules
from poker_harness.engine.table import DecisionContext, LegalActions, SeatView
from poker_harness.game import play_session

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate  # noqa: E402
from bots import BOTS, PLAYABLE, PREMIUM, hand_class, made_hand, make_bot  # noqa: E402


def ctx(hole, board=(), call=0, to_match=2, raise_type="raise", street="preflop", pot=3, stack=198) -> DecisionContext:
    seats = [SeatView(0, "seat0", stack, 0, ""), SeatView(1, "seat1", 200 - to_match, to_match, "")]
    legal = LegalActions(call == 0, call == 0, call, raise_type, 2 * to_match, stack)
    return DecisionContext(1, 30, street, 0, 0, 5, list(hole), list(board), pot, seats, [], legal)


def decide(kind, c):
    return asyncio.run(make_bot(kind, "seat1").decide(c, ""))


def test_hand_classes_and_ranges() -> None:
    assert hand_class(["Ks", "Ah"]) == "AKo" and hand_class(["9d", "Td"]) == "T9s" and hand_class(["7c", "7d"]) == "77"
    assert "AA" in PREMIUM and "TT" in PREMIUM and "99" in PLAYABLE and not PREMIUM & PLAYABLE


def test_made_hand() -> None:
    assert made_hand(["As", "Kd"], ["Ah", "7c", "2d"]) == "pair"  # top pair
    assert made_hand(["Qs", "Qd"], ["Jh", "7c", "2d"]) == "pair"  # overpair
    assert made_hand(["7s", "6d"], ["Ah", "7c", "2d"]) == "weak"  # middle pair
    assert made_hand(["9s", "8d"], ["Kh", "Kc", "2d"]) == "weak"  # the board's pair isn't ours
    assert made_hand(["Ks", "2s"], ["Kh", "7c", "2d"]) == "strong"  # two pair
    assert made_hand(["9s", "8d"], ["7h", "6c", "5d"]) == "strong"  # straight


def test_shover_shoves_pairs_and_aces_and_folds_junk() -> None:
    facing = {"call": 40, "to_match": 40}
    for hole in (["2c", "2d"], ["As", "3d"]):
        d = decide("shover", ctx(hole, **facing))
        assert d.type == "raise" and d.amount == 198
    assert decide("shover", ctx(["9c", "4d"], **facing)).type == "fold"


def test_tag_raises_premiums_opens_playables_and_folds_junk() -> None:
    assert decide("tag", ctx(["As", "Ad"], call=2)).amount == 6  # open to 3 BB
    assert decide("tag", ctx(["As", "Ad"], call=10, to_match=12)).amount == 36  # re-raise 3x
    assert decide("tag", ctx(["8s", "8d"], call=2)).amount == 6
    assert decide("tag", ctx(["8s", "8d"], call=10, to_match=12)).type == "call"  # cheap raise
    assert decide("tag", ctx(["8s", "8d"], call=100, to_match=102)).type == "fold"  # expensive raise
    assert decide("tag", ctx(["7s", "2d"], call=2)).type == "fold"
    flop = {"board": ["Ah", "7c", "2d"], "street": "flop", "pot": 20}
    assert decide("tag", ctx(["As", "Kd"], **flop)).amount == 10  # bet half pot with top pair
    assert decide("tag", ctx(["9s", "8d"], call=20, to_match=20, **flop)).type == "fold"


def test_every_bot_plays_a_full_event_game_without_an_illegal_move() -> None:
    rules = table_rules(simulate.event_params(7, None))
    for seed in range(4):
        players = [make_bot(kind, f"seat{i}", seed) for i, kind in enumerate(BOTS)]
        log = asyncio.run(play_session(f"t{seed}", seed, players, rules))
        actions = [a for h in log["hands"] for a in h["actions"]]
        assert actions and not [a for a in actions if a["model_response"].get("fallback_used")]
        assert sorted(log["ranking"]["places"]) == sorted(p.name for p in players)


def test_lineup_rotates_and_entries_resolve() -> None:
    lineup = ["a", "b", "c", "d", "e"]
    assert simulate.lineup_for(lineup, 0) == lineup and simulate.lineup_for(lineup, 6) == ["b", "c", "d", "e", "a"]
    assert simulate.model_for("tag", "p/x") is None and simulate.model_for("model", "p/x") == "p/x"
    assert simulate.model_for("house", "p/x") == simulate.HOUSE_MODEL and simulate.model_for("p/y", "p/x") == "p/y"
    with pytest.raises(ValueError):
        simulate.model_for("gto", "p/x")


def gateway() -> FastAPI:
    """Every model seat calls, and updates its scratchpad after each hand."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        last = str(body["messages"][-1].get("content", ""))
        name, arguments = (
            ("update_scratchpad", {"contents": "n"}) if "is over" in last else ("submit_action", {"action": "call"})
        )
        call = {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return {
            "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [call]}}],
            "usage": {},
        }

    return app


def test_simulate_runs_games_and_writes_a_scorecard(monkeypatch, tmp_path, capsys) -> None:
    real = httpx.AsyncClient

    class Client(real):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.ASGITransport(app=gateway())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(simulate.httpx, "AsyncClient", Client)
    monkeypatch.setattr(simulate, "GATEWAY_URL", "http://gw/v1")
    monkeypatch.setenv("AC2_API_KEY", "k")
    argv = [
        "simulate.py",
        "--model",
        "p/entry",
        "--lineup",
        "shover,tag,model,nit,maniac",
        "--opponent-model",
        "p/base",
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--games", "3", "--hands", "4", "--out", str(tmp_path)])
    simulate.main()

    out = capsys.readouterr().out
    assert "p/entry over 3 games" in out and "failed decisions 0/" in out
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["summary"]["games"] == 3 and len(list(tmp_path.glob("sim-*.json"))) == 3
    assert {g["lineup"]["seat1"] for g in saved["games"]} == {"shover", "tag", "model"}  # rotated

    for g in saved["games"]:  # chips credited to opponents add up to seat 0's net
        assert sum(g["chips_by_opponent"].values()) == pytest.approx(g["net_bb"] * 2)

    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["simulate.py", "--report", str(tmp_path)])
    simulate.main()
    assert "p/entry over 3 games" in capsys.readouterr().out

    first = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("sim-*.json")}
    monkeypatch.setattr(sys, "argv", [*argv, "--games", "4", "--hands", "4", "--out", str(tmp_path)])  # 3 reused
    simulate.main()
    out = capsys.readouterr().out
    assert out.count("reusing") == 3 and "p/entry over 4 games" in out
    assert all(tmp_path.joinpath(n).stat().st_mtime_ns == t for n, t in first.items())
