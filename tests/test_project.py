import importlib.util
import json
import sys
from pathlib import Path

import pytest
from ac2.runtime import FunctionCall, Message, Reasoning

from vals_tools.score import summarize
from vals_tools.tasks import GAME_SPLITS, SPLITS, Opponents, datasets, game_datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "opsd"), str(ROOT / "analysis")]

import build_dataset  # noqa: E402
import game_stats  # noqa: E402
import train as opsd_train  # noqa: E402

OPPONENT = Opponents("my-project/opponent")


def test_splits_share_no_deal_and_cycle_every_seat() -> None:
    built = datasets("p", OPPONENT)
    seeds = [t.env_params["seed"] for name, tasks in built.items() if not name.endswith("table-smoke") for t in tasks]
    assert len(seeds) == len(set(seeds)) == sum(n for n, _ in SPLITS.values())
    train = built["p-train"]
    assert {t.env_params["policy_seats"][0] for t in train[:6]} == set(range(6))
    params = train[0].env_params
    assert params["seats"] == 6 and params["opponent_model"] == "my-project/opponent" and params["ev_labels"]
    assert params["opponent_api"] == "responses" and params["opponent_reasoning_effort"] == "medium"
    assert params["preset"] == "event" and built["p-table-smoke"][0].env_params["preset"] == "event"
    assert params["total_hands"] == 1 and "chat_template_kwargs" not in params  # thinking comes from the preset
    assert "policy_seats" not in built["p-table-smoke"][0].env_params  # every seat is the evaluated model



def test_game_splits_are_full_event_games_on_their_own_deals() -> None:
    games = game_datasets("p", OPPONENT)
    hands = {t.env_params["seed"] for tasks in datasets("p", OPPONENT).values() for t in tasks}
    seeds = [t.env_params["seed"] for tasks in games.values() for t in tasks]
    assert len(seeds) == len(set(seeds)) == sum(n for n, _, _ in GAME_SPLITS.values())
    assert not hands & set(seeds)  # no game replays a single-hand deal
    tasks = games["p-games"]
    assert [t.env_params["policy_seats"][0] for t in tasks[:6]] == list(range(6))
    params = tasks[0].env_params
    assert params["preset"] == "event" and params["total_hands"] == 30
    assert {t.env_params["total_hands"] for t in games["p-games-10x10h"]} == {10}
    assert {t.env_params["total_hands"] for t in games["p-games-10x30h"]} == {30}
    # Only the game length differs from the preset: stacks carry over, the scratchpad persists, seats can bust.
    assert "reset_stacks" not in params and "scratchpad" not in params and "ranking" not in params


def test_game_place_follows_elimination_ranking() -> None:
    stacks = game_stats.parse_stacks("Pot: seat2 won 40.\nStacks now: seat0 0 (out), seat1 300, seat2 300, seat3 600.")
    assert stacks == {"seat0": 0, "seat1": 300, "seat2": 300, "seat3": 600}
    assert game_stats.place("seat1", [(30, stacks)]) == (2, None)  # tied survivors share the better place
    before = {"seat0": 50, "seat1": 0, "seat2": 90, "seat3": 100}
    after = {"seat0": 0, "seat1": 0, "seat2": 0, "seat3": 240}
    # seat0 and seat2 bust on the same hand: the bigger stack before it ranks higher; survivors rank above both
    assert game_stats.place("seat0", [(11, before), (12, after)]) == (3, 12)
    assert game_stats.place("seat2", [(11, before), (12, after)]) == (2, 12)

def row(ev: float) -> dict:
    return {"ev_bb": ev, "realized_bb": ev, "policy_format_errors": 0}


def test_summary_pairs_by_deal() -> None:
    rows = {"a": row(-0.5), "b": row(4.0), "c": row(2.0)}
    base = {"a": row(-3.0), "b": row(1.0), "c": row(1.0)}
    out = summarize(rows, base)
    assert out["hands"] == 3 and out["ev_bb_per_hand"] == pytest.approx(5.5 / 3, abs=1e-3)
    assert out["vs_base"]["paired_hands"] == 3 and out["vs_base"]["diff"] == pytest.approx(6.5 / 3, abs=1e-3)


def test_opsd_builder_makes_one_hinted_task_per_clean_decision() -> None:
    system = Message(role="system", content="rules")
    user = Message(role="user", content="state")
    good = [Reasoning(content="think"), FunctionCall(call_id="c1", name="submit_action", arguments='{"action":"fold"}')]
    bad = [Message(role="assistant", content="no tool call")]

    def span(start: int, reply: list) -> dict:
        dump = lambda items: json.dumps([i.model_dump(mode="json") for i in items])  # noqa: E731
        return {
            "name": "get_completion",
            "status": "OK",
            "start_time": start,
            "input": dump([system, user]),
            "output": dump(reply),
            "tokens_out": 10,
        }

    tasks, stats = build_dataset.decision_tasks({"trace_id": "t", "spans": [span(1, good), span(2, bad)]}, "hint", 100)
    assert stats == {"decisions": 2, "kept": 1, "cleanup": 0, "format": 1, "truncated": 0}
    (task,) = tasks
    (selection,) = task.opsd_hint_selections
    assert len(task.input) == 4 and selection.rollout_prefix_len == 2
    assert selection.mutation.item_boundary == 1 and selection.mutation.action == "append_to_previous_item"
    assert [t["name"] for t in task.env_params["tools"]] == ["submit_action", "update_scratchpad"]
    assert set(task.env_params["tools"][0]["parameters"]["properties"]) == {"action", "amount", "say"}
    assert "hint" not in json.dumps([i.model_dump(mode="json") for i in task.input])  # only the teacher sees it


def test_training_configs_send_only_explicit_settings() -> None:
    """Unset flags are not sent, so AC2 keeps its defaults; a flag that is set shows up in the configuration."""
    spec = importlib.util.spec_from_file_location("rl_train", ROOT / "rl" / "train.py")
    grpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grpo)
    config = grpo.run_config(grpo.parser().parse_args([]), "n").to_json()
    text = json.dumps(config, default=str)
    assert "dr-grpo" in text and "pokerbots-train" in text and "pokerbots-select" in text
    assert "learning_rate" not in text
    # The hand's token budget is set explicitly: AC2's default (8,192) runs out on multi-decision hands.
    assert config["experiment_cfg"]["rollout"]["max_response_len"] == 24576
    # Replica counts are set: AC2's default for this model (4 + 4 replicas) needs more GPUs than a team's quota.
    assert config["experiment_cfg"]["rollout"]["n_inference_replicas"] == 1
    assert config["experiment_cfg"]["eval"]["n_inference_replicas"] == 1
    tuned = grpo.run_config(grpo.parser().parse_args(["--lr", "1e-6", "--max-response-len", "10000"]), "n").to_json()
    assert tuned["experiment_cfg"]["trainer"]["learning_rate"] == 1e-6
    assert tuned["experiment_cfg"]["rollout"]["max_response_len"] == 10000

    opsd = opsd_train.run_config(opsd_train.parser().parse_args(["--dataset", "d"]), "n").to_json()
    text = json.dumps(opsd, default=str)
    assert "offline" in text and '"num_train_steps": 20' in text
    assert "learning_rate" not in text and "max_response_len" not in text
    assert opsd["experiment_cfg"]["rollout"] == {"n_inference_replicas": 0}


def test_every_deployment_is_exactly_one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """One replica, never scaled out, and no flag to ask for more."""
    from vals_tools import deploy

    assert (deploy.SCALING.min_replicas, deploy.SCALING.max_replicas) == (1, 1)
    monkeypatch.setattr(sys, "argv", ["deploy.py", "--name", "x", "--model", "m", "--replicas", "2"])
    with pytest.raises(SystemExit) as exit_info:
        deploy.main()
    assert exit_info.value.code == 2  # argparse rejects the unknown flag before anything reaches AC2
