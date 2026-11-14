# Architecture

The harness plays games on our machine; AC2 serves the model, records every call, and trains. A connector joins them over an **outbound** connection, so the harness never needs a public address.

```
our machine                                   AC2 (project vals-hackathon)
┌──────────────────────────────┐              ┌───────────────────────────────────┐
│ harness (+ shard router)     │◀── local ──▶ │                                   │
│ 127.0.0.1:8765               │    HTTP      │ training run / eval               │
│                              │              │   sends tasks, reads scores       │
│ connector                    │── outbound ─▶│ relay: our seat's calls → model    │
│ `ac2 harness connect`        │              │   being trained/evaluated; records │
└──────────────────────────────┘              │ gateway: opponent seats →          │
                                              │   vals-hackathon/ac-vals-model     │
                                              │ deployments, checkpoints, arena    │
                                              └───────────────────────────────────┘
```

## The parts

| Part | What it is |
| --- | --- |
| Org | The team's AC2 org: members, the 10-GPU quota, everything below. |
| Project | Workspace holding datasets, runs, checkpoints, evals, deployments, code versions, and one connector. `vals-hackathon` is pre-seeded. |
| Project version | Code snapshot AC2 installs per run. Ours is the empty `src/pokerbots/` stub; `.ac2ignore` keeps everything else out. One must exist (`ac2 project init`). |
| Dataset / task | List of tasks; here a task is one game spec: seed, policy seat(s), opponents, `env_params`. |
| Harness | FastAPI server in `harness/`: deals, asks every seat to act, plays out, grades. |
| Connector | `ac2 harness connect`, one per project; forwards AC2's requests to the harness. |
| Shard router | `harness/run_sharded.sh` + `shard_router.py`: front door to `SHARDS` single-process harnesses. |
| Inference deployment | A model on 1 GPU behind an OpenAI-compatible API, named `<project>/<name>`. Base model, checkpoints, and external opponent models are all deployments. |
| Training run | GPU job that plays tasks through the harness and updates weights (GRPO) or learns from a teacher (OPSD). Checkpoints every N steps. |
| Eval | Plays a dataset with a fixed model; keeps scores. |
| Arena | Where submitted deployments play each other; submission happens in the AC2 console. |

## One rollout

1. AC2 picks a task and sends it through the connector to the harness (`/submit`).
2. The harness deals. On the policy seat's turn it calls AC2's **relay**, which routes to the model under training/eval and records the call. The five opponent seats call the opponent model through AC2's **gateway** with `AC2_API_KEY`.
3. The hand plays out; the seat gets its cleanup turn.
4. AC2 asks the harness to grade (`/submit_grading` … `/collect_score`); the sample grader returns the reward ([rewards.md](rewards.md)).
5. Training uses the score as the reward; an eval just keeps it.

## Harness endpoints

All POST except `/health`. Rollout lifecycle: `/submit` → `/get_status` → `/collect_outputs`. Grader lifecycle: `/submit_grading` → `/get_grading_status` → `/collect_score`.

- `/submit` is idempotent on `rollout_id` (body or `Idempotency-Key`); `env_params.seed` is required (422 otherwise).
- `/submit_grading` requires `Idempotency-Key: <rollout_id>:grading`; one grading per rollout.
- `/collect_score` returns `{"score", "reasoning", "artifacts"}`, 409 until grading completes.
- `/collect_outputs` returns `log`, `rules`, `policy_seats`, `error`, bounded to **1 MiB** (the connector's per-rollout limit). The harness trims multi-hand logs to fit: scratchpad tables keep each seat's final scratchpad and cleanup counts, then model text goes first. Grading reads the full log.
- `POKER_HARNESS_TOKEN`, if set, requires `Authorization: Bearer <token>` on every request; the connector sends it via `--upstream-token`.

Other env: `AC2_API_KEY` (opponent seats; unneeded in pure self-play), `POKER_OPPONENT_BASE_URL` / `POKER_OPPONENT_API_KEY` (any OpenAI-compatible opponent server), `POKER_HARNESS_RUNS` (log dir, default `runs`).

## Why shards, not workers

Tables live in process memory, so every call for a rollout must reach the process that ran its `/submit`. `uvicorn --workers` breaks that. The router starts `SHARDS` (default 4) servers on `SHARD_BASE_PORT + i` (default 9000) and routes by `hash(rollout_id) % SHARDS`, keyed on the body's `rollout_id` or else the `Idempotency-Key`. `/health` aggregates shards; dead shards restart. One shard handles a few dozen concurrent tables.

Consequence: **restarting the harness loses in-flight tables and their logs.** Grading then fails with `unknown rollout`. Restart only between runs.

## Game logs

Every finished table is written to `harness/runs/<rollout_id>.json` (gitignored). Top level: `game_id`, `session_info`, `hands`, `final_stacks`, `net_chips`, `ranking`, `eliminated_on_hand`, `complete`, `private`. Each action carries `player`, `seat`, `street`, `action`, `chip_change`, `stack_after`, `pot_before`, `say`, and a `model_response` with `submitted_by_tool`, `format_error`, `error`, token counts, and `private` (`reasoning`, `say_raw`).

- Failed decisions add `format_error`, `fallback_used`, `error`, `afk`, `afk_reason`.
- Privacy boundary: the model's own text is under `model_response.private`; scratchpads and cleanup turns are under `log["private"][player]`. Nothing private lives anywhere else, so stripping those two keys makes a log publishable.
- Event tables also log per hand `blinds`, `stacks_before`, `blinds_posted`, `pot_won`, `eliminated`.
- Evaluator-only labels models never see: per-hand `ev` (`ev.py`, when `ev_labels` is on) and `fold_regret_bb` on folds (`labels.py`: value of calling vs. the live hands, floored at 0). Useful for analysis and reward design.

## Engine parity

`harness/` plays exactly like the event harness (Applied Compute reference `954d83651`) minus research-only code. The engine is a port of Vals' engine on PokerKit; `harness/tests/test_engine_parity.py` replays 2,000+ seeded hands against recorded digests. A change that moves those digests changes game behaviour and diverges from the arena.

## Testing without a model

`poker_harness.engine.ScriptedPlayer` plays a seat with archetypes `random`, `maniac`, `nit`, `calling_station`. Use it in tests (pattern: `harness/tests/test_harness.py`), via `play_local.py --opponents bots`, or with `poker_harness.game.play_session` directly:

```python
import asyncio, random
from poker_harness.completions import table_rules, with_preset
from poker_harness.engine import ScriptedPlayer
from poker_harness.game import play_session

params = with_preset({"preset": "event", "seed": 7, "total_hands": 2})
players = [ScriptedPlayer(f"seat{i}", random.Random(i).choice(["nit", "calling_station"]), 7) for i in range(6)]
log = asyncio.run(play_session("local-7", 7, players, table_rules(params)))
```

Run from `harness/` with `uv run python`. To test a modified harness while a run is live, start a second one on another port.
