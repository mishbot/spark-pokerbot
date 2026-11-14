# Datasets and tasks

A dataset is a list of tasks; the harness reads everything about a task from its `env_params`. The deck depends only on `seed`, so two runs on the same dataset play identical deals and can be compared hand by hand. Task builders live in `vals_tools/tasks.py`; `rl/make_dataset.py` uploads the starter splits.

## Starter splits

| Dataset | Tasks | Base seed | Role |
| --- | --- | --- | --- |
| `pokerbots-train` | 2,000 | 6,610,001 | training |
| `pokerbots-select` | 500 | 6,620,002 | in-run evals; picking a checkpoint |
| `pokerbots-eval` | 3,000 | 6,630,003 | held out; the verdict |
| `pokerbots-smoke` | 4 | 6,640,004 | smoke test with real opponents |
| `pokerbots-table-smoke` | 2 | 6,650,005 | two 3-hand self-play tables; no opponent endpoint |

Each split draws fresh 48-bit seeds from its own base seed, so no deal appears in two splits. A new split takes a base seed no other split uses.

Every starter task is **one hand starting fresh at 100 BB**, the policy seat cycling `i % 6` so every position is equally represented (button starts on the last seat: seat 0 = SB, 1 = BB, 2 = UTG, 3 = HJ, 4 = CO, 5 = BTN). The cleanup turn still runs so the model sees the event tools and prompt.

## Anatomy of a task

```json
{
  "input": [{"role": "user", "content": "Play one hand of 6-seat no-limit hold'em from seat 0."}],
  "env_params": {
    "preset": "event",
    "submit_fields": ["action", "amount"],
    "decision_timeout_seconds": 600,
    "seed": 55064965336171,
    "seats": 6,
    "total_hands": 1,
    "policy_seats": [0],
    "ev_labels": true,
    "ev_runouts": 64,
    "opponent_model": "vals-hackathon/ac-vals-model",
    "opponent_timeout_seconds": 60
  }
}
```

- `preset: "event"` brings in every event rule; **the task's own keys win**. Here `total_hands: 1` overrides the event's 30.
- Table talk on → the harness adds `say` to `submit_fields`.
- `decision_timeout_seconds` is only a safety net against stuck calls; the event itself has no time limit.

## env_params: harness defaults differ from the event

Without `preset: "event"` the harness falls back to research defaults. The ones that bite:

| Key | Harness default | Event |
| --- | --- | --- |
| `seats` | 10 | 6 |
| `total_hands` | 3 | 30 |
| `reset_stacks` | `true` | `false` (stacks carry) |
| `submit_fields` | action, amount, p_win, read | action, amount (+ say) |
| `table_talk` | `false` | `true` |
| `ev_labels`, `ev_runouts` | `false`, 32 | starter sets `true`, 64 |

Other keys: `seed` (required), `decision_timeout_seconds` (300; a slower reply checks/folds), `small_blind`/`big_blind`/`starting_stack` (1/2/200), `policy_seats` (default every seat), `opponent_model`, `opponent_api` (`chat` | `responses`), `opponent_reasoning_effort`, `opponent_thinking`, `opponent_timeout_seconds`. A task may also carry `grader_params`; the sample grader accepts none and fails on unknown keys ([rewards.md](rewards.md#extension-points)). The authoritative list is `harness/README.md` and `completions.py`.

## Opponents

Opponents are a property of the **dataset**, not of `rl/train.py`. Default: the five other seats call `vals-hackathon/ac-vals-model` through AC2's gateway with `AC2_API_KEY` (≈60 req/s per org, shared by everything the team runs).

- Different AC2 model: `rl/make_dataset.py --opponent-model vals-hackathon/<name> --opponent-api responses --opponent-effort medium`. OpenAI reasoning models need `responses` for tool calls with reasoning.
- Any OpenAI-compatible server: start the harness with `POKER_OPPONENT_BASE_URL` and `POKER_OPPONENT_API_KEY`.
- Self-play: omit `opponent_model` and `policy_seats` (as `pokerbots-table-smoke` does). Several policy seats: list them in `policy_seats`.

## Beyond single hands

The arena plays 30-hand games with carried stacks, busting, and a live scratchpad; the starter data exercises none of these. Directions the format already supports:

- **Longer games**: raise `total_hands`, keep the preset's `reset_stacks: false`. Stacks carry, players bust, the scratchpad carries notes.
- **Chosen spots**: pick seats and positions deliberately; vary `starting_stack` for short- and deep-stack play.
- **Other opponents**: stronger or stranger models, multiple own seats, self-play.
- **A matching reward**: game-level `hand_reward` that values rank and survival ([rewards.md](rewards.md)).

Multi-hand tasks have two size limits: the connector's **1 MiB per rollout** (the harness trims logs to fit; keep any added log fields small) and the training **per-hand token budget**, which becomes per-task, so raise `--max-response-len` accordingly ([training.md](training.md#token-budget-per-hand)).

## Filtering for real decisions

Trash hands are folded by every GRPO sample, so the whole group gets the same reward and teaches nothing. Hole cards are known from the seed before play:

```python
from poker_harness.spots import chen_score, hole_cards
cards = hole_cards(55064965336171, 0, seats=6)   # hand 1, seat 0, exactly as dealt
chen_score(cards)                                 # ['9c', 'Qd'] -> 5
```

Pass `seats=6`; the default is 10. Chen score: `AA` 20, `AKs` 12, `22` 5, `72o` −1. Keep some folds: preflop folds from the blinds and late position are real decisions too.

## Building a dataset

Use the helpers in `vals_tools/tasks.py` (`TABLE_RULES`, `Opponents`) and upload like `make_dataset.py`: `client.datasets.create(name, exist_ok=True)`, then `client.datasets.add_tasks` in batches of 500. Name it `pokerbots-<something>`.

Done-check for a new dataset: `--dry-run` output reviewed, `vals_tools.eval` on a 4-task slice succeeds with `format_error_hands` near zero, and the training run's `--max-response-len` is sized for the longest task.
