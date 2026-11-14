# Training

Two launchers, both thin: `rl/train.py` (GRPO through the harness) and `opsd/train.py` (offline OPSD from a hinted dataset). Every setting each sends is a flag; anything unset is AC2's default. `--dry-run` prints exactly what a launch sends, and the GPU estimate; `--help` lists the flags.

Team orgs are **BYOH-only**: allowed are GRPO through our harness, harness-graded evals, data-only offline OPSD, deployments, and arena entries. Ruled out: online OPSD, custom images or launch args, SSH into jobs, choosing a cluster (AC2 places every job; passing one fails the launch).

## GRPO (`rl/train.py`)

Each task is played several times (the group); attempts above the group mean get positive advantage, below get negative. A hand bad for every attempt teaches little. A KL term keeps the policy near base.

Recipe defaults worth knowing (the rest: `--dry-run`):

- 60 steps × 16 hands × 16 samples (`--steps`, `--hands-per-step`, `--samples-per-hand`), Dr. GRPO, `kl_coef 0.01`.
- Checkpoint every 10 steps; in-run eval on `pokerbots-select` at step 0 and every 10.
- `--max-response-len 24576` (AC2's default is 8,192) and `--max-total-len 32768`.
- 1 rollout replica and 1 eval replica (AC2 default is 4 each).
- Left to AC2 unless set: `--lr`, `--concurrency`, `--keep-checkpoints`, `--eval-concurrency`, `--grading-timeout`.

The harness plays every rollout, so harness + connector must stay up for the whole run.

### GPU cost

About **8 of the team's 10 GPUs**: 4 train + 2 rollout engine + 2 in-run eval engine (6 with `--no-eval`). Each training inference engine uses 2 GPUs for this model and no setting changes that, which is why replicas are pinned to 1: AC2's default of 4 each would request 20 GPUs. Over-quota launches queue silently.

### Harness capacity

A step plays 16 × 16 = 256 games. `--concurrency` is unset by default, so AC2 may send a whole step at once. With `SHARDS=4`, pass `--concurrency 64`, or run more shards on a bigger machine. Size shards for the run's concurrency **plus** its in-run eval. Opponent calls run well over a thousand per step against a shared ~60 req/s limit.

### Token budget per hand

In BYOH training, `max_response_len` budgets **everything after the task's first prompt**: every decision's reasoning (kept inline), every observation, and the cleanup turn. Once a hand spends it, the trainer returns empty replies with `finish_reason=length`; the harness logs "reply hit the cap", checks/folds, and the hand scores 0. Training stays correct but learns nothing from that hand.

Size it for the longest hands (multi-hand tasks multiply it) and raise `--max-total-len` with it. Separately, the event caps each decision at 5,000 and each cleanup at 10,000: watch decision lengths in the traces as training proceeds, and confirm with `play_local.py` before submitting.

### Launch protocol

1. `uv run python -m vals_tools.doctor --datasets --harness http://127.0.0.1:8765`: every line `OK`.
2. `uv run python rl/train.py --concurrency 64 --dry-run`: settings and `GPUs while running` reviewed against free quota.
3. `uv run python rl/train.py --steps 5`: a smoke run; a mistake costs minutes, not hours. Done when it reaches step 5 without mass format errors.
4. The real launch; record the `train_id`.
5. Watch: `ac2 train get|logs <id>`, `uv run python -m vals_tools.score --run <id> --step N`, `ac2 train checkpoints <id>`.

Stop a run whose `pokerbots-select` eval keeps worsening or whose decisions turn into format errors: `ac2 train stop <id> --stop-evals`. Without `--stop-evals` its evals keep running and keep their GPUs.

Then judge as in [rewards.md](rewards.md#judging-a-checkpoint).

## Offline OPSD (`opsd/`)

The harness owns the system prompt at the event, so strategy must be in the weights. OPSD distills a prompt into them: the **teacher** is base with a hint appended to its system prompt, the **student** is base without it, and the loss is reverse KL on replies base actually wrote. No grader, environment, or opponents run during training; ~4 GPUs.

Pipeline:

1. Base plays the **train** deals (keep `pokerbots-eval` held out): `uv run python -m vals_tools.eval --dataset pokerbots-train --deployment vals-hackathon/base`.
2. `uv run python opsd/build_dataset.py --eval <eval id> --dataset pokerbots-opsd-train [--dry-run]`. Downloads traces to `traces/<eval id>.jsonl` (or `--traces`), keeps only decisions with exactly one `submit_action` and under `--max-tokens` (5,000), skips cleanup turns, shuffles, prints `{"decisions", "kept", "cleanup", "format", "truncated", "tasks"}`.
3. `uv run python opsd/train.py --dataset pokerbots-opsd-train [--dry-run]`: 20 steps × 64 decisions, checkpoint every 10.
4. Evaluate the student exactly like a GRPO checkpoint. The harness never shows the hint, so the eval measures what the weights learned.

The `HintedTask` contract (the builder constructs it; there is no SDK helper): `input` = the decision's prompt plus base's reply (reasoning, text, `submit_action`); one `HintSelection` at the end of the prompt with `HintMutation(item_boundary=1, action="append_to_previous_item")` so the hint appends to the system message for the teacher only; `env_params.tools` = `opsd/tools.json` (training rejects tool-call transcripts whose tools aren't listed). Dataset created with `required_types=["hinted"]`.

The hint in `opsd/hint.py` is a deliberately generic tight-aggressive example; writing a better one is the point. Good hint material: bet sizing, adapting to scratchpad notes on opponents, short/deep-stack play late in a game, and thinking little enough to stay well under 5,000 tokens.

Limit: offline mode replays base's own tokens, so it can only **reweight** behaviour base already produces. Teaching something base never writes needs online OPSD (`rollout_mode="one_step"`), which BYOH-only orgs can't run.

## Traces into datasets

A deployment's traces are its gateway spans: `client.analytics.query`, the console's Requests tab, or `ac2 traces download <run or eval id>`. `opsd/build_dataset.py` is the worked example: one task per `get_completion` span.
