# AGENTS.md

Starter repo for the Vals poker event: fine-tune `Qwen/Qwen3.6-35B-A3B` on AC2 to play 6-max no-limit hold'em, deploy it, enter it in the Vals arena. The repo is BYOH ("bring your own harness"): `harness/` plays every game on our machine, AC2 serves and trains the model, and nothing in this repo runs on AC2 (`src/pokerbots/` is an empty stub on purpose).

## The invariant

The arena plays the **official, unmodified `harness/`** under `env_params.preset: "event"`. Training may use anything (forked harness, custom rewards, other opponents), but the entered model must still honour the arena contract: one `submit_action` per decision, ≤5,000 output tokens per decision and ≤10,000 per cleanup turn (thinking included), one `update_scratchpad` per cleanup turn. Read [docs/arena-contract.md](docs/arena-contract.md) before changing anything the model sees or produces.

## Conventions

- Run everything from the repo root with `uv run`, except harness work, which runs from `harness/` (its own `pyproject.toml` and venv).
- Every tool defaults to AC2 project `vals-hackathon` and dataset prefix `pokerbots-`.
- Every launcher has `--dry-run`; run it before any real launch and read what it will send. Flags live in each script's `--help`.
- Tests: `uv run pytest` (tasks, scoring, OPSD builder, training configs) and `cd harness && uv run pytest` (harness, engine parity, grader). Both green before a commit that touches their code.
- `README.md` is the human source of truth; when it disagrees with `docs/`, the README wins and `docs/` gets fixed.

## Docs

Read the one whose trigger matches the task:

- [docs/arena-contract.md](docs/arena-contract.md): **event rules** and the model's exact view: prompts, tools, invalid turns, scratchpad, ranking. Touching prompts, tools, token budgets, or anything about how an entry plays.
- [docs/architecture.md](docs/architecture.md): **how BYOH works**: harness, connector, relay, shards, rollout and grading lifecycles, game logs. Touching `harness/` server code or debugging the AC2 ↔ harness path.
- [docs/rewards.md](docs/rewards.md): **grader and scoring**: the reward formula, its invariants, extension points, reading `vals_tools.score`. Changing the reward or judging a checkpoint.
- [docs/datasets.md](docs/datasets.md): **tasks and `env_params`**: splits, seeds, opponents, multi-hand tasks, custom datasets. Building or changing training data.
- [docs/training.md](docs/training.md): **GRPO and OPSD runs**: GPU cost, token budget per hand, harness concurrency, launch and stop. Launching or tuning a run.
- [docs/runbook.md](docs/runbook.md): **end-to-end operations**: setup, connect, baseline, train, evaluate, deploy, submit, each step with its done-check; the GPU budget. Operating AC2 for real.
- [docs/troubleshooting.md](docs/troubleshooting.md): **symptom → fix** table. Something failed.
