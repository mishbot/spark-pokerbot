# Runbook

Clone to arena entry. Each step ends on a **done-check**; move on only when it passes. Steps 1–6 are one-time setup; 7–12 repeat as the team iterates. Commands run from the repo root unless they start with `cd harness`. Placeholders like `<train id>` are replaced whole, brackets included.

These steps create AC2 resources, hold GPUs, and submit entries; confirm with the user before launching training, deploying, deleting, or submitting.

## What the org starts with

Teammates as org admins; a 10-GPU quota; project `vals-hackathon` (no code version until step 1); deployment `vals-hackathon/ac-vals-model` (the provided opponent and smoke-test model); a Vals Arena page in the console sidebar.

## Setup

Prereqs: Python 3.12+, `uv`, git, an AC2 login, and a machine that stays on to run the harness during training (a laptop suffices for smoke tests only).

```bash
curl -fsSL https://api.appliedcompute.com/install.sh | sh   # ac2 CLI, version ≥ 0.3.97
ac2 login                                                   # or --no-browser, or export AC2_API_KEY (+ AC2_ORG_ID)
uv sync && (cd harness && uv sync)
uv run python -m vals_tools.doctor
```

If `uv sync` can't fetch `ac2`: `export UV_INDEX_AC2_USERNAME=__token__ UV_INDEX_AC2_PASSWORD=<API key>`.

Done: doctor prints `OK` for SDK, CLI, project, and `ac-vals-model`, plus exactly one expected `FAIL` (`no code version yet`).

Optional: `ac2 mcp install --agent claude` gives a coding agent read-only access to AC2 docs, runs, datasets, deployments, and traces.

## 1. Link the project

`ac2 project init`. Done: doctor shows no `FAIL`; record the printed `prj_...` id.

## 2. Opponent model

Default `vals-hackathon/ac-vals-model` needs nothing. For an OpenAI opponent: `ac2 secrets put --key OPENAI_API_KEY --value <key>`, then `ac2 inference external --name opponent --provider openai --model <m> --base-url https://api.openai.com/v1 --protocol openai_responses`, and pass `--opponent-model vals-hackathon/opponent --opponent-api responses --opponent-effort medium` to step 3. Done: `ac2 inference list | grep vals-hackathon` lists it.

## 3. Upload datasets

`uv run python rl/make_dataset.py --dry-run`, then without `--dry-run`. Done: `uv run python -m vals_tools.doctor --datasets` prints five `OK dataset` lines.

## 4. Start the harness (terminal 1, long-lived)

```bash
cd harness
export AC2_API_KEY=<key>
export POKER_HARNESS_TOKEN=$(openssl rand -hex 16); echo "$POKER_HARNESS_TOKEN"
SHARDS=4 uv run ./run_sharded.sh
```

Done: `curl -s http://127.0.0.1:8765/health` shows `"status": "ok"` and every shard `ok`.

## 5. Connect to AC2 (terminal 2, long-lived)

An org admin runs `ac2 harness credentials create` once (the secret is shown only once). Then, in a terminal of its own:

```bash
export AC2_SERVICE_CLIENT_ID=svc_... AC2_SERVICE_CLIENT_SECRET=...
ac2 harness connect --project-id <prj_...> --upstream http://127.0.0.1:8765 --upstream-token <token from step 4>
```

Keep `AC2_SERVICE_CLIENT_*` confined to this terminal: in `.env`, `ac2 config`, or any other shell they override the user login and break every other command.

Done: `Connected to the AC2 harness gateway`, and from a clean terminal `uv run python -m vals_tools.doctor --datasets --harness http://127.0.0.1:8765` is all `OK`, ending `a connector is connected to vals-hackathon`.

## 6. Smoke test

```bash
uv run python -m vals_tools.eval --dataset pokerbots-table-smoke --deployment vals-hackathon/ac-vals-model
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/ac-vals-model
```

Done: `ac2 eval get <eval id>` shows `succeeded`, and `vals_tools.score --eval <id>` on the second prints `"hands": 4`.

## 7. Baseline (1 GPU until deleted)

```bash
uv run python -m vals_tools.deploy --name base --model Qwen/Qwen3.6-35B-A3B
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/base
uv run python -m vals_tools.eval --dataset pokerbots-eval  --deployment vals-hackathon/base
```

`deploy` waits until the model actually answers with `submit_action` (up to 20 min) and prints `deployment` (for `--deployment`) and `deployment_id` (`infer_...`, for the arena). Done: the `pokerbots-eval` eval id is recorded as **the base eval** and `score` reports ~3,000 hands. Keep `base` for OPSD; otherwise `vals_tools.deploy --name base --delete`.

## 8. Train with GRPO (~8 GPUs)

[training.md](training.md#grpo-rltrainpy): launch protocol, watching, stopping.

## 9. Evaluate a checkpoint (1 GPU)

Pick from the in-run evals on `pokerbots-select`, then:

```bash
uv run python -m vals_tools.deploy --name step20 --checkpoint <ckpt_...>
uv run python -m vals_tools.eval --dataset pokerbots-smoke --deployment vals-hackathon/step20
uv run python -m vals_tools.eval --dataset pokerbots-eval  --deployment vals-hackathon/step20
uv run python -m vals_tools.score --eval <checkpoint eval> --base <base eval>
```

Done: a verdict per [rewards.md](rewards.md#judging-a-checkpoint), and the deployment deleted unless it becomes the entry.

## 10. Train with OPSD (~4 GPUs)

[training.md](training.md#offline-opsd-opsd): needs the `base` deployment from step 7; evaluate its checkpoints as in step 9.

## 11. Deploy the entry

```bash
uv run python -m vals_tools.deploy --name entry --checkpoint <best ckpt_...>   # or --model Qwen/Qwen3.6-35B-A3B
cd harness && AC2_API_KEY=<key> uv run python play_local.py --model vals-hackathon/entry --hands 6
```

`play_local.py` plays under the official event rules (provided model in the other seats; `--opponents bots` for scripted ones) and is the only real check that the entry tool-calls within budget. Done: `N decisions by vals-hackathon/entry, 0 failed`.

## 12. Submit

In the AC2 console → Vals Arena, select `entry`, name it, submit. The latest submission counts. Done: the page shows the entry; the arena's `Reply with OK.` probe passes; first rating ~15–25 minutes later. Games and replays: pokerbots.ac.

The entered deployment holds 1 GPU for as long as it is entered; keep it running, and free quota by deleting other deployments.

## GPU budget (10 per org)

| Job | GPUs |
| --- | --- |
| GRPO run | 8 (6 with `--no-eval`) |
| Offline OPSD run | 4 |
| Each deployment (base, checkpoint, entry) | 1, until deleted |
| Eval through the harness | 0 (uses the given deployment) |

Fits: GRPO + 2 deployments; OPSD + 6 deployments; GRPO `--no-eval` + OPSD. Two GRPO runs = 16, so the second queues. Once an entry is live, a GRPO run leaves room for one more deployment.

Habits: delete idle deployments (`vals_tools.deploy --name <n> --delete`); stop failing runs with `--stop-evals`; keep few checkpoints (`--keep-checkpoints`, `ac2 checkpoints list vals-hackathon <train id>`, `ac2 checkpoints delete <train id> <step> --format all --cluster all`); check usage with `ac2 train list` and `ac2 inference list`.

## Team lanes

Harness/prompt/strategy (`harness/`) · datasets and OPSD hints (`rl/make_dataset.py`, `vals_tools/tasks.py`, `opsd/hint.py`) · RL launches and the reward (`grading.py`) · evals and analysis (evals vs base, replays, traces, `harness/runs/`) · deployment and submission. Submit something early (even base) to get on the leaderboard, then resubmit when a checkpoint beats base.
