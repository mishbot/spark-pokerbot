# Troubleshooting

Start with `uv run python -m vals_tools.doctor --datasets --harness http://127.0.0.1:8765`: one line per check, each `FAIL` saying what to do.

| Symptom | Cause → fix |
| --- | --- |
| `uv sync` can't find `ac2` | Export `UV_INDEX_AC2_USERNAME=__token__` and `UV_INDEX_AC2_PASSWORD=<API key>`, rerun. |
| doctor: `can't reach AC2`, or 401/403 | Not logged in or wrong org: `ac2 login`, or export `AC2_API_KEY` (+ `AC2_ORG_ID`). Check `AC2_SERVICE_CLIENT_*` is unset in this shell. |
| `ProjectNotFoundError` | `[project].name` in `pyproject.toml` must be `vals-hackathon` (or a project that exists); `ac2 project init`. Tools resolve the project from the folder they run in. |
| doctor: `no code version yet` | `ac2 project init` from the repo root. |
| doctor: `no vals-hackathon/ac-vals-model` | Organizers must provision it; meanwhile register an own opponent (runbook step 2). |
| `422 ... base_url and protocol are required` | Add `--base-url https://api.openai.com/v1 --protocol openai_responses` to `ac2 inference external`. |
| `run_sharded.sh: uvicorn: not found` | Run `uv sync` in `harness/`, then `uv run ./run_sharded.sh` from there. |
| doctor: `no connector is connected` | Start the connector (runbook step 5). One connector per project: stop the old one; AC2 may take ~1 min to admit a new one. |
| `409 harness_connector_outdated` | `ac2` CLI older than 0.3.97: `ac2 upgrade`. |
| Run or deployment stays queued | At the GPU quota. `ac2 train list`, `ac2 inference list`; delete idle deployments, stop runs with `--stop-evals`. |
| Launch fails mentioning a cluster/placement | A cluster was passed; drop it. AC2 places every job. |
| Rollouts fail immediately | Harness or connector down, or connector token ≠ `POKER_HARNESS_TOKEN`. Doctor with `--harness`. |
| Rollouts fail with an opponent error | `AC2_API_KEY` not exported in the harness terminal, or the dataset's `opponent_model` doesn't exist. |
| Rollouts time out under load | Harness saturated: raise `SHARDS` (and CPUs) or lower `--concurrency`. |
| Rollouts fail at collect, mentioning 1 MiB | Over the connector's per-rollout limit: shrink custom log fields or task length. Full log stays in `harness/runs/`. |
| Many "reply hit the cap" late in training hands | Per-hand token budget exhausted: raise `--max-response-len` and `--max-total-len` ([training.md](training.md#token-budget-per-hand)). |
| Grading: `unknown grader_params` | Task key missing from `GRADER_PARAMS` in `grading.py`: add it or fix the dataset. |
| Grading: `unknown rollout` | Harness restarted since the table was played; restart only between runs. |
| Deployment "ready" but calls 503/time out | Still loading. `vals_tools.deploy` waits for a real answer; otherwise `ac2 inference get <infer_...>` and wait. Send evals only to a deployment that has answered. |
| `vals_tools.deploy`: `never answered` | No tool call within 20 min. `ac2 inference get <infer_...>`; checkpoints must finish converting (`promote`) first. |
| `play_local.py`: `export AC2_API_KEY first` | It calls every seat through the AC2 inference API: `AC2_API_KEY=<key> uv run python play_local.py ...`. |
| Arena seats keep checking/folding | Entry likely can't call `submit_action` (the arena probe checks only plain text). Run `play_local.py` against it; >50% of the last 40 decisions failing auto-pauses the entry until resubmission. |
