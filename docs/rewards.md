# Rewards and scoring

A run that names no grader is graded by our harness through AC2's Grader API, using `harness/src/poker_harness/grading.py`.

## The sample reward

```
hand_reward = clip(EV_bb, -100, +100) / 1      # EV_CLIP_BB = 100, EV_UNIT_BB = 1
score       = 6 + clip(hand_reward, -5, +5)    # every real hand lands in [1, 11]
failure     = 0                                 # format error, reply cut at cap, failed/truncated rollout, table error
```

Why this shape:

- 1 BB per reward point makes typical single-hand pots span most of the band, so GRPO sees real gaps within a group.
- The +6 is a shift, not a rescale: group-relative advantages are unchanged and rewards are never negative.
- Failures score exactly 0, below every real hand, and still train (not dropped), so the model learns failing never pays.

**Invariant for any custom reward: failures at 0, every real hand strictly above 0.** Break it and the model can learn that failing pays.

`GradingError` (unknown `grader_params`, or a harness restarted since the table was played) is different: AC2 drops the sample (`GRADER_ERROR`) instead of training on a made-up reward.

## EV labels

With `env_params.ev_labels` on (starter datasets: `ev_runouts: 64`), a pre-river all-in counts at its expected share of the pot, averaged over board runouts from the cards unseen when the last chip went in; opponents keep their dealt hands. This cuts all-in-hand standard deviation by ~40% without moving the mean. Hands without showdown stay as dealt. Without labels the grader scores chips actually won.

## Extension points

Search `grading.py` for `EXTENSION POINT`:

1. `hand_reward(result: SeatResult, params) -> float`: raw reward for a non-failed game, before clipping and the +6. `SeatResult` carries `ev_bb`, `realized_bb`, `method`, `decisions`, `format_errors`, `folded`, `hands`; read more from the log in `seat_result`.
2. `GRADER_PARAMS`: the `grader_params` keys tasks may pass. Unknown keys fail grading by design, so a dataset typo can't silently change the reward.
3. `artifacts`: extra per-rollout numbers stored with each score row on AC2.

Done-check for a reward change: `cd harness && uv run pytest` green, invariant above holds, harness restarted only after the current run finishes (a live run grades with whatever harness is up). Replies over 1 MiB are trimmed, artifacts first.

The sample grader scores chips only; on single-hand tasks the scratchpad starts blank and is never read, so nothing rewards scratchpad use or survival. A game-level reward valuing rank and survival is open territory ([datasets.md](datasets.md#beyond-single-hands)).

## Score artifacts

`vals_tools.score` reads these, not the shifted reward: `ev_bb`, `realized_bb`, `ev_method` (`realized`, `allin`, `showdown`, …), `policy_decisions`, `policy_format_errors`, `folded`, `hands`.

## Judging a checkpoint

```bash
uv run python -m vals_tools.score --eval <checkpoint eval id> --base <base eval id>
uv run python -m vals_tools.score --run <train id> --step 20      # a training run's in-run eval
```

Output fields: `ev_bb_per_hand` (mean), `ci95` (1.96·sd/√n half-width), `format_error_hands` (share of hands with ≥1 format error), and `vs_base.diff` / `vs_base.ci95`: mean per-deal difference from base. Both evals must use the same dataset so each hand is **paired** with base on identical cards.

The checkpoint beats base when `diff − ci95 > 0`.

Split discipline: **pick** on `pokerbots-select` (the in-run evals), **judge** on `pokerbots-eval` (held out). A verdict computed on the deals used to pick the checkpoint is selection bias. Score early checkpoints too: later is not automatically better.
