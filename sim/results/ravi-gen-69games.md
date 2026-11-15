# SFT data: 69 simulated games (base Qwen in seat 0)

`ravi-gen-69games.tar.gz` (5.1 MB; 39 MB unpacked) holds 69 clean 30-hand games, each as a game log (`sim-*.json`) and a transcript (`transcript-*.json`). The transcript records every request seat 0 sent and the exact reply it got, which is the raw material for `sim/build_sft.py`.

```bash
mkdir -p harness/runs/sim && tar -xzf sim/results/ravi-gen-69games.tar.gz -C harness/runs/sim
uv run python sim/build_sft.py harness/runs/sim/ravi-gen* --dataset pokerbots-sft-ravi-sim --dry-run
```

## The games

| Batch | Seat 0 | Opponents (seats 1–5, rotating) | Games | Seat 0 decisions | Mean place | Busted |
|---|---|---|---|---|---|---|
| ravi-gen2-mixed | eval-base | tag, shover, maniac, nit, calling_station | 33 | 834 | 3.21 | 20/33 |
| ravi-gen2-shovers | eval-base | shover, shover, tag, nit, calling_station | 25 | 693 | 3.84 | 16/25 |
| ravi-gen-shovers | base_mihail | shover, shover, tag, nit, calling_station | 5 | 43 | 5.00 | 5/5 |
| ravi-gen-mixed | base_mihail | tag, shover, maniac, nit, calling_station | 4 | 22 | 4.50 | 4/4 |
| ravi-gen2-qwen | eval-base | base Qwen ×2, tag, shover, maniac | 2 | 42 | 2.50 | 1/2 |
| **Total** | | | **69** | **1,634** | **3.62** | **46/69** |

`eval-base` and `base_mihail` serve the same checkpoint with the same settings. The two `base_mihail` batches are the games that finished first, before their runs were stopped, so they lean toward early busts.

**Left out:** games 32, 33 and 35 of ravi-gen2-mixed. `eval-base` went away during them, and 23 of seat 0's turns were played as automatic check/fold.

## What the builder makes of them (current rules)

| Outcome | Decisions |
|---|---|
| kept as written | 1,358 |
| relabeled to fold: junk all-in call | 25 (×3 copies) |
| relabeled to fold: junk call of a preflop raise | 51 (×3 copies) |
| dropped: failed decision (token cap, illegal move) | 26 |
| dropped: the hand seat 0 busted on | 25 |
| dropped: later in a hand after a dropped decision | 149 |
| **tasks** | **1,586** |

Open question before upload: the call-off range. It's currently TT+, AK, AQs. 77+, AJ+, KQs would stop relabeling reasonable calls against wide shovers (99, AJs, A9s, KQs).
