# Duel: is model B better than model A?

A fixed head-to-head eval under the event's rules that runs in under 30 minutes on your machine. A is the current
model, B the candidate. Each model plays three seats of a six-seat table; every deal is played twice with the seats
swapped; all-in luck is removed with EV labels; 128 tables play at once; and the run stops at a hard 28-minute
deadline. It also grades both models' preflop decisions against *Modern Poker Theory*, at no extra cost.

```bash
# from the repo root, with AC2_API_KEY exported (source ../.env)
uv run --project harness python duel/duel.py --a vals-hackathon/<current> --b vals-hackathon/<candidate> --dry-run
uv run --project harness python duel/duel.py --a vals-hackathon/<current> --b vals-hackathon/<candidate>
uv run --project harness python duel/duel.py --a ... --b ... --block 1      # a second run on fresh deals, to pool

uv run --project harness python duel/duel.py --report duel/runs/<run>                # rescore a run
uv run --project harness python duel/duel.py --report duel/runs/<r1> duel/runs/<r2>  # pool runs of one match-up
uv run --project harness python duel/mpt.py duel/runs/<run>                          # preflop detail
uv run --project harness python duel/calibrate.py duel/runs/<run>                    # noise and timing by hand
uv run --project harness python -m pytest duel -q                                    # tests (bots, no model calls)
```

Both deployments must be up: the run sends one tiny request to each first and stops in seconds if one fails. Logs
go to `duel/runs/<time>/` (git-ignored): one JSON per game plus `summary.json`.

## How it works

| Piece | Why |
| --- | --- |
| Event preset, unmodified harness engine | Same rules as the arena: 6 seats, 100 BB, carried stacks, busting, scratchpad and cleanup turn, talk, token caps. |
| Models are told the game is 30 hands; it stops after `--hands` (default 10) | A full 30-hand game takes ~50 minutes of sequential calls. Telling them 30 keeps their play as in the arena's early and middle game. |
| A in seats 0, 2, 4 and B in 1, 3, 5 | Every hand is A against B, and chips are zero-sum between the two teams. |
| Every seed played twice, seats swapped | Each model gets the other's cards in the same seats, so card luck partly cancels. The pair is the unit of every statistic. |
| Fixed seed blocks (base seed 6,710,011, used by no other split) | Block 0 is the standard set, so every candidate is measured on the same deals. Blocks 1, 2, ... are separate streams of fresh deals for extra runs; no two blocks share a deal. |
| EV labels on every hand | A pre-river all-in counts at its equity, not its runout. |
| 64 seeds = 128 tables in parallel, hard deadline | The most tables one deployment per model can serve in the budget (measured below). Tables still playing at 28 minutes stop, and each pair is scored on the hands both of its games completed. |

## Output and verdict

```
B edge, EV-adjusted : -16.9 bb/100 per seat  (95% CI -49.1 .. +15.3)    <- chips, luck-adjusted: the sharper signal
B edge, as dealt    : -7.7 bb/100  (+/- 32.7)
B place win rate    : 0.514  ->  Elo +10  (95% CI -22 .. +41)           <- finishing places: what the arena rates
verdict             : no clear difference
```

- Intervals are 95% Student-t over mirrored pairs.
- The verdict calls chips and places separately, and says "B better" or "B worse" only when both agree; otherwise
  `mixed` or `no clear difference`. Chips and places can genuinely disagree: in bot tests a maniac beat a random bot
  by +385 bb/100 but lost on places (Elo −105) because it busts.
- **INVALID** replaces the verdict when more than 2% of one side's decisions failed on the deployment or network.
  Format errors the model causes itself (cap hits, illegal amounts) count as play and are listed per model.
- `summary.json` also has a bootstrap `p_b_better_*`; don't decide on it (it read 0.98 in a null test).

## What one run can tell you

Measured on base_mihail (A) vs negreanu-x (B), 2026-11-14/15, plus a null test of negreanu-x against itself.

| Run | Pairs (tables) | Wall time | Hands per game | B edge, chips (95% CI) | B edge, places | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| first run, 16 seeds, scored at 10 hands | 16 (32) | 23.6 min to hand 10 | 10 | +59.0 (+1.5 .. +116.4) | +31 Elo (−28 .. +93) | mixed |
| block 1, 32 seeds | 32 (64) | 18.0 min | 10 | −6.3 (−32.6 .. +19.9) | −5 Elo (−42 .. +31) | none |
| block 2, 64 seeds | 64 (128) | 27.5 min | 10 | −12.1 (−36.2 .. +12.1) | +0 Elo (−31 .. +32) | none |
| **default command** (block 0, 64 seeds) | 64 (128) | 28.0 min, 13 tables cut | 7–10, median 10 | −16.9 (−49.1 .. +15.3) | +10 Elo (−22 .. +41) | none |
| default + block 2 pooled (no shared deals) | 128 | two runs | | −14.5 (−34.3 .. +5.4) | +5 Elo (−17 .. +27) | none |
| **null: negreanu-x vs itself** (block 3) | 64 (128) | 20.5 min | 10 | +18.5 (−0.2 .. +37.2) | +18 Elo (−9 .. +46) | none |
| block 4, 90 seeds, 7 hands | 90 (180) | 28.0 min, 23 tables cut | 4–7, median 7 | +19.4 (−13.9 .. +52.8) | +19 Elo (−5 .. +44) | none |
| blocks 0, 2, 4 pooled at 7 hands (no shared deals) | 203 | three runs | 7 | −4.6 (−24.3 .. +15.0) | +4 Elo (−12 .. +21) | none |

- **The limit is the slower model's decision throughput, not the table count.** base_mihail served about 11.7k
  decisions per 28 minutes however they were spread (11,733 at 128 tables × 10 hands; 11,716 at 180 tables × 7
  hands). Spreading them over more, shorter games made each pair noisier (sd 148 vs 100 bb/100) and cut tables
  short, so 180 × 7 detected less (48 vs 35 bb/100). Of the shapes tried (32 uncapped, 64 × 10, 128 × 10,
  180 × 7), 128 × 10 detects the smallest gap. The only way to a lower floor inside 30 minutes is more serving
  capacity for the slower model (another replica roughly doubles the pairs: ~25 bb/100); otherwise pool blocks.
- **128 tables is the ceiling.** Both 128-table runs used the whole budget; in the default run base_mihail's median
  decision time rose to 9.7 s and 13 tables were stopped between hands 7 and 9 (still scored, on the hands both
  twins finished). 64 tables took 18 minutes. If a deployment is shared with other traffic, use `--seeds 32`.
- **Run length is set by the slower thinker.** Each deployment serves three seats of every table, so the per-model
  load is the same whatever the opponent. Two slow thinkers make each hand longer and the deadline stops games
  earlier; the noise is about the same from hand 7 on (`calibrate.py`), so that costs little.
- **More pairs beat more hands.** Past hand 7 the spread of a pair's score barely falls; doubling pairs cuts it by
  √2. From all 240 pairs above (within-run spread 99.5 bb/100 in chips, 0.170 in place rate):

| Mirrored pairs | Smallest edge a run finds 80% of the time: chips | places | Cost |
| --- | --- | --- | --- |
| 32 | 49 bb/100 | 59 Elo | one 18-minute run (`--seeds 32`) |
| 64 | 35 bb/100 | 42 Elo | one 28-minute run (the default) |
| 128 | 25 bb/100 | 29 Elo | two runs: blocks 0 and 1, pooled |
| 256 | 17 bb/100 | 21 Elo | four runs |

- **The null test is the honesty check.** One model against itself must come out "no clear difference", and did.
  Its chips interval only just covered zero, and the run's original percentile-bootstrap interval did not; that is
  why the intervals are Student-t. Expect roughly 1 null run in 20 to graze an edge on one measure by chance; the
  both-measures rule is what stops that from becoming a wrong "B better".
- **One small run can mislead.** The first 16-seed run, scored at 10 hands, said B better on chips (+59 bb/100);
  every larger run, and the pools, say otherwise. Hence the 128-table default, and a second block for close calls.
- **Mirroring helped 1.0–1.7x** across runs. It holds until a seat busts, after which the twins' deals differ.

What this says about the two models: they are close. The three independent runs pooled (203 pairs, scored at 7
hands) bound B's chip edge to −24 .. +15 bb/100 and its place edge to −12 .. +21 Elo.

## Preflop grading against Modern Poker Theory

`mpt.py` grades two spots from the game logs, against Acevedo's ch. 5 equilibrium ranges (6-max, 100 BB):

- **RFI**: folded to the hero. LJ, HJ, CO and BN raise or fold at 2.5x; the SB plays (raise 3x or limp) or folds.
- **BB defence**: one open of at most 4 BB, everyone else folded. Defend (call or 3-bet) or fold.

The ten charts were transcribed from the book's images; each one's share of hands matches the book's stated number
(`mpt.py --check`). Only clear hands are graded (cells the book splits between actions are skipped), only at full
six-handed tables with every relevant stack at least 50 BB, and only decisions the model made.

From the first (16-seed) run:

| Spot | base_mihail | negreanu-x |
| --- | --- | --- |
| RFI LJ | 83.5% of 170 (25 too loose) | 91.4% of 174 (13 too tight) |
| RFI CO | 74.6% of 114 (21 too loose, 4 limps) | 87.6% of 113 (11 too tight) |
| RFI BN | 78.6% of 98 (12 too loose, 7 limps) | 72.2% of 72 (20 too tight) |
| RFI SB | 74.1% of 58 | 63.4% of 41 (15 too tight) |
| BB vs SB | 100% of 6 | 38.9% of 18 (11 too tight) |
| Open size | 3.0 BB | 2.5 BB |
| All graded | 79.7% of 660 | 83.0% of 593 |

`base_mihail` is too loose and open-limps (K9o and 98o from the CO). `negreanu-x` opens the book's 2.5x but folds
too much late and in the blinds, e.g. QJs and K6s in the BB against an SB open.

Read agreement as a style diagnostic, not a score: the book assumes 5% rake and equilibrium opponents, while the
arena is rake-free against LLMs, so somewhat wider ranges than the book's are reasonable there.
