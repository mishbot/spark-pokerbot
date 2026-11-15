# Baseline: base Qwen vs the mixed bots (10 games × 30 hands)

```bash
uv run --project harness python sim/simulate.py --games 10 --hands 30 --seed 1 --parallel 5 \
  --out harness/runs/sim/ravi-base-mixed
```

Setup: seat 0 is `vals-hackathon/base_mihail`. Seats 1–5 are `tag, shover, maniac, nit, calling_station`, and the lineup moves one seat each game. Event rules apply. The full numbers are in `ravi-base-mixed.summary.json`. The raw game logs are in `ravi-base-mixed/`, one JSON per game with every action, the model's reasoning and its scratchpad. Rescore them with `uv run --project harness python sim/simulate.py --report sim/results/ravi-base-mixed`.

## Scorecard

| Game | Place | Busted | Net BB | Failed decisions |
|---|---|---|---|---|
| 0 | 4 | – | −55.0 | 1/86 |
| 1 | 4 | hand 9 | −100.0 | 0/13 |
| 2 | **1** | – | +201.5 | 0/56 |
| 3 | 4 | – | −12.5 | 0/42 |
| 4 | 4 | hand 1 | −100.0 | 0/1 |
| 5 | 5 | hand 11 | −100.0 | 0/21 |
| 6 | 3 | – | +0.5 | 3/78 |
| 7 | 3 | – | +27.5 | 2/83 |
| 8 | **1** | – | +174.5 | 2/39 |
| 9 | **1** | – | +280.0 | 0/43 |

- **Mean place: 3.00.** 3.5 means even. Places: 1st ×3, 3rd ×2, 4th ×4, 5th ×1.
- **Busted in 3 of 10 games**, every time after calling an all-in.
- **Mean net: +31.6 BB per game**, carried by the three wins.
- **Failed decisions: 8 of 462 (1.7%).** 6 replies were cut off at the 5,000-token cap, and 2 tried to check when facing a bet.

## Net BB against each opponent type

| Bot | Seat 0 net | Bot's mean place |
|---|---|---|
| maniac | +171.8 | 4.50 |
| calling_station | +141.3 | 4.00 |
| tag | +85.5 | 2.40 |
| nit | −27.0 | 2.80 |
| **shover** | **−55.1** | 3.30 |

Qwen beats the loose bots. It loses chips to the bot that shoves all-in, and pays off the tight bot.

## Facing an all-in (18 spots)

- **Called (8):** 72o, KQo, 87s, 32s, 98o, ATs, KK, AQo
- **Folded (10):** A9s, 94s, 77, Q3o, 94o, J3o, A8o, 75o, 44, QJs

It folds most trash, but some bad calls get through, and one of them can end the game.

## Key hands

- **Game 4, hand 1: busted.** It called off all 200 chips from the small blind with **2♦3♦** after a raise and two all-ins. Its reasoning: *"My 2-3 suited hand likely has about 25-30% equity against random ranges... the pot odds make the call reasonable."* It treated the shoves as random hands, and it ignored that busting on hand 1 is close to the worst possible result.
- **Game 1, hand 9: busted.** It re-raised with K♠Q♣, then called a 403-chip shove and lost to A7: *"QK is a premium hand... roughly 55-60% equity... clear call."* KQ has about 45% equity against a range of any pair or any ace. Qwen had already seen that seat shove A5 in hand 4.
- **Game 1, hand 4.** It called an all-in with **7♣2♠** against A5 and won by luck.
- **Game 5, hand 7: misread its own hand.** It held AK on J-Q-A-8-4 and thought *"my trips are likely good"*, when it had one pair. It lost 111 chips to a straight.

## Leaks, in priority order

1. **Loose all-in calls.** This is what busts it. Under elimination ranking, an early bust costs more than any chip gain.
2. **Passive preflop.** It called 77 times and raised only 31.
3. **No opponent notes.** In all 10 games the scratchpad holds only its own results and the stacks, never a read like "seat 1 shoves any ace". The 30-hand games make those reads free, and it doesn't take them.
4. **Format failures** in 1.7% of decisions: cut off at the token cap, or an illegal check.

**Caveat:** this is 10 games against weak, predictable bots, so the mean place is noisy. The leak patterns are the signal, and they match the arena traces for `base_mihail`.
