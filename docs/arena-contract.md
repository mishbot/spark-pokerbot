# Arena contract

What the arena enforces and exactly what an entered model sees. The arena runs the official `harness/` with `preset: "event"`; every rule below follows from that preset. The values live in `EVENT_PARAMS` (`harness/src/poker_harness/completions.py`); the system prompt and state messages in `harness/src/poker_harness/prompts.py`; the tool schemas in `opsd/tools.json` (a harness test keeps them in sync).

## The game

- 6 seats (`seat0`–`seat5`, clockwise), blinds 1/2, everyone starts at 200 chips (100 BB).
- 30 hands (5 orbits). Stacks **carry over**, no rebuys, a busted seat is out. The game ends early if one seat holds all the chips.
- The button moves one live seat clockwise per hand; the two live seats after it post the blinds. Position counts from the small blind: position 0 = SB, 5 = button.
- Bet and raise amounts are **"to" amounts**: the total committed on the current street.
- Cards are two characters, rank then suit (`T` = ten; suits `c d h s`).

## Ranking and rating

The goal is **rank**, not chips in a hand. Seats still in at the end rank above every busted seat, ordered by chips; busted seats rank by when they busted (later is higher). Survival therefore dominates: busting on hand 12 ranks below every survivor at hand 30.

The arena turns each game's ranking into an OpenSkill update and reports bb/100. Its **EV bb/100** column values pre-river all-ins at equity to show luck; it never affects rating.

**Training reward ≠ arena score.** The sample grader rewards single hands by EV in BB; the arena ranks 30-hand games with carried stacks and busting. The starter datasets never exercise carried stacks, busting, or the scratchpad. Closing that gap is the open problem; see [datasets.md](datasets.md#beyond-single-hands).

## A hand's conversation

Each hand is a fresh conversation per seat. Nothing from earlier hands is shown except the seat's own scratchpad.

1. **system**: the event system prompt. The harness owns it; teams cannot add to it at the event, so any strategy must live in the **weights**.
2. **user, private**: the scratchpad block with its character count (`<<<SCRATCHPAD … SCRATCHPAD>>>`, `n/5000 characters`).
3. **user**: hand number, seat and position, street, hole cards, board, pot, every seat's stack and bet, actions so far, then the legal line, e.g. `Legal actions: fold, call 1, raise to 4-200`.
4. **assistant**: reasoning (Qwen thinking is on) and one `submit_action` call.
5. **tool**: what was applied, e.g. `Applied: call 1.`, with the reason if the request was changed.
6. **user**: `Since your last action:` with others' moves and talk, then the new state. Steps 4–6 repeat.
7. **user, private (cleanup)**: what the seat hasn't seen yet, showdown, pot winner, all stacks, and the request to call `update_scratchpad`.
8. **assistant**: one `update_scratchpad` call.

Private messages are never shown to other seats or published; logs keep them under `private` keys.

## Tools

- `submit_action(action, amount?, say?)`: `action` ∈ `fold check call bet raise`; `amount` required for bet/raise. Event tasks set `submit_fields: ["action", "amount"]` and table talk adds `say`. (The harness's research default also offers `p_win` and `read`.)
- `update_scratchpad(contents)`: full replacement, ≤5,000 characters.

## Invalid turns

Strict actions: a turn without a valid `submit_action` is played as **check if legal, else fold**, no retry, logged `afk: true` and `format_error: true`, and the hand scores 0 under the sample grader. Causes and the recorded error:

| Cause | Error |
| --- | --- |
| Reply cut at the 5,000-token cap, even if it holds an action | `reply hit the 5000-token cap` |
| `update_scratchpad` during the hand | `update_scratchpad called during the hand` |
| Arguments not JSON / not an object | `submit_action arguments are not valid JSON` / `... not an object` |
| Bet/raise without amount | `raise without an amount` |
| Amount out of range | `raise to 300 is outside 4-200` |
| Raise impossible | `raise is not legal here` |
| Check facing a bet | `check is not legal facing a bet` |
| No call at all, or the model call failed/timed out | the exception text |

Normalized, not rejected: `bet` and `raise` are one action; `call` with nothing to call is a check; `fold` when checking is free is a check; several `submit_action` calls in one reply use the last (`Ignored: one submit_action per turn; the last call was used.`). After a failed turn the next user message starts `Your last turn made no usable submit_action call.`

The harness also parses a JSON action from plain text when no tool call is present (a legacy Vals fallback). Train for the tool call; treat the fallback as a safety net only.

**Auto-pause:** in ranked games, if more than half of an entry's last 40 decisions fail, its seats check/fold until a deployment is resubmitted. The arena's entry probe only checks a plain-text reply, so a deployment that can't tool-call passes the probe and then fails every decision. `harness/play_local.py` is the real check (see [runbook.md](runbook.md#11-deploy-the-entry)).

## Scratchpad and cleanup turn

- ≤5,000 characters, starts empty, persists for the game, shown at the start of every hand. It is the model's only cross-hand memory.
- After every hand the seat was dealt into: one cleanup turn, ≤10,000 output tokens, expecting exactly one `update_scratchpad`.
- Over-length or malformed update: rejected with the reason, one retry. If the retry also fails, an over-length update is truncated to 5,000 and a malformed one changes nothing.
- No call, or a reply cut at the cap: scratchpad unchanged.

## Table talk

Optional `say`, sanitized before anyone sees it: lowercased, whitespace → space, only `a-z 0-9` space `. , ! ? - : ;` survive, spaces collapsed, trimmed, capped at 70 characters; empty means no talk. Others see it on the speaker's action line (`flop: seat4 (seat 4) raise to 12 and says: "i have the nuts"`); a seat never sees its own. The system prompt calls talk unverified chatter. Talk changes only what models see: identical decisions give identical chips with talk on or off. The log keeps the sanitized text in `say` and the raw text in `model_response.private.say_raw`.

## Token budgets

- Event: 5,000 output tokens per decision and 10,000 per cleanup turn, separately, reasoning included. The harness sends these caps with every call.
- Training is different: AC2's `max_response_len` budgets a **whole hand**. See [training.md](training.md#token-budget-per-hand).
- Logs keep the last 2,000 characters of each decision's text in `raw_output` and `reasoning`, so a forced check/fold shows what the model was thinking when it ran out. Reasoning is never sent back to the model.

## Entries

An entry is a running deployment of `Qwen/Qwen3.6-35B-A3B`: the base model or a fine-tune from one of the team's AC2 training-run checkpoints. The latest submission counts. Thinking is on (`enable_thinking: true`); no time limit.
