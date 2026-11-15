"""Opponents for local simulations: the harness's four scripted archetypes plus two of our own.

`shover` goes all-in with any pair or any ace and folds everything else. It tests the leak we saw in the arena: base
Qwen calling all-ins with junk. `tag` is a plain tight-aggressive player (a fixed preflop range, raises its good
hands, bets made hands after the flop, folds the rest), so a table has at least one opponent that plays sensibly.

Both are rules, not models: they answer instantly and cost nothing. They live here, not in `harness/`, because the
arena runs the official harness unmodified.
"""

from __future__ import annotations

import random
from typing import Any

from poker_harness.engine import ScriptedPlayer
from poker_harness.engine.scripted import ARCHETYPES
from poker_harness.engine.table import Decision, DecisionContext
from pokerkit import Label, StandardHighHand

RANKS = "23456789TJQKA"
BOTS = (*ARCHETYPES, "shover", "tag")


def _ranks(hole: list[str]) -> tuple[int, int]:
    """Hole-card ranks, high first (2 = 0 ... A = 12)."""
    a, b = sorted((RANKS.index(c[0]) for c in hole), reverse=True)
    return a, b


def hand_class(hole: list[str]) -> str:
    """ "AA", "AKs" or "AKo"."""
    a, b = _ranks(hole)
    if a == b:
        return RANKS[a] * 2
    return RANKS[a] + RANKS[b] + ("s" if hole[0][1] == hole[1][1] else "o")


def _pairs(low: str) -> set[str]:
    return {r * 2 for r in RANKS[RANKS.index(low) :]}


# Raise and re-raise these; never fold them before the flop.
PREMIUM = _pairs("T") | {"AKs", "AKo", "AQs"}
# Open these; call a raise with them only when it is cheap.
PLAYABLE = (_pairs("2") - PREMIUM) | {"AQo", "AJs", "ATs", "AJo", "KQs", "KQo", "KJs", "QJs", "JTs", "T9s"}


def _response(name: str) -> dict[str, Any]:
    """The bookkeeping fields the game log expects on every action, as the harness's own scripted seats fill them."""
    return {
        "model_name": name,
        "provider": "scripted",
        "latency_seconds": 0.0,
        "in_tokens": 0,
        "out_tokens": 0,
        "reasoning_tokens": 0,
        "cost": 0.0,
        "reasoning": "",
        "fallback_used": False,
        "error": None,
    }


def made_hand(hole: list[str], board: list[str]) -> str:
    """After the flop: "strong" (two pair or better that uses a hole card), "pair" (top pair or an overpair) or
    "weak". A board that pairs itself doesn't count as our hand."""
    label = StandardHighHand.from_game("".join(hole), "".join(board)).entry.label
    hole_ranks = {c[0] for c in hole}
    board_ranks = [c[0] for c in board]
    uses_hole = len(hole_ranks) == 1 or bool(hole_ranks & set(board_ranks))
    if label not in (Label.HIGH_CARD, Label.ONE_PAIR) and (uses_hole or label in (Label.STRAIGHT, Label.FLUSH)):
        return "strong"
    if label == Label.ONE_PAIR:
        top = max(RANKS.index(r) for r in board_ranks)
        a, b = _ranks(hole)
        if a == b and a > top:  # overpair
            return "pair"
        if any(RANKS.index(r) == top for r in hole_ranks):  # top pair
            return "pair"
    return "weak"


class RuleBot:
    """Base for our bots: a sync `choose`, wrapped the way the game expects."""

    kind = ""

    def __init__(self, name: str, seed: int = 0) -> None:
        self.name = name
        self.rng = random.Random(f"{name}:{seed}")

    def choose(self, ctx: DecisionContext) -> Decision:
        raise NotImplementedError

    async def decide(self, ctx: DecisionContext, history: str) -> Decision:
        decision = self.choose(ctx)
        decision.model_response = _response(self.name)
        return decision

    @staticmethod
    def passive(ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "call")

    @staticmethod
    def give_up(ctx: DecisionContext) -> Decision:
        return Decision("check" if ctx.legal.can_check else "fold")

    @staticmethod
    def raise_to(ctx: DecisionContext, amount: int) -> Decision:
        """A bet or raise to `amount`, kept inside the legal range; a call when no raise is allowed."""
        legal = ctx.legal
        if legal.raise_type is None:
            return RuleBot.passive(ctx)
        return Decision(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, int(amount))))

    @staticmethod
    def me(ctx: DecisionContext):
        return next(s for s in ctx.seats if s.seat == ctx.seat)


class Shover(RuleBot):
    """All-in with any pair or any ace before the flop, and with a pair or better after it; otherwise check or fold."""

    kind = "shover"

    def choose(self, ctx: DecisionContext) -> Decision:
        if ctx.street == "preflop":
            a, b = _ranks(ctx.hole_cards)
            good = a == b or a == RANKS.index("A")
        else:
            good = made_hand(ctx.hole_cards, ctx.board) != "weak" or len(set(_ranks(ctx.hole_cards))) == 1
        if not good:
            return self.give_up(ctx)
        return self.raise_to(ctx, ctx.legal.max_raise_to)


class Tag(RuleBot):
    """Tight-aggressive. Preflop: raise PREMIUM (and re-raise), open PLAYABLE to 3 BB or call a cheap raise with it,
    fold the rest. After the flop: raise strong hands, bet or call cheaply with top pair, check or fold otherwise."""

    kind = "tag"
    big_blind = 2

    def choose(self, ctx: DecisionContext) -> Decision:
        legal = ctx.legal
        to_match = max(s.bet for s in ctx.seats)
        stack = self.me(ctx).stack
        if ctx.street == "preflop":
            hand = hand_class(ctx.hole_cards)
            opened = to_match > self.big_blind
            if hand in PREMIUM:
                return self.raise_to(ctx, 3 * to_match if opened else 3 * self.big_blind)
            if hand in PLAYABLE:
                if not opened:
                    return self.raise_to(ctx, 3 * self.big_blind)
                if legal.call_amount * 10 <= stack:  # a raise up to a tenth of our stack: see a flop
                    return self.passive(ctx)
            return self.give_up(ctx)

        strength = made_hand(ctx.hole_cards, ctx.board)
        if strength == "strong":
            return self.raise_to(ctx, 3 * to_match if legal.call_amount else ctx.pot * 3 // 4)
        if strength == "pair":
            if not legal.call_amount:
                return self.raise_to(ctx, ctx.pot // 2)
            if legal.call_amount * 2 <= ctx.pot:  # at most a half-pot bet
                return self.passive(ctx)
        return self.give_up(ctx)


def make_bot(kind: str, name: str, seed: int = 0):
    """A seat played by rules: one of BOTS."""
    if kind == "shover":
        return Shover(name, seed)
    if kind == "tag":
        return Tag(name, seed)
    if kind in ARCHETYPES:
        return ScriptedPlayer(name, kind, seed)
    raise ValueError(f"unknown bot {kind!r}; choose from {', '.join(BOTS)}")
