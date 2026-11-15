"""Checking actions against what is legal, and building legal ones."""

from pokerlab.datatypes.actions import CALL, CHECK, FOLD, Action, ActionKind
from pokerlab.datatypes.observation import LegalActions


def can_check(legal: LegalActions) -> bool:
    return legal.to_call == 0


def can_raise(legal: LegalActions) -> bool:
    return legal.min_raise_to is not None


def is_legal(action: Action, legal: LegalActions) -> bool:
    match action.kind:
        case ActionKind.FOLD:
            return legal.can_fold
        case ActionKind.CHECK:
            return can_check(legal)
        case ActionKind.CALL:
            return not can_check(legal)
        case ActionKind.RAISE:
            return (
                can_raise(legal)
                and action.amount is not None
                and legal.min_raise_to <= action.amount <= legal.max_raise_to
            )
    return False


def check_or_call(legal: LegalActions) -> Action:
    return CHECK if can_check(legal) else CALL


def check_or_fold(legal: LegalActions) -> Action:
    return CHECK if can_check(legal) else FOLD


def raise_to_nearest(legal: LegalActions, amount: int) -> Action:
    """Raise to `amount`, moved into the legal range; check or call if raising is not allowed."""
    if not can_raise(legal):
        return check_or_call(legal)
    return Action(ActionKind.RAISE, min(max(amount, legal.min_raise_to), legal.max_raise_to))


def all_in(legal: LegalActions) -> Action:
    return raise_to_nearest(legal, legal.max_raise_to) if can_raise(legal) else check_or_call(legal)
