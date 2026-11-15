"""What a player answers: the action, and the decision around it."""

from dataclasses import dataclass
from enum import StrEnum


class ActionKind(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"


@dataclass(frozen=True)
class Action:
    """One action. For a raise, `amount` is the total bet this street ("raise to")."""

    kind: ActionKind
    amount: int | None = None


FOLD = Action(ActionKind.FOLD)
CHECK = Action(ActionKind.CHECK)
CALL = Action(ActionKind.CALL)


@dataclass(frozen=True)
class ModelTrace:
    """Everything about one model call: what was sent, what came back, and what it cost."""

    system_prompt: str
    user_prompt: str
    reply: str  # the final answer, after any reasoning
    reasoning: str | None  # the thinking text, when the server separates it out
    prompt_tokens: int
    completion_tokens: int  # includes reasoning tokens
    finish_reason: str | None  # "length" means the token budget ran out
    seconds: float
    error: str | None = None  # set when the request itself failed


@dataclass(frozen=True)
class Decision:
    """What a player answered.

    `action` is None when the answer could not be understood at all. The
    table, not the player, decides whether an action is legal.
    `stated_hand` and `trace` are filled in by model seats only.
    """

    action: Action | None
    stated_hand: str | None = None
    trace: ModelTrace | None = None


@dataclass(frozen=True)
class ParsedReply:
    """What the model said. `action` is None when no action could be read."""

    action: Action | None
    stated_hand: str | None


@dataclass(frozen=True)
class ActionValue:
    """An action with the teacher's utility for it, in chips (see bots/field/postflop.py)."""

    action: Action
    utility: float
