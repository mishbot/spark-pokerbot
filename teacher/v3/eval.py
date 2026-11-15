"""Made-hand, draws, and board-texture analysis in pure rank/suit logic.

Deliberately does NOT use the generator's strength helpers or pokelab, so the independent
validator can re-derive the same facts a different way (phevaluator) and a shared bug cannot
pass both. Cards are 2-char strings like 'As', 'Td', '7c' (rank in '23456789TJQKA', suit in 'cdhs').

A made hand is classed from the best 5 of the 5-7 available cards. Relation to the board (top
pair / overpair / second pair ...) is what the tier reads, so we compute it explicitly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from phevaluator import evaluate_cards  # trusted oracle; used only for the "plays the board" check

RANK_ORDER = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANK_ORDER)}  # 2->0 ... A->12

# Made-hand categories, weakest to strongest (index is the ordering).
CATEGORIES = (
    "high_card", "pair", "two_pair", "trips", "straight", "flush",
    "full_house", "quads", "straight_flush",
)
CAT_RANK = {c: i for i, c in enumerate(CATEGORIES)}


def rank_of(card: str) -> int:
    return RANK_VALUE[card[0]]


def suit_of(card: str) -> str:
    return card[1]


def _straight_high(ranks: set[int]) -> int | None:
    """Highest card rank (0..12) completing a 5-straight from the given rank set, or None.
    Handles the wheel (A-2-3-4-5): ace counts low as well."""
    present = set(ranks)
    if 12 in present:  # ace plays low for the wheel
        present = present | {-1}
    best = None
    for high in range(12, 2, -1):  # high card from A(12) down to 6(4); low end >= 2(0)/wheel(-1)
        window = {high, high - 1, high - 2, high - 3, high - 4}
        if window <= present:
            best = high
            break
    return best


@dataclass
class HandInfo:
    category: str                 # one of CATEGORIES
    made: str                     # human label, e.g. "top pair", "overpair", "set", "two pair"
    uses_hole: bool               # the made hand uses at least one hole card (not playing the board)
    pair_rank: int | None         # rank (0..12) of the relevant pair/trips, when applicable
    flush_draw: bool
    oesd: bool                    # open-ended straight draw (8 outs)
    gutshot: bool                 # inside straight draw (4 outs)
    board_tag: str                # dry | two_tone | connected | wet | monotone | paired | four_flush | four_straight
    dangerous: bool               # the board carries a threat that out-ranks a plain made hand
    max_threat: int = -1          # CAT_RANK of the best hand the board threatens (-1 = none)
    plays_board: bool = False     # river: the hole cards do not improve on the board (can only chop)
    flush_board_count: int = 0    # how many of the flush suit are already on the board
    flush_hole_high: int = -1     # highest rank (0..12) of our hole card in the flush suit (-1 = none)
    flush_better_count: int = 0   # higher flush cards still unseen (opponents could hold them to beat us)
    draw_text: str = field(default="")

    @property
    def has_pair_or_better(self) -> bool:
        return CAT_RANK[self.category] >= CAT_RANK["pair"]


def _board_tag(board: list[str]) -> tuple[str, bool]:
    """Coarse texture + whether it is 'dangerous' (a nut-changing feature is on board)."""
    if len(board) < 3:
        return "preflop", False
    ranks = [rank_of(c) for c in board]
    suits = [suit_of(c) for c in board]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    paired = any(v >= 2 for v in rank_counts.values())
    max_suit = max(suit_counts.values())
    uniq = sorted(set(ranks))
    # longest run of consecutive ranks on the board (wheel-aware)
    run = _longest_run(set(uniq))
    four_straight = run >= 4
    if max_suit >= 4:
        return "four_flush", True
    if paired:
        return "paired", True
    if four_straight:
        return "four_straight", True
    if max_suit == len(board) and len(board) >= 3:
        return "monotone", True   # all one suit (flush already possible)
    connected = _connected(uniq)
    if max_suit >= 2 and connected:
        return "wet", False       # two-tone AND coordinated
    if connected:
        return "connected", False  # coordinated rainbow (e.g. 9-J-8); NOT dry
    if max_suit >= 2:
        return "two_tone", False
    return "dry", False


def _threats(board: list[str]) -> int:
    """CAT_RANK of the best hand the BOARD threatens a villain could hold, or -1 if none.
    A flush is live with 3+ of a suit on board; a straight with 4+ to a straight; a full house
    whenever the board is paired. We use the highest such threat to cap our non-nut made hands."""
    if len(board) < 3:
        return -1
    suits = Counter(suit_of(c) for c in board)
    ranks = [rank_of(c) for c in board]
    rc = Counter(ranks)
    threats = []
    if max(suits.values()) >= 3:
        threats.append(CAT_RANK["flush"])
    if _longest_run(set(ranks)) >= 4:
        threats.append(CAT_RANK["straight"])
    if any(v >= 2 for v in rc.values()):
        threats.append(CAT_RANK["full_house"])
    return max(threats) if threats else -1


def _connected(uniq: list[int]) -> bool:
    """True if any three board ranks sit within a 5-rank window (wheel-aware)."""
    u = set(uniq)
    if 12 in u:
        u = u | {-1}
    s = sorted(u)
    return any(s[i + 2] - s[i] <= 4 for i in range(len(s) - 2))


def _longest_run(ranks: set[int]) -> int:
    present = set(ranks)
    if 12 in present:
        present = present | {-1}
    best = 0
    for start in sorted(present):
        length = 0
        r = start
        while r in present:
            length += 1
            r += 1
        best = max(best, length)
    return best


def _draws(hole: list[str], board: list[str], made_cat: str) -> tuple[bool, bool, bool]:
    """flush_draw, oesd, gutshot — only meaningful on flop/turn (4-5 cards total before river)."""
    if len(board) >= 5 or len(board) < 3:
        return False, False, False
    all_cards = hole + board
    # flush draw: exactly 4 of a suit including >=1 hole card, and not already a flush
    flush_draw = False
    if made_cat not in ("flush", "straight_flush"):
        for s in "cdhs":
            total = sum(1 for c in all_cards if suit_of(c) == s)
            in_hole = any(suit_of(c) == s for c in hole)
            if total == 4 and in_hole:
                flush_draw = True
    oesd = gutshot = False
    if made_cat not in ("straight", "straight_flush"):
        ranks = {rank_of(c) for c in all_cards}
        hole_ranks = {rank_of(c) for c in hole}
        present = set(ranks)
        if 12 in present:
            present = present | {-1}
        # count distinct straight-completing ranks (outs), requiring a hole card to participate
        outs = set()
        for draw_rank in range(-1, 13):
            if draw_rank in present:
                continue
            filled = present | {draw_rank}
            hi = _straight_high_from(filled)
            if hi is None:
                continue
            window = {hi, hi - 1, hi - 2, hi - 3, hi - 4}
            # the completed straight must use at least one hole card
            if window & {r if r != -1 else 12 for r in hole_ranks | ({-1} if 12 in hole_ranks else set())} or \
               window & hole_ranks:
                outs.add(draw_rank if draw_rank != -1 else 12)
        if len(outs) >= 2:
            oesd = True
        elif len(outs) == 1:
            gutshot = True
    return flush_draw, oesd, gutshot


def _straight_high_from(present: set[int]) -> int | None:
    for high in range(12, 2, -1):
        window = {high, high - 1, high - 2, high - 3, high - 4}
        if window <= present:
            return high
    return None


def analyze(hole: list[str], board: list[str]) -> HandInfo:
    """Classify the made hand (best 5 of hole+board), its relation to the board, and draws."""
    hole = list(hole)
    board = list(board)
    all_cards = hole + board
    ranks_all = [rank_of(c) for c in all_cards]
    suits_all = [suit_of(c) for c in all_cards]
    rank_counts = Counter(ranks_all)
    suit_counts = Counter(suits_all)

    # flush / straight flush
    flush_suit = next((s for s, n in suit_counts.items() if n >= 5), None)
    straight_flush_high = None
    if flush_suit is not None:
        sf_ranks = {rank_of(c) for c in all_cards if suit_of(c) == flush_suit}
        straight_flush_high = _straight_high(sf_ranks)
    straight_high = _straight_high(set(ranks_all))

    counts = sorted(rank_counts.values(), reverse=True)
    board_ranks = [rank_of(c) for c in board]
    board_rank_counts = Counter(board_ranks)
    hole_ranks = [rank_of(c) for c in hole]

    tag, _ = _board_tag(board)
    max_threat = _threats(board)       # the best hand the board itself threatens
    dangerous = max_threat >= 0
    plays_board = len(board) == 5 and evaluate_cards(*hole, *board) == evaluate_cards(*board)

    def info(category, made, pair_rank, uses_hole, flush_board_count=0, flush_hole_high=-1,
             flush_better_count=0):
        fd, oesd, gut = _draws(hole, board, category)
        dt = []
        if fd: dt.append("flush draw")
        if oesd: dt.append("open-ended")
        elif gut: dt.append("gutshot")
        return HandInfo(category=category, made=made, uses_hole=uses_hole, pair_rank=pair_rank,
                        flush_draw=fd, oesd=oesd, gutshot=gut, board_tag=tag, dangerous=dangerous,
                        max_threat=max_threat, plays_board=plays_board, draw_text=", ".join(dt),
                        flush_board_count=flush_board_count, flush_hole_high=flush_hole_high,
                        flush_better_count=flush_better_count)

    if straight_flush_high is not None:
        return info("straight_flush", "straight flush", None, True)
    if counts[0] == 4:
        quad = max(r for r, n in rank_counts.items() if n == 4)
        return info("quads", "quads", quad, quad in hole_ranks)
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return info("full_house", "full house", None, True)
    if flush_suit is not None:
        bcount = sum(1 for c in board if suit_of(c) == flush_suit)
        board_suit_ranks = {rank_of(c) for c in board if suit_of(c) == flush_suit}
        hole_suit = [rank_of(c) for c in hole if suit_of(c) == flush_suit]
        hi = max(hole_suit) if hole_suit else -1
        # higher flush cards an opponent could still hold (not on the board) = hands that beat us
        better = sum(1 for r in range(hi + 1, 13) if r not in board_suit_ranks)
        return info("flush", "flush", None, bool(hole_suit),
                    flush_board_count=bcount, flush_hole_high=hi, flush_better_count=better)
    if straight_high is not None:
        window = {straight_high, straight_high - 1, straight_high - 2, straight_high - 3, straight_high - 4}
        window = {12 if r == -1 else r for r in window}
        uses = bool(window & set(hole_ranks))
        return info("straight", "straight", None, uses)
    if counts[0] == 3:
        trip_rank = max(r for r, n in rank_counts.items() if n == 3)
        is_set = hole_ranks.count(trip_rank) == 2  # pocket pair + board card
        return info("trips", "set" if is_set else "trips", trip_rank, trip_rank in hole_ranks)
    if counts[0] == 2 and counts.count(2) >= 2:
        pair_ranks = sorted((r for r, n in rank_counts.items() if n == 2), reverse=True)
        uses = any(r in hole_ranks for r in pair_ranks)
        return info("two_pair", "two pair", pair_ranks[0], uses)
    if counts[0] == 2:
        pair_rank = max(r for r, n in rank_counts.items() if n == 2)
        made = _pair_label(pair_rank, hole_ranks, board_ranks, board_rank_counts)
        uses = pair_rank in hole_ranks
        return info("pair", made, pair_rank, uses)
    return info("high_card", "high card", None, False)


def _pair_label(pair_rank: int, hole_ranks: list[int], board_ranks: list[int],
                board_rank_counts: Counter) -> str:
    """Relation of a one-pair hand to the board: overpair / top pair / second pair / weak pair /
    board pair (we do not hold it)."""
    pocket = hole_ranks.count(pair_rank) == 2
    live_board = sorted(set(board_ranks), reverse=True)
    if pocket:
        if not live_board or pair_rank > live_board[0]:
            return "overpair"
        return "pocket pair below top"
    if pair_rank not in hole_ranks:
        return "board pair"          # the pair is on the board; we do not hold it (play the board / kicker)
    # a hole card pairs the board
    if live_board and pair_rank == live_board[0]:
        return "top pair"
    if len(live_board) >= 2 and pair_rank == live_board[1]:
        return "second pair"
    return "weak pair"
