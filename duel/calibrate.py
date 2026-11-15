"""Size the duel from a saved run: for each cut-off (hands per game), how long tables took to get there, how noisy a
mirrored pair is, and the smallest edge the eval would reliably detect with that many pairs.

    uv run --project harness python duel/calibrate.py duel/runs/<run>

"Detectable" is the edge the eval finds 80% of the time at the 5% level: 2.8 x sd(pair) / sqrt(pairs). Chips are
EV-adjusted bb/100 per seat; places are the share of A-seat vs B-seat match-ups B finishes ahead in (0.5 = even),
also shown as Elo.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duel import elo, pair_scores


def hand_minutes(logs: list[dict], hands: int) -> list[float]:
    """Minutes each table took to finish `hands` hands (tables that got that far)."""
    out = []
    for log in logs:
        ends = log["duel"].get("hand_end_seconds") or []
        if len(ends) >= hands:
            out.append(ends[hands - 1] / 60)
    return out


def main() -> None:
    run = Path(sys.argv[1])
    logs = [json.loads(p.read_text()) for p in sorted(run.glob("*.json")) if p.name[0].isdigit() and "partial" not in p.name]
    most = max(len(log["hands"]) for log in logs)
    resp = [a.get("model_response") or {} for log in logs for h in log["hands"] for a in h["actions"]]
    lat = sorted(r["latency_seconds"] for r in resp if r.get("latency_seconds") is not None)
    print(f"{run.name}: {len(logs)} tables, up to {most} hands; decision latency median {statistics.median(lat):.1f} s, "
          f"p90 {lat[int(0.9 * len(lat))]:.1f} s over {len(lat)} decisions\n")
    print(" hands | pairs | minutes to reach (median / max) | chips: B edge, sd(pair), detectable | places: B rate, sd, detectable (Elo)")
    for at in range(2, most + 1):
        rows = pair_scores(logs, at)
        if len(rows) < 3:
            continue
        mins = hand_minutes(logs, at)
        when = f"{statistics.median(mins):5.1f} / {max(mins):5.1f}" if mins else "   n/a      "
        n = len(rows)
        ev = [r["b_ev_bb100"] for r in rows]
        pl = [r["b_place_wins"] for r in rows]
        sd_ev, sd_pl = statistics.stdev(ev), statistics.stdev(pl)
        mde_ev, mde_pl = 2.8 * sd_ev / math.sqrt(n), 2.8 * sd_pl / math.sqrt(n)
        print(f" {at:5d} | {n:5d} | {when:>31} | {statistics.fmean(ev):+7.1f} {sd_ev:7.1f} {mde_ev:7.1f}"
              f" | {statistics.fmean(pl):.3f} {sd_pl:.3f} {mde_pl:.3f} ({elo(0.5 + mde_pl):+.0f})")


if __name__ == "__main__":
    main()
