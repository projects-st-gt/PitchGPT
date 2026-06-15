"""Score matchup-card predictions against actual game results.

Joins predicted per-(pitcher, batter) outcome distributions to real PA
events from the actuals table. Computes per-PA log-loss, top-1 accuracy,
and per-class calibration.

Usage:
    uv run python -m scripts.mcsim.score_predictions [--dates 2026-06-04,2026-06-05]

Omit --dates to score all dates that have both predictions AND actuals.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from hitter.eval import pa_logloss, pa_outcome_class
from mcsim.storage import DEFAULT_DB_PATH, init_db

PA_CLASSES = ["K", "BB", "1B", "2B", "3B", "HR", "out"]

_EXCLUDE_EVENT_TYPES = {
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "wild_pitch", "passed_ball", "balk", "other_advance",
}


def _event_type_to_class(event_type: str) -> str | None:
    if event_type in _EXCLUDE_EVENT_TYPES:
        return None
    return pa_outcome_class(event_type)


def _load_scorable_dates(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("""
        SELECT DISTINCT p.prediction_date
        FROM predictions p
        JOIN actuals a ON p.game_pk = a.game_pk
        WHERE p.app = 'matchup_card'
          AND a.matchup_events_json IS NOT NULL
        ORDER BY p.prediction_date
    """).fetchall()
    return [r[0] for r in rows]


def _score_date(conn: sqlite3.Connection, date: str) -> dict:
    preds = conn.execute("""
        SELECT p.game_pk, p.payload_json, a.matchup_events_json
        FROM predictions p
        JOIN actuals a ON p.game_pk = a.game_pk
        WHERE p.prediction_date = ? AND p.app = 'matchup_card'
          AND a.matchup_events_json IS NOT NULL
    """, (date,)).fetchall()

    matched_dists: list[dict] = []
    matched_actuals: list[str] = []
    n_games = 0
    n_cells_matched = 0
    n_events_total = 0
    n_events_excluded = 0
    trust_counts: Counter = Counter()

    for row in preds:
        n_games += 1
        payload = json.loads(row[1])
        events = json.loads(row[2])

        event_lookup: dict[tuple[int, int], list[dict]] = {}
        for ev in events:
            key = (ev["pitcher_id"], ev["batter_id"])
            event_lookup.setdefault(key, []).append(ev)

        for pitcher_row in payload["rows"]:
            pid = pitcher_row["pitcher_id"]
            for cell in pitcher_row["cells"]:
                bid = cell["batter_id"]
                trust_counts[cell.get("trust_state", "unknown")] += 1
                real_pas = event_lookup.get((pid, bid), [])
                if not real_pas:
                    continue
                n_cells_matched += 1
                pred_dist = cell["predicted_outcome_dist"]
                for pa in real_pas:
                    n_events_total += 1
                    outcome = _event_type_to_class(pa["event_type"])
                    if outcome is None:
                        n_events_excluded += 1
                        continue
                    matched_dists.append(pred_dist)
                    matched_actuals.append(outcome)

    if not matched_actuals:
        return {"date": date, "n_games": n_games, "n_scored_pas": 0}

    logloss = pa_logloss(matched_dists, matched_actuals)

    top1_preds = [max(d, key=d.get) for d in matched_dists]
    top1_acc = sum(1 for p, a in zip(top1_preds, matched_actuals) if p == a) / len(matched_actuals)

    class_counts: dict[str, dict] = {}
    for cls in PA_CLASSES:
        mask = [a == cls for a in matched_actuals]
        n_true = sum(mask)
        if n_true == 0:
            continue
        pred_probs = [d.get(cls, 0.0) for d in matched_dists]
        mean_pred = np.mean(pred_probs)
        true_rate = n_true / len(matched_actuals)
        in_class_mean_pred = np.mean([p for p, m in zip(pred_probs, mask) if m])
        class_counts[cls] = {
            "n": n_true,
            "true_rate": round(true_rate, 4),
            "mean_pred_prob": round(float(mean_pred), 4),
            "mean_pred_when_true": round(float(in_class_mean_pred), 4),
        }

    return {
        "date": date,
        "n_games": n_games,
        "n_cells_matched": n_cells_matched,
        "n_scored_pas": len(matched_actuals),
        "n_events_excluded": n_events_excluded,
        "logloss": round(logloss, 4),
        "top1_acc": round(top1_acc, 4),
        "trust_dist": dict(trust_counts),
        "per_class": class_counts,
        "matched_dists": matched_dists,
        "matched_actuals": matched_actuals,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score matchup-card predictions vs actuals.")
    ap.add_argument("--dates", type=str, default=None,
                    help="Comma-separated dates to score (default: all scorable dates)")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()

    conn = init_db(args.db)

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
    else:
        dates = _load_scorable_dates(conn)
        if not dates:
            print("No dates with both predictions and actuals found.")
            return

    print(f"Scoring {len(dates)} date(s): {', '.join(dates)}\n")

    all_dists: list[dict] = []
    all_actuals: list[str] = []
    total_games = 0

    for date in dates:
        result = _score_date(conn, date)
        n = result["n_scored_pas"]
        total_games += result["n_games"]

        if n == 0:
            print(f"  {date}: {result['n_games']} games, 0 matched PAs (no actuals?)")
            continue

        print(f"  {date}: {result['n_games']} games, {result['n_cells_matched']} cells matched, "
              f"{n} PAs scored")
        print(f"    log-loss: {result['logloss']:.4f}  |  top-1 acc: {result['top1_acc']:.4f}")
        print(f"    trust: {result['trust_dist']}")
        print(f"    per-class:")
        for cls in PA_CLASSES:
            if cls in result["per_class"]:
                c = result["per_class"][cls]
                print(f"      {cls:>3}: n={c['n']:>4}  true_rate={c['true_rate']:.3f}  "
                      f"mean_pred={c['mean_pred_prob']:.3f}  mean_pred|true={c['mean_pred_when_true']:.3f}")
        print()

        all_dists.extend(result.get("matched_dists", []))
        all_actuals.extend(result.get("matched_actuals", []))

    if len(all_actuals) < 2:
        print("Not enough matched PAs for aggregate scoring.")
        return

    print("=" * 60)
    print(f"AGGREGATE: {total_games} games, {len(all_actuals)} PAs across {len(dates)} dates")
    print("=" * 60)
    agg_ll = pa_logloss(all_dists, all_actuals)
    agg_top1 = [max(d, key=d.get) for d in all_dists]
    agg_acc = sum(1 for p, a in zip(agg_top1, all_actuals) if p == a) / len(all_actuals)
    print(f"  log-loss: {agg_ll:.4f}")
    print(f"  top-1 acc: {agg_acc:.4f}")

    print(f"\n  per-class calibration:")
    print(f"  {'class':>5}  {'n':>5}  {'true_rate':>10}  {'mean_pred':>10}  {'gap':>8}")
    for cls in PA_CLASSES:
        mask = [a == cls for a in all_actuals]
        n_true = sum(mask)
        if n_true == 0:
            continue
        true_rate = n_true / len(all_actuals)
        mean_pred = np.mean([d.get(cls, 0.0) for d in all_dists])
        gap = float(mean_pred) - true_rate
        print(f"  {cls:>5}  {n_true:>5}  {true_rate:>10.4f}  {float(mean_pred):>10.4f}  {gap:>+8.4f}")


if __name__ == "__main__":
    main()
