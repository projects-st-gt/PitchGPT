"""Quick backtest analysis: compare game sim predictions vs actuals.

Usage:
    python -m scripts.mcsim.analyze_backtest [--db data/mcsim.sqlite]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from mcsim.storage import DEFAULT_DB_PATH


def analyze(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)

    rows = conn.execute("""
        SELECT p.game_pk, p.prediction_date, p.payload_json,
               a.final_score_home, a.final_score_away, a.winner
        FROM predictions p
        JOIN actuals a ON p.game_pk = a.game_pk
        WHERE p.app = 'score_prediction'
          AND a.winner IS NOT NULL
        ORDER BY p.prediction_date, p.game_pk
    """).fetchall()

    if not rows:
        print("No games with both predictions and actuals.")
        return

    wp_home = []
    actual_home_win = []
    dates = set()

    for game_pk, date, payload_json, score_h, score_a, winner in rows:
        payload = json.loads(payload_json)
        wp = payload.get("win_prob_home", 0.5)
        home_won = 1 if winner == "home" else 0
        wp_home.append(wp)
        actual_home_win.append(home_won)
        dates.add(date)

    wp = np.array(wp_home)
    actual = np.array(actual_home_win)
    n = len(wp)

    # Accuracy: did the favored team win?
    predicted_home = wp > 0.5
    correct = (predicted_home == actual.astype(bool))
    acc = correct.mean()

    # Exclude toss-ups (WP 0.45-0.55)
    not_tossup = (wp < 0.45) | (wp > 0.55)
    acc_ex = correct[not_tossup].mean() if not_tossup.sum() > 0 else float("nan")

    # Brier score and skill
    brier = np.mean((wp - actual) ** 2)
    base_rate = actual.mean()
    brier_ref = np.mean((base_rate - actual) ** 2)
    brier_skill = 1 - brier / brier_ref if brier_ref > 0 else 0

    # Home-field analysis
    mean_wp_home = wp.mean()
    actual_home_rate = actual.mean()
    home_gap = mean_wp_home - actual_home_rate

    # Calibration by WP bucket
    edges = [0.0, 0.35, 0.45, 0.55, 0.65, 1.01]
    labels = ["<0.35", "0.35-0.45", "0.45-0.55", "0.55-0.65", ">0.65"]

    print(f"Game-Sim Backtest: {n} games, {len(dates)} dates ({min(dates)} to {max(dates)})")
    print(f"  Overall accuracy:      {acc:.1%} ({correct.sum()}/{n})")
    print(f"  Excl toss-ups:         {acc_ex:.1%} ({correct[not_tossup].sum()}/{not_tossup.sum()})")
    print(f"  Brier score:           {brier:.4f}")
    print(f"  Brier skill (vs base): {brier_skill:+.4f}")
    print(f"  Mean predicted WP(H):  {mean_wp_home:.3f}")
    print(f"  Actual home win rate:  {actual_home_rate:.3f}")
    print(f"  Home-field gap:        {home_gap:+.3f}")
    print()
    print(f"  {'WP bucket':>12}  {'n':>4}  {'pred WP':>8}  {'actual':>8}  {'acc':>6}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}")
    for i, lbl in enumerate(labels):
        mask = (wp >= edges[i]) & (wp < edges[i + 1])
        if mask.sum() == 0:
            continue
        pred_mean = wp[mask].mean()
        act_mean = actual[mask].mean()
        bucket_acc = correct[mask].mean()
        print(f"  {lbl:>12}  {mask.sum():>4}  {pred_mean:>8.3f}  {act_mean:>8.3f}  {bucket_acc:>6.1%}")

    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = ap.parse_args()
    analyze(args.db)
