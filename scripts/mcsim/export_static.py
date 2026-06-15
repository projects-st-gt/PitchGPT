"""Export all game sim data as static JSON for GitHub Pages deployment.

Reads from the SQLite database and writes JSON files that the frontend can
load directly without a backend API.

    python -m scripts.mcsim.export_static

Output goes to frontend/public/data/ — Vite serves these as static assets.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcsim.storage import DEFAULT_DB_PATH, init_db, read_actual, read_predictions_for_date

GAMESIM_APP = "score_prediction"
CARD_APP = "matchup_card"
OUT_DIR = Path("frontend/public/data")


def main() -> None:
    conn = init_db(DEFAULT_DB_PATH)

    # Find all dates with score predictions
    rows = conn.execute(
        "SELECT DISTINCT prediction_date FROM predictions WHERE app = ? ORDER BY prediction_date",
        (GAMESIM_APP,),
    ).fetchall()
    dates = [r["prediction_date"] for r in rows]
    print(f"Found {len(dates)} dates with score predictions")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write dates index
    dates_payload = {"dates": list(reversed(dates)), "default_date": dates[-1] if dates else ""}
    (OUT_DIR / "gamesim-dates.json").write_text(json.dumps(dates_payload))
    print(f"  wrote gamesim-dates.json")

    for date in dates:
        preds = read_predictions_for_date(conn, prediction_date=date, app=GAMESIM_APP)
        games = []
        for r in preds:
            payload = r.get("payload") or {}
            actual = read_actual(conn, game_pk=r["game_pk"])
            has_actual = actual is not None
            final_h = (actual or {}).get("final_score_home")
            final_a = (actual or {}).get("final_score_away")
            winner = (actual or {}).get("winner")
            wp_h = payload.get("win_prob_home", 0.5)
            predicted_winner = "home" if wp_h > 0.5 else "away"
            correct = (predicted_winner == winner) if winner else None

            games.append({
                "game_pk": r["game_pk"],
                "prediction_date": r["prediction_date"],
                "home_team": payload.get("home_team", "?"),
                "away_team": payload.get("away_team", "?"),
                "win_prob_home": wp_h,
                "win_prob_away": payload.get("win_prob_away", 0.5),
                "projected_home": payload.get("projected_score", {}).get("home", 0),
                "projected_away": payload.get("projected_score", {}).get("away", 0),
                "projected_total": payload.get("projected_total_runs", 0),
                "home_starter_name": payload.get("home_starter_name"),
                "away_starter_name": payload.get("away_starter_name"),
                "has_actual": has_actual,
                "final_score_home": final_h,
                "final_score_away": final_a,
                "winner": winner,
                "predicted_winner_correct": correct,
            })

        list_payload = {"date": date, "count": len(games), "games": games}
        (OUT_DIR / f"gamesim-{date}.json").write_text(
            json.dumps(list_payload, separators=(",", ":"))
        )

        # Per-game detail files
        for r in preds:
            game_pk = r["game_pk"]
            payload = r.get("payload") or {}
            actual = read_actual(conn, game_pk=game_pk)

            # Pull staff from card
            home_staff = []
            away_staff = []
            card_rows = read_predictions_for_date(conn, prediction_date=date, app=CARD_APP)
            for cr in card_rows:
                if cr["game_pk"] != game_pk:
                    continue
                cp = cr.get("payload") or {}
                ht = cp.get("home_team", "")
                at = cp.get("away_team", "")
                sh = (cp.get("starter_home") or {}).get("pitcher_id")
                sa = (cp.get("starter_away") or {}).get("pitcher_id")
                for row in cp.get("rows", []):
                    pid = row.get("pitcher_id")
                    is_starter = row.get("is_starter", False) or pid in (sh, sa)
                    ps = {
                        "pitcher_id": pid,
                        "name": row.get("name", "Unknown"),
                        "throws": row.get("throws", "?"),
                        "is_starter": is_starter,
                    }
                    team = row.get("team", "")
                    if team == ht:
                        home_staff.append(ps)
                    elif team == at:
                        away_staff.append(ps)
                break

            detail_payload = {
                "game_pk": game_pk,
                "prediction_date": date,
                "home_team": payload.get("home_team", "?"),
                "away_team": payload.get("away_team", "?"),
                "sim": payload,
                "has_actual": actual is not None,
                "final_score_home": (actual or {}).get("final_score_home"),
                "final_score_away": (actual or {}).get("final_score_away"),
                "winner": (actual or {}).get("winner"),
                "home_staff": home_staff,
                "away_staff": away_staff,
            }
            (OUT_DIR / f"gamesim-{date}-{game_pk}.json").write_text(
                json.dumps(detail_payload, separators=(",", ":"))
            )

        print(f"  {date}: {len(games)} games exported")

    conn.close()
    total_files = len(list(OUT_DIR.glob("*.json")))
    print(f"\nDone: {total_files} JSON files in {OUT_DIR}")


if __name__ == "__main__":
    main()
