"""Read-only HTTP endpoints for MCSim App B — predictions + actuals overlay.

Serves the stored matchup cards (:mod:`mcsim.storage`) for the date-carousel UI
and a per-game detail view, with the real game result stamped onto each
(pitcher, batter) cell once the game has finished.

Read-only and model-free: these endpoints only touch SQLite, so they mount as a
standalone ``APIRouter`` without loading ``NuisanceModels`` (the heavy model app
is ``inference/api.py``). DB access goes through the :func:`get_conn` dependency
so tests can point it at a temp database.

Language discipline: these cards are predictive rollouts, not causal estimates —
this layer just serves and overlays them; it adds no causal-worded copy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from mcsim.storage import (
    DEFAULT_DB_PATH,
    init_db,
    read_actual,
    read_prediction,
    read_predictions_for_date,
)

MCSIM_APP = "matchup_card"

_workload_cache: dict[int, int] | None = None


def _get_workload_table() -> dict[int, int]:
    global _workload_cache
    if _workload_cache is None:
        import json
        workload_path = Path("data/run_value/pitcher_workload.json")
        if workload_path.exists():
            with open(workload_path) as f:
                raw = json.load(f)
            _workload_cache = {int(k): v for k, v in raw.items()}
        else:
            _workload_cache = {}
    return _workload_cache

router = APIRouter(prefix="/mcsim", tags=["mcsim"])


# ============================================================
# DB dependency
# ============================================================


def _db_path() -> Path:
    return Path(os.environ.get("MCSIM_DB_PATH", str(DEFAULT_DB_PATH)))


def get_conn():
    """Yield a SQLite connection for one request (override in tests)."""
    conn = init_db(_db_path())
    try:
        yield conn
    finally:
        conn.close()


# ============================================================
# Response models
# ============================================================


class PredictionSummary(BaseModel):
    game_pk: int
    prediction_date: str
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    made_at: Optional[str] = None
    n_cells: Optional[int] = None
    has_actual: bool


class PredictionsForDateResponse(BaseModel):
    date: str
    count: int
    predictions: list[PredictionSummary]


class MatchupCardResponse(BaseModel):
    game_pk: int
    prediction_date: str
    made_at: Optional[str] = None
    model_ckpt_hash: Optional[str] = None
    has_actual: bool
    final_score_home: Optional[int] = None
    final_score_away: Optional[int] = None
    winner: Optional[str] = None
    unmatched_event_count: int
    card: dict


class GameSimSummary(BaseModel):
    game_pk: int
    prediction_date: str
    home_team: str
    away_team: str
    win_prob_home: float
    win_prob_away: float
    projected_home: float
    projected_away: float
    projected_total: float
    home_starter_name: Optional[str] = None
    away_starter_name: Optional[str] = None
    has_actual: bool
    final_score_home: Optional[int] = None
    final_score_away: Optional[int] = None
    winner: Optional[str] = None
    predicted_winner_correct: Optional[bool] = None


class GameSimListResponse(BaseModel):
    date: str
    count: int
    games: list[GameSimSummary]


class PitcherStaff(BaseModel):
    pitcher_id: int
    name: str
    throws: str
    is_starter: bool
    workload_bf: Optional[int] = None


class GameSimDetailResponse(BaseModel):
    game_pk: int
    prediction_date: str
    home_team: str
    away_team: str
    sim: dict
    has_actual: bool
    final_score_home: Optional[int] = None
    final_score_away: Optional[int] = None
    winner: Optional[str] = None
    home_staff: list[PitcherStaff] = []
    away_staff: list[PitcherStaff] = []


# ============================================================
# Overlay helper (pure)
# ============================================================


def _overlay_actuals_on_card(card: dict, actual: Optional[dict]) -> tuple[dict, int]:
    """Stamp each (pitcher, batter) cell with the real PA(s) that occurred.

    Returns ``(overlaid_card, unmatched_event_count)``:

    - A cell whose (pitcher, batter) pair occurred gets
      ``actual = {"pa_count": n, "events": [...]}`` (a *list* — a starter sees a
      hitter multiple times, and averaging 1-3 PAs would be misleading).
    - A cell that never came up gets ``actual = None``.
    - Real PAs whose pair isn't in the predicted grid (pinch-hitters, call-ups)
      are NOT dropped — they're counted as ``unmatched`` and surfaced at the
      game level.

    The input card is not mutated (a shallow copy of the touched levels is made).
    """
    events = (actual or {}).get("matchup_events") or []
    by_pair: dict[tuple, list] = {}
    for e in events:
        key = (e.get("pitcher_id"), e.get("batter_id"))
        by_pair.setdefault(key, []).append(e)

    matched_keys: set = set()
    new_rows = []
    for row in card.get("rows", []):
        pid = row.get("pitcher_id")
        new_cells = []
        for cell in row.get("cells", []):
            key = (pid, cell.get("batter_id"))
            cell_events = by_pair.get(key)
            new_cell = dict(cell)
            if cell_events:
                matched_keys.add(key)
                new_cell["actual"] = {"pa_count": len(cell_events), "events": cell_events}
            else:
                new_cell["actual"] = None
            new_cells.append(new_cell)
        new_row = dict(row)
        new_row["cells"] = new_cells
        new_rows.append(new_row)

    new_card = dict(card)
    new_card["rows"] = new_rows
    unmatched = sum(len(v) for k, v in by_pair.items() if k not in matched_keys)
    return new_card, unmatched


# ============================================================
# Endpoints
# ============================================================


@router.get("/predictions", response_model=PredictionsForDateResponse)
def list_predictions(
    date: str = Query(..., description="YYYY-MM-DD"),
    conn=Depends(get_conn),
) -> PredictionsForDateResponse:
    """Lightweight per-game summaries for one date (the carousel)."""
    rows = read_predictions_for_date(conn, prediction_date=date, app=MCSIM_APP)
    summaries = []
    for r in rows:
        payload = r.get("payload") or {}
        summaries.append(PredictionSummary(
            game_pk=r["game_pk"],
            prediction_date=r["prediction_date"],
            home_team=payload.get("home_team"),
            away_team=payload.get("away_team"),
            made_at=r.get("made_at"),
            n_cells=payload.get("n_cells"),
            has_actual=read_actual(conn, game_pk=r["game_pk"]) is not None,
        ))
    return PredictionsForDateResponse(date=date, count=len(summaries), predictions=summaries)


@router.get("/predictions/{game_pk}", response_model=MatchupCardResponse)
def get_prediction(
    game_pk: int,
    date: str = Query(..., description="YYYY-MM-DD"),
    conn=Depends(get_conn),
) -> MatchupCardResponse:
    """One game's full card, with actuals stamped onto each cell if the game finished."""
    pred = read_prediction(conn, game_pk=game_pk, prediction_date=date, app=MCSIM_APP)
    if pred is None:
        raise HTTPException(
            status_code=404,
            detail=f"no matchup_card prediction for game_pk={game_pk} on {date}",
        )
    actual = read_actual(conn, game_pk=game_pk)
    overlaid, unmatched = _overlay_actuals_on_card(pred.get("payload") or {}, actual)
    return MatchupCardResponse(
        game_pk=game_pk,
        prediction_date=date,
        made_at=pred.get("made_at"),
        model_ckpt_hash=pred.get("model_ckpt_hash"),
        has_actual=actual is not None,
        final_score_home=(actual or {}).get("final_score_home"),
        final_score_away=(actual or {}).get("final_score_away"),
        winner=(actual or {}).get("winner"),
        unmatched_event_count=unmatched,
        card=overlaid,
    )


GAMESIM_APP = "score_prediction"


class AvailableDatesResponse(BaseModel):
    dates: list[str]
    default_date: str


@router.get("/gamesim/dates", response_model=AvailableDatesResponse)
def list_gamesim_dates(conn=Depends(get_conn)) -> AvailableDatesResponse:
    """Return all dates that have score_prediction rows, newest first."""
    rows = conn.execute(
        "SELECT DISTINCT prediction_date FROM predictions WHERE app = ? ORDER BY prediction_date DESC",
        (GAMESIM_APP,),
    ).fetchall()
    dates = [r["prediction_date"] for r in rows]
    return AvailableDatesResponse(
        dates=dates,
        default_date=dates[0] if dates else "",
    )


@router.get("/gamesim", response_model=GameSimListResponse)
def list_gamesim(
    date: str = Query(..., description="YYYY-MM-DD"),
    conn=Depends(get_conn),
) -> GameSimListResponse:
    """Per-game sim summaries for one date — win probs, projected scores, actuals."""
    rows = read_predictions_for_date(conn, prediction_date=date, app=GAMESIM_APP)
    games = []
    for r in rows:
        payload = r.get("payload") or {}
        actual = read_actual(conn, game_pk=r["game_pk"])
        has_actual = actual is not None
        final_h = (actual or {}).get("final_score_home")
        final_a = (actual or {}).get("final_score_away")
        winner = (actual or {}).get("winner")

        wp_h = payload.get("win_prob_home", 0.5)
        predicted_winner = "home" if wp_h > 0.5 else "away"
        correct = (predicted_winner == winner) if winner else None

        games.append(GameSimSummary(
            game_pk=r["game_pk"],
            prediction_date=r["prediction_date"],
            home_team=payload.get("home_team", "?"),
            away_team=payload.get("away_team", "?"),
            win_prob_home=wp_h,
            win_prob_away=payload.get("win_prob_away", 0.5),
            projected_home=payload.get("projected_score", {}).get("home", 0),
            projected_away=payload.get("projected_score", {}).get("away", 0),
            projected_total=payload.get("projected_total_runs", 0),
            home_starter_name=payload.get("home_starter_name"),
            away_starter_name=payload.get("away_starter_name"),
            has_actual=has_actual,
            final_score_home=final_h,
            final_score_away=final_a,
            winner=winner,
            predicted_winner_correct=correct,
        ))
    return GameSimListResponse(date=date, count=len(games), games=games)


@router.get("/gamesim/{game_pk}", response_model=GameSimDetailResponse)
def get_gamesim(
    game_pk: int,
    date: str = Query(..., description="YYYY-MM-DD"),
    conn=Depends(get_conn),
) -> GameSimDetailResponse:
    """Full sim detail for one game — per-inning breakdown, score distributions."""
    pred = read_prediction(conn, game_pk=game_pk, prediction_date=date, app=GAMESIM_APP)
    if pred is None:
        raise HTTPException(
            status_code=404,
            detail=f"no score_prediction for game_pk={game_pk} on {date}",
        )
    actual = read_actual(conn, game_pk=game_pk)
    payload = pred.get("payload") or {}

    home_staff: list[PitcherStaff] = []
    away_staff: list[PitcherStaff] = []
    card = read_prediction(conn, game_pk=game_pk, prediction_date=date, app=MCSIM_APP)
    if card:
        card_payload = card.get("payload") or {}
        home_team_name = card_payload.get("home_team", "")
        away_team_name = card_payload.get("away_team", "")
        starter_home_id = (card_payload.get("starter_home") or {}).get("pitcher_id")
        starter_away_id = (card_payload.get("starter_away") or {}).get("pitcher_id")
        workload = _get_workload_table()
        home_wl = payload.get("home_starter_workload", 24)
        away_wl = payload.get("away_starter_workload", 24)
        for row in card_payload.get("rows", []):
            pid = row.get("pitcher_id")
            is_starter = row.get("is_starter", False) or pid in (starter_home_id, starter_away_id)
            wl = None
            if is_starter:
                wl = workload.get(pid)
                if wl is None:
                    wl = home_wl if pid == starter_home_id else away_wl if pid == starter_away_id else 24
            ps = PitcherStaff(
                pitcher_id=pid,
                name=row.get("name", "Unknown"),
                throws=row.get("throws", "?"),
                is_starter=is_starter,
                workload_bf=wl,
            )
            team = row.get("team", "")
            if team == home_team_name:
                home_staff.append(ps)
            elif team == away_team_name:
                away_staff.append(ps)

    return GameSimDetailResponse(
        game_pk=game_pk,
        prediction_date=date,
        home_team=payload.get("home_team", "?"),
        away_team=payload.get("away_team", "?"),
        sim=payload,
        has_actual=actual is not None,
        final_score_home=(actual or {}).get("final_score_home"),
        final_score_away=(actual or {}).get("final_score_away"),
        winner=(actual or {}).get("winner"),
        home_staff=home_staff,
        away_staff=away_staff,
    )
