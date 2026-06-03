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
