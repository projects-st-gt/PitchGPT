"""MLB Stats API client for post-game ACTUALS — final score + per-PA events.

Pulls the live game feed (``/api/v1.1/game/{game_pk}/feed/live``) after a game
ends and extracts (a) the final score + winner and (b) every completed plate
appearance with the situation it started in. This serves two consumers:

- the demo overlay ("predicted X, actually got Y" once a game finishes), and
- the per-PA calibration eval (each real PA is one Bernoulli draw against the
  model's predicted probability for that matchup/context — pooled across a
  season into a reliability diagram; a single cell is never validated alone).

Real MLB data only — outcomes and contexts come straight from the official
feed; nothing is fabricated. Reuses :func:`mcsim.mlb_api._get_json` as the
single mockable network seam.

**Base-state caveat (verified, not assumed):** ``matchup.splits.menOnBase`` is
a *stat-split label*, not the live base state — it reports "RISP" even for a
leadoff hitter with the bases empty. The correct start context is reconstructed
from ``runners[].movement.originBase`` (the bases a runner already occupied when
the PA began) and the first pitch event's ``count.outs``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mcsim.mlb_api import _get_json

API_FEED_BASE = "https://statsapi.mlb.com/api/v1.1"


@dataclass
class GameActuals:
    game_pk: int
    status: str
    is_final: bool
    final_score_home: Optional[int]
    final_score_away: Optional[int]
    winner: Optional[str]            # "home" / "away" / "tie" / None
    matchup_events: list             # list[dict], one per completed PA


def _winner(home: Optional[int], away: Optional[int]) -> Optional[str]:
    if home is None or away is None:
        return None
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "tie"


def _bases_at_start(play: dict) -> list:
    """Bases occupied at PA start, from runners who already held a base.

    A runner's ``movement.originBase`` is the base they started the play on
    ('1B'/'2B'/'3B'); the batter has ``originBase`` None. So the set of
    non-null originBases is exactly the base state when the PA began.
    """
    occ = set()
    for r in play.get("runners", []):
        origin = (r.get("movement") or {}).get("originBase")
        if origin:
            occ.add(origin)
    return sorted(occ)


def _parse_pa(play: dict) -> dict:
    m = play["matchup"]
    result = play.get("result", {})
    about = play.get("about", {})
    count = play.get("count", {})
    pitches = [e for e in play.get("playEvents", []) if e.get("isPitch")]
    # outs don't change mid-PA, so the first pitch's count.outs is the start state;
    # fall back to the play-level count if a PA somehow has no pitch events.
    outs_start = (
        pitches[0].get("count", {}).get("outs") if pitches else count.get("outs")
    )
    runs_scored = sum(
        1 for r in play.get("runners", [])
        if (r.get("movement") or {}).get("end") == "score"
    )
    return {
        "pitcher_id": int(m["pitcher"]["id"]),
        "batter_id": int(m["batter"]["id"]),
        "bat_side": (m.get("batSide") or {}).get("code"),
        "pitch_hand": (m.get("pitchHand") or {}).get("code"),
        "inning": about.get("inning"),
        "half": about.get("halfInning"),
        "outs_start": outs_start,
        "bases_start": _bases_at_start(play),
        "balls_end": count.get("balls"),
        "strikes_end": count.get("strikes"),
        "event": result.get("event"),
        "event_type": result.get("eventType"),
        "rbi": result.get("rbi"),
        "runs_scored": runs_scored,
        "pitch_types": [
            e["details"]["type"]["code"]
            for e in pitches
            if (e.get("details") or {}).get("type", {}).get("code")
        ],
    }


def get_game_actuals(game_pk: int) -> GameActuals:
    """Fetch and parse one finished game's actuals from the live feed."""
    data = _get_json(f"{API_FEED_BASE}/game/{game_pk}/feed/live")
    status_obj = data.get("gameData", {}).get("status", {})
    live = data.get("liveData", {})
    teams = live.get("linescore", {}).get("teams", {})
    home = teams.get("home", {}).get("runs")
    away = teams.get("away", {}).get("runs")
    plays = live.get("plays", {}).get("allPlays", [])
    events = [
        _parse_pa(p) for p in plays if (p.get("about") or {}).get("isComplete")
    ]
    return GameActuals(
        game_pk=game_pk,
        status=status_obj.get("detailedState", ""),
        is_final=status_obj.get("abstractGameState") == "Final",
        final_score_home=home,
        final_score_away=away,
        winner=_winner(home, away),
        matchup_events=events,
    )
