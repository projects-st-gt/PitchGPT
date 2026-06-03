"""MLB Stats API client for MCSim App B — schedule + active rosters.

Pulls REAL games, probable pitchers, and active rosters from the public
statsapi.mlb.com endpoints (no auth). Returns the PitcherSpec/BatterSpec value
objects that mcsim.matchup_card.compute_matchup_card consumes.

Only roster/schedule METADATA comes from here — never pitches. The model still
conditions on real Statcast trailing-window profiles (hard rule #1).

Lineups are deliberately NOT used: MLB posts confirmed lineups only ~2-4h
before first pitch, whereas active rosters are known the night before. The
matchup card grids the full roster (all pitchers x all opposing position
players), which is also a better dugout document — it helps build a lineup,
not just react to one.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional

from mcsim.matchup_card import BatterSpec, PitcherSpec

API_BASE = "https://statsapi.mlb.com/api/v1"


@dataclass
class GameInfo:
    game_pk: int
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    home_probable_pitcher_id: Optional[int]
    away_probable_pitcher_id: Optional[int]


def _get_json(url: str, *, timeout: float = 15.0) -> dict:
    """GET a URL and parse JSON. The single network seam — tests patch this."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _opt_id(obj: Optional[dict]) -> Optional[int]:
    if obj and obj.get("id") is not None:
        return int(obj["id"])
    return None


def get_schedule(date: str) -> list[GameInfo]:
    """Return one GameInfo per scheduled game on ``date`` (YYYY-MM-DD)."""
    url = f"{API_BASE}/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    data = _get_json(url)
    games: list[GameInfo] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            try:
                home = g["teams"]["home"]
                away = g["teams"]["away"]
                games.append(GameInfo(
                    game_pk=int(g["gamePk"]),
                    home_team_id=int(home["team"]["id"]),
                    away_team_id=int(away["team"]["id"]),
                    home_team=home["team"]["name"],
                    away_team=away["team"]["name"],
                    home_probable_pitcher_id=_opt_id(home.get("probablePitcher")),
                    away_probable_pitcher_id=_opt_id(away.get("probablePitcher")),
                ))
            except KeyError as e:
                print(f"[mlb_api] skipping malformed game "
                      f"{g.get('gamePk', '?')}: missing key {e}", file=sys.stderr)
                continue
    return games
