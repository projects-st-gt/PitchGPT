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


def _resolve_stand(bat_side_code: str, *, switch_default: str = "L") -> str:
    """Map a batSide code to 'R'/'L' (build_synthetic_ab requires those).

    Switch hitters ('S') resolve to a fixed side for v1 (default 'L', the
    platoon side vs the more common RHP). Per-pitcher resolution is a
    documented follow-up — compute_matchup_card uses a fixed stand per
    BatterSpec, so true per-cell resolution would need it to vary by pitcher.
    """
    if bat_side_code in ("R", "L"):
        return bat_side_code
    return switch_default


def get_active_roster(
    team_id: int,
    date: str,
    *,
    probable_pitcher_id: Optional[int] = None,
) -> tuple[list[PitcherSpec], list[BatterSpec]]:
    """Return (pitchers, position_players) for a team's active roster on ``date``.

    Splits by position type; attaches handedness from the person hydrate. The
    probable starter (if its id matches a rostered pitcher) gets is_starter=True.
    """
    url = (f"{API_BASE}/teams/{team_id}/roster?rosterType=active"
           f"&date={date}&hydrate=person")
    data = _get_json(url)
    pitchers: list[PitcherSpec] = []
    hitters: list[BatterSpec] = []
    for entry in data.get("roster", []):
        try:
            person = entry.get("person", {})
            pid = int(person["id"])
            name = person.get("fullName", str(pid))
            if entry["position"]["type"] == "Pitcher":
                throws = (person.get("pitchHand") or {}).get("code", "R")
                pitchers.append(PitcherSpec(
                    id=pid,
                    name=name,
                    throws=throws if throws in ("R", "L") else "R",
                    is_starter=(probable_pitcher_id is not None
                                and pid == probable_pitcher_id),
                ))
            else:
                stand = _resolve_stand((person.get("batSide") or {}).get("code", "R"))
                hitters.append(BatterSpec(id=pid, name=name, stand=stand))
        except KeyError as e:
            print(f"[mlb_api] skipping malformed roster entry "
                  f"(team {team_id}): missing key {e}", file=sys.stderr)
            continue
    return pitchers, hitters


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
