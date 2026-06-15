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
    venue_id: Optional[int] = None


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

    If the probable pitcher isn't on the active roster (common when MLB announces
    the starter before the corresponding roster move), fetches their info from
    the person endpoint and injects them as the starter.

    Rotation members (GS-heavy pitchers who aren't today's starter) are marked
    ``is_rotation=True`` so the bullpen policy can exclude them.
    """
    url = (f"{API_BASE}/teams/{team_id}/roster?rosterType=active"
           f"&date={date}&hydrate=person")
    data = _get_json(url)
    pitchers: list[PitcherSpec] = []
    hitters: list[BatterSpec] = []
    roster_pitcher_ids: set[int] = set()
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
                roster_pitcher_ids.add(pid)
            else:
                stand = _resolve_stand((person.get("batSide") or {}).get("code", "R"))
                hitters.append(BatterSpec(id=pid, name=name, stand=stand))
        except KeyError as e:
            print(f"[mlb_api] skipping malformed roster entry "
                  f"(team {team_id}): missing key {e}", file=sys.stderr)
            continue

    if probable_pitcher_id and probable_pitcher_id not in roster_pitcher_ids:
        spec = _fetch_pitcher_as_starter(probable_pitcher_id)
        if spec:
            pitchers.append(spec)
            print(f"[mlb_api] injected off-roster probable pitcher "
                  f"{spec.name} (id={spec.id}) for team {team_id}",
                  file=sys.stderr)

    _mark_rotation_members(pitchers, team_id, date, probable_pitcher_id)

    return pitchers, hitters


def _mark_rotation_members(
    pitchers: list[PitcherSpec],
    team_id: int,
    date: str,
    probable_pitcher_id: Optional[int],
) -> None:
    """Mark non-starting rotation arms so they're excluded from bullpen duty.

    Scans the schedule for the previous 6 days and marks any pitcher who
    started a game in that window as a rotation member. Pitchers who started
    recently are resting and would never pitch in relief.
    """
    from datetime import datetime, timedelta

    try:
        d = datetime.strptime(date, "%Y-%m-%d")
        start = (d - timedelta(days=6)).strftime("%Y-%m-%d")
        url = (f"{API_BASE}/schedule?sportId=1&startDate={start}&endDate={date}"
               f"&teamId={team_id}&hydrate=probablePitcher")
        data = _get_json(url)
    except Exception as e:
        print(f"[mlb_api] could not fetch recent schedule for rotation "
              f"classification (team {team_id}): {e}", file=sys.stderr)
        return

    recent_starter_ids: set[int] = set()
    for day in data.get("dates", []):
        for g in day.get("games", []):
            for side in ("home", "away"):
                team_data = g.get("teams", {}).get(side, {})
                if team_data.get("team", {}).get("id") == team_id:
                    pp = team_data.get("probablePitcher", {})
                    if pp.get("id"):
                        recent_starter_ids.add(pp["id"])

    marked = []
    for p in pitchers:
        if p.id in recent_starter_ids and p.id != probable_pitcher_id and not p.is_starter:
            p.is_rotation = True
            marked.append(p.name)

    if marked:
        print(f"[mlb_api] rotation arms excluded from bullpen (team {team_id}): "
              f"{', '.join(marked)}", file=sys.stderr)


def _fetch_pitcher_as_starter(pitcher_id: int) -> Optional[PitcherSpec]:
    """Fetch a single player's info and return a PitcherSpec with is_starter=True."""
    try:
        url = f"{API_BASE}/people/{pitcher_id}"
        data = _get_json(url)
        person = data.get("people", [{}])[0]
        name = person.get("fullName", str(pitcher_id))
        throws = (person.get("pitchHand") or {}).get("code", "R")
        return PitcherSpec(
            id=pitcher_id,
            name=name,
            throws=throws if throws in ("R", "L") else "R",
            is_starter=True,
        )
    except Exception as e:
        print(f"[mlb_api] failed to fetch pitcher {pitcher_id}: {e}",
              file=sys.stderr)
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
                    venue_id=_opt_id(g.get("venue")),
                ))
            except KeyError as e:
                print(f"[mlb_api] skipping malformed game "
                      f"{g.get('gamePk', '?')}: missing key {e}", file=sys.stderr)
                continue
    return games
