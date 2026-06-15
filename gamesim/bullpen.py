"""Bullpen policy: when to pull the starter and which reliever comes in.

The policy decides two things each PA:
1. Should the current pitcher be replaced?
2. If so, who comes in?

The starter gets pulled based on his typical workload (batters faced per
start, derived from real data). Relievers are ordered by a simple priority
and cycle when exhausted.

Platoon matching: when possible, bring in a reliever whose throwing hand
has the platoon advantage against the current batter (RHP vs RHB, LHP vs LHB).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


RELIEVER_BF_LIMIT = 5

@dataclass
class PitcherInfo:
    pitcher_id: int
    name: str
    throws: str          # "L" or "R"
    is_starter: bool
    team: str
    workload_bf: int = 24   # typical batters faced per start (starters only)
    used: bool = False


@dataclass
class BullpenPolicy:
    """Rule-based pitching change policy.

    Starter pulled when batters_faced >= starter's workload_bf.
    Relievers chosen with platoon preference, cycling through the pen.
    """

    home_pitchers: list[PitcherInfo] = field(default_factory=list)
    away_pitchers: list[PitcherInfo] = field(default_factory=list)
    batter_stand_lookup: dict[int, str] = field(default_factory=dict)

    _home_reliever_idx: int = 0
    _away_reliever_idx: int = 0

    def _get_relievers(self, team_is_home: bool) -> list[PitcherInfo]:
        pitchers = self.home_pitchers if team_is_home else self.away_pitchers
        return [p for p in pitchers if not p.is_starter]

    def _get_starter(self, team_is_home: bool) -> PitcherInfo | None:
        pitchers = self.home_pitchers if team_is_home else self.away_pitchers
        for p in pitchers:
            if p.is_starter:
                return p
        return None

    def maybe_change_pitcher(self, state) -> None:
        """Check if pitching team should make a change. Mutates state in place.

        Starters get pulled at their personal BF threshold.
        Relievers get pulled after RELIEVER_BF_LIMIT batters (~1 inning).
        """
        from gamesim.state import PitchingChange

        pitching_team = state.pitching_team
        is_home = state.is_top

        starter = self._get_starter(is_home)
        current_pid = pitching_team.current_pitcher_id
        is_starter_pitching = starter is not None and current_pid == starter.pitcher_id

        should_change = False
        if is_starter_pitching:
            if pitching_team.batters_faced >= starter.workload_bf:
                should_change = True
        else:
            if pitching_team.batters_faced >= RELIEVER_BF_LIMIT:
                should_change = True

        if should_change:
            old_pid = current_pid
            old_bf = pitching_team.batters_faced
            self._bring_in_reliever(state, is_home)
            if pitching_team.current_pitcher_id != old_pid:
                change = PitchingChange(
                    inning=state.inning,
                    is_top=state.is_top,
                    batters_faced_by_outgoing=old_bf,
                    outgoing_pitcher_id=old_pid,
                    incoming_pitcher_id=pitching_team.current_pitcher_id,
                )
                if is_home:
                    state.home_pitching_changes.append(change)
                else:
                    state.away_pitching_changes.append(change)

    def _bring_in_reliever(self, state, is_home: bool) -> None:
        """Select and bring in the next reliever."""
        relievers = self._get_relievers(is_home)
        if not relievers:
            return

        batter_id = state.current_batter_id()
        raw_stand = self.batter_stand_lookup.get(batter_id)

        best = None
        for rp in relievers:
            if rp.used:
                continue
            batter_stand = _resolve_stand(raw_stand, rp.throws)
            if batter_stand and rp.throws == batter_stand:
                best = rp
                break

        if best is None:
            for rp in relievers:
                if not rp.used:
                    best = rp
                    break

        if best is None:
            best = relievers[-1]

        best.used = True
        state.pitching_team.current_pitcher_id = best.pitcher_id
        state.pitching_team.batters_faced = 0


def _resolve_stand(raw_stand: str | None, pitcher_throws: str) -> str | None:
    """Resolve batter handedness. Switch hitters (S) bat opposite the pitcher."""
    if raw_stand is None:
        return None
    if raw_stand == "S":
        return "L" if pitcher_throws == "R" else "R"
    return raw_stand


def load_batter_stand_lookup() -> dict[int, str]:
    """Load the batter_id -> stand lookup from data/run_value/batter_stand.json."""
    from pathlib import Path
    import json
    path = Path("data/run_value/batter_stand.json")
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_bullpen_policy_from_card(
    payload: dict,
    workload_table: dict[int, int] | None = None,
    batter_stand_lookup: dict[int, str] | None = None,
    rotation_pitcher_ids: set[int] | None = None,
) -> BullpenPolicy:
    """Build a BullpenPolicy from a matchup card payload.

    Args:
        payload: the matchup card JSON payload (from SQLite)
        workload_table: optional {pitcher_id: typical_batters_faced} lookup.
            If None, uses a default of 24 BF for all starters.
        batter_stand_lookup: optional {batter_id: "R"/"L"/"S"} for platoon matching.
        rotation_pitcher_ids: optional set of pitcher IDs to exclude from bullpen
            (rotation starters who aren't today's starter). Supplements the
            is_rotation flag in the card payload for older cards that lack it.
    """
    home_team = payload.get("home_team", "")
    away_team = payload.get("away_team", "")
    starter_home_id = (payload.get("starter_home") or {}).get("pitcher_id")
    starter_away_id = (payload.get("starter_away") or {}).get("pitcher_id")

    home_pitchers = []
    away_pitchers = []
    _rotation_ids = rotation_pitcher_ids or set()

    for pr in payload["rows"]:
        pid = pr["pitcher_id"]
        is_starter = pr.get("is_starter", False) or pid in (starter_home_id, starter_away_id)
        is_rotation = pr.get("is_rotation", False) or (pid in _rotation_ids)
        team = pr.get("team", "")

        if is_rotation and not is_starter:
            continue

        default_bf = 24
        if workload_table and pid in workload_table:
            bf = workload_table[pid]
        else:
            bf = default_bf

        info = PitcherInfo(
            pitcher_id=pid,
            name=pr.get("name", "Unknown"),
            throws=pr.get("throws", "R"),
            is_starter=is_starter,
            team=team,
            workload_bf=bf if is_starter else 0,
        )

        if team == home_team:
            home_pitchers.append(info)
        elif team == away_team:
            away_pitchers.append(info)

    stand_lut = batter_stand_lookup or {}
    return BullpenPolicy(
        home_pitchers=home_pitchers,
        away_pitchers=away_pitchers,
        batter_stand_lookup=stand_lut,
    )


def build_pitcher_workload(
    raw_dir: str = "data/raw",
    seasons: list[int] | None = None,
    decay_halflife_starts: int = 5,
) -> dict[int, int]:
    """Compute recency-weighted typical batters-faced-per-start for each pitcher.

    Uses exponential decay weighting (recent starts count more). Returns
    {pitcher_id: rounded BF/start}, clamped to [18, 30].
    """
    from pathlib import Path

    if seasons is None:
        seasons = list(range(2021, 2024))

    all_starts: list[dict] = []

    for season in seasons:
        season_dir = Path(raw_dir) / str(season)
        if not season_dir.exists():
            continue
        for pq in sorted(season_dir.glob("*.parquet")):
            df = pd.read_parquet(pq, columns=["game_pk", "pitcher", "at_bat_number", "inning", "events"])
            pa_df = df[df["events"].notna()].copy()
            if pa_df.empty:
                continue
            # Find starters: the pitcher who threw the first pitch of the game for each team
            first_ab = pa_df.sort_values("at_bat_number").groupby("game_pk").first()
            starters = set(zip(first_ab.index, first_ab["pitcher"]))

            for (gk, pid) in starters:
                game_pas = pa_df[(pa_df["game_pk"] == gk) & (pa_df["pitcher"] == pid)]
                bf = len(game_pas)
                all_starts.append({"pitcher_id": int(pid), "game_pk": int(gk), "bf": bf})

    if not all_starts:
        return {}

    starts_df = pd.DataFrame(all_starts)
    starts_df = starts_df.sort_values(["pitcher_id", "game_pk"]).reset_index(drop=True)

    decay = np.log(2) / decay_halflife_starts
    result = {}
    for pid, grp in starts_df.groupby("pitcher_id"):
        bfs = grp["bf"].values
        n = len(bfs)
        if n < 3:
            continue
        weights = np.exp(-decay * np.arange(n - 1, -1, -1))
        weighted_bf = np.average(bfs, weights=weights)
        result[int(pid)] = int(np.clip(round(weighted_bf), 18, 30))

    return result
