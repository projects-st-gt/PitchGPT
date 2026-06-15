"""Monte Carlo game simulation: run N games and aggregate into projections.

This is the top-level engine. It takes a matchup card (the per-PA outcome
distributions from the pitchGPT model), lineups, and a transition matrix,
and produces projected scores, win probability, and run distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collections import Counter

from gamesim.state import GameResult, PitchingChange, simulate_one_game
from gamesim.transition import BaseOutTransition


@dataclass
class GameSimResult:
    """Aggregated results from N Monte Carlo game simulations."""

    n_sims: int
    home_team: str
    away_team: str
    home_starter: int
    away_starter: int

    home_scores: np.ndarray = field(repr=False)
    away_scores: np.ndarray = field(repr=False)

    n_backstop: int = 0

    inning_runs_home_avg: list[float] = field(default_factory=list)
    inning_runs_away_avg: list[float] = field(default_factory=list)

    bullpen_stats: dict = field(default_factory=dict)

    @property
    def win_prob_home(self) -> float:
        return float((self.home_scores > self.away_scores).mean())

    @property
    def win_prob_away(self) -> float:
        return float((self.away_scores > self.home_scores).mean())

    @property
    def tie_pct(self) -> float:
        return float((self.home_scores == self.away_scores).mean())

    @property
    def projected_home(self) -> float:
        return float(self.home_scores.mean())

    @property
    def projected_away(self) -> float:
        return float(self.away_scores.mean())

    @property
    def projected_total(self) -> float:
        return float((self.home_scores + self.away_scores).mean())

    @property
    def median_home(self) -> float:
        return float(np.median(self.home_scores))

    @property
    def median_away(self) -> float:
        return float(np.median(self.away_scores))

    @property
    def median_total(self) -> float:
        return float(np.median(self.home_scores + self.away_scores))

    def total_runs_distribution(self) -> dict[int, float]:
        """Histogram of total runs across all simulations."""
        totals = self.home_scores + self.away_scores
        unique, counts = np.unique(totals, return_counts=True)
        return {int(u): float(c / self.n_sims) for u, c in zip(unique, counts)}

    def score_margin_distribution(self) -> dict[int, float]:
        """Histogram of (home - away) score margin."""
        margins = self.home_scores - self.away_scores
        unique, counts = np.unique(margins, return_counts=True)
        return {int(u): float(c / self.n_sims) for u, c in zip(unique, counts)}

    def percentile_total(self, pct: float) -> float:
        return float(np.percentile(self.home_scores + self.away_scores, pct))

    def summary(self) -> dict:
        """Human-readable summary for display and storage."""
        return {
            "n_sims": self.n_sims,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "win_prob_home": round(self.win_prob_home, 4),
            "win_prob_away": round(self.win_prob_away, 4),
            "projected_score": {
                "home": round(self.projected_home, 2),
                "away": round(self.projected_away, 2),
            },
            "median_score": {
                "home": round(self.median_home, 1),
                "away": round(self.median_away, 1),
            },
            "projected_total_runs": round(self.projected_total, 2),
            "total_runs_90pct_band": [
                round(self.percentile_total(5), 1),
                round(self.percentile_total(95), 1),
            ],
            "n_backstop": self.n_backstop,
            "inning_runs_home": [round(x, 3) for x in self.inning_runs_home_avg],
            "inning_runs_away": [round(x, 3) for x in self.inning_runs_away_avg],
            "total_runs_dist": {str(k): round(v, 4) for k, v in self.total_runs_distribution().items()},
            "margin_dist": {str(k): round(v, 4) for k, v in self.score_margin_distribution().items()},
            "bullpen_stats": self.bullpen_stats,
        }


def _aggregate_bullpen_stats(
    home_hook_innings: list[int],
    away_hook_innings: list[int],
    home_hook_bf: list[int],
    away_hook_bf: list[int],
    home_reliever_usage: Counter,
    away_reliever_usage: Counter,
    n_sims: int,
) -> dict:
    """Aggregate pitching change data across all sims into a summary dict."""

    def _hook_summary(innings: list[int], bfs: list[int]) -> dict:
        if not innings:
            return {"hooked_pct": 0.0}
        arr_inn = np.array(innings)
        arr_bf = np.array(bfs)
        inn_counts = Counter(innings)
        inn_dist = {str(k): round(v / len(innings), 4) for k, v in sorted(inn_counts.items())}
        return {
            "hooked_pct": round(len(innings) / n_sims, 4),
            "avg_hook_inning": round(float(arr_inn.mean()), 2),
            "median_hook_inning": int(np.median(arr_inn)),
            "avg_bf_at_hook": round(float(arr_bf.mean()), 1),
            "hook_inning_dist": inn_dist,
        }

    def _reliever_summary(usage: Counter) -> list[dict]:
        total = sum(usage.values())
        if total == 0:
            return []
        return [
            {"pitcher_id": pid, "appearances": count, "pct": round(count / n_sims, 4)}
            for pid, count in usage.most_common()
        ]

    return {
        "home": {
            "starter_hook": _hook_summary(home_hook_innings, home_hook_bf),
            "relievers_used": _reliever_summary(home_reliever_usage),
        },
        "away": {
            "starter_hook": _hook_summary(away_hook_innings, away_hook_bf),
            "relievers_used": _reliever_summary(away_reliever_usage),
        },
    }


def simulate_game(
    matchup_dists: dict[tuple[int, int], dict[str, float]],
    home_lineup: list[int],
    away_lineup: list[int],
    home_starter: int,
    away_starter: int,
    transition: BaseOutTransition,
    n_sims: int = 10_000,
    seed: int = 42,
    home_team: str = "HOME",
    away_team: str = "AWAY",
    bullpen_policy_factory=None,
    park_factors: dict[str, float] | None = None,
) -> GameSimResult:
    """Run N Monte Carlo game simulations and aggregate results.

    Args:
        matchup_dists: {(pitcher_id, batter_id): {outcome: prob}}
        home_lineup: 9 batter IDs in order
        away_lineup: 9 batter IDs in order
        home_starter: pitcher ID
        away_starter: pitcher ID
        transition: loaded BaseOutTransition
        n_sims: number of simulations (default 10,000)
        seed: random seed for reproducibility
        home_team: team name/abbr for display
        away_team: team name/abbr for display
        bullpen_policy_factory: callable returning a fresh BullpenPolicy per sim
        park_factors: optional {outcome: multiplier} for the home ballpark

    Returns:
        GameSimResult with per-sim score arrays and aggregate stats.
    """
    rng = np.random.default_rng(seed)

    home_scores = np.zeros(n_sims, dtype=np.int32)
    away_scores = np.zeros(n_sims, dtype=np.int32)
    n_backstop = 0

    all_inning_home: list[list[int]] = []
    all_inning_away: list[list[int]] = []

    home_hook_innings: list[int] = []
    away_hook_innings: list[int] = []
    home_hook_bf: list[int] = []
    away_hook_bf: list[int] = []
    home_reliever_usage: Counter = Counter()
    away_reliever_usage: Counter = Counter()

    for i in range(n_sims):
        policy = bullpen_policy_factory() if bullpen_policy_factory else None
        result = simulate_one_game(
            matchup_dists=matchup_dists,
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            home_starter=home_starter,
            away_starter=away_starter,
            transition=transition,
            rng=rng,
            bullpen_policy=policy,
            park_factors=park_factors,
        )
        home_scores[i] = result.score_home
        away_scores[i] = result.score_away
        all_inning_home.append(result.inning_runs_home)
        all_inning_away.append(result.inning_runs_away)
        if result.backstop_fired:
            n_backstop += 1

        for chg in result.home_pitching_changes:
            if chg.outgoing_pitcher_id == home_starter:
                home_hook_innings.append(chg.inning)
                home_hook_bf.append(chg.batters_faced_by_outgoing)
            home_reliever_usage[chg.incoming_pitcher_id] += 1
        for chg in result.away_pitching_changes:
            if chg.outgoing_pitcher_id == away_starter:
                away_hook_innings.append(chg.inning)
                away_hook_bf.append(chg.batters_faced_by_outgoing)
            away_reliever_usage[chg.incoming_pitcher_id] += 1

    max_innings = max(max(len(r) for r in all_inning_home), 9)
    inning_home_avg = []
    inning_away_avg = []
    for inn in range(max_innings):
        h_runs = [r[inn] if inn < len(r) else 0 for r in all_inning_home]
        a_runs = [r[inn] if inn < len(r) else 0 for r in all_inning_away]
        inning_home_avg.append(float(np.mean(h_runs)))
        inning_away_avg.append(float(np.mean(a_runs)))

    bullpen_stats = _aggregate_bullpen_stats(
        home_hook_innings, away_hook_innings,
        home_hook_bf, away_hook_bf,
        home_reliever_usage, away_reliever_usage,
        n_sims,
    )

    return GameSimResult(
        n_sims=n_sims,
        home_team=home_team,
        away_team=away_team,
        home_starter=home_starter,
        away_starter=away_starter,
        home_scores=home_scores,
        away_scores=away_scores,
        n_backstop=n_backstop,
        inning_runs_home_avg=inning_home_avg,
        inning_runs_away_avg=inning_away_avg,
        bullpen_stats=bullpen_stats,
    )


def simulate_from_card(
    payload: dict,
    transition: BaseOutTransition,
    n_sims: int = 10_000,
    seed: int | None = None,
    workload_table: dict[int, int] | None = None,
    batter_stand_lookup: dict[int, str] | None = None,
    park_factors_table: dict[str, dict[str, float]] | None = None,
    rotation_pitcher_ids: set[int] | None = None,
) -> GameSimResult:
    """Convenience: simulate a game directly from a matchup card payload.

    Extracts lineups, starters, matchup distributions, and builds a bullpen
    policy automatically from the card's pitcher metadata.
    """
    from gamesim.bullpen import build_bullpen_policy_from_card

    home_team = payload.get("home_team", "HOME")
    away_team = payload.get("away_team", "AWAY")
    starter_home = (payload.get("starter_home") or {})
    starter_away = (payload.get("starter_away") or {})
    home_starter_id = starter_home.get("pitcher_id", 0)
    away_starter_id = starter_away.get("pitcher_id", 0)

    # Infer missing starters from is_starter flag or first pitcher on that team
    if not home_starter_id:
        for pr in payload["rows"]:
            if pr.get("team") == home_team and pr.get("is_starter"):
                home_starter_id = pr["pitcher_id"]
                break
    if not home_starter_id:
        for pr in payload["rows"]:
            if pr.get("team") == home_team:
                home_starter_id = pr["pitcher_id"]
                break

    if not away_starter_id:
        for pr in payload["rows"]:
            if pr.get("team") == away_team and pr.get("is_starter"):
                away_starter_id = pr["pitcher_id"]
                break
    if not away_starter_id:
        for pr in payload["rows"]:
            if pr.get("team") == away_team:
                away_starter_id = pr["pitcher_id"]
                break

    matchup_dists: dict[tuple[int, int], dict[str, float]] = {}
    home_lineup: list[int] = []
    away_lineup: list[int] = []

    for pr in payload["rows"]:
        pid = pr["pitcher_id"]
        batters = [c["batter_id"] for c in pr["cells"]]
        for cell in pr["cells"]:
            dist = cell.get("predicted_outcome_dist")
            if dist:
                matchup_dists[(pid, cell["batter_id"])] = dist

        # Home pitchers face away batters, away pitchers face home batters
        if pid == home_starter_id and not away_lineup:
            away_lineup = batters[:9]
        elif pid == away_starter_id and not home_lineup:
            home_lineup = batters[:9]

    # Fallback: use any pitcher from that team with is_starter or first available
    if not away_lineup:
        for pr in payload["rows"]:
            if pr.get("team") == home_team:
                away_lineup = [c["batter_id"] for c in pr["cells"]][:9]
                break
    if not home_lineup:
        for pr in payload["rows"]:
            if pr.get("team") == away_team:
                home_lineup = [c["batter_id"] for c in pr["cells"]][:9]
                break

    if not home_lineup or not away_lineup:
        raise ValueError(f"Cannot extract lineups from card for {away_team} @ {home_team}")

    game_seed = seed if seed is not None else payload.get("game_pk", 42)

    # Build pitcher name lookup from the card
    pitcher_names: dict[int, str] = {}
    pitcher_throws: dict[int, str] = {}
    for pr in payload["rows"]:
        pitcher_names[pr["pitcher_id"]] = pr.get("name", "Unknown")
        pitcher_throws[pr["pitcher_id"]] = pr.get("throws", "?")

    # Extract batter stand from card cells if present, merge with external lookup
    card_stands: dict[int, str] = {}
    for pr in payload["rows"]:
        for cell in pr["cells"]:
            if "stand" in cell:
                card_stands[cell["batter_id"]] = cell["stand"]
    merged_stands = dict(batter_stand_lookup or {})
    merged_stands.update(card_stands)

    def policy_factory():
        return build_bullpen_policy_from_card(
            payload, workload_table, merged_stands,
            rotation_pitcher_ids=rotation_pitcher_ids)


    # Resolve park factors for this game's home ballpark
    game_park_factors = None
    if park_factors_table:
        # Try team abbreviation lookup; cards store full team names
        from gamesim.park import resolve_park_factors
        game_park_factors = resolve_park_factors(home_team, park_factors_table)

    result = simulate_game(
        matchup_dists=matchup_dists,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        home_starter=home_starter_id,
        away_starter=away_starter_id,
        transition=transition,
        n_sims=n_sims,
        seed=game_seed,
        home_team=home_team,
        away_team=away_team,
        bullpen_policy_factory=policy_factory,
        park_factors=game_park_factors,
    )

    # Attach metadata for display
    result.pitcher_names = pitcher_names
    result.pitcher_throws = pitcher_throws
    result.home_starter_name = pitcher_names.get(home_starter_id, "Unknown")
    result.away_starter_name = pitcher_names.get(away_starter_id, "Unknown")
    result.home_starter_workload = workload_table.get(home_starter_id, 24) if workload_table else 24
    result.away_starter_workload = workload_table.get(away_starter_id, 24) if workload_table else 24

    return result
