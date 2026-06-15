"""Game state machine: tracks the full state of a baseball game.

Manages innings, outs, base runners, score, lineup cycling, and pitcher
changes. Knows the rules: 9 innings, walk-off, ghost runner in extras,
3 outs per half-inning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gamesim.outcomes import BASES_EMPTY, INNING_OVER, ON_2B
from gamesim.transition import BaseOutTransition

MAX_INNINGS = 30

# Empirical TTO multipliers from 340K starter PAs (2021-2023).
# Each maps outcome -> ratio vs TTO-1 baseline.
TTO_MULTIPLIERS = {
    2: {"K": 0.870, "BB": 0.985, "1B": 1.042, "2B": 1.096, "3B": 1.184, "HR": 1.069, "out": 1.045},
    3: {"K": 0.804, "BB": 0.994, "1B": 1.082, "2B": 1.151, "3B": 1.118, "HR": 1.183, "out": 1.054},
}


@dataclass
class TeamState:
    lineup: list[int]         # 9 batter IDs in batting order
    lineup_ptr: int = 0       # index into lineup (0-8, wraps)
    starter_id: int = 0
    current_pitcher_id: int = 0
    batters_faced: int = 0

    def current_batter(self) -> int:
        return self.lineup[self.lineup_ptr % 9]

    def advance_lineup(self) -> None:
        self.lineup_ptr = (self.lineup_ptr + 1) % 9


@dataclass
class GameState:
    """Full state of a game in progress."""

    inning: int = 1
    is_top: bool = True       # True = away batting, False = home batting
    outs: int = 0
    base_state: int = BASES_EMPTY
    score_home: int = 0
    score_away: int = 0

    away: TeamState = field(default_factory=TeamState)
    home: TeamState = field(default_factory=TeamState)

    game_over: bool = False
    backstop_fired: bool = False
    total_pas: int = 0
    home_pitching_changes: list = field(default_factory=list)
    away_pitching_changes: list = field(default_factory=list)

    inning_runs_away: list[int] = field(default_factory=lambda: [0])
    inning_runs_home: list[int] = field(default_factory=lambda: [0])

    @property
    def batting_team(self) -> TeamState:
        return self.away if self.is_top else self.home

    @property
    def pitching_team(self) -> TeamState:
        return self.home if self.is_top else self.away

    def current_batter_id(self) -> int:
        return self.batting_team.current_batter()

    def current_pitcher_id(self) -> int:
        return self.pitching_team.current_pitcher_id

    def step(
        self,
        outcome: str,
        transition: BaseOutTransition,
        rng: np.random.Generator,
    ) -> int:
        """Advance the game by one PA with the given outcome.

        Returns the number of runs scored on this PA.
        """
        to_state_idx, runs = transition.sample(
            self.base_state, self.outs, outcome, rng
        )

        if self.is_top:
            self.score_away += runs
            self.inning_runs_away[self.inning - 1] += runs
        else:
            self.score_home += runs
            self.inning_runs_home[self.inning - 1] += runs

        self.batting_team.advance_lineup()
        self.pitching_team.batters_faced += 1
        self.total_pas += 1

        if to_state_idx == INNING_OVER:
            self._end_half_inning()
        else:
            self.base_state = to_state_idx // 3
            self.outs = to_state_idx % 3

        self._check_game_end()
        return runs

    def _end_half_inning(self) -> None:
        """Flip to next half-inning, reset bases/outs."""
        if not self.is_top:
            self.inning += 1
        self.is_top = not self.is_top
        self.outs = 0
        self.base_state = BASES_EMPTY

        while len(self.inning_runs_away) < self.inning:
            self.inning_runs_away.append(0)
        while len(self.inning_runs_home) < self.inning:
            self.inning_runs_home.append(0)

        if self.inning > 9 and not self.game_over:
            self.base_state = ON_2B

    def _check_game_end(self) -> None:
        """Check if the game is over."""
        if self.inning > 9 and self.outs == 0 and self.base_state in (BASES_EMPTY, ON_2B):
            if self.is_top and self.score_home != self.score_away:
                if self.inning > 9 or (self.inning == 10 and self.is_top):
                    pass

        if not self.is_top and self.score_home > self.score_away:
            if self.inning >= 9 or (self.inning > 9):
                self.game_over = True
                return

        at_natural_break = (self.outs == 0 and self.base_state in (BASES_EMPTY, ON_2B))
        if at_natural_break and self.inning > 9:
            if self.score_home != self.score_away:
                self.game_over = True
                return

        if self.inning > MAX_INNINGS:
            self.game_over = True
            self.backstop_fired = True

    def is_walk_off_possible(self) -> bool:
        """True when home team is batting in bottom of 9th or later."""
        return not self.is_top and self.inning >= 9


@dataclass
class PitchingChange:
    """Record of a pitching change during simulation."""
    inning: int
    is_top: bool
    batters_faced_by_outgoing: int
    outgoing_pitcher_id: int
    incoming_pitcher_id: int


@dataclass
class GameResult:
    """Outcome of a single simulated game."""

    score_home: int
    score_away: int
    total_innings: int
    total_pas: int
    backstop_fired: bool = False
    home_pitching_changes: list[PitchingChange] = field(default_factory=list)
    away_pitching_changes: list[PitchingChange] = field(default_factory=list)
    inning_runs_away: list[int] = field(default_factory=list)
    inning_runs_home: list[int] = field(default_factory=list)

    @property
    def home_wins(self) -> bool:
        return self.score_home > self.score_away

    @property
    def total_runs(self) -> int:
        return self.score_home + self.score_away


def simulate_one_game(
    matchup_dists: dict[tuple[int, int], dict[str, float]],
    home_lineup: list[int],
    away_lineup: list[int],
    home_starter: int,
    away_starter: int,
    transition: BaseOutTransition,
    rng: np.random.Generator,
    bullpen_policy=None,
    apply_tto: bool = True,
    park_factors: dict[str, float] | None = None,
) -> GameResult:
    """Simulate one complete game.

    Args:
        matchup_dists: {(pitcher_id, batter_id): {outcome: prob}} from matchup cards
        home_lineup: list of 9 batter IDs in batting order
        away_lineup: list of 9 batter IDs in batting order
        home_starter: pitcher ID for home team starter
        away_starter: pitcher ID for away team starter
        transition: loaded base-out transition matrix
        rng: numpy random generator (for reproducibility)
        bullpen_policy: optional pitching change policy
        apply_tto: if True, apply times-through-order adjustments to starters
        park_factors: optional {outcome: multiplier} for the home ballpark
    """
    state = GameState(
        away=TeamState(lineup=list(away_lineup), starter_id=away_starter, current_pitcher_id=away_starter),
        home=TeamState(lineup=list(home_lineup), starter_id=home_starter, current_pitcher_id=home_starter),
    )

    outcome_classes = list(matchup_dists.get(
        (away_starter, home_lineup[0]),
        {"K": 0.2, "BB": 0.08, "1B": 0.15, "2B": 0.05, "3B": 0.005, "HR": 0.03, "out": 0.485}
    ).keys())

    starters = {home_starter, away_starter}
    # Track (pitcher_id, batter_id) -> times faced this game
    matchup_count: dict[tuple[int, int], int] = {}

    while not state.game_over:
        if bullpen_policy is not None:
            bullpen_policy.maybe_change_pitcher(state)

        pitcher_id = state.current_pitcher_id()
        batter_id = state.current_batter_id()

        dist = matchup_dists.get((pitcher_id, batter_id))
        if dist is None:
            dist = _league_average_dist()

        probs = np.array([dist.get(oc, 0.0) for oc in outcome_classes])
        total = probs.sum()
        if total <= 0:
            probs = np.array([0.2, 0.08, 0.15, 0.05, 0.005, 0.03, 0.485])
            total = probs.sum()
        probs = probs / total

        matchup_key = (pitcher_id, batter_id)
        matchup_count[matchup_key] = matchup_count.get(matchup_key, 0) + 1
        tto = matchup_count[matchup_key]

        if apply_tto and pitcher_id in starters and tto >= 2:
            tto_key = min(tto, 3)
            mults = TTO_MULTIPLIERS.get(tto_key)
            if mults:
                for i, oc in enumerate(outcome_classes):
                    probs[i] *= mults.get(oc, 1.0)
                probs = probs / probs.sum()

        if park_factors:
            for i, oc in enumerate(outcome_classes):
                probs[i] *= park_factors.get(oc, 1.0)
            probs = probs / probs.sum()

        chosen_idx = rng.choice(len(outcome_classes), p=probs)
        outcome = outcome_classes[chosen_idx]

        state.step(outcome, transition, rng)

        if state.total_pas > 200:
            state.game_over = True
            state.backstop_fired = True

    return GameResult(
        score_home=state.score_home,
        score_away=state.score_away,
        total_innings=state.inning if state.is_top else state.inning,
        total_pas=state.total_pas,
        backstop_fired=state.backstop_fired,
        home_pitching_changes=state.home_pitching_changes,
        away_pitching_changes=state.away_pitching_changes,
        inning_runs_away=state.inning_runs_away,
        inning_runs_home=state.inning_runs_home,
    )


def _league_average_dist() -> dict[str, float]:
    """Fallback outcome distribution when a matchup isn't in the card."""
    return {
        "K": 0.220,
        "BB": 0.087,
        "1B": 0.145,
        "2B": 0.046,
        "3B": 0.004,
        "HR": 0.034,
        "out": 0.464,
    }
