"""Trust-region-restricted causal recommendation.

Given a pitch-state (pitcher, batter, count, runners, outs, history-so-far in
the AB), :func:`rank.rank_pitch_types` returns a ranked list of pitch-type
candidates by expected run value (lowest = best for the pitcher), with the
positivity gate refusing candidates the data can't confidently support.

See :doc:`docs/recommender_brainstorm` for the design decisions (D1–D6).
The public surface today:

- :class:`rank.CandidateRanking`     — one ranked or refused candidate
- :class:`rank.RankedRecommendations` — the full result envelope
- :func:`rank.rank_pitch_types`      — the ranking function
"""
from recommender.rank import (
    CandidateRanking,
    RankedRecommendations,
    rank_pitch_types,
)

__all__ = ["CandidateRanking", "RankedRecommendations", "rank_pitch_types"]
