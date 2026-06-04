"""Tests for the profile cache loader.

Builds tiny synthetic per-player + league caches in a tmp directory, then
verifies the loader's lookup logic: per-player blending, league fallback,
zero fallback, and the schema-version refusal path.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.profile_cache import (
    BATTER_VECTOR_LEN,
    PITCHER_FEATURE_INDEX,
    PITCHER_VECTOR_LEN,
    PROFILE_SCHEMA_VERSION,
)
from data.profile_cache_loader import ProfileCache


def _write_cache(
    tmp_path,
    role,
    fold_id,
    player_rows,
    league_rows,
    *,
    skip_league=False,
    schema_version_override=None,
):
    """Helper: write per-player and league parquets to tmp_path."""
    sv = schema_version_override or PROFILE_SCHEMA_VERSION
    pdf = pd.DataFrame([
        {
            "player_id": int(pid), "asof_date": pd.Timestamp(asof),
            "asof_game_num": int(num), "fold_id": int(fold_id),
            "schema_version": sv, "vector": list(vec),
        }
        for (pid, asof, num, vec) in player_rows
    ])
    pdf.to_parquet(tmp_path / f"{role}_fold_{fold_id}.parquet", index=False)

    if not skip_league:
        ldf = pd.DataFrame([
            {
                "asof_date": pd.Timestamp(asof),
                "asof_game_num": int(num),
                "fold_id": int(fold_id),
                "schema_version": sv,
                "vector": list(vec),
                "n_players_in_mean": 99,
            }
            for (asof, num, vec) in league_rows
        ])
        ldf.to_parquet(tmp_path / f"league_{role}_fold_{fold_id}.parquet", index=False)


def _vec(value, length=PITCHER_VECTOR_LEN):
    """Make a length-N vector filled with ``value``."""
    return np.full(length, value, dtype=np.float32)


# ============================================================
# Loading + schema verification
# ============================================================


def test_loader_reads_per_player_and_league_caches(tmp_path):
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, _vec(1.0))],
        league_rows=[("2024-04-15", 1, _vec(10.0))],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    assert len(pc) == 1


def test_loader_refuses_stale_schema_version(tmp_path):
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, _vec(1.0))],
        league_rows=[("2024-04-15", 1, _vec(10.0))],
        schema_version_override=PROFILE_SCHEMA_VERSION - 1,
    )
    with pytest.raises(RuntimeError, match="schema_version"):
        ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)


def test_loader_validates_vector_length(tmp_path):
    bad_vec = np.full(PITCHER_VECTOR_LEN - 5, 1.0, dtype=np.float32)
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, bad_vec)],
        league_rows=[("2024-04-15", 1, bad_vec)],
    )
    with pytest.raises(RuntimeError, match="vector length"):
        ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)


def test_loader_raises_on_missing_per_player_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="per-player cache"):
        ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)


def test_loader_tolerates_missing_league_cache(tmp_path):
    """League-mean missing is non-fatal; loader falls back to zero-fill."""
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, _vec(1.0))],
        league_rows=[],
        skip_league=True,
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(123, "2024-04-15", 1)
    # Per-player vec has all 1.0s, no NaNs → blend with zero-fallback returns
    # values close to 1.0 weighted by confidence.
    assert out["source"] == "per_player_blended"


# ============================================================
# Lookup behavior
# ============================================================


def _confidence_idx_pitcher():
    return PITCHER_FEATURE_INDEX["profile_confidence"]


def test_lookup_per_player_full_confidence_returns_per_player_unchanged(tmp_path):
    """Confidence = 1.0 → blend should give exactly the per-player vec."""
    per = _vec(2.0)
    per[_confidence_idx_pitcher()] = 1.0
    league = _vec(10.0)
    league[_confidence_idx_pitcher()] = 0.5
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, per)],
        league_rows=[("2024-04-15", 1, league)],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(123, "2024-04-15", 1)
    assert out["source"] == "per_player_blended"
    np.testing.assert_allclose(out["vector"], per, atol=1e-6)


def test_lookup_per_player_zero_confidence_returns_league(tmp_path):
    per = _vec(2.0)
    per[_confidence_idx_pitcher()] = 0.0
    league = _vec(10.0)
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, per)],
        league_rows=[("2024-04-15", 1, league)],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(123, "2024-04-15", 1)
    np.testing.assert_allclose(out["vector"], league, atol=1e-6)


def test_lookup_missing_player_returns_league_only(tmp_path):
    """Debut player at a known asof: fall back to the league mean."""
    league = _vec(10.0)
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, _vec(2.0))],
        league_rows=[("2024-04-15", 1, league)],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(999, "2024-04-15", 1)  # 999 not in cache
    assert out["source"] == "league_only"
    np.testing.assert_allclose(out["vector"], league)


def test_lookup_missing_player_and_league_returns_zero(tmp_path):
    """Edge case: asof falls outside any cached league entry too."""
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, _vec(2.0))],
        league_rows=[("2024-04-15", 1, _vec(10.0))],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(999, "2025-09-01", 1)  # neither player nor asof match
    assert out["source"] == "zero_fallback"
    assert np.allclose(out["vector"], 0.0)


def test_lookup_nan_slots_in_per_player_get_league_values(tmp_path):
    """The canonical debut/early-corpus case: per-player has NaN where data
    is missing; league mean fills in; final vector has no NaN."""
    per = _vec(np.nan)
    per[_confidence_idx_pitcher()] = 0.0  # debut — no data
    league = _vec(10.0)
    league[_confidence_idx_pitcher()] = 0.5  # league has reasonable confidence
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2024-04-15", 1, per)],
        league_rows=[("2024-04-15", 1, league)],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(123, "2024-04-15", 1)
    # With confidence=0, all slots take league value (NaN slots filled, then
    # weighted with confidence=0 → 100% league).
    assert not np.any(np.isnan(out["vector"]))
    np.testing.assert_allclose(out["vector"], league, atol=1e-6)


# ============================================================
# Both roles
# ============================================================


def test_loader_works_for_batter_role(tmp_path):
    league_b = np.full(BATTER_VECTOR_LEN, 1.0, dtype=np.float32)
    per_b = np.full(BATTER_VECTOR_LEN, 0.5, dtype=np.float32)
    _write_cache(
        tmp_path, "batter", 0,
        player_rows=[(456, "2024-04-15", 1, per_b)],
        league_rows=[("2024-04-15", 1, league_b)],
    )
    pc = ProfileCache(role="batter", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(456, "2024-04-15", 1)
    assert out["vector"].shape == (BATTER_VECTOR_LEN,)


def test_loader_rejects_unknown_role():
    with pytest.raises(ValueError, match="role"):
        ProfileCache(role="catcher", fold_id=0)


# ============================================================
# Final zero-fill: NaN-NaN cascade is safe for the model
# ============================================================


def test_lookup_final_output_is_always_nan_free(tmp_path):
    """Even when both per-player and league-mean have NaN in the same slot,
    the final returned vector must be NaN-free so model gradients don't
    poison."""
    per = _vec(np.nan)
    per[_confidence_idx_pitcher()] = 0.0
    league = _vec(np.nan)  # also NaN — first-corpus-day case
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2017-04-02", 1, per)],
        league_rows=[("2017-04-02", 1, league)],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(123, "2017-04-02", 1)
    assert not np.any(np.isnan(out["vector"])), \
        "loader must not return NaN even when both per-player and league are NaN"


def test_lookup_league_only_path_also_nan_free(tmp_path):
    """Debut player whose league mean has NaNs: still must return clean vec."""
    league = _vec(np.nan)
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[(123, "2017-04-02", 1, _vec(1.0))],
        league_rows=[("2017-04-02", 1, league)],
    )
    pc = ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)
    out = pc.lookup(999, "2017-04-02", 1)  # unknown player
    assert out["source"] == "league_only"
    assert not np.any(np.isnan(out["vector"]))


# ============================================================
# As-of fallback (pre-game prediction beyond the cache horizon)
# ============================================================

def _asof_cache(tmp_path):
    """Player 1 has profiles at three dates; league skipped so the as-of path
    returns the selected player vector verbatim (clean equality checks)."""
    _write_cache(
        tmp_path, "pitcher", 0,
        player_rows=[
            (1, "2024-05-01", 1, _vec(1.0)),
            (1, "2024-06-01", 1, _vec(2.0)),
            (1, "2024-07-01", 1, _vec(3.0)),
        ],
        league_rows=[],
        skip_league=True,
    )
    return ProfileCache(role="pitcher", fold_id=0, profiles_dir=tmp_path)


def test_asof_exact_date_still_wins(tmp_path):
    pc = _asof_cache(tmp_path)
    out = pc.lookup(1, "2024-06-01", 1, as_of_fallback=True)
    assert out["source"] == "per_player_blended"
    np.testing.assert_allclose(out["vector"], _vec(2.0))


def test_asof_returns_latest_prior_when_exact_missing(tmp_path):
    pc = _asof_cache(tmp_path)
    out = pc.lookup(1, "2024-08-01", 1, as_of_fallback=True)  # beyond all profiles
    assert out["source"] == "per_player_asof"
    np.testing.assert_allclose(out["vector"], _vec(3.0))  # latest (2024-07-01)


def test_asof_is_strictly_before_no_leakage(tmp_path):
    pc = _asof_cache(tmp_path)
    # mid-range date: must pick 2024-06-01 (strictly before), NOT 2024-07-01
    out = pc.lookup(1, "2024-06-15", 1, as_of_fallback=True)
    assert out["source"] == "per_player_asof"
    np.testing.assert_allclose(out["vector"], _vec(2.0))
    # same date but EARLIER game_num than the stored (07-01, 1): must not grab
    # the 07-01 profile (not strictly before) -> falls back to 06-01
    out2 = pc.lookup(1, "2024-07-01", 0, as_of_fallback=True)
    assert out2["source"] == "per_player_asof"
    np.testing.assert_allclose(out2["vector"], _vec(2.0))


def test_asof_off_by_default(tmp_path):
    pc = _asof_cache(tmp_path)
    out = pc.lookup(1, "2024-08-01", 1)  # default: no as-of
    assert out["source"] == "zero_fallback"  # league skipped -> zero


def test_asof_no_fabrication_for_player_without_history(tmp_path):
    pc = _asof_cache(tmp_path)
    out = pc.lookup(999, "2024-08-01", 1, as_of_fallback=True)  # unknown player
    assert out["source"] == "zero_fallback"  # as-of never invents a profile
