"""Unit tests for extraction checkpoint, parquet writing, and the timeout wrapper.

No network. The live pybaseball call is exercised separately via ``make extract-test``.
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import pytest

import data.extract_statcast as ext
from data.extract_statcast import (
    _daterange,
    fetch_day,
    load_checkpoint,
    save_checkpoint,
    write_day_parquet,
)


def test_load_missing_checkpoint(tmp_path):
    state = load_checkpoint(tmp_path / "missing.json")
    assert state == {"completed_dates": []}


def test_save_then_load_checkpoint_roundtrips(tmp_path):
    cp = tmp_path / "cp.json"
    save_checkpoint(cp, {"completed_dates": ["2024-04-01", "2024-04-02"]})
    state = load_checkpoint(cp)
    assert state == {"completed_dates": ["2024-04-01", "2024-04-02"]}


def test_save_checkpoint_leaves_no_tmp_file(tmp_path):
    cp = tmp_path / "cp.json"
    save_checkpoint(cp, {"completed_dates": ["2024-04-01"]})
    assert not cp.with_suffix(".tmp").exists()


def test_write_day_parquet_uses_year_subdirectory(tmp_path):
    df = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [95.0, 87.0]})
    out = write_day_parquet(df, date(2024, 4, 1), tmp_path)
    assert out == tmp_path / "2024" / "2024-04-01.parquet"
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 2


def test_daterange_inclusive_on_both_ends():
    days = list(_daterange(date(2024, 4, 1), date(2024, 4, 3)))
    assert days == [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 3)]


def test_daterange_single_day():
    days = list(_daterange(date(2024, 4, 1), date(2024, 4, 1)))
    assert days == [date(2024, 4, 1)]


# ---------- fetch_day timeout behavior ----------


class _FakePybaseball:
    """Stand-in for pybaseball with a controllable per-call delay."""

    def __init__(self, sleep_secs: float, df=None, exc=None):
        self.sleep_secs = sleep_secs
        self.df = df
        self.exc = exc
        self.call_count = 0

    def statcast(self, start_dt, end_dt):
        self.call_count += 1
        time.sleep(self.sleep_secs)
        if self.exc is not None:
            raise self.exc
        return self.df


def test_fetch_day_returns_dataframe_when_pybaseball_responds(monkeypatch):
    fake_df = pd.DataFrame({"pitch_type": ["FF"], "release_speed": [95.0]})
    fake = _FakePybaseball(sleep_secs=0.0, df=fake_df)
    monkeypatch.setattr(ext, "pybaseball", fake)

    out = fetch_day(date(2024, 4, 1))
    assert len(out) == 1
    assert fake.call_count == 1


def test_fetch_day_gives_up_after_max_consecutive_timeouts(monkeypatch):
    """Hung pybaseball calls eventually return an empty DataFrame, not block forever."""
    fake = _FakePybaseball(sleep_secs=10.0)  # always exceeds the patched timeout
    monkeypatch.setattr(ext, "pybaseball", fake)
    monkeypatch.setattr(ext, "PER_DAY_TIMEOUT_SEC", 0.2)
    monkeypatch.setattr(ext, "MAX_TIMEOUTS_PER_DAY", 2)
    monkeypatch.setattr(ext, "INITIAL_BACKOFF_SEC", 0.05)
    monkeypatch.setattr(ext, "MAX_BACKOFF_SEC", 0.05)

    started = time.monotonic()
    out = fetch_day(date(2024, 4, 1))
    elapsed = time.monotonic() - started

    assert out.empty
    # Two timeouts × 0.2s + one backoff × 0.05s ≈ 0.45s; allow generous slack
    # for thread scheduling. Critically, must NOT hang for 10s+.
    assert elapsed < 3.0, f"fetch_day did not give up in time: elapsed={elapsed:.2f}s"


def test_fetch_day_retries_then_succeeds_after_transient_error(monkeypatch):
    """A one-shot exception triggers backoff, then the next call succeeds."""

    class _Flaky:
        def __init__(self):
            self.calls = 0

        def statcast(self, start_dt, end_dt):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("simulated transient")
            return pd.DataFrame({"pitch_type": ["FF"]})

    fake = _Flaky()
    monkeypatch.setattr(ext, "pybaseball", fake)
    monkeypatch.setattr(ext, "INITIAL_BACKOFF_SEC", 0.01)
    monkeypatch.setattr(ext, "MAX_BACKOFF_SEC", 0.01)

    out = fetch_day(date(2024, 4, 1))
    assert len(out) == 1
    assert fake.calls == 2
