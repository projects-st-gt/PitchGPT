"""Unit tests for game-metadata parsing.

No network. The full scrape is exercised separately via ``make extract-meta-test``
once Statcast extraction has produced day files.
"""

from __future__ import annotations

import json

import pytest

from data.extract_game_metadata import (
    _parse_wind,
    _safe_int,
    append_ndjson,
    load_checkpoint,
    parse_game_feed,
    save_checkpoint,
)


# ---------- _safe_int ----------


def test_safe_int_handles_strings_and_none():
    assert _safe_int("72") == 72
    assert _safe_int("0") == 0
    assert _safe_int(None) is None
    assert _safe_int("") is None
    assert _safe_int("not a number") is None


# ---------- _parse_wind ----------


def test_parse_wind_normal_case():
    speed, direction = _parse_wind("7 mph, In From CF")
    assert speed == 7
    assert direction == "In From CF"


def test_parse_wind_zero_with_none_direction():
    # The API uses literal 'None' when wind speed is 0.
    speed, direction = _parse_wind("0 mph, None")
    assert speed == 0
    assert direction is None


def test_parse_wind_empty_or_missing():
    assert _parse_wind(None) == (None, None)
    assert _parse_wind("") == (None, None)


# ---------- parse_game_feed ----------


def _fake_feed(weather=None, officials=None, datetime=None, venue=None):
    return {
        "gameData": {
            "weather": weather or {},
            "datetime": datetime or {},
            "venue": venue or {},
        },
        "liveData": {"boxscore": {"officials": officials or []}},
    }


def test_parse_game_feed_full_response():
    feed = _fake_feed(
        weather={"condition": "Sunny", "temp": "68", "wind": "12 mph, Out To CF"},
        officials=[
            {
                "officialType": "Home Plate",
                "official": {"id": 427520, "fullName": "Larry Vanover"},
            },
            {"officialType": "First Base", "official": {"id": 100001}},
            {"officialType": "Second Base", "official": {"id": 100002}},
            {"officialType": "Third Base", "official": {"id": 100003}},
        ],
        datetime={"officialDate": "2024-04-01", "dateTime": "2024-04-01T19:07:00Z"},
        venue={"id": 1, "name": "Yankee Stadium"},
    )
    row = parse_game_feed(747218, feed)

    assert row["game_pk"] == 747218
    assert row["hp_umpire_id"] == 427520
    assert row["hp_umpire_name"] == "Larry Vanover"
    assert row["fb_umpire_id"] == 100001
    assert row["sb_umpire_id"] == 100002
    assert row["tb_umpire_id"] == 100003
    assert row["temp_f"] == 68
    assert row["weather_condition"] == "Sunny"
    assert row["wind_speed_mph"] == 12
    assert row["wind_direction"] == "Out To CF"
    assert row["roof_closed"] is False
    assert row["venue_id"] == 1
    assert row["venue_name"] == "Yankee Stadium"
    assert row["game_date"] == "2024-04-01"


def test_parse_game_feed_indoor_game_marked_roof_closed():
    feed = _fake_feed(
        weather={"condition": "Roof Closed", "temp": "72", "wind": "0 mph, None"}
    )
    row = parse_game_feed(1, feed)
    assert row["roof_closed"] is True
    assert row["wind_speed_mph"] == 0
    assert row["wind_direction"] is None
    assert row["temp_f"] == 72


def test_parse_game_feed_handles_missing_everything():
    row = parse_game_feed(1, {})
    assert row["game_pk"] == 1
    assert row["hp_umpire_id"] is None
    assert row["temp_f"] is None
    assert row["weather_condition"] is None
    assert row["roof_closed"] is False
    assert row["wind_speed_mph"] is None


def test_parse_game_feed_handles_partial_officials():
    # Some games have fewer than 4 listed officials.
    feed = _fake_feed(
        officials=[
            {"officialType": "Home Plate", "official": {"id": 999, "fullName": "X"}}
        ]
    )
    row = parse_game_feed(1, feed)
    assert row["hp_umpire_id"] == 999
    assert row["fb_umpire_id"] is None
    assert row["sb_umpire_id"] is None
    assert row["tb_umpire_id"] is None


# ---------- checkpoint + ndjson ----------


def test_load_missing_checkpoint(tmp_path):
    assert load_checkpoint(tmp_path / "missing.json") == set()


def test_save_then_load_checkpoint_roundtrips(tmp_path):
    cp = tmp_path / "cp.json"
    save_checkpoint(cp, {747218, 747058, 746899})
    assert load_checkpoint(cp) == {747218, 747058, 746899}


def test_append_ndjson_writes_one_line_per_call(tmp_path):
    out = tmp_path / "games.ndjson"
    append_ndjson(out, {"game_pk": 1, "hp_umpire_id": 100})
    append_ndjson(out, {"game_pk": 2, "hp_umpire_id": 200})
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["game_pk"] == 1
    assert json.loads(lines[1])["game_pk"] == 2
