"""SQLite persistence for MCSim predictions + actuals.

Single-file SQLite at ``data/mcsim.sqlite`` (gitignored). Three tables —
the schema is defined inline so a fresh DB bootstraps from
:func:`init_db`:

- ``predictions``     — one row per ``(game_pk, prediction_date, app)``;
                        ``payload_json`` carries the full per-game matchup
                        grid or score distribution.
- ``actuals``         — one row per ``game_pk`` once the game ends;
                        ``matchup_events_json`` carries the per-(P, B) ABs
                        that actually occurred.
- ``model_versions``  — provenance: which checkpoint produced which
                        predictions, with a human label.

Stdlib only (``sqlite3``). No ORM, no migrations beyond ``CREATE TABLE IF
NOT EXISTS`` — version with ``PRAGMA user_version`` so the next migration
gets a clear hook.

Public API:

- :func:`init_db`                 — open/create the DB; idempotent.
- :func:`write_prediction`        — upsert one (game_pk, date, app) row.
- :func:`read_prediction`         — fetch one row.
- :func:`read_predictions_for_date` — list payloads for a date (carousel).
- :func:`write_actual`            — upsert one game_pk row.
- :func:`read_actual`             — fetch one row.
- :func:`register_model_version`  — idempotent provenance insert.

JSON is decoded into Python dicts on read; callers don't need to call
``json.loads``.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path("data/mcsim.sqlite")
SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk         INTEGER NOT NULL,
    prediction_date TEXT    NOT NULL,                  -- YYYY-MM-DD (the *game* date)
    made_at         TEXT    NOT NULL,                  -- ISO 8601 UTC timestamp
    model_ckpt_hash TEXT    NOT NULL,
    app             TEXT    NOT NULL,                  -- "matchup_card" | "score_prediction"
    payload_json    TEXT    NOT NULL,
    UNIQUE (game_pk, prediction_date, app)
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions (prediction_date);

CREATE TABLE IF NOT EXISTS actuals (
    game_pk             INTEGER PRIMARY KEY,
    fetched_at          TEXT NOT NULL,                 -- ISO 8601 UTC
    final_score_home    INTEGER,
    final_score_away    INTEGER,
    winner              TEXT,                          -- home | away | tie | NULL (not finished)
    matchup_events_json TEXT                           -- per-(P,B) actual ABs
);

CREATE TABLE IF NOT EXISTS model_versions (
    ckpt_hash  TEXT PRIMARY KEY,
    trained_at TEXT,
    label      TEXT,                                   -- e.g. "tiny-fold0-v7"
    notes      TEXT
);
"""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN, +/-Inf) with None.

    NaN run-values arise legitimately for all-truncated cells (debut/low-data
    players whose rollout paths never terminate). Python's ``json`` emits these
    as bare ``NaN``/``Infinity`` tokens, which are invalid JSON and rejected by
    strict parsers (notably JavaScript ``JSON.parse`` in the frontend). Coercing
    to ``null`` keeps the stored payload spec-valid; such cells read as "no value".
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection to the MCSim DB, creating tables if absent.

    Idempotent: safe to call on every script start. Sets
    ``PRAGMA user_version = SCHEMA_VERSION`` so a future migration can
    branch on the on-disk version.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI runs a sync dependency and its endpoint in
    # different threadpool threads, so a per-request connection is created in one
    # thread and used in another. We never use one connection concurrently (each
    # request gets its own, used sequentially), so this is safe; single-threaded
    # callers (runner, tests) are unaffected.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    # PRAGMA needs its own statement to read back the value
    cur = conn.execute("PRAGMA user_version")
    current = int(cur.fetchone()[0])
    if current == 0:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    elif current != SCHEMA_VERSION:
        raise RuntimeError(
            f"mcsim DB at {db_path} has user_version={current} but code expects "
            f"{SCHEMA_VERSION}. A migration is needed before continuing."
        )
    conn.commit()
    return conn


# ============================================================
# Predictions
# ============================================================


def write_prediction(
    conn: sqlite3.Connection,
    *,
    game_pk: int,
    prediction_date: str,
    app: str,
    payload: dict,
    model_ckpt_hash: str,
    made_at: Optional[str] = None,
) -> int:
    """Upsert one prediction row. Returns the rowid.

    Uses ``ON CONFLICT … DO UPDATE`` so re-running the nightly job (or a
    day-of refresh per D5) overwrites the prior row for the same key.
    """
    if app not in ("matchup_card", "score_prediction"):
        raise ValueError(f"unknown app {app!r}; expected 'matchup_card' or 'score_prediction'")
    made_at = made_at or _now_utc_iso()
    # allow_nan=False rejects bare NaN/Infinity (valid in Python's json but not
    # in the JSON spec — JS JSON.parse on the frontend chokes on them); _json_safe
    # first coerces legitimate non-finite values (e.g. an all-truncated cell's run
    # value) to null so the write succeeds and the cell reads as "no value".
    payload_json = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    cur = conn.execute(
        """
        INSERT INTO predictions
            (game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_pk, prediction_date, app) DO UPDATE SET
            made_at = excluded.made_at,
            model_ckpt_hash = excluded.model_ckpt_hash,
            payload_json = excluded.payload_json
        """,
        (game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json),
    )
    conn.commit()
    return int(cur.lastrowid)


def read_prediction(
    conn: sqlite3.Connection,
    *,
    game_pk: int,
    prediction_date: str,
    app: str,
) -> Optional[dict]:
    """Fetch one prediction row by its unique key; ``None`` if absent.

    Returns a plain dict with the row columns plus ``payload`` (the
    decoded JSON, not the raw string).
    """
    row = conn.execute(
        """
        SELECT id, game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json
        FROM predictions
        WHERE game_pk = ? AND prediction_date = ? AND app = ?
        """,
        (game_pk, prediction_date, app),
    ).fetchone()
    if row is None:
        return None
    return _row_to_prediction_dict(row)


def read_predictions_for_date(
    conn: sqlite3.Connection,
    *,
    prediction_date: str,
    app: Optional[str] = None,
) -> list[dict]:
    """List all predictions for one date (and optionally one app).

    Powers the date-carousel UI: one call returns every game's card for the
    selected day. Ordered by ``game_pk`` for stable presentation.
    """
    if app is None:
        rows = conn.execute(
            """
            SELECT id, game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json
            FROM predictions
            WHERE prediction_date = ?
            ORDER BY game_pk, app
            """,
            (prediction_date,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, game_pk, prediction_date, made_at, model_ckpt_hash, app, payload_json
            FROM predictions
            WHERE prediction_date = ? AND app = ?
            ORDER BY game_pk
            """,
            (prediction_date, app),
        ).fetchall()
    return [_row_to_prediction_dict(r) for r in rows]


def _row_to_prediction_dict(row: sqlite3.Row) -> dict:
    return {
        "id": int(row["id"]),
        "game_pk": int(row["game_pk"]),
        "prediction_date": row["prediction_date"],
        "made_at": row["made_at"],
        "model_ckpt_hash": row["model_ckpt_hash"],
        "app": row["app"],
        "payload": json.loads(row["payload_json"]),
    }


# ============================================================
# Actuals (post-game)
# ============================================================


def write_actual(
    conn: sqlite3.Connection,
    *,
    game_pk: int,
    final_score_home: Optional[int] = None,
    final_score_away: Optional[int] = None,
    winner: Optional[str] = None,
    matchup_events: Optional[list] = None,
    fetched_at: Optional[str] = None,
) -> None:
    """Upsert one ``actuals`` row.

    Two-pass design (see D5 / open questions in the brainstorm): a first
    pass right after the game ends populates line-score fields; a second
    pass the next morning, once Statcast enrichment lands, fills the
    per-(P, B) ``matchup_events_json``. Either pass can call this with
    only the fields it has — missing fields stay as in the existing row
    (no nulling out).
    """
    if winner is not None and winner not in ("home", "away", "tie"):
        raise ValueError(f"winner must be 'home', 'away', or 'tie'; got {winner!r}")
    fetched_at = fetched_at or _now_utc_iso()
    matchup_events_json = (
        json.dumps(matchup_events, sort_keys=True, separators=(",", ":"))
        if matchup_events is not None else None
    )
    # COALESCE-on-update keeps existing fields when this call doesn't carry them.
    conn.execute(
        """
        INSERT INTO actuals
            (game_pk, fetched_at, final_score_home, final_score_away, winner, matchup_events_json)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_pk) DO UPDATE SET
            fetched_at = excluded.fetched_at,
            final_score_home    = COALESCE(excluded.final_score_home, actuals.final_score_home),
            final_score_away    = COALESCE(excluded.final_score_away, actuals.final_score_away),
            winner              = COALESCE(excluded.winner, actuals.winner),
            matchup_events_json = COALESCE(excluded.matchup_events_json, actuals.matchup_events_json)
        """,
        (game_pk, fetched_at, final_score_home, final_score_away, winner, matchup_events_json),
    )
    conn.commit()


def read_actual(conn: sqlite3.Connection, *, game_pk: int) -> Optional[dict]:
    """Fetch one actuals row by game_pk; ``None`` if not yet ingested."""
    row = conn.execute(
        """
        SELECT game_pk, fetched_at, final_score_home, final_score_away,
               winner, matchup_events_json
        FROM actuals WHERE game_pk = ?
        """,
        (game_pk,),
    ).fetchone()
    if row is None:
        return None
    return {
        "game_pk": int(row["game_pk"]),
        "fetched_at": row["fetched_at"],
        "final_score_home": row["final_score_home"],
        "final_score_away": row["final_score_away"],
        "winner": row["winner"],
        "matchup_events": (
            json.loads(row["matchup_events_json"]) if row["matchup_events_json"] else None
        ),
    }


# ============================================================
# Model versions (provenance)
# ============================================================


def register_model_version(
    conn: sqlite3.Connection,
    *,
    ckpt_hash: str,
    label: Optional[str] = None,
    trained_at: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Idempotently record which checkpoint was used.

    Called once per checkpoint at MCSim startup; subsequent calls are no-ops
    if the hash already exists.
    """
    conn.execute(
        """
        INSERT INTO model_versions (ckpt_hash, trained_at, label, notes)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (ckpt_hash) DO NOTHING
        """,
        (ckpt_hash, trained_at, label, notes),
    )
    conn.commit()


def lookup_model_version(conn: sqlite3.Connection, *, ckpt_hash: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT ckpt_hash, trained_at, label, notes FROM model_versions WHERE ckpt_hash = ?",
        (ckpt_hash,),
    ).fetchone()
    if row is None:
        return None
    return {
        "ckpt_hash": row["ckpt_hash"],
        "trained_at": row["trained_at"],
        "label": row["label"],
        "notes": row["notes"],
    }
