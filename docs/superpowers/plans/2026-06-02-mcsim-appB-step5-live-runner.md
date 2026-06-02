# MCSim App B Step 5+6 — Live MLB-API Matchup-Card Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI that turns a calendar date into persisted pre-game matchup cards, pulling real games + active rosters live from the MLB Stats API and gridding all rostered pitchers × all opposing position players.

**Architecture:** Two units. `mcsim/mlb_api.py` is an isolated HTTP client (schedule + roster → `PitcherSpec`/`BatterSpec`) with a single mockable network seam. `scripts/mcsim/run_matchup_cards.py` orchestrates: load nuisance once, fetch schedule, fetch both rosters per game, feed full rosters into the unchanged `compute_matchup_card`, and persist via `storage.write_prediction`. Sequential v1; no multiprocessing.

**Tech Stack:** Python, `urllib`/`requests`, `argparse`, SQLite (`mcsim.storage`), PyTorch (CPU), pytest with monkeypatched HTTP.

**Key facts verified during design (do not re-litigate):**
- statsapi player IDs are MLBAM = same namespace as Statcast `pitcher`/`batter`. No crosswalk.
- Active roster: `GET /api/v1/teams/{id}/roster?rosterType=active&date=YYYY-MM-DD&hydrate=person`. Split by `entry["position"]["type"] == "Pitcher"`. Handedness from `person["pitchHand"]["code"]` / `person["batSide"]["code"]` (codes `"R"`/`"L"`/`"S"`).
- Schedule: `GET /api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher`. Probable id at `g["teams"][side].get("probablePitcher",{}).get("id")` (may be absent).
- `compute_matchup_card(nuisance, *, game_pk, game_date, home_team, away_team, home_pitchers, away_pitchers, home_lineup, away_lineup, ballpark_id=0, umpire_id=0, catcher_home_id=0, catcher_away_id=0, n_paths=1000, rng_seed=None, context=None) -> dict` already loops arbitrary-length lists. **Do not modify it.**
- Missing profiles do NOT raise — `ProfileCache.lookup` falls back to league-mean then zeros. No per-cell skip needed; keep only a per-GAME try/except.
- `write_prediction(conn, *, game_pk, prediction_date, app, payload, model_ckpt_hash, made_at=None)`; valid `app` keys are `"matchup_card"`/`"score_prediction"` only.
- `init_db(db_path=DEFAULT_DB_PATH) -> sqlite3.Connection`; `DEFAULT_DB_PATH = Path("data/mcsim.sqlite")`.
- `register_model_version(conn, *, ckpt_hash, label=None, trained_at=None, notes=None)`.
- `NuisanceModels(ckpt_path, device="cpu")`; default ckpt `checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt`.
- All test infra forces CPU: `torch.backends.mps.is_available = lambda: False` (MPS AB-outcome gather bug).

---

## File Structure

- Create: `mcsim/mlb_api.py` — MLB Stats API client (`GameInfo`, `get_schedule`, `get_active_roster`, `_get_json` seam).
- Create: `scripts/mcsim/__init__.py` — package marker (empty).
- Create: `scripts/mcsim/run_matchup_cards.py` — `compute_ckpt_hash`, `run_matchup_cards`, `main`.
- Create: `tests/test_mcsim_mlb_api.py` — client parsing tests (mocked HTTP).
- Create: `tests/test_mcsim_run_matchup_cards.py` — hash test + end-to-end orchestration test.
- Unchanged: `mcsim/matchup_card.py`, `mcsim/storage.py`, `causal/*`.

---

## Task 1: `mcsim/mlb_api.py` — `GameInfo` + `get_schedule`

**Files:**
- Create: `mcsim/mlb_api.py`
- Test: `tests/test_mcsim_mlb_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcsim_mlb_api.py
"""Tests for mcsim.mlb_api — schedule + roster parsing against canned JSON.

No live network in CI: every test patches mcsim.mlb_api._get_json (the single
network seam) with canned statsapi-shaped dicts.
"""
from __future__ import annotations

import mcsim.mlb_api as mlb_api
from mcsim.matchup_card import BatterSpec, PitcherSpec

_SCHED_JSON = {
    "dates": [{"games": [{
        "gamePk": 776543,
        "teams": {
            "home": {"team": {"id": 139, "name": "Tampa Bay Rays"},
                     "probablePitcher": {"id": 111, "fullName": "Home SP"}},
            "away": {"team": {"id": 116, "name": "Detroit Tigers"},
                     "probablePitcher": {"id": 222, "fullName": "Away SP"}},
        },
    }]}]
}


def test_get_schedule_parses_games(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: _SCHED_JSON)
    games = mlb_api.get_schedule("2026-06-02")
    assert len(games) == 1
    g = games[0]
    assert g.game_pk == 776543
    assert g.home_team_id == 139 and g.away_team_id == 116
    assert g.home_team == "Tampa Bay Rays" and g.away_team == "Detroit Tigers"
    assert g.home_probable_pitcher_id == 111
    assert g.away_probable_pitcher_id == 222


def test_get_schedule_handles_missing_probable(monkeypatch):
    j = {"dates": [{"games": [{
        "gamePk": 1, "teams": {
            "home": {"team": {"id": 1, "name": "H"}},
            "away": {"team": {"id": 2, "name": "A"}},
        }}]}]}
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: j)
    g = mlb_api.get_schedule("2026-06-02")[0]
    assert g.home_probable_pitcher_id is None
    assert g.away_probable_pitcher_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcsim_mlb_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcsim.mlb_api'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mcsim/mlb_api.py
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


def get_schedule(date: str) -> list[GameInfo]:
    """Return one GameInfo per scheduled game on ``date`` (YYYY-MM-DD)."""
    url = f"{API_BASE}/schedule?sportId=1&date={date}&hydrate=probablePitcher"
    data = _get_json(url)
    games: list[GameInfo] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
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
    return games
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcsim_mlb_api.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mcsim/mlb_api.py tests/test_mcsim_mlb_api.py
git commit -m "feat(mcsim): MLB Stats API schedule client (App B Step 5)"
```

---

## Task 2: `mcsim/mlb_api.py` — `get_active_roster` + handedness + switch-hitter resolution

**Files:**
- Modify: `mcsim/mlb_api.py`
- Test: `tests/test_mcsim_mlb_api.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_mcsim_mlb_api.py

_ROSTER_JSON = {
    "roster": [
        {"position": {"type": "Pitcher", "abbreviation": "P"},
         "person": {"id": 111, "fullName": "Righty Starter",
                    "pitchHand": {"code": "R"}, "batSide": {"code": "R"}}},
        {"position": {"type": "Pitcher", "abbreviation": "P"},
         "person": {"id": 112, "fullName": "Lefty Reliever",
                    "pitchHand": {"code": "L"}, "batSide": {"code": "L"}}},
        {"position": {"type": "Infielder", "abbreviation": "2B"},
         "person": {"id": 201, "fullName": "Switch Hitter",
                    "pitchHand": {"code": "R"}, "batSide": {"code": "S"}}},
        {"position": {"type": "Outfielder", "abbreviation": "CF"},
         "person": {"id": 202, "fullName": "Lefty Bat",
                    "pitchHand": {"code": "L"}, "batSide": {"code": "L"}}},
    ]
}


def test_get_active_roster_splits_and_handedness(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: _ROSTER_JSON)
    pitchers, hitters = mlb_api.get_active_roster(139, "2026-06-02",
                                                  probable_pitcher_id=111)
    assert [p.id for p in pitchers] == [111, 112]
    assert [h.id for h in hitters] == [201, 202]
    assert all(isinstance(p, PitcherSpec) for p in pitchers)
    assert all(isinstance(h, BatterSpec) for h in hitters)
    # handedness
    assert pitchers[0].throws == "R" and pitchers[1].throws == "L"
    # probable starter flagged
    assert pitchers[0].is_starter is True and pitchers[1].is_starter is False
    # switch hitter resolves to 'L' for v1; explicit-side hitter unchanged
    assert hitters[0].stand == "L"   # was "S"
    assert hitters[1].stand == "L"


def test_get_active_roster_no_probable(monkeypatch):
    monkeypatch.setattr(mlb_api, "_get_json", lambda url, **kw: _ROSTER_JSON)
    pitchers, _ = mlb_api.get_active_roster(139, "2026-06-02")
    assert all(p.is_starter is False for p in pitchers)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcsim_mlb_api.py -k roster -v`
Expected: FAIL — `AttributeError: module 'mcsim.mlb_api' has no attribute 'get_active_roster'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to mcsim/mlb_api.py

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
    return pitchers, hitters
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcsim_mlb_api.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add mcsim/mlb_api.py tests/test_mcsim_mlb_api.py
git commit -m "feat(mcsim): MLB active-roster client w/ handedness + switch-hitter resolution"
```

---

## Task 3: `compute_ckpt_hash` helper in the runner module

**Files:**
- Create: `scripts/mcsim/__init__.py` (empty)
- Create: `scripts/mcsim/run_matchup_cards.py` (hash helper only for now)
- Test: `tests/test_mcsim_run_matchup_cards.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcsim_run_matchup_cards.py
"""Tests for the MCSim App B runner — checkpoint hashing + end-to-end orchestration."""
from __future__ import annotations

from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from scripts.mcsim import run_matchup_cards as runner


def test_compute_ckpt_hash_is_stable_and_truncated(tmp_path):
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"hello world")
    h1 = runner.compute_ckpt_hash(f)
    h2 = runner.compute_ckpt_hash(f)
    assert h1 == h2                      # deterministic
    assert len(h1) == 16                 # truncated
    # sha256("hello world") starts with b94d27b9934d3e08...
    assert h1 == "b94d27b9934d3e08"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcsim_run_matchup_cards.py::test_compute_ckpt_hash_is_stable_and_truncated -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.mcsim'`.

- [ ] **Step 3: Write minimal implementation**

Create empty `scripts/mcsim/__init__.py`:

```python
```

Create `scripts/mcsim/run_matchup_cards.py`:

```python
"""CLI runner for MCSim App B — pull real games + rosters from the MLB Stats API,
compute one matchup card per game, and persist via mcsim.storage.

Grid: all rostered pitchers x all opposing position players (both halves). See
docs/superpowers/specs/2026-06-02-mcsim-appB-step5-live-runner-design.md.

Sequential v1 (no multiprocessing). Per-cell cost is ~linear in n_paths
(~40ms/path on CPU); an all-vs-all game is ~338 cells, so tune --n-paths and
--max-games for the run budget. Missing profiles do not crash — debut players
get a league-mean fallback (data/profile_cache_loader.py).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

torch.backends.mps.is_available = lambda: False  # pre-existing AB-outcome MPS bug

from causal.nuisance import NuisanceModels
from mcsim.matchup_card import compute_matchup_card
from mcsim.mlb_api import get_active_roster, get_schedule
from mcsim.storage import (
    DEFAULT_DB_PATH,
    init_db,
    register_model_version,
    write_prediction,
)

DEFAULT_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")


def compute_ckpt_hash(ckpt_path: Path, *, n_chars: int = 16) -> str:
    """Truncated sha256 of the checkpoint file bytes — provenance for which
    exact weights produced a prediction. No such helper existed elsewhere."""
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcsim_run_matchup_cards.py::test_compute_ckpt_hash_is_stable_and_truncated -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/mcsim/__init__.py scripts/mcsim/run_matchup_cards.py tests/test_mcsim_run_matchup_cards.py
git commit -m "feat(mcsim): checkpoint-hash provenance helper for App B runner"
```

---

## Task 4: `run_matchup_cards` orchestration + `main` CLI

**Files:**
- Modify: `scripts/mcsim/run_matchup_cards.py`
- Test: `tests/test_mcsim_run_matchup_cards.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_mcsim_run_matchup_cards.py

import numpy as np
import pandas as pd
import pytest

from causal.g_computation import AB_OUTCOME_NAMES
from causal.nuisance import NuisanceModels
from mcsim.mlb_api import GameInfo
from mcsim.matchup_card import BatterSpec, PitcherSpec
from mcsim.storage import init_db, read_prediction

V7_CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")
VAL_DIR = Path("data/augmented/2024")

requires_v7 = pytest.mark.skipif(
    not V7_CKPT.exists(), reason=f"v7 checkpoint not present at {V7_CKPT}"
)


@requires_v7
def test_run_matchup_cards_end_to_end(tmp_path, monkeypatch, capsys):
    """Monkeypatch the API client so no network is touched; feed 2 real pitcher
    ids + 2 real batter ids (so profile lookups succeed) through the real
    nuisance + compute_matchup_card, and assert a row lands in SQLite.

    Prints a NAMED numerical output (a real cell's median RV + modal-type pi-hat)
    per CLAUDE.md bug-prevention discipline.
    """
    nuisance = NuisanceModels(V7_CKPT, device="cpu")
    df = pd.read_parquet(sorted(VAL_DIR.glob("2024-*.parquet"))[0])
    pids = [int(x) for x in df["pitcher"].drop_duplicates().head(2)]
    bids = [int(x) for x in df["batter"].drop_duplicates().head(2)]

    game = GameInfo(game_pk=776543, home_team_id=139, away_team_id=116,
                    home_team="HOME", away_team="AWAY",
                    home_probable_pitcher_id=pids[0],
                    away_probable_pitcher_id=pids[1])
    rosters = {
        139: ([PitcherSpec(pids[0], "HSP", "R", True), PitcherSpec(pids[1], "HRP", "L")],
              [BatterSpec(bids[0], "HB1", "R"), BatterSpec(bids[1], "HB2", "L")]),
        116: ([PitcherSpec(pids[1], "ASP", "L", True), PitcherSpec(pids[0], "ARP", "R")],
              [BatterSpec(bids[1], "AB1", "L"), BatterSpec(bids[0], "AB2", "R")]),
    }
    monkeypatch.setattr(runner, "get_schedule", lambda date: [game])
    monkeypatch.setattr(runner, "get_active_roster",
                        lambda team_id, date, probable_pitcher_id=None: rosters[team_id])

    conn = init_db(tmp_path / "t.sqlite")
    cards = runner.run_matchup_cards(
        nuisance, conn, date="2026-06-02", game_pks=None,
        n_paths=50, rng_seed=7, model_ckpt_hash="testhash",
    )

    assert len(cards) == 1
    got = read_prediction(conn, game_pk=776543, prediction_date="2026-06-02",
                          app="matchup_card")
    assert got is not None
    assert got["model_ckpt_hash"] == "testhash"   # provenance column round-trips
    card = got["payload"]
    # 2 halves x (2 pitchers x 2 hitters) = 8 cells
    assert card["n_cells"] == 8
    first_cell = card["rows"][0]["cells"][0]
    assert first_cell["predicted_top1_outcome"] in AB_OUTCOME_NAMES
    assert np.isfinite(first_cell["predicted_rv_median"])
    print(f"\n[runner E2E] {card['rows'][0]['name']} vs {first_cell['batter_name']}: "
          f"median RV={first_cell['predicted_rv_median']:+.4f}  "
          f"modal {first_cell['modal_type']} pi-hat={first_cell['p_hat_top_type']:.3f}  "
          f"trust={first_cell['trust_state']}")
    captured = capsys.readouterr()
    assert "median RV" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcsim_run_matchup_cards.py::test_run_matchup_cards_end_to_end -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'run_matchup_cards'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/mcsim/run_matchup_cards.py

def run_matchup_cards(
    nuisance: NuisanceModels,
    conn,
    *,
    date: str,
    game_pks: "list[int] | None",
    n_paths: int,
    rng_seed: "int | None",
    model_ckpt_hash: str,
    dry_run: bool = False,
    max_games: "int | None" = None,
) -> list[dict]:
    """Fetch the schedule for ``date``, compute one all-vs-all matchup card per
    game, and (unless dry_run) persist it. Returns the list of card payloads.

    A failure on one game is logged and skipped so it can't abort the batch.
    """
    games = get_schedule(date)
    if game_pks:
        wanted = set(game_pks)
        games = [g for g in games if g.game_pk in wanted]
    if max_games is not None:
        games = games[:max_games]

    cards: list[dict] = []
    for g in games:
        try:
            home_p, home_h = get_active_roster(
                g.home_team_id, date, probable_pitcher_id=g.home_probable_pitcher_id)
            away_p, away_h = get_active_roster(
                g.away_team_id, date, probable_pitcher_id=g.away_probable_pitcher_id)
            card = compute_matchup_card(
                nuisance,
                game_pk=g.game_pk,
                game_date=date,
                home_team=g.home_team,
                away_team=g.away_team,
                home_pitchers=home_p,
                away_pitchers=away_p,
                home_lineup=home_h,
                away_lineup=away_h,
                n_paths=n_paths,
                rng_seed=rng_seed,
            )
            if not dry_run:
                write_prediction(
                    conn,
                    game_pk=g.game_pk,
                    prediction_date=date,
                    app="matchup_card",
                    payload=card,
                    model_ckpt_hash=model_ckpt_hash,
                )
            cards.append(card)
            print(f"[{g.away_team} @ {g.home_team}] game_pk={g.game_pk}: "
                  f"{card['n_cells']} cells"
                  f"{' (dry-run, not written)' if dry_run else ' written'}")
        except Exception as e:  # one bad game must not abort the batch
            print(f"  SKIP game_pk={g.game_pk}: {type(e).__name__}: {e}")
    return cards


def main() -> None:
    ap = argparse.ArgumentParser(description="MCSim App B matchup-card runner")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--game-pk", type=int, action="append", dest="game_pks",
                    help="restrict to these game_pks (repeatable)")
    ap.add_argument("--n-paths", type=int, default=250)
    ap.add_argument("--rng-seed", type=int, default=None)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nuisance = NuisanceModels(args.ckpt, device="cpu")
    ckpt_hash = compute_ckpt_hash(args.ckpt)
    conn = init_db(args.db_path)
    register_model_version(conn, ckpt_hash=ckpt_hash, label=args.ckpt.parent.name)
    print(f"ckpt={args.ckpt}  hash={ckpt_hash}  n_paths={args.n_paths}  db={args.db_path}")

    cards = run_matchup_cards(
        nuisance, conn,
        date=args.date,
        game_pks=args.game_pks,
        n_paths=args.n_paths,
        rng_seed=args.rng_seed,
        model_ckpt_hash=ckpt_hash,
        dry_run=args.dry_run,
        max_games=args.max_games,
    )
    print(f"done: {len(cards)} card(s) for {args.date}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcsim_run_matchup_cards.py -v -s`
Expected: PASS. The `-s` shows the named line, e.g. `[runner E2E] HSP vs AB1: median RV=-0.1000 modal FS pi-hat=0.327 trust=green`. (If `V7_CKPT` is absent, the E2E test SKIPS; the hash test still passes.)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: all prior tests still pass (was 347) plus the new mlb_api (4) and runner (2) tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/mcsim/run_matchup_cards.py tests/test_mcsim_run_matchup_cards.py
git commit -m "feat(mcsim): live MLB-API matchup-card runner + CLI (App B Step 5+6)"
```

---

## Task 5: Manual end-to-end validation on one real game (named numbers + wall-clock)

**Files:** none (manual run + ContextSwitcher update).

- [ ] **Step 1: Pick one real game for a near date**

Run:
```bash
uv run python -c "import mcsim.mlb_api as m; gs=m.get_schedule('2026-06-02'); print(gs[0].game_pk, gs[0].away_team,'@',gs[0].home_team)"
```
Expected: prints a real game_pk + matchup (live network call — this step is not CI).

- [ ] **Step 2: Run the runner on that one game at modest n_paths**

Run (substitute the game_pk from Step 1):
```bash
uv run python -m scripts.mcsim.run_matchup_cards --date 2026-06-02 --game-pk <GAME_PK> --n-paths 100 --rng-seed 1 --db-path data/mcsim.sqlite
```
Expected: a `[AWAY @ HOME] game_pk=… N cells written` line, then `done: 1 card(s)`. Note the wall-clock (`time` it). Record the true all-vs-all cell count + minutes — this is the number that decides whether multiprocessing is needed for a nightly batch.

- [ ] **Step 3: Confirm the row persisted and print a NAMED numerical output**

Run:
```bash
uv run python -c "
from mcsim.storage import init_db, read_prediction
c=init_db('data/mcsim.sqlite')
p=read_prediction(c, game_pk=<GAME_PK>, prediction_date='2026-06-02', app='matchup_card')['payload']
cell=p['rows'][0]['cells'][0]
print(f\"{p['rows'][0]['name']} vs {cell['batter_name']}: median RV={cell['predicted_rv_median']:+.4f} modal {cell['modal_type']} pi-hat={cell['p_hat_top_type']:.3f} trust={cell['trust_state']}  | n_cells={p['n_cells']}\")
"
```
Expected: one named line (e.g. `… median RV=-0.083 modal FF pi-hat=0.31 trust=green | n_cells=338`). Do NOT claim success without this output (CLAUDE.md rule).

- [ ] **Step 4: Update ContextSwitcher.md**

Mark Step 5+6 done; record the measured all-vs-all wall-clock + cell count; set next step to Step 7 (post-game actuals fetcher). Note whether multiprocessing is now warranted based on the measured time.

- [ ] **Step 5: Commit + push**

```bash
git add docs/ContextSwitcher.md
git commit -m "docs: Step 5+6 done — measured all-vs-all wall-clock; Step 7 next"
git push origin mcsim-app-b-matchup-card
```

---

## Self-Review Notes

- **Spec coverage:** mlb_api client (Tasks 1–2), ckpt hash (Task 3), runner+CLI+persist+grid+error-handling (Task 4), manual validation/wall-clock (Task 5). All spec sections mapped.
- **Missing-profile handling:** spec said "skip with log"; verification showed lookups never raise (league-mean fallback), so v1 keeps only a per-GAME try/except. Documented in runner docstring + this plan's header.
- **No `compute_matchup_card` change:** confirmed it loops arbitrary-length lists; runner feeds full rosters.
- **Type consistency:** `GameInfo`, `PitcherSpec(id,name,throws,is_starter)`, `BatterSpec(id,name,stand)`, `get_schedule(date)`, `get_active_roster(team_id,date,*,probable_pitcher_id)`, `run_matchup_cards(...)`, `compute_ckpt_hash(path)` used identically across tasks.
- **Known v1 simplification:** switch hitters resolve to a fixed `'L'` (vs RHP); per-pitcher resolution deferred (would require compute_matchup_card to vary stand by pitcher).
