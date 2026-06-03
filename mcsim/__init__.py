"""MCSim — Monte Carlo simulation apps for baseball.

Two pre-game apps planned (see ``docs/MCSim_brainstorm.md``):

- **App A — daily score prediction.** Simulates whole games from 0–0 top-of-1
  for a daily prediction site. Needs a multi-AB state machine (not yet built).
- **App B — pre-game matchup report card.** Per-game grid of (my pitcher × their
  batter) cells with expected run-value distributions; nightly batch; stored
  for date-carousel browsing. Uses single-AB ``g_compute`` directly.

This package is App B v1. App A lives in a separate module when built.

Public surface today:

- :mod:`mcsim.storage` — SQLite persistence (predictions + actuals + model
  versions). Single file at ``data/mcsim.sqlite`` (gitignored).
"""
