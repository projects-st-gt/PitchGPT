"""Dedicated hitter / swing-decision model (XGBoost) — see docs/Hitter_Swing_Model.md.

Phase 1: per-pitch swing -> whiff -> contact decomposition, composed with the
pitch sequence model inside the causal rollout. This package holds the label
extraction, feature builder, and per-node models.
"""
