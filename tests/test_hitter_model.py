"""Tests for hitter.model — HitterModel loads the persisted cascade nodes and
reproduces train-time predictions (round-trip), plus the foul-rate lookup.

Self-contained: trains tiny synthetic nodes, saves them, loads via HitterModel.
No dependency on the full real training run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hitter.train import train_node, save_models
from hitter.model import HitterModel


def _synth(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(2.0 * x1 - x2)))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y)


@pytest.fixture
def saved_dir(tmp_path):
    """Train tiny synthetic nodes (one per cascade node) + save them."""
    feats = ["x1", "x2"]
    nodes = {}
    for node in ("swing", "called_strike", "whiff"):
        Xtr, ytr = _synth(seed=hash(node) % 100)
        Xva, yva = _synth(seed=hash(node) % 100 + 50)
        nodes[node] = train_node(Xtr, ytr, Xva, yva, objective="binary",
                                 feature_names=feats, n_estimators=40)
    # contact_quality is regression
    rng = np.random.default_rng(7)
    xq = rng.normal(size=2000)
    yq = 0.35 + 0.15 * xq + rng.normal(scale=0.1, size=2000)
    Xq = pd.DataFrame({"x1": xq, "x2": rng.normal(size=2000)})
    nodes["contact_quality"] = train_node(
        Xq.iloc[:1500], pd.Series(yq[:1500]), Xq.iloc[1500:], pd.Series(yq[1500:]),
        objective="regression", feature_names=feats, n_estimators=40)
    bundle = {"nodes": nodes,
              "foul_rate_by_count": {(0, 0): 0.40, (0, 1): 0.35, (2, 2): 0.28}}
    save_models(bundle, str(tmp_path))
    return tmp_path, bundle


def test_hittermodel_loads_all_nodes(saved_dir):
    d, _ = saved_dir
    hm = HitterModel(str(d))
    assert set(hm.nodes) == {"swing", "called_strike", "whiff", "contact_quality"}


def test_predict_node_reproduces_train_time(saved_dir):
    """The whole point of persistence: loaded model == train-time predict()."""
    d, bundle = saved_dir
    hm = HitterModel(str(d))
    Xtest, _ = _synth(seed=999)
    for node in ("swing", "whiff", "contact_quality"):
        got = hm.predict_node(node, Xtest)
        want = bundle["nodes"][node]["predict"](Xtest)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # a named numerical check: P(swing) on a high-x1 row is high
    hi = pd.DataFrame({"x1": [3.0], "x2": [0.0]})
    p = float(hm.predict_node("swing", hi)[0])
    print(f"\nP(swing | x1=3) = {p:.3f}")
    assert p > 0.7


def test_foul_rate_lookup_and_fallback(saved_dir):
    d, _ = saved_dir
    hm = HitterModel(str(d))
    assert hm.foul_rate(0, 1) == pytest.approx(0.35)
    # unseen count falls back to a sane default in [0,1], not a crash
    fb = hm.foul_rate(3, 0)
    assert 0.0 <= fb <= 1.0


def test_predict_cascade_returns_all_arrays(saved_dir):
    d, _ = saved_dir
    hm = HitterModel(str(d))
    X, _ = _synth(seed=3, n=50)
    out = hm.predict_cascade(X)
    for k in ("swing", "called_strike", "whiff", "contact_quality"):
        assert k in out and len(out[k]) == 50
    # binary nodes in [0,1]
    assert out["swing"].min() >= 0 and out["swing"].max() <= 1
