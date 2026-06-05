import torch
from model.config import PitchGPTConfig
from model.heads import LocationMDN


def _head(d=32, k=5):
    cfg = PitchGPTConfig(d_model=d, mdn_components=k)
    return LocationMDN(cfg, d_in=d), cfg


def test_mdn_params_shapes():
    head, cfg = _head()
    h = torch.randn(2, 4, cfg.d_model)
    w, mu, log_std = head(h)
    assert w.shape == (2, 4, cfg.mdn_components)
    assert mu.shape == (2, 4, cfg.mdn_components, 2)
    assert log_std.shape == (2, 4, cfg.mdn_components, 2)
    assert torch.allclose(w.exp().sum(-1), torch.ones(2, 4), atol=1e-5)   # weights normalize
    assert (log_std >= cfg.mdn_logstd_floor - 1e-6).all()                 # floor honored


def test_mdn_nll_lower_for_closer_target():
    torch.manual_seed(0)
    head, cfg = _head()
    h = torch.randn(1, 1, cfg.d_model)
    w, mu, log_std = head(h)
    center = mu[0, 0, w[0, 0].argmax()]               # dominant component mean
    near = head.nll(w, mu, log_std, center.view(1, 1, 2))
    far = head.nll(w, mu, log_std, (center + 5.0).view(1, 1, 2))
    assert near.item() < far.item()


def test_mdn_sample_in_range_and_seeded():
    head, cfg = _head()
    h = torch.randn(3, 2, cfg.d_model)
    w, mu, log_std = head(h)
    g = torch.Generator().manual_seed(1)
    s1 = head.sample(w, mu, log_std, generator=g)
    g2 = torch.Generator().manual_seed(1)
    s2 = head.sample(w, mu, log_std, generator=g2)
    assert s1.shape == (3, 2, 2)
    assert torch.allclose(s1, s2)
    assert torch.isfinite(s1).all()
