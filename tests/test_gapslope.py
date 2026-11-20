"""The period-specific ERA5 slope (beta3 * post_gap * ERA5) of the sensitivity models."""

import numpy as np
import pandas as pd
import pytest
import torch

import baselines
import gp
from style import model_color, model_label

CFG = {"n_inducing": 60, "restarts": 1, "max_iter": 80, "seed": 0, "jitter": 1e-6, "gap_step": True,
       "init_sigma_A": 5.0, "init_sigma_B": 5.0, "init_ell_space_A_km": 2000.0, "init_ell_season_days": 60.0,
       "init_ell_space_B_km": 4000.0, "init_ell_time_years": 1.0, "init_sigma0": 5.0, "init_sigma1": 5.0}
P = {"sigma_A": 0.9, "sigma_B": 0.6, "ell_space_A": 0.7, "ell_season": 0.9, "ell_space_B": 1.3, "ell_time": 1.5,
     "sigma0": 0.3, "sigma1": 0.5, "beta0": 0.1, "beta1": 0.4, "beta2": 0.8, "beta3": -0.3}


def synthetic(m=500, seed=11, b3=0.3):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-60, 60, m), rng.uniform(-180, 180, m)
    time = pd.Timestamp("2020-01-15") + pd.to_timedelta(rng.integers(0, 81, m) * 30.44, unit="D")
    s = np.asarray(time >= pd.Timestamp("2021-09-01"), float)
    x = gp.embed(lat, lon, time, "2020-01-01")
    e = rng.normal(230, 25, m)
    y = 20 + 0.6 * e + 40 * s + b3 * s * e + 5 * np.sin(np.radians(lon)) + rng.normal(0, 3, m)
    return x, y, e, s


def tensors(p):
    return {k: torch.tensor(v, dtype=gp.DTYPE) for k, v in p.items()}


@pytest.mark.parametrize("family", gp.KERNELS)
def test_bound_with_slope_change_is_exact_and_chunks(family):
    rng = np.random.default_rng(3)
    x = gp.embed(rng.uniform(-60, 60, 100), rng.uniform(-180, 180, 100),
                 pd.Timestamp("2020-01-15") + pd.to_timedelta(rng.integers(0, 81, 100) * 30.44, unit="D"), "2020-01-01")
    xt = torch.as_tensor(x, dtype=gp.DTYPE)
    y, e = (torch.as_tensor(rng.normal(size=100), dtype=gp.DTYPE) for _ in range(2))
    s = torch.as_tensor((rng.uniform(size=100) > 0.4).astype(float), dtype=gp.DTYPE)
    n = torch.as_tensor(rng.integers(1, 4, 100).astype(float), dtype=gp.DTYPE)
    p = tensors(P)
    model = gp.GP(family, {"jitter": 1e-10, "gap_step": True}, z=x, gap_slope=True)
    exact = float(gp.exact_log_marginal(family, p, xt, y, e, n, s))
    assert float(model.bound(p, xt, y, e, n, s=s)) == pytest.approx(exact, rel=1e-6)
    assert float(model.bound_chunked(p, xt, y, e, n, 40, s)) == pytest.approx(exact, rel=1e-6)
    no_slope = {k: v for k, v in p.items() if k != "beta3"}
    assert exact != pytest.approx(float(gp.exact_log_marginal(family, no_slope, xt, y, e, n, s)), rel=1e-6)


def test_fit_recovers_the_slope_change_and_round_trips():
    x, y, e, s = synthetic()
    model = gp.GP("matern32", CFG, gap_slope=True).fit(x, y, e, np.ones(len(y)), s=s)
    b0, b1, b2 = model.coefficients()
    assert b1 == pytest.approx(0.6, abs=0.1) and model.slope_change() == pytest.approx(0.3, abs=0.1)
    assert model.hyperparameters()["beta3"] == model.slope_change()
    # the post minus pre mean function is b2 + b3 e: the step now depends on ERA5
    pre, post = (model.predict(x[:50], e[:50], s=np.full(50, v)) for v in (0.0, 1.0))
    assert np.allclose(post["mean_fn"] - pre["mean_fn"], b2 + model.slope_change() * e[:50])
    back = gp.GP.from_state(model.state())
    assert back.slope and np.allclose(back.predict(x[:50], e[:50], s=np.ones(50))["mean"], post["mean"])
    # the model without the term keeps beta3 out of its parameters and its state
    plain = gp.GP("matern32", dict(CFG, max_iter=5)).fit(x, y, e, np.ones(len(y)), s=s)
    assert "beta3" not in plain.raw and plain.slope_change() == 0.0 and "beta3" not in plain.hyperparameters()
    with pytest.raises(ValueError):
        gp.GP("matern32", dict(CFG, gap_step=False), gap_slope=True)


def test_coefficient_covariance_is_the_exact_gls_covariance_when_z_is_x():
    x, y, e, s = synthetic(m=150, seed=5)
    n = np.random.default_rng(1).integers(1, 4, len(y)).astype(float)
    model = gp.GP("exponential", dict(CFG, n_inducing=150, max_iter=30, jitter=1e-10), z=x, gap_slope=True)
    model.fit(x, y, e, n, s=s)
    names, cov = model.mean_coef_covariance(x, e, n, s, chunk=60)
    assert names == ["beta0", "beta1", "beta2", "beta3"]
    p = model._post["p"]
    es = (e - model.stats["e_mean"]) / model.stats["e_sd"]
    h = torch.as_tensor(np.column_stack([np.ones(len(e)), es, s, s * es]), dtype=gp.DTYPE)
    xt, nt = (torch.as_tensor(v, dtype=gp.DTYPE) for v in (x, n))
    sigma = gp.kernel("exponential", xt, xt, p) + torch.diag(p["sigma0"] ** 2 + p["sigma1"] ** 2 / nt)
    exact = torch.linalg.inv(h.T @ torch.linalg.solve(sigma, h)).detach().numpy()
    assert np.allclose(cov, exact, rtol=1e-5)
    se = model.physical_se(names, cov)
    assert se["beta3"] == pytest.approx(np.sqrt(exact[3, 3]) * model.stats["y_sd"] / model.stats["e_sd"])


def test_linear_model_with_slope_change_and_registry():
    x, y, e, s = synthetic(b3=-0.2)
    m = baselines.make("linear_gapslope", CFG).fit(x, y, e, np.ones(len(y)), s=s)
    assert m.slope_change() == pytest.approx(-0.2, abs=0.05) and 0 < m.slope_change_se() < 0.05
    hp = m.hyperparameters()
    assert hp["beta3"] == m.slope_change() and hp["beta3_se"] == m.slope_change_se()
    pre, post = (m.predict(x[:20], e[:20], s=np.full(20, v)) for v in (0.0, 1.0))
    assert np.allclose(post["mean"] - pre["mean"], m.coefficients()[2] + m.slope_change() * e[:20])
    back = baselines.LinearModel.from_state(m.state())
    assert np.allclose(back.predict(x[:20], e[:20], s=np.ones(20))["mean"], post["mean"])
    # registry: the sensitivity ids parse as their base model and carry the slope
    assert baselines.parse("exponential_gapslope") == ("exponential", True)
    assert baselines.parse("linear_gapslope") == (None, True)
    assert isinstance(baselines.make("exponential_gapslope", CFG), gp.GP)
    assert baselines.make("exponential_gapslope", CFG).slope and not baselines.make("exponential", CFG).slope
    assert {"exponential_gapslope", "linear_gapslope"} <= set(baselines.ALL_IDS)
    assert "period-specific slope" in model_label("exponential_gapslope") and model_color("linear_gapslope")
    with pytest.raises(ValueError):
        baselines.parse("exponential_noera5_gapslope")
