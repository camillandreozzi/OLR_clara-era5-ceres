"""Baselines (regression only, GP without ERA5), the model registry and the Slurm task lists."""

import numpy as np
import pandas as pd
import pytest

import baselines
import gp
import run_cv

CFG_GP = {"n_inducing": 40, "restarts": 1, "max_iter": 20, "seed": 0, "jitter": 1e-6,
          "init_sigma_A": 5.0, "init_sigma_B": 5.0, "init_ell_space_A_km": 2000.0, "init_ell_season_days": 60.0,
          "init_ell_space_B_km": 4000.0, "init_ell_time_years": 1.0, "init_sigma0": 5.0, "init_sigma1": 5.0}


def synthetic(m, seed):
    """Heteroscedastic data y = 20 + 0.8 e + N(0, 3^2 + 6^2 / n) with a little space-time structure."""
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-60, 60, m), rng.uniform(-180, 180, m)
    time = pd.Timestamp("2021-01-15") + pd.to_timedelta(rng.integers(0, 24, m) * 30, unit="D")
    x = gp.embed(lat, lon, time, "2020-01-01")
    e = rng.normal(230, 25, m)
    n = rng.integers(1, 6, m).astype(float)
    y = 20 + 0.8 * e + 2 * np.sin(np.radians(lon)) + rng.normal(0, np.sqrt(3 ** 2 + 6 ** 2 / n))
    return x, y, e, n


@pytest.fixture(scope="module")
def linear_fit():
    x, y, e, n = synthetic(20000, seed=1)
    return baselines.LinearModel(CFG_GP).fit(x, y, e, n), (x, y, e, n)


def test_linear_recovers_mean_and_noise(linear_fit):
    model, _ = linear_fit
    b0, b1, b2 = model.coefficients()
    hp = model.hyperparameters()
    assert b1 == pytest.approx(0.8, abs=0.02) and b0 == pytest.approx(20, abs=3)
    assert hp["sigma0_wm2"] == pytest.approx(np.sqrt(3 ** 2 + 2 ** 2 / 2), rel=0.15)  # + the unmodelled sine
    assert hp["sigma1_wm2"] == pytest.approx(6, rel=0.15)


def test_linear_coefficients_are_gls_at_fitted_noise(linear_fit):
    model, (_, y, e, n) = linear_fit
    hp = model.hyperparameters()
    sw = 1 / np.sqrt(hp["sigma0_wm2"] ** 2 + hp["sigma1_wm2"] ** 2 / n)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(e)), e]) * sw[:, None], y * sw, rcond=None)[0]
    assert np.allclose(model.coefficients()[:2], beta, rtol=1e-8) and model.coefficients()[2] == 0.0


def test_linear_prediction_and_training_statistics(linear_fit):
    model, (x, y, e, n) = linear_fit
    stats = dict(model.stats)
    p = model.predict(x[:500], e[:500] + 10, n[:500])
    assert np.array_equal(p["mean"], p["mean_fn"])
    assert np.all(p["latent_var"] > 0) and np.all(p["obs_var"] > p["latent_var"])
    assert model.stats == stats and stats["y_mean"] == pytest.approx(y.mean())
    back = baselines.LinearModel.from_state(model.state())
    assert np.allclose(back.predict(x[:50], e[:50], n[:50])["obs_var"], model.predict(x[:50], e[:50], n[:50])["obs_var"])


@pytest.mark.parametrize("model_id", ["linear_noera5", "matern32_noera5"])
def test_no_era5_models_never_read_era5(model_id):
    x, y, e, n = synthetic(300, seed=2)
    family, era5 = baselines.parse(model_id)
    z = gp.inducing_inputs(x, 40, CFG_GP, 0) if family else None
    model = baselines.make(model_id, CFG_GP, z).fit(x, y, np.full(len(y), np.nan), n)
    assert model.coefficients()[1] == 0.0
    p_nan = model.predict(x[:60], np.full(60, np.nan), n[:60])
    p_era = model.predict(x[:60], e[:60], n[:60])
    for k in ("mean", "mean_fn", "latent_var", "obs_var"):
        assert np.all(np.isfinite(p_nan[k])) and np.array_equal(p_nan[k], p_era[k])
    if family:
        assert float(model.raw["beta1"]) == 0.0  # fixed, never optimised
        back = gp.GP.from_state(model.state())
        assert not back.era5 and np.allclose(back.predict(x[:60], e[:60])["mean"], p_nan["mean"])


def test_gp_states_without_era5_flag_load_with_era5():
    """full_model.pt files written before the baselines existed have no 'era5' key."""
    x, y, e, n = synthetic(200, seed=3)
    st = gp.GP("exponential", dict(CFG_GP, max_iter=5), z=x[:30]).fit(x, y, e, n).state()
    del st["era5"]
    assert gp.GP.from_state(st).era5


def test_registry():
    assert baselines.parse("matern32") == ("matern32", True)
    assert baselines.parse("gaussian_noera5") == ("gaussian", False)
    assert baselines.parse("linear") == (None, True) and baselines.parse("linear_noera5") == (None, False)
    for bad in ("linear_era5", "foo_noera5", "matern52"):
        with pytest.raises(ValueError):
            baselines.parse(bad)
    assert isinstance(baselines.make("exponential_noera5", CFG_GP), gp.GP)
    assert isinstance(baselines.make("linear", CFG_GP), baselines.LinearModel)
    cfg = {"baselines": {"gp_noera5_kernels": ["matern32"]}}
    assert baselines.baseline_models(cfg) == ["linear", "linear_noera5", "matern32_noera5"]
    assert baselines.gp_baselines({}) == []
    with pytest.raises(ValueError):
        baselines.gp_baselines({"baselines": {"gp_noera5_kernels": ["matern52"]}})


def test_kernel_task_list_unchanged_by_baselines():
    """Euler array tasks find their fit by index: the default list must not depend on [baselines]."""
    folds = pd.DataFrame({"cv_type": ["spatial", "spatial", "temporal", "joint"], "fold": [0, 1, 0, 0]})
    kernels = ["exponential", "gaussian", "matern32"]
    plain = {"gp": {"kernels": kernels}}
    with_b = plain | {"baselines": {"gp_noera5_kernels": ["gaussian"]}}
    expected = [(k, "full", -1) for k in kernels] + [(k, t, f) for t, f in
                                                     (("spatial", 0), ("spatial", 1), ("temporal", 0), ("joint", 0))
                                                     for k in kernels]
    assert run_cv.task_list(plain, folds) == run_cv.task_list(with_b, folds) == expected
    assert run_cv.task_list(plain, folds, "baselines") == []
    tasks = run_cv.task_list(with_b, folds, "baselines")
    assert tasks[0] == ("gaussian_noera5", "full", -1) and {m for m, _, _ in tasks} == {"gaussian_noera5"}
    assert len(tasks) == 5


def test_linear_gap_step_is_gls():
    x, y, e, n = synthetic(20000, seed=4)
    s = (np.arange(len(y)) % 3 == 0).astype(float)
    y = y + 40 * s
    cfg = dict(CFG_GP, gap_step=True)
    model = baselines.LinearModel(cfg).fit(x, y, e, n, s=s)
    b0, b1, b2 = model.coefficients()
    assert b2 == pytest.approx(40, abs=0.5) and b1 == pytest.approx(0.8, abs=0.02)
    hp = model.hyperparameters()
    assert hp["beta2_wm2"] == b2
    sw = 1 / np.sqrt(hp["sigma0_wm2"] ** 2 + hp["sigma1_wm2"] ** 2 / n)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(e)), e, s]) * sw[:, None], y * sw, rcond=None)[0]
    assert np.allclose(model.coefficients(), beta, rtol=1e-8)
    with pytest.raises(ValueError):
        model.predict(x[:5], e[:5], n[:5])
    pre, post = (model.predict(x[:20], e[:20], n[:20], s=np.full(20, v)) for v in (0.0, 1.0))
    assert np.allclose(post["mean"] - pre["mean"], b2)
    back = baselines.LinearModel.from_state(model.state())
    assert np.allclose(back.predict(x[:20], e[:20], n[:20], s=s[:20])["mean"],
                       model.predict(x[:20], e[:20], n[:20], s=s[:20])["mean"])
    const = baselines.make("linear_noera5", cfg).fit(x, y, np.full(len(y), np.nan), n, s=s)
    assert const.coefficients()[1] == 0.0 and const.coefficients()[2] == pytest.approx(40, abs=1.5)
    for model_id in ("linear", "matern32"):  # one regime only: beta2 is not identifiable
        z = gp.inducing_inputs(x[:300], 20, CFG_GP, 0) if model_id == "matern32" else None
        with pytest.raises(ValueError, match="not identifiable"):
            baselines.make(model_id, cfg, z).fit(x[:300], y[:300], e[:300], n[:300], s=np.ones(300))
