"""Baseline models scored on the same CV folds as the GP kernels, and the registry of model ids.

The models form a 2 x 2 ablation of the GP model of src/gp.py; the noise model is the same in all four, and
so is the gap step b2 s (s = post_gap, 0 before the 2021-22 CLARA gap and 1 from its start; cfg gap_step):

                          with ERA5                                   without ERA5
    GP (mean + kernel)    <kernel>         b0 + b1 e + b2 s + f(x)    <kernel>_noera5   b0 + b2 s + f(x)
    regression only       linear           b0 + b1 e + b2 s           linear_noera5     b0 + b2 s (a constant per period)

    y_i = mean_i + eps_i,   eps_i ~ N(0, sigma0^2 + sigma1^2 / n_i),   n_i overpasses behind y_i

so <kernel> vs linear is what the kernel adds with ERA5, and <kernel> vs <kernel>_noera5 what ERA5 adds to
the GP. Without ERA5 the GP is gp.GP(..., era5=False) and ERA5 is never read.

LinearModel (regression only) is fitted by exact Gaussian maximum likelihood. For fixed noise (sigma0,
sigma1) the coefficients are the GLS estimate (closed form), so L-BFGS runs over log sigma0 and log sigma1
only, with beta profiled out (design [1, e~, s] or a subset). latent_var is the GLS variance of the fitted mean function,
x*^T (X^T W X)^-1 x*; obs_var adds the noise of a new cell-month mean from n overpasses. As in gp.GP, y and
ERA5 are standardised with training statistics only and everything returned is in physical units.
"""

import math

import numpy as np
import torch

import gp

LINEAR = ("linear", "linear_noera5")
NOERA5 = "_noera5"
ALL_IDS = gp.KERNELS + tuple(k + NOERA5 for k in gp.KERNELS) + LINEAR
MAX_ITER = 200  # two noise parameters: cheap, so not tied to the GP budget gp.max_iter


def parse(model):
    """(kernel family, or None for regression only; whether ERA5 is a predictor) of a model id."""
    if model in gp.KERNELS:
        return model, True
    if model.endswith(NOERA5) and model[:-len(NOERA5)] in gp.KERNELS:
        return model[:-len(NOERA5)], False
    if model in LINEAR:
        return None, model == "linear"
    raise ValueError(f"unknown model {model!r}; expected one of {ALL_IDS}")


def make(model, cfg_gp, z=None):
    """An unfitted model of the given id (z: inducing inputs, GP models only)."""
    family, era5 = parse(model)
    return gp.GP(family, cfg_gp, z=z, era5=era5) if family else LinearModel(cfg_gp, era5=era5)


def gp_baselines(cfg):
    """GP models without ERA5 to fit (config [baselines] gp_noera5_kernels; the winning kernel)."""
    kernels = cfg.get("baselines", {}).get("gp_noera5_kernels", [])
    for k in kernels:
        parse(k)
    return [k + NOERA5 for k in kernels]


def baseline_models(cfg):
    """Every baseline model id: the two regression-only models and the configured GPs without ERA5."""
    return list(LINEAR) + gp_baselines(cfg)


class LinearModel:
    """Regression only (no kernel): y = b0 [+ b1 e] [+ b2 s] + eps with the GP's heteroscedastic noise."""

    family = None

    def __init__(self, cfg_gp, era5=True):
        self.cfg, self.era5 = cfg_gp, era5
        self.step = bool(cfg_gp.get("gap_step", False))  # b2 * post_gap in the mean function
        self.stats = None

    _step = gp.GP._step

    def _design(self, e, s=None):
        """Design matrix [1, e~ (standardised ERA5, with ERA5), s (post_gap, with the gap step)]."""
        e = np.asarray(e, float)
        cols = [np.ones(len(e))]
        if self.era5:
            cols.append((e - self.stats["e_mean"]) / self.stats["e_sd"])
        if self.step:
            cols.append(np.asarray(s, float))
        return np.column_stack(cols)

    @staticmethod
    def _profile(raw, x, y, n):
        """Profile log-likelihood at noise exp(raw), with the GLS coefficients and their covariance."""
        s0, s1 = torch.exp(raw) + 1e-6
        noise = s0 ** 2 + s1 ** 2 / n
        xtw = x.T / noise
        a = xtw @ x
        beta = torch.linalg.solve(a, xtw @ y)
        r = y - x @ beta
        ll = -0.5 * (len(y) * math.log(2 * math.pi) + torch.log(noise).sum() + (r * r / noise).sum())
        return ll, beta, torch.linalg.inv(a), (s0, s1)

    def fit(self, x, y, e, n, verbose=False, s=None):
        """Fit on training rows (x is accepted for the common interface and not used); s = post_gap."""
        y, n = np.asarray(y, float), np.asarray(n, float)
        s = self._step(s, len(y), training=True)
        e = np.asarray(e, float) if self.era5 else np.zeros_like(y)
        self.stats = {"y_mean": y.mean(), "y_sd": y.std(), "e_mean": e.mean(), "e_sd": e.std() if self.era5 else 1.0}
        sy = self.stats["y_sd"]
        xt = torch.as_tensor(self._design(e, s), dtype=gp.DTYPE)
        yt = torch.as_tensor((y - self.stats["y_mean"]) / sy, dtype=gp.DTYPE)
        nt = torch.as_tensor(n, dtype=gp.DTYPE)
        raw = torch.tensor([math.log(self.cfg["init_sigma0"] / sy), math.log(self.cfg["init_sigma1"] / sy)],
                           dtype=gp.DTYPE, requires_grad=True)
        opt = torch.optim.LBFGS([raw], lr=1.0, max_iter=MAX_ITER, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)
        n_eval = [0]

        def closure():
            opt.zero_grad()
            loss = -self._profile(raw, xt, yt, nt)[0] / len(y)
            loss.backward()
            n_eval[0] += 1
            return loss

        opt.step(closure)
        with torch.no_grad():
            ll, beta, cov, (s0, s1) = self._profile(raw, xt, yt, nt)
        self.beta, self.cov = beta.numpy(), cov.numpy()
        self.sigma = (float(s0), float(s1))
        self.bound_value, self.n_evals = float(ll), n_eval[0]
        if verbose:
            print(f"  linear{'' if self.era5 else NOERA5}: log-likelihood {self.bound_value:.2f} after "
                  f"{self.n_evals} evaluations")
        return self

    def coefficients(self):
        """(beta0, beta1, beta2) in physical units: CLARA = beta0 + beta1 * ERA5 + beta2 * post_gap (beta1 = 0
        without ERA5, beta2 = 0 without the gap step)."""
        s = self.stats
        b1 = float(self.beta[1]) * s["y_sd"] / s["e_sd"] if self.era5 else 0.0
        b2 = float(self.beta[1 + self.era5]) * s["y_sd"] if self.step else 0.0
        return s["y_mean"] + float(self.beta[0]) * s["y_sd"] - b1 * s["e_mean"], b1, b2

    def predict(self, x, e, n=None, batch=None, s=None):
        """Same output as gp.GP.predict; mean equals mean_fn (there is no GP term)."""
        sy = self.stats["y_sd"]
        d = self._design(e if self.era5 else np.zeros(len(x)), self._step(s, len(x)))
        mean = self.stats["y_mean"] + (d @ self.beta) * sy
        res = {"mean_fn": mean, "mean": mean.copy(), "latent_var": np.einsum("ij,jk,ik->i", d, self.cov, d) * sy ** 2}
        if n is not None:
            s0, s1 = self.sigma
            res["obs_var"] = res["latent_var"] + (s0 ** 2 + s1 ** 2 / np.asarray(n, float)) * sy ** 2
        return res

    def hyperparameters(self):
        """Fitted parameters in physical units ('bound' is the exact log-likelihood, standardised units)."""
        b0, b1, b2 = self.coefficients()
        sy = self.stats["y_sd"]
        return {"kernel": "none", "bound": self.bound_value, "n_evals": self.n_evals, "beta0_wm2": b0, "beta1": b1,
                "beta2_wm2": b2,
                "sigma0_wm2": self.sigma[0] * sy, "sigma1_wm2": self.sigma[1] * sy}

    def state(self):
        return {"family": None, "cfg": self.cfg, "era5": self.era5, "stats": self.stats, "beta": self.beta,
                "cov": self.cov, "sigma": self.sigma, "bound_value": self.bound_value, "n_evals": self.n_evals}

    @classmethod
    def from_state(cls, st):
        model = cls(st["cfg"], era5=st["era5"])
        model.stats, model.beta, model.cov, model.sigma = st["stats"], st["beta"], st["cov"], st["sigma"]
        model.bound_value, model.n_evals = st["bound_value"], st["n_evals"]
        return model
