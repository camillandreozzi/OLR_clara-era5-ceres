"""Checkpoint before the cross-validation run: is the sparse GP accurate and affordable?

1. VFE vs exact GP on a random 3,000-cell-month subset of the real observations, per kernel: hyperparameters
   fitted by VFE (M = n_inducing), then at those hyperparameters the exact log marginal likelihood vs the
   VFE bound, and exact vs VFE predictions on 1,000 held-out rows of the same subset.
2. M sensitivity on one spatial fold (default: the median-sized one): out-of-sample RMSE, coverage, bound and runtime per kernel for
   M in --m-values, to check that the kernel ranking is not an artefact of the approximation.
3. Restart agreement: bound and hyperparameters of each restart on that fold (verbose fit log).

Output: results/model/benchmark/{vfe_vs_exact.csv, m_sensitivity.csv, summary.md}

Usage:
    python src/benchmark_gp.py [--fold 7] [--m-values 250 500 1000] [--kernels matern32]
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

import config
import cv
import gp
import run_cv


def vfe_vs_exact(cfg, obs, kernels, n_sub=3000, n_test=1000):
    rng = np.random.default_rng(cfg["gp"]["seed"])
    idx = rng.choice(len(obs), n_sub + n_test, replace=False)
    tr, te = idx[:n_sub], idx[n_sub:]
    x = run_cv.inputs(obs, cfg)
    e = obs.era5_night_r0.to_numpy()
    rows = []
    for kernel in kernels:
        t0 = time.time()
        cfg_k = dict(cfg["gp"], restarts=1, n_inducing=1000)
        model = gp.GP(kernel, cfg_k).fit(x[tr], obs.clara.to_numpy()[tr], e[tr], obs.n_overpasses.to_numpy()[tr])
        t_fit = time.time() - t0
        p = model._post["p"]
        yt, et = (torch.as_tensor(v, dtype=gp.DTYPE)
                  for v in model._standardise(obs.clara.to_numpy()[tr], e[tr]))
        xt, nt = torch.as_tensor(x[tr], dtype=gp.DTYPE), torch.as_tensor(obs.n_overpasses.to_numpy()[tr], dtype=gp.DTYPE)
        with torch.no_grad():
            exact_lml = float(gp.exact_log_marginal(kernel, p, xt, yt, et, nt))
            ys, es = model._standardise(obs.clara.to_numpy()[te], e[te])
            m_ex, v_ex = gp.exact_predict(kernel, p, xt, yt, et, nt, torch.as_tensor(x[te], dtype=gp.DTYPE),
                                          torch.as_tensor(es, dtype=gp.DTYPE))
        sy, ym = model.stats["y_sd"], model.stats["y_mean"]
        m_ex = ym + m_ex.numpy() * sy
        v_ex = v_ex.numpy() * sy ** 2
        pv = model.predict(x[te], e[te], obs.n_overpasses.to_numpy()[te])
        y_te = obs.clara.to_numpy()[te]
        rows.append({"kernel": kernel, "n_train": n_sub, "n_inducing": len(model.z), "vfe_bound": model.bound_value,
                     "exact_lml": exact_lml, "bound_gap_per_obs": (exact_lml - model.bound_value) / n_sub,
                     "rmse_vfe": np.sqrt(np.mean((pv["mean"] - y_te) ** 2)),
                     "rmse_exact": np.sqrt(np.mean((m_ex - y_te) ** 2)),
                     "max_abs_mean_diff": np.abs(pv["mean"] - m_ex).max(),
                     "rms_mean_diff": np.sqrt(np.mean((pv["mean"] - m_ex) ** 2)),
                     "median_sd_ratio_vfe_over_exact": np.median(np.sqrt(pv["latent_var"] / np.maximum(v_ex, 1e-12))),
                     "fit_seconds": t_fit} | {k: v for k, v in model.hyperparameters().items()
                                               if k.endswith(("_km", "_days", "_years", "_wm2")) or k == "beta1"})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def m_sensitivity(cfg, obs, fold, m_values, kernels):
    folds = run_cv.check_folds()
    train, test = cv.split(folds, "spatial", fold, obs.obs_id.to_numpy())
    x = run_cv.inputs(obs, cfg)
    radius, _ = run_cv.choose_radius(obs, train, cfg["era5"]["smoothing_radii_km"])
    e = obs[f"era5_night_r{int(radius)}"].to_numpy()
    rows = []
    for m in m_values:
        z = gp.inducing_inputs(x[train], m, cfg["gp"], cfg["gp"]["seed"])
        for kernel in kernels:
            t0 = time.time()
            print(f"M={m} {kernel}:", flush=True)
            # restarts only at the configured M (restart agreement); one start elsewhere to bound the runtime
            restarts = cfg["gp"]["restarts"] if m == cfg["gp"]["n_inducing"] else 1
            model = gp.GP(kernel, dict(cfg["gp"], n_inducing=m, restarts=restarts), z=z).fit(
                x[train], obs.clara[train], e[train], obs.n_overpasses[train], verbose=True)
            pred = model.predict(x[test], e[test], obs.n_overpasses[test].to_numpy())
            met = cv.metrics(obs.clara[test], pred["mean"], pred["obs_var"], obs.cell_area_km2[test], pred["mean_fn"])
            rows.append({"M": m, "kernel": kernel, "fold": fold, "n_train": int(train.sum()), "n_test": int(test.sum()),
                         "bound_per_obs": model.bound_value / train.sum(), "rmse": met["rmse"],
                         "cover95": met["cover95"], "nlpd": met["nlpd"], "n_evals": model.n_evals,
                         "fit_seconds": time.time() - t0, "seconds_per_eval": (time.time() - t0) / model.n_evals}
                        | {k: v for k, v in model.hyperparameters().items()
                           if k.endswith(("_km", "_days", "_years", "_wm2")) or k == "beta1"})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, default=None, help="spatial fold for the M sensitivity (default: largest)")
    parser.add_argument("--m-values", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--kernels", nargs="+", choices=gp.KERNELS, default=list(gp.KERNELS),
                        help="kernels to benchmark (run one per process to parallelise)")
    args = parser.parse_args()
    tag = "" if len(args.kernels) == 3 else "_" + "_".join(args.kernels)
    cfg = config.load()
    torch.set_num_threads(cfg["gp"]["threads"])
    obs = run_cv.load_obs(cfg)
    out = config.MODEL_DIR / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_exact:
        ve = vfe_vs_exact(cfg, obs, args.kernels)
        ve.to_csv(out / f"vfe_vs_exact{tag}.csv", index=False, float_format="%.5g")
    fold = args.fold
    if fold is None:
        summ = pd.read_csv(config.MODEL_DIR / "fold_summary.csv")
        sp = summ[summ.cv_type == "spatial"].sort_values("test")
        fold = int(sp.fold.iloc[len(sp) // 2])  # the median-sized spatial fold
    ms = m_sensitivity(cfg, obs, fold, args.m_values, args.kernels)
    ms.to_csv(out / f"m_sensitivity{tag}.csv", index=False, float_format="%.5g")
    print(ms[["M", "kernel", "rmse", "cover95", "bound_per_obs", "n_evals", "fit_seconds"]].round(3).to_string())


if __name__ == "__main__":
    main()
