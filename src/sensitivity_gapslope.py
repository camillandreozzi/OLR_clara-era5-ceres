"""Sensitivity to a period-specific ERA5 slope: y = b0 + b1 e + b2 s + b3 s e + f(x) + eps.

The paper's model has one ERA5 slope for both calibration periods, but the CLARA-on-ERA5 slope of the cell-month
means differs between them (eda/joint_stats.csv). The variants

    exponential_gapslope   the current exponential GP + b3 s e
    linear_gapslope        the regression on ERA5 (no GP) + b3 s e

are fitted on the same 49 folds (folds.csv, checked against folds_sha256.txt) and on all data, into results/model/
next to the paper's fits, with the paper's settings (config/pipeline_euler.toml). On Euler the GP variant reuses
every fold's cached inducing inputs (euler/submit_sensitivity.sh), so it pairs exactly with the current GP; the
regression has no inducing inputs and runs anywhere in seconds.

Outputs, results/model/ (each table is written once its inputs exist; report.py turns them into macros):
    gapslope_scores.csv          primary score and mean fold RMSE per CV type, current models and variants
    gapslope_pairwise.csv        variant minus current, paired over folds per CV type: mean RMSE difference with a
                                 fold-bootstrap 95 % interval (cv.paired_bootstrap)
    gapslope_coverage.csv        pooled coverage of the central 50 / 80 / 95 % predictive intervals per CV type
    gapslope_beta.csv            full-data b1 (pre-gap slope), b3 (post minus pre) and b1 + b3, with GLS standard
                                 errors at the fitted hyperparameters (gp.GP.mean_coef_covariance)
    gapslope_global_mean.csv     monthly global mean G_t of the current and the variant GP, and their difference
    gapslope_global_summary.csv  per period (pre-gap, gap, post-gap, all): max and mean |difference| of G_t, and the
                                 months in which |difference| exceeds 2 SD of the current model's G_t

Usage:
    python src/sensitivity_gapslope.py --fit linear_gapslope        # seconds, anywhere
    bash euler/submit_sensitivity.sh                                # on Euler: the GP variant, then --summarise
    python src/sensitivity_gapslope.py --summarise
"""

import argparse

import numpy as np
import pandas as pd
import torch

import baselines
import config
import cv
import kernel_global_means
import predict_maps
import run_cv

CONFIG = config.ROOT / "config" / "pipeline_euler.toml"
PAIRS = [("exponential_gapslope", "exponential"), ("linear_gapslope", "linear")]  # (variant, current model)
VARIANTS = [v for v, _ in PAIRS]


def setup(path=None):
    cfg = config.load(str(path or CONFIG))
    config.MODEL_DIR = config.RESULTS_DIR / cfg.get("output", {}).get("model_dir", "model")
    return cfg


def fit(cfg, models, verbose):
    for model in models:
        run_cv.run_model(cfg, model, run_cv.CV_TYPES, verbose)
        if not (config.MODEL_DIR / model / "full_model.pt").exists():
            run_cv.run_full(cfg, model, verbose)


def predictions(cfg, model):
    """Pooled out-of-sample predictions of one model (every CV type), or None if any CV type is missing. Per-fold
    files (array tasks do not pool them) are pooled first; without them the existing tables are kept."""
    for t in run_cv.CV_TYPES:
        run_cv.collect(cfg, model, t)
    parts = [sorted((config.MODEL_DIR / model).glob(f"oos_predictions_{t}*.csv")) for t in run_cv.CV_TYPES]
    if not all(parts):
        return None
    return pd.concat([pd.read_csv(f) for files in parts for f in files], ignore_index=True)


def complete(cfg, model, pred):
    """True when the model has predictions on every fold of every CV type."""
    folds = run_cv.check_folds(cfg)
    want = folds.groupby("cv_type").fold.nunique()
    return pred is not None and (pred.groupby("cv_type").fold.nunique().reindex(want.index) == want).all()


def evaluate(cfg):
    """Scores, paired differences and coverage of every pair whose two models are complete."""
    fm, cover, rows = [], [], []
    for variant, current in PAIRS:
        preds = {m: predictions(cfg, m) for m in (variant, current)}
        if not all(complete(cfg, m, p) for m, p in preds.items()):
            print(f"{variant} vs {current}: not every fold is fitted yet; skipped")
            continue
        for m, pred in preds.items():
            fm.append(pd.DataFrame([{"kernel": m, "cv_type": t, "fold": f}
                                    | cv.metrics(g.clara_obs, g.pred_mean, g.pred_var_obs, g.cell_area_km2, g.mean_fn)
                                    for (t, f), g in pred.groupby(["cv_type", "fold"])]))
            cover += [{"model": m, "cv_type": t}
                      | {k: v for k, v in cv.metrics(g.clara_obs, g.pred_mean, g.pred_var_obs, g.cell_area_km2,
                                                     g.mean_fn).items() if k.startswith("cover")}
                      for t, g in pred.groupby("cv_type")]
        both = pd.concat(fm[-2:], ignore_index=True)
        for t in run_cv.CV_TYPES:
            piv = both[both.cv_type == t].pivot(index="fold", columns="kernel", values="rmse")[[variant, current]]
            d, lo, hi = cv.paired_bootstrap(piv.dropna(), variant, current, cfg["cv"]["bootstrap_samples"],
                                            cfg["gp"]["seed"])
            rows.append({"variant": variant, "current": current, "cv_type": t, "mean_rmse_diff": d, "ci95_lo": lo,
                         "ci95_hi": hi, "folds": len(piv.dropna()), "distinguishable": not (lo <= 0 <= hi)})
    if not fm:
        return None, None, None
    fm = pd.concat(fm, ignore_index=True)
    score, per_type = cv.primary_score(fm)
    scores = per_type.add_prefix("mean_fold_rmse_").assign(primary_score=score).rename_axis("model")
    return scores, pd.DataFrame(rows), pd.DataFrame(cover)


def full_fit(cfg, model):
    """(fitted model, ERA5 radius, training x, e, n, s) of a full-data fit, as run_cv.run_full made it."""
    st = torch.load(config.MODEL_DIR / model / "full_model.pt", weights_only=False)
    family, _ = baselines.parse(model)
    m = (baselines.gp.GP if family else baselines.LinearModel).from_state(st)
    obs = run_cv.load_obs(cfg)
    radius = st["era5_radius_km"]
    return m, radius, run_cv.inputs(obs, cfg), obs[f"era5_night_r{int(radius)}"].to_numpy(), \
        obs.n_overpasses.to_numpy(), obs.post_gap.to_numpy()


def slopes(cfg):
    """Full-data b1 (pre-gap slope), b3 and b1 + b3 (post-gap slope) of every fitted variant, with GLS SEs."""
    rows = []
    for model in VARIANTS:
        if not (config.MODEL_DIR / model / "full_model.pt").exists():
            continue
        m, radius, x, e, n, s = full_fit(cfg, model)
        if isinstance(m, baselines.LinearModel):
            names, cov = ["beta0", "beta1", "beta2", "beta3"], m.cov
        else:
            names, cov = m.mean_coef_covariance(x, e, n, s, chunk=cfg["gp"]["chunk_size"])
        scale = m.stats["y_sd"] / m.stats["e_sd"]
        i1, i3 = names.index("beta1"), names.index("beta3")
        b1, b3 = m.coefficients()[1], m.slope_change()
        rows.append({"model": model, "era5_radius_km": radius, "beta1": b1, "beta1_se": np.sqrt(cov[i1, i1]) * scale,
                     "beta3": b3, "beta3_se": np.sqrt(cov[i3, i3]) * scale, "slope_post": b1 + b3,
                     "slope_post_se": np.sqrt(cov[i1, i1] + cov[i3, i3] + 2 * cov[i1, i3]) * scale})
    return pd.DataFrame(rows) if rows else None


def global_means(cfg):
    """G_t of the current and the variant full-data GP fits (predict_maps.global_mean), their difference, and a
    summary per period."""
    variant, current = PAIRS[0]
    if not all((config.MODEL_DIR / m / "full_model.pt").exists() for m in (variant, current)):
        return None, None
    out = {}
    for model in (current, variant):
        m, radius, *_ = full_fit(cfg, model)
        grid, x, ok = predict_maps.predict_grid(cfg, m, radius)
        out[model] = predict_maps.global_mean(cfg, m, grid, x, ok, radius).set_index("time")
    g = pd.DataFrame({"current": out[current].global_mean, "variant": out[variant].global_mean,
                      "sd_current": out[current].sd_full_covariance, "sd_variant": out[variant].sd_full_covariance})
    g["diff"] = g.variant - g.current
    g["period"] = kernel_global_means.period(g.index, cfg)
    rows = []
    for name in ("pre-gap", "gap", "post-gap", "all"):
        sel = g if name == "all" else g[g.period == name]
        d = sel["diff"].abs()
        rows.append({"months": name, "n": len(sel), "max_abs_diff": d.max(), "month_of_max": f"{d.idxmax():%Y-%m}",
                     "mean_abs_diff": d.mean(), "mean_diff": sel["diff"].mean(),
                     "months_above_2sd": int((d > 2 * sel.sd_current).sum())})
    return g, pd.DataFrame(rows)


def summarise(cfg):
    out = config.MODEL_DIR
    pd.set_option("display.width", 200)
    scores, pw, cover = evaluate(cfg)
    tables = {"gapslope_scores.csv": scores, "gapslope_pairwise.csv": pw, "gapslope_coverage.csv": cover,
              "gapslope_beta.csv": slopes(cfg)}
    g, gs = global_means(cfg)
    tables |= {"gapslope_global_mean.csv": g, "gapslope_global_summary.csv": gs}
    for name, t in tables.items():
        if t is None:
            continue
        t.to_csv(out / name, index=name in ("gapslope_scores.csv", "gapslope_global_mean.csv"), float_format="%.5f")
        print(f"{name}\n{t.round(3).to_string()}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", nargs="+", choices=VARIANTS, metavar="MODEL", help=f"fit models: {VARIANTS}")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", help=f"configuration (default {CONFIG.relative_to(config.ROOT)})")
    args = parser.parse_args()
    cfg = setup(args.config)
    torch.set_num_threads(cfg["gp"]["threads"])
    if args.fit:
        fit(cfg, args.fit, args.verbose)
    if args.summarise:
        summarise(cfg)


if __name__ == "__main__":
    main()
