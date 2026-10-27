"""Phase 5: blocked spatio-temporal cross-validation of the three GP kernels and of the baselines, and the
full-data fits.

Models (src/baselines.py): the GP kernels exponential | gaussian | matern32 (mean b0 + b1 ERA5 + b2 post_gap),
and the baselines linear (b0 + b1 ERA5 + b2 post_gap, no kernel), linear_noera5 (b0 + b2 post_gap) and
<kernel>_noera5 (GP, mean b0 + b2 post_gap); b2 is there when gp.gap_step is true. All are scored on the same
folds (SHA-256 checked) with the same metrics.

make-folds refuses to replace the folds while results/model holds fits, fold predictions or inducing inputs
made with other folds (e.g. after a grid change): archive or delete results/model first.

Steps (each resumable; finished folds are skipped):
    python src/run_cv.py --make-folds                    write results/model/folds.csv (+ summary, SHA-256)
    python src/run_cv.py --model matern32                every fold of every validation type for one model
    python src/run_cv.py --model matern32 --full         full-data fit: in-sample fitted values + saved model
    python src/run_cv.py --model linear linear_noera5    several models (--kernel is an alias of --model)
    python src/run_cv.py --summarise                     collect, metrics, kernel and model comparisons, figures

On a cluster (euler/): one array task per fit.
    python src/run_cv.py --list-tasks                    number of tasks (full fits + kernel x fold)
    python src/run_cv.py --prepare-inducing              k-means inducing inputs of every fold (run once first)
    python src/run_cv.py --task-index 17                 one fit (a full-data fit or one kernel on one fold)
    --group baselines                                    with the two above: the GPs without ERA5 instead

Per fold, with training rows only: the ERA5 smoothing radius R (highest correlation with CLARA among
era5.smoothing_radii_km), the standardisation of y and ERA5 (inside gp.GP.fit), the k-means inducing inputs
(cached per fold, so every kernel uses the same ones) and all hyperparameters. Test rows are predicted
from the training rows only; buffer rows are neither trained on nor scored.

Outputs (results/model/; <model> is a kernel or a baseline id, and the `kernel` column of the tables holds it):
    folds.csv, fold_summary.csv, folds_sha256.txt
    <model>/folds/<cv_type>_<fold>.parquet                   per-fold out-of-sample predictions (resume state)
    <model>/oos_predictions_<cv_type>[_partN].csv            out-of-sample predictions, sample_type=out_of_sample
    <model>/insample_fitted.csv                              full-data fitted values, sample_type=in_sample
    <model>/hyperparameters_<cv_type>.csv, hyperparameters_full.csv, config.json, full_model.pt
    <model>/metrics_by_fold.csv, metrics_summary.csv, metrics_breakdown.csv
    kernel_comparison.csv, kernel_pairwise.csv, missing_tasks.csv     GP kernels only (they pick the winner)
    model_comparison.csv, model_pairwise.csv, missing_baseline_tasks.csv   every model, baselines included
    figures/*.png
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

import baselines
import config
import cv
import gp
from style import (CLARA_UNIT, INK, KERNEL_COLORS, KERNEL_LABELS, MUTED, model_color, model_label, plt, save,
                   style_axes, title)

CV_TYPES = ("spatial", "temporal", "joint")
GROUPS = ("kernels", "baselines")  # task lists: the GP kernels, or the GPs without ERA5
PART_ROWS = 100_000


def load_obs(cfg):
    obs = pd.read_parquet(config.PROCESSED_DIR / f"harmonised_{cfg['grid']['res_deg']}deg_monthly.parquet")
    obs["band"] = cv.lat_band(obs.lat)
    obs["season"] = obs.month.map(cv.SEASONS)
    return obs


def inputs(df, cfg):
    return gp.embed(df.lat, df.lon, df.model_time, cfg["time"]["origin"])


def choose_radius(obs, train, radii):
    """ERA5 smoothing radius with the highest correlation with CLARA on the training rows."""
    r = {rad: np.corrcoef(obs.clara[train], obs[f"era5_night_r{int(rad)}"][train])[0, 1] for rad in radii}
    return max(r, key=r.get), r


def era5_predictor(cfg, obs, train, era5):
    """(radius, correlations by radius, ERA5 predictor of every row) of one fit; NaN for a model without ERA5."""
    if not era5:
        return np.nan, {}, np.full(len(obs), np.nan)
    radius, rs = choose_radius(obs, train, cfg["era5"]["smoothing_radii_km"])
    return radius, rs, obs[f"era5_night_r{int(radius)}"].to_numpy()


def inducing(cfg, x_train, key):
    """k-means inducing inputs of one fold, cached so every kernel of the fold uses the same ones."""
    path = config.MODEL_DIR / "inducing" / f"{key}_{cfg['grid']['res_deg']}deg_M{cfg['gp']['n_inducing']}.npy"
    if path.exists():
        return np.load(path)
    z = gp.inducing_inputs(x_train, cfg["gp"]["n_inducing"], cfg["gp"], cfg["gp"]["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, z)
    os.replace(tmp, path)  # atomic: concurrent jobs never read a half-written file
    return z


def predictions_frame(obs, rows, pred, kernel, cv_type, fold, block_id, sample_type, e, radius):
    d = obs.loc[rows]
    e = e[rows]
    out = pd.DataFrame({
        "obs_id": d.obs_id.to_numpy(), "lat": d.lat.to_numpy(), "lon": d.lon.to_numpy(), "year": d.year.to_numpy(),
        "month": d.month.to_numpy(), "time": d.time.to_numpy(), "clara_obs": d.clara.to_numpy(),
        "era5_night": e, "era5_radius_km": radius, "n_overpasses": d.n_overpasses.to_numpy(),
        "pred_mean": pred["mean"], "mean_fn": pred["mean_fn"], "latent_var": pred["latent_var"],
        "latent_sd": np.sqrt(pred["latent_var"]), "pred_var_obs": pred["obs_var"],
        "pred_sd_obs": np.sqrt(pred["obs_var"]), "residual": d.clara.to_numpy() - pred["mean"],
        "post_gap": d.post_gap.to_numpy(),
        "kernel": kernel, "cv_type": cv_type, "fold": fold, "block_id": block_id, "sample_type": sample_type,
        "cell_area_km2": d.cell_area_km2.to_numpy(), "period": d.period.to_numpy(), "band": d.band.to_numpy(),
        "season": d.season.to_numpy()})
    out["pit"] = stats.norm.cdf(out.residual / out.pred_sd_obs)  # Phi((y - mu) / sd)
    return out


FOLD_SECTIONS = ("qc", "grid", "era5", "time", "cv")  # sections that define the data and the folds


def check_folds(cfg=None):
    """Fold table, after checking its SHA-256 and (when cfg is given) that cfg defines the same data and
    folds as the configuration the folds were made with (e.g. config/pipeline_euler.toml may only differ in [gp])."""
    folds = pd.read_csv(config.MODEL_DIR / "folds.csv")
    expected = (config.MODEL_DIR / "folds_sha256.txt").read_text().strip()
    if cv.folds_hash(folds) != expected:
        raise RuntimeError("folds.csv does not match folds_sha256.txt; refusing to run with different folds")
    made_with = config.MODEL_DIR / "config_folds.json"
    if cfg is not None and made_with.exists():
        ref = json.loads(made_with.read_text())
        diff = [k for k in FOLD_SECTIONS if json.loads(json.dumps(cfg.get(k), default=str)) != ref.get(k)]
        if diff:
            raise RuntimeError(f"config sections {diff} differ from those the folds were made with ({made_with})")
    return folds


def stale_outputs(new_hash):
    """Fits, fold predictions and inducing inputs under MODEL_DIR that were made with folds other than
    new_hash (the folds_sha256.txt on disk differs); empty when the folds are unchanged or nothing exists."""
    sha = config.MODEL_DIR / "folds_sha256.txt"
    if not sha.exists() or sha.read_text().strip() == new_hash:
        return []
    d = config.MODEL_DIR
    return sorted(d.glob("*/folds/*.parquet")) + sorted(d.glob("*/full_model.pt")) + sorted(d.glob("inducing/*.npy"))


def make_folds(cfg):
    obs = load_obs(cfg)
    folds = cv.make_folds(obs, cfg)
    stale = stale_outputs(cv.folds_hash(folds))
    if stale:
        raise RuntimeError(f"{len(stale)} outputs in {config.MODEL_DIR} were made with other folds (e.g. "
                           f"{stale[0].relative_to(config.MODEL_DIR)}); the runs would skip them as done. "
                           f"Archive or delete {config.MODEL_DIR} first, then make the folds again.")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(config.MODEL_DIR / "folds.csv", index=False)
    (config.MODEL_DIR / "folds_sha256.txt").write_text(cv.folds_hash(folds) + "\n")
    config.save({k: cfg[k] for k in FOLD_SECTIONS}, config.MODEL_DIR / "config_folds.json")
    summ = cv.fold_summary(folds, len(obs))
    summ.to_csv(config.MODEL_DIR / "fold_summary.csv", index=False)
    print(summ.groupby("cv_type")[["test", "buffer", "train"]].describe().round(0).T.to_string())
    print(f"wrote {len(folds):,} fold rows; SHA-256 {cv.folds_hash(folds)[:12]}")


def fit_fold(cfg, obs, folds, x_all, kernel, cv_type, fold, verbose=False):
    """Fit one model (a kernel or a baseline id) on the training rows of one fold and save its out-of-sample
    predictions."""
    out_dir = config.MODEL_DIR / kernel
    path = out_dir / "folds" / f"{cv_type}_{fold:02d}.parquet"
    if path.exists():
        print(f"{kernel} {cv_type} fold {fold}: done already ({path.name})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    family, era5 = baselines.parse(kernel)
    train, test = cv.split(folds, cv_type, fold, obs.obs_id.to_numpy())
    block_id = folds[(folds.cv_type == cv_type) & (folds.fold == fold)].block_id.iloc[0]
    radius, rs, e = era5_predictor(cfg, obs, train, era5)
    z = inducing(cfg, x_all[train], f"{cv_type}_{fold:02d}") if family else None
    t0 = time.time()
    s = obs.post_gap.to_numpy()
    model = baselines.make(kernel, cfg["gp"], z).fit(x_all[train], obs.clara[train], e[train],
                                                     obs.n_overpasses[train], verbose=verbose, s=s[train])
    pred = model.predict(x_all[test], e[test], obs.n_overpasses[test].to_numpy(), s=s[test])
    frame = predictions_frame(obs, np.flatnonzero(test), pred, kernel, cv_type, fold, block_id,
                              "out_of_sample", e, radius)
    hp = model.hyperparameters() | {"cv_type": cv_type, "fold": fold, "block_id": block_id,
                                    "era5_radius_km": radius, "n_train": int(train.sum()),
                                    "n_test": int(test.sum()), "fit_seconds": time.time() - t0} \
        | {f"train_r_R{int(k)}": v for k, v in rs.items()}
    frame.attrs = {}
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    pd.DataFrame([hp]).to_parquet(path.with_suffix(".hp.parquet"), index=False)
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # the prediction file appears only when complete (it marks the fold as done)
    m = cv.metrics(frame.clara_obs, frame.pred_mean, frame.pred_var_obs, frame.cell_area_km2, frame.mean_fn)
    print(f"{kernel} {cv_type} fold {fold:2d} {block_id}: n_test {m['n']:5d}, RMSE {m['rmse']:6.2f}, "
          f"cover95 {m['cover95']:.2f}, R {radius:g} km, {hp['n_evals']} evals, {hp['fit_seconds']:.0f} s",
          flush=True)


def run_model(cfg, kernel, cv_types, verbose):
    torch.set_num_threads(cfg["gp"]["threads"])
    obs = load_obs(cfg)
    folds = check_folds(cfg)
    config.save(cfg, config.MODEL_DIR / kernel / "config.json")
    x_all = inputs(obs, cfg)
    for cv_type in cv_types:
        for fold in sorted(folds[folds.cv_type == cv_type].fold.unique()):
            fit_fold(cfg, obs, folds, x_all, kernel, cv_type, int(fold), verbose)
        collect(cfg, kernel, cv_type)


def model_tasks(folds, models):
    """[(model, cv_type, fold)]: the full-data fits first (largest), then every model on every fold.
    cv_type 'full' with fold -1 is a full-data fit."""
    tasks = [(k, "full", -1) for k in models]
    for t in CV_TYPES:
        for f in sorted(folds[folds.cv_type == t].fold.unique()):
            tasks += [(k, t, int(f)) for k in models]
    return tasks


def task_list(cfg, folds, group="kernels"):
    """Tasks of a Slurm array. 'kernels' (the default, indices fixed while an array runs): the GP kernels.
    'baselines': the GPs without ERA5 ([baselines] gp_noera5_kernels); the regression-only baselines take
    seconds and run in one process (--model linear linear_noera5)."""
    if group not in GROUPS:
        raise ValueError(f"unknown task group {group!r}; expected one of {GROUPS}")
    return model_tasks(folds, cfg["gp"]["kernels"] if group == "kernels" else baselines.gp_baselines(cfg))


def run_task(cfg, index, verbose, group="kernels"):
    folds = check_folds(cfg)
    tasks = task_list(cfg, folds, group)
    kernel, cv_type, fold = tasks[index]
    print(f"task {index} of {len(tasks)}: {kernel} {cv_type} {fold}", flush=True)
    torch.set_num_threads(cfg["gp"]["threads"])
    config.save(cfg, config.MODEL_DIR / kernel / "config.json")
    if cv_type == "full":
        if (config.MODEL_DIR / kernel / "full_model.pt").exists():
            print(f"{kernel} full fit: done already")
            return
        run_full(cfg, kernel, verbose)
        return
    obs = load_obs(cfg)
    fit_fold(cfg, obs, folds, inputs(obs, cfg), kernel, cv_type, fold, verbose)


def prepare_inducing(cfg):
    """Compute and cache the inducing inputs of every fold and of the full fit (one process, no races)."""
    obs = load_obs(cfg)
    folds = check_folds(cfg)
    x_all = inputs(obs, cfg)
    t0 = time.time()
    inducing(cfg, x_all, "full")
    for t in CV_TYPES:
        for f in sorted(folds[folds.cv_type == t].fold.unique()):
            train, _ = cv.split(folds, t, f, obs.obs_id.to_numpy())
            inducing(cfg, x_all[train], f"{t}_{int(f):02d}")
    print(f"inducing inputs (M = {cfg['gp']['n_inducing']}) cached for every fold in {time.time() - t0:.0f} s")


def write_parts(df, path):
    """CSV, split into <= PART_ROWS-row parts (path_part1.csv, ...) when larger."""
    for old in path.parent.glob(path.stem + "_part*.csv"):
        old.unlink()
    if len(df) <= PART_ROWS:
        df.to_csv(path, index=False, float_format="%.4f")
        return
    if path.exists():
        path.unlink()
    for k, start in enumerate(range(0, len(df), PART_ROWS), 1):
        df.iloc[start:start + PART_ROWS].to_csv(path.with_name(f"{path.stem}_part{k}.csv"), index=False,
                                                float_format="%.4f")


def collect(cfg, kernel, cv_type):
    out_dir = config.MODEL_DIR / kernel
    files = sorted((out_dir / "folds").glob(f"{cv_type}_*[0-9].parquet"))
    if not files:
        return
    pred = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    assert (pred.sample_type == "out_of_sample").all()
    write_parts(pred, out_dir / f"oos_predictions_{cv_type}.csv")
    hp = pd.concat([pd.read_parquet(f.with_suffix(".hp.parquet")) for f in files], ignore_index=True)
    hp.to_csv(out_dir / f"hyperparameters_{cv_type}.csv", index=False, float_format="%.6g")


def run_full(cfg, kernel, verbose):
    torch.set_num_threads(cfg["gp"]["threads"])
    obs = load_obs(cfg)
    out_dir = config.MODEL_DIR / kernel
    out_dir.mkdir(parents=True, exist_ok=True)
    x = inputs(obs, cfg)
    family, era5 = baselines.parse(kernel)
    allrows = np.ones(len(obs), bool)
    radius, rs, e = era5_predictor(cfg, obs, allrows, era5)
    z = inducing(cfg, x, "full") if family else None
    t0 = time.time()
    s = obs.post_gap.to_numpy()
    model = baselines.make(kernel, cfg["gp"], z).fit(x, obs.clara, e, obs.n_overpasses, verbose=verbose, s=s)
    pred = model.predict(x, e, obs.n_overpasses.to_numpy(), s=s)
    frame = predictions_frame(obs, np.arange(len(obs)), pred, kernel, "none", -1, "all", "in_sample", e, radius)
    write_parts(frame, out_dir / "insample_fitted.csv")
    hp = model.hyperparameters() | {"era5_radius_km": radius, "n_train": len(obs), "fit_seconds": time.time() - t0} \
        | {f"train_r_R{int(k)}": v for k, v in rs.items()}
    pd.DataFrame([hp]).to_csv(out_dir / "hyperparameters_full.csv", index=False, float_format="%.6g")
    torch.save(model.state() | {"era5_radius_km": radius, "origin": cfg["time"]["origin"]}, out_dir / "full_model.pt")
    m = cv.metrics(frame.clara_obs, frame.pred_mean, frame.pred_var_obs, frame.cell_area_km2, frame.mean_fn)
    print(f"{kernel} full fit: in-sample RMSE {m['rmse']:.2f}, R {radius:g} km, {hp['n_evals']} evals, "
          f"{hp['fit_seconds']:.0f} s")
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in hp.items()}, indent=1))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def load_oos(kernel):
    files = sorted((config.MODEL_DIR / kernel / "folds").glob("*_[0-9][0-9].parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def completeness(cfg, models):
    """Fits (full and per fold) of the given models whose output does not exist yet."""
    folds = check_folds(cfg)
    missing = []
    for k, t, f in model_tasks(folds, models):
        done = (config.MODEL_DIR / k / "full_model.pt") if t == "full" else \
            (config.MODEL_DIR / k / "folds" / f"{t}_{f:02d}.parquet")
        if not done.exists():
            missing.append({"kernel": k, "cv_type": t, "fold": f})
    return pd.DataFrame(missing, columns=["kernel", "cv_type", "fold"])


COMPARE_COLS = ("rmse_area", "mae", "bias", "cover50", "cover80", "cover95", "nlpd", "crps", "width95", "r2",
                "r2_mean_fn")


def comparison_table(fm):
    """Primary score, mean per-fold RMSE by validation type and the other mean per-fold metrics, best first."""
    score, per_type = cv.primary_score(fm)
    comp = per_type.add_prefix("mean_fold_rmse_")
    comp["primary_score"] = score
    for col in COMPARE_COLS:
        comp = comp.join(fm.groupby("kernel")[col].mean().rename(f"mean_fold_{col}"))
    return comp.sort_values("primary_score")


def summarise(cfg):
    fig_dir = config.MODEL_DIR / "figures"
    kernels = list(cfg["gp"]["kernels"])
    missing = completeness(cfg, kernels)
    missing.to_csv(config.MODEL_DIR / "missing_tasks.csv", index=False)
    if len(missing):
        print(f"WARNING: {len(missing)} fits missing ({config.MODEL_DIR / 'missing_tasks.csv'}); summaries are "
              f"incomplete:\n"
              f"{missing.groupby(['kernel', 'cv_type']).size().to_string()}")
    base = baselines.baseline_models(cfg)
    missing_b = completeness(cfg, base).rename(columns={"kernel": "model"})
    missing_b.to_csv(config.MODEL_DIR / "missing_baseline_tasks.csv", index=False)
    if len(missing_b):
        print(f"note: {len(missing_b)} baseline fits missing ({config.MODEL_DIR / 'missing_baseline_tasks.csv'}):\n"
              f"{missing_b.groupby(['model', 'cv_type']).size().to_string()}")
    models = kernels + base
    for kernel in models:
        for t in CV_TYPES:
            collect(cfg, kernel, t)
    allpred, fold_rows = {}, []
    for kernel in models:
        pred = load_oos(kernel)
        if pred is None:
            continue
        allpred[kernel] = pred
        out_dir = config.MODEL_DIR / kernel
        fm = pd.DataFrame([{"kernel": kernel, "cv_type": t, "fold": f, "block_id": g.block_id.iloc[0]}
                           | cv.metrics(g.clara_obs, g.pred_mean, g.pred_var_obs, g.cell_area_km2, g.mean_fn)
                           for (t, f), g in pred.groupby(["cv_type", "fold"])])
        fm.to_csv(out_dir / "metrics_by_fold.csv", index=False, float_format="%.4f")
        fold_rows.append(fm)
        summ = pd.DataFrame([{"kernel": kernel, "cv_type": t, "pooled": True}
                             | cv.metrics(g.clara_obs, g.pred_mean, g.pred_var_obs, g.cell_area_km2, g.mean_fn)
                             for t, g in pred.groupby("cv_type")])
        agg = fm.groupby("cv_type")[["rmse", "rmse_area", "mae", "bias", "cover95", "nlpd", "crps"]].agg(["mean", "std"])
        agg.columns = [f"fold_{a}_{b}" for a, b in agg.columns]
        summ = summ.merge(agg.reset_index(), on="cv_type")
        summ.to_csv(out_dir / "metrics_summary.csv", index=False, float_format="%.4f")
        bd = pd.concat([cv.breakdown(pred, ["cv_type", by]).rename(columns={by: "group_value"}).assign(group=by)
                        for by in ("band", "season", "period", "year")], ignore_index=True)
        bd.to_csv(out_dir / "metrics_breakdown.csv", index=False, float_format="%.4f")
    if not fold_rows:
        print("no predictions yet")
        return
    fm_all = pd.concat(fold_rows, ignore_index=True)
    model_comparison(cfg, fm_all, fig_dir)
    fm = fm_all[fm_all.kernel.isin(kernels)]
    allpred = {k: v for k, v in allpred.items() if k in kernels}
    if fm.empty:
        print("no GP kernel predictions yet")
        return
    comp = comparison_table(fm)
    comp.to_csv(config.MODEL_DIR / "kernel_comparison.csv", float_format="%.4f")

    rows = []
    best = comp.index[0]
    for t in CV_TYPES:
        piv = fm[fm.cv_type == t].pivot(index="fold", columns="kernel", values="rmse").dropna()
        for k in piv.columns:
            if k == best:
                continue
            d, lo, hi = cv.paired_bootstrap(piv, k, best, cfg["cv"]["bootstrap_samples"], cfg["gp"]["seed"])
            rows.append({"cv_type": t, "kernel": k, "vs": best, "mean_rmse_diff": d, "ci95_lo": lo, "ci95_hi": hi,
                         "folds": len(piv),  # a bootstrap over fewer than 5 blocks says nothing
                         "distinguishable": (not (lo <= 0 <= hi)) if len(piv) >= 5 else None})
    pw = pd.DataFrame(rows)
    pw.to_csv(config.MODEL_DIR / "kernel_pairwise.csv", index=False, float_format="%.4f")
    figures(fm, allpred, fig_dir)
    print(comp.round(3).to_string())
    print(pw.round(3).to_string(index=False))


def model_comparison(cfg, fm, fig_dir):
    """Every model present (GP kernels and baselines), scored on the same folds: model_comparison.csv, paired
    block-bootstrap differences of every pair on their common folds (model_pairwise.csv) and a figure."""
    comp = comparison_table(fm)
    comp.index.name = "model"
    parsed = [baselines.parse(m) for m in comp.index]
    comp.insert(0, "gp", [f is not None for f, _ in parsed])
    comp.insert(1, "era5", [e for _, e in parsed])
    comp.to_csv(config.MODEL_DIR / "model_comparison.csv", float_format="%.4f")
    order = list(comp.index)
    rows = []
    for t in CV_TYPES:
        piv = fm[fm.cv_type == t].pivot(index="fold", columns="kernel", values="rmse")
        for i, b in enumerate(order):
            for a in order[i + 1:]:  # a ranks below b, so the difference is mostly positive
                if a not in piv or b not in piv:
                    continue
                pair = piv[[a, b]].dropna()
                if pair.empty:
                    continue
                d, lo, hi = cv.paired_bootstrap(pair, a, b, cfg["cv"]["bootstrap_samples"], cfg["gp"]["seed"])
                rows.append({"cv_type": t, "model": a, "vs": b, "mean_rmse_diff": d, "ci95_lo": lo, "ci95_hi": hi,
                             "folds": len(pair),
                             "distinguishable": (not (lo <= 0 <= hi)) if len(pair) >= 5 else None})
    pd.DataFrame(rows, columns=["cv_type", "model", "vs", "mean_rmse_diff", "ci95_lo", "ci95_hi", "folds",
                                "distinguishable"]).to_csv(config.MODEL_DIR / "model_pairwise.csv", index=False,
                                                           float_format="%.4f")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained", sharey=True)
    rng = np.random.default_rng(0)
    for ax, t in zip(axes, CV_TYPES):
        for i, m in enumerate(order):
            v = fm[(fm.cv_type == t) & (fm.kernel == m)].rmse.to_numpy()
            if not len(v):
                continue
            hollow = m.endswith(baselines.NOERA5)
            ax.plot(i + rng.uniform(-0.15, 0.15, len(v)), v, "o", ms=4, alpha=0.6, color=model_color(m),
                    mfc="white" if hollow else model_color(m), mew=1)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color=model_color(m), lw=2.5)
        ax.set_xticks(range(len(order)), [model_label(m) for m in order], rotation=30, ha="right", fontsize=8)
        ax.set(ylabel=f"RMSE ({CLARA_UNIT})")
        style_axes(ax)
        title(ax, f"{t}: per-fold RMSE (bars: mean over folds)")
    fig.suptitle("GP kernels and baselines on identical folds (hollow: no ERA5)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    save(fig, fig_dir / "model_comparison.png")
    print(comp[["gp", "era5", "primary_score"] + [c for c in comp if c.startswith("mean_fold_rmse_")]]
          .round(3).to_string())


def figures(fm, allpred, fig_dir):
    kernels = list(allpred)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained", sharey=True)
    for ax, t in zip(axes, CV_TYPES):
        g = fm[fm.cv_type == t]
        for k_i, k in enumerate(kernels):
            v = g[g.kernel == k].sort_values("fold")
            ax.plot(v.fold + (k_i - 1) * 0.18, v.rmse, "o", color=KERNEL_COLORS[k], ms=6, mec="white", mew=1,
                    label=KERNEL_LABELS[k])
            ax.hlines(v.rmse.mean(), v.fold.min() - 0.4, v.fold.max() + 0.4, color=KERNEL_COLORS[k], lw=1.5)
        ax.set(xlabel="fold (held-out block)", ylabel=f"RMSE ({CLARA_UNIT})")
        style_axes(ax)
        title(ax, f"{t}: per-fold RMSE (lines: mean over folds)")
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, fig_dir / "rmse_by_fold.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained", sharey=True)
    levels = np.array([0.5, 0.8, 0.95])
    for ax, t in zip(axes, CV_TYPES):
        ax.plot([0.4, 1], [0.4, 1], color=MUTED, ls="--", lw=1, label="perfect calibration")
        for k in kernels:
            p = allpred[k][allpred[k].cv_type == t]
            z = np.abs(p.residual / p.pred_sd_obs)
            emp = [np.mean(z <= stats.norm.ppf(0.5 + lv / 2)) for lv in levels]
            ax.plot(levels, emp, color=KERNEL_COLORS[k], lw=2, marker="o", ms=6, label=KERNEL_LABELS[k])
        ax.set(xlabel="nominal coverage", ylabel="empirical coverage", xlim=(0.4, 1), ylim=(0.4, 1))
        style_axes(ax)
        title(ax, f"{t}: central predictive intervals")
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, fig_dir / "coverage.png")

    fig, axes = plt.subplots(len(kernels), 3, figsize=(14, 3 * len(kernels)), layout="constrained", sharex=True)
    axes = np.atleast_2d(axes)
    for r, k in enumerate(kernels):
        for c, t in enumerate(CV_TYPES):
            p = allpred[k][allpred[k].cv_type == t]
            axes[r, c].hist(p.pit, bins=20, range=(0, 1), color=KERNEL_COLORS[k], edgecolor="white", density=True)
            axes[r, c].axhline(1, color=MUTED, ls="--", lw=1)
            style_axes(axes[r, c])
            title(axes[r, c], f"{KERNEL_LABELS[k]}, {t}", 9)
    fig.suptitle("PIT histograms (flat = calibrated; U = intervals too narrow; hump = too wide)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    save(fig, fig_dir / "pit.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="configuration file (default config/pipeline.toml)")
    parser.add_argument("--make-folds", action="store_true")
    parser.add_argument("--model", "--kernel", dest="model", nargs="+", choices=baselines.ALL_IDS,
                        help="GP kernel(s) or baseline model(s) to cross-validate (src/baselines.py)")
    parser.add_argument("--cv-type", nargs="+", choices=CV_TYPES, default=list(CV_TYPES))
    parser.add_argument("--full", action="store_true", help="full-data fit of each --model")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-tasks", action="store_true", help="print the tasks and, last, their number")
    parser.add_argument("--task-index", type=int, help="run one task of --list-tasks (0-based), for job arrays")
    parser.add_argument("--prepare-inducing", action="store_true", help="cache every fold's inducing inputs")
    parser.add_argument("--group", choices=GROUPS, default="kernels",
                        help="task list of --list-tasks / --task-index: the GP kernels (default) or the GPs "
                             "without ERA5")
    args = parser.parse_args()
    cfg = config.load(args.config)
    config.MODEL_DIR = config.RESULTS_DIR / cfg.get("output", {}).get("model_dir", "model")
    if args.make_folds:
        make_folds(cfg)
    if args.list_tasks:
        tasks = task_list(cfg, check_folds(cfg), args.group)
        for i, t in enumerate(tasks):
            print(i, *t)
        print(len(tasks))
    if args.prepare_inducing:
        prepare_inducing(cfg)
    if args.task_index is not None:
        run_task(cfg, args.task_index, args.verbose, args.group)
    for model in args.model or []:
        if args.full:
            run_full(cfg, model, args.verbose)
        else:
            run_model(cfg, model, args.cv_type, args.verbose)
    if args.summarise:
        summarise(cfg)


if __name__ == "__main__":
    main()
