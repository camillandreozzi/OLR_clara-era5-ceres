"""Phase 7: final report answering the nine questions, from the generated tables only.

Every number in results/report/report.md is read from an output of an earlier phase; the status table at
the end records which outputs exist (were executed) and when they were written.

Usage:
    python src/report.py
"""

import datetime as dt
import json

import numpy as np
import pandas as pd

import config
from style import KERNEL_LABELS, model_label

R = config.RESULTS_DIR
OUT = config.REPORT_DIR
KERNELS = ("exponential", "gaussian", "matern32")
CV_TYPES = ("spatial", "temporal", "joint")
LINEAR = ("linear", "linear_noera5")  # regression-only baselines (src/baselines.py)


def rel(path):
    """Link to a results path from results/report/."""
    return "../" + str(path.relative_to(R))


def md(df, floatfmt="{:.2f}", index=False):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def status_table():
    items = [
        ("Phase 1 inventory", R / "eda/inventory/dataset_comparison.md"),
        ("ERA5 night-time product", config.PROCESSED_DIR / "era5_night_1deg.nc"),
        ("Phase 2 EDA", R / "eda/summary.md"),
        ("Phase 3 harmonised dataset", config.PROCESSED_DIR / f"harmonised_{config.load()['grid']['res_deg']}deg_monthly.parquet"),
        ("Phase 3 decisions", R / "preprocessing/decisions.md"),
        ("CV folds", R / "model/folds.csv"),
        ("Benchmark: VFE vs exact GP (3,000-row subset)", R / "model/benchmark/vfe_vs_exact_exponential.csv"),
        ("Benchmark: M sensitivity at full scale", R / "model/benchmark/m_sensitivity_exponential.csv"),
    ]
    for k in KERNELS:
        for t in ("spatial", "temporal", "joint"):
            p = R / f"model/{k}/oos_predictions_{t}.csv"
            if not p.exists():
                p = R / f"model/{k}/oos_predictions_{t}_part1.csv"
            items.append((f"CV {k} {t}", p))
        items.append((f"Full fit {k}", R / f"model/{k}/full_model.pt"))
    items += [("Kernel comparison", R / "model/kernel_comparison.csv")]
    comp = read(R / "model/kernel_comparison.csv")
    if comp is not None:
        w = comp.kernel.iloc[0]
        items += [(f"Monthly maps ({w})", R / f"model/{w}/global_mean_monthly.csv"),
                  (f"Whole-grid predictions ({w})", R / f"model/{w}/grid_predictions.csv")]
    for m in LINEAR + ((f"{w}_noera5",) if comp is not None else ()):
        items.append((f"Baseline {m} (CV metrics)", R / f"model/{m}/metrics_summary.csv"))
    items.append(("Model comparison (kernels and baselines)", R / "model/model_comparison.csv"))
    rows = []
    for name, p in items:
        ok = p.exists()
        when = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if ok else ""
        rows.append({"stage": name, "status": "executed" if ok else "NOT RUN", "output": str(p.relative_to(config.ROOT)),
                     "written": when})
    return pd.DataFrame(rows)


def contrast(pw, cv_type, a, b):
    """(rmse(a) - rmse(b), ci lo, ci hi, folds, distinguishable) from model_pairwise.csv, which stores each pair
    once in either order; None when the pair is missing."""
    for x, y, sign in ((a, b, 1), (b, a, -1)):
        r = pw[(pw.cv_type == cv_type) & (pw.model == x) & (pw.vs == y)]
        if len(r):
            r = r.iloc[0]
            lo, hi = sorted((sign * r.ci95_lo, sign * r.ci95_hi))
            return sign * r.mean_rmse_diff, lo, hi, int(r.folds), r.distinguishable
    return None


def baseline_section(winner):
    """2 x 2 ablation of the winning GP: without its kernel, without ERA5, without both (same folds)."""
    noera5 = f"{winner}_noera5"
    lines = ["### What the kernel and ERA5 each add (baselines)",
             "The winner is compared on the same folds with three baselines: the same model without the GP kernel "
             "(regression only, β₀ + β₁·ERA5 + β₂·post-gap with the same noise model), without ERA5 (the GP with "
             "mean β₀ + β₂·post-gap) and without both (a constant per period, β₀ + β₂·post-gap). Values are the "
             "primary score (W m⁻² sr⁻¹, lower is better).", ""]
    mc = read(R / "model/model_comparison.csv")
    if mc is None or not mc.model.isin(LINEAR).any():
        return lines + ["**Not available: the baselines have not been run yet** (`run_cv.py --model linear "
                        "linear_noera5`; `euler/submit_baselines.sh`).", ""]
    mc = mc.set_index("model")
    score = lambda m: mc.primary_score.get(m, np.nan)
    grid = pd.DataFrame({"with ERA5": [score(winner), score("linear")],
                         "without ERA5": [score(noera5), score("linear_noera5")]},
                        index=pd.Index([f"GP ({KERNEL_LABELS[winner]})", "regression only (no kernel)"], name="model"))
    cols = ["primary_score", "mean_fold_rmse_spatial", "mean_fold_rmse_temporal", "mean_fold_rmse_joint",
            "mean_fold_r2", "mean_fold_cover95", "mean_fold_crps"]
    t = mc.loc[[m for m in (winner, noera5) + LINEAR if m in mc.index], cols].reset_index()
    t["model"] = t.model.map(model_label)
    lines += [md(grid, "{:.2f}", index=True), "", md(t, "{:.3f}"), ""]
    if noera5 not in mc.index:
        lines += [f"The GP without ERA5 ({noera5}) has not been run yet: set `[baselines] gp_noera5_kernels = "
                  f"[\"{winner}\"]` and run `euler/submit_baselines.sh`.", ""]
    pw = read(R / "model/model_pairwise.csv")
    rows = []
    for name, a, b in (("the kernel, with ERA5", "linear", winner), ("the kernel, without ERA5", "linear_noera5", noera5),
                       ("ERA5, with the kernel", noera5, winner), ("ERA5, without the kernel", "linear_noera5", "linear")):
        for cv_type in CV_TYPES:
            c = contrast(pw, cv_type, a, b) if pw is not None else None
            if c is not None:
                rows.append({"removed": name, "validation": cv_type, "RMSE increase": c[0], "ci95_lo": c[1],
                             "ci95_hi": c[2], "folds": c[3], "distinguishable": c[4]})
    if rows:
        lines += ["Effect of removing one component: mean per-fold RMSE of the reduced model minus that of the fuller "
                  "model (positive = the component helps), with a paired block-bootstrap 95 % interval over folds:", "",
                  md(pd.DataFrame(rows), "{:.2f}"), ""]
    fig = R / "model/figures/model_comparison.png"
    return lines + [f"Figure: [{rel(fig)}]({rel(fig)}).", ""]


def model_section(cfg):
    """The statistical model of every fit as equations: the regression (mean function), the GP on its residual,
    prediction, estimation and the baselines. M and the starts come from the configuration the fits used."""
    fitted = [json.loads(p.read_text())["gp"] for k in KERNELS if (p := R / f"model/{k}/config.json").exists()]
    g = fitted[0] if fitted else cfg["gp"]
    res = cfg["grid"]["res_deg"]
    radii = ", ".join(f"{r:g}" for r in cfg["era5"]["smoothing_radii_km"])
    starts = ", ".join(f"×{f:g}" for f in (1, 0.5, 2)[:g["restarts"]])
    hp = [h for k in KERNELS if (h := read(R / f"model/{k}/hyperparameters_full.csv")) is not None]
    n_full = f"{int(hp[0].n_train.iloc[0]):,}" if hp else "all"
    return [
        "## Statistical model",
        "Every kernel and every CV fold uses the same model; only the correlation function κ changes. Index $i$ is "
        f"a cell-month: one {res}° × {res}° cell in one calendar month.",
        "",
        "### Regression (mean function)",
        "",
        "```math",
        r"\mathrm{OLR}^{\mathrm{CLARA}}_i = \underbrace{\beta_0 + \beta_1\,\mathrm{OLR}^{\mathrm{ERA5}}_i"
        r" + \beta_2\,s_i}_{\text{regression}} + \underbrace{f(\mathbf{x}_i)}_{\text{GP}} + \varepsilon_i",
        "```",
        "",
        r"- $y_i \equiv \mathrm{OLR}^{\mathrm{CLARA}}_i$ is the mean of the quality-controlled CLARA samples in the "
        r"cell-month, a radiance in W m⁻² sr⁻¹. They come from $n_i$ distinct overpasses.",
        r"- $e_i \equiv \mathrm{OLR}^{\mathrm{ERA5}}_i$ is the night-time ERA5 OLR of the same cell and month "
        r"(a flux, W m⁻²), "
        rf"smoothed over spherical caps of radius $R$. $R \in \lbrace {radii} \rbrace$ km is chosen in each fold as the "
        "radius with the highest correlation with CLARA on the training rows.",
        rf"- $s_i = \mathbf{{1}}[t_i \ge \text{{{cfg['time']['gap_start']}}}]$ is the post-gap indicator. $\beta_2$ "
        r"is the calibration step across the data gap at fixed ERA5, $\beta_0$ the pre-gap intercept and "
        r"$\beta_1$ the change in CLARA (W m⁻² sr⁻¹) per W m⁻² of ERA5; it also converts ERA5's flux into "
        r"CLARA's radiance units (sr⁻¹). $\beta_0$, $\beta_2$ and every RMSE, SD and noise scale below are in "
        r"CLARA's units, W m⁻² sr⁻¹.",
        r"- In vector form the mean is $\mathbf{h}_i^\top\boldsymbol\beta$, with $\mathbf{h}_i = (1, e_i, s_i)^\top$ "
        r"and $\boldsymbol\beta = (\beta_0, \beta_1, \beta_2)^\top$.",
        "",
        "### Gaussian process on the residual",
        "The regression residual is split into a spatio-temporally correlated part $f$ and independent noise "
        r"$\varepsilon$, with $f$ independent of $\varepsilon$:",
        "",
        "```math",
        r"y_i - \mathbf{h}_i^\top\boldsymbol\beta = f(\mathbf{x}_i) + \varepsilon_i, \qquad "
        r"f \sim \mathcal{GP}\big(0,\ k_\theta(\mathbf{x}, \mathbf{x}')\big), \qquad "
        r"\varepsilon_i \overset{\text{ind}}{\sim} \mathcal{N}\!\left(0,\ \sigma_0^2 + \frac{\sigma_1^2}{n_i}\right)",
        "```",
        "",
        r"- **Noise.** $\sigma_1^2$ is the sampling variance of one overpass. Samples along one overpass are strongly "
        r"correlated, so the overpass is the independent unit and this part shrinks as $1/n_i$. $\sigma_0^2$ is "
        "noise that does not average out with more overpasses.",
        r"- **Joint likelihood.** Stacked over the $N$ training cell-months:",
        "",
        "```math",
        r"\mathbf{y} \sim \mathcal{N}\big(\mathbf{H}\boldsymbol\beta,\ \mathbf{K}_\theta + \mathbf{D}\big), \qquad "
        r"[\mathbf{K}_\theta]_{ij} = k_\theta(\mathbf{x}_i, \mathbf{x}_j), \qquad "
        r"\mathbf{D} = \mathrm{diag}\!\left(\sigma_0^2 + \sigma_1^2/n_i\right)",
        "```",
        "",
        r"  $\boldsymbol\beta$ and $\theta$ are estimated **jointly** from this likelihood. It is not a two-step fit "
        "(regression first, then a GP on its residuals): as in generalised least squares, the regression "
        "coefficients account for the correlation of the residuals.",
        "",
        "### Covariance function",
        r"Each cell-month enters through $\mathbf{x} = (\mathbf{u}, \mathbf{c}, t)$:",
        "",
        "```math",
        r"\mathbf{u} = (\cos\varphi\cos\lambda,\ \cos\varphi\sin\lambda,\ \sin\varphi), \qquad "
        r"\mathbf{c} = (\cos 2\pi\tau,\ \sin 2\pi\tau), \qquad "
        rf"t = \text{{years since {cfg['time']['origin']}}}",
        "```",
        "",
        r"Here $\varphi, \lambda$ are the latitude and longitude of the cell centre, and $\tau \in [0, 1)$ is the "
        r"fraction of the year at mid-month. $\lVert\mathbf{u} - \mathbf{u}'\rVert = 2\sin(d / 2R_\oplus)$ is the "
        r"chord between two cells $d$ km apart on the sphere (Earth radius $R_\oplus$ = 6371 km), so the poles and "
        "the dateline need no special case. "
        r"On the seasonal circle $\mathbf{c}$, December is next to January. The covariance has two additive "
        "components:",
        "",
        "```math",
        r"k_\theta(\mathbf{x}, \mathbf{x}') = \sigma_A^2\,\kappa(r_A) + \sigma_B^2\,\kappa(r_B)",
        "```",
        "",
        "```math",
        r"r_A^2 = \frac{\lVert\mathbf{u} - \mathbf{u}'\rVert^2}{\ell_{\text{space},A}^2} "
        r"+ \frac{\lVert\mathbf{c} - \mathbf{c}'\rVert^2}{\ell_{\text{season}}^2}, \qquad "
        r"r_B^2 = \frac{\lVert\mathbf{u} - \mathbf{u}'\rVert^2}{\ell_{\text{space},B}^2} "
        r"+ \frac{(t - t')^2}{\ell_{\text{time}}^2}",
        "```",
        "",
        "- **Component A (space × season)** is a CLARA − ERA5 discrepancy pattern that recurs every year.",
        "- **Component B (space × long-term time)** holds changes that do not recur: drift beyond the step, and "
        "any regional structure of the step.",
        r"- $\theta = (\sigma_A, \ell_{\text{space},A}, \ell_{\text{season}}, \sigma_B, \ell_{\text{space},B}, "
        r"\ell_{\text{time}}, \sigma_0, \sigma_1)$.",
        "",
        "The three kernels differ only in κ:",
        "",
        "| kernel | κ(r) | sample paths of f |",
        "|---|---|---|",
        r"| Exponential | $e^{-r}$ | continuous, not differentiable |",
        r"| Gaussian (SE) | $e^{-r^2/2}$ | infinitely differentiable |",
        r"| Matérn 3/2 | $(1 + \sqrt{3}r)e^{-\sqrt{3}r}$ | once differentiable |",
        "",
        "All three are valid covariances here because $r_A$ and $r_B$ are scaled Euclidean distances (chords). With "
        "great-circle distance, the Gaussian and Matérn 3/2 kernels would not be positive definite on the sphere.",
        "",
        "### Prediction",
        "At a new cell-month (marked ∗), the target is its monthly-mean CLARA OLR, with the covariances to the "
        "training cell-months collected in a vector:",
        "",
        "```math",
        r"\mu_{\ast} = \mathbf{h}_{\ast}^\top\boldsymbol\beta + f(\mathbf{x}_{\ast}), \qquad "
        r"\mathbf{h}_{\ast} = (1, e_{\ast}, s_{\ast})^\top, \qquad "
        r"[\mathbf{k}_{\ast}]_i = k_\theta(\mathbf{x}_i, \mathbf{x}_{\ast})",
        "```",
        "",
        "For an exact GP its posterior is",
        "",
        "```math",
        r"\mathbb{E}[\mu_{\ast} \mid \mathbf{y}] = \mathbf{h}_{\ast}^\top\hat{\boldsymbol\beta} "
        r"+ \mathbf{k}_{\ast}^\top(\mathbf{K}_\theta + \mathbf{D})^{-1}(\mathbf{y} - \mathbf{H}\hat{\boldsymbol\beta}), "
        r"\qquad \mathrm{Var}[\mu_{\ast} \mid \mathbf{y}] = k_\theta(\mathbf{x}_{\ast}, \mathbf{x}_{\ast}) "
        r"- \mathbf{k}_{\ast}^\top(\mathbf{K}_\theta + \mathbf{D})^{-1}\mathbf{k}_{\ast}",
        "```",
        "",
        "The fitted model uses the sparse version of the same formulas (next subsection):",
        "",
        "```math",
        r"\mathbb{E}[\mu_{\ast} \mid \mathbf{y}] = \mathbf{h}_{\ast}^\top\hat{\boldsymbol\beta} "
        r"+ \mathbf{k}_{\ast u}\boldsymbol\Sigma\mathbf{K}_{uf}\mathbf{D}^{-1}(\mathbf{y} - \mathbf{H}\hat{\boldsymbol\beta}), "
        r"\qquad \mathrm{Var}[\mu_{\ast} \mid \mathbf{y}] = k_{\ast\ast} - \mathbf{k}_{\ast u}\mathbf{K}_{uu}^{-1}\mathbf{k}_{u\ast} "
        r"+ \mathbf{k}_{\ast u}\boldsymbol\Sigma\mathbf{k}_{u\ast}, \qquad "
        r"\boldsymbol\Sigma = \big(\mathbf{K}_{uu} + \mathbf{K}_{uf}\mathbf{D}^{-1}\mathbf{K}_{fu}\big)^{-1}",
        "```",
        "",
        r"- **Maps.** The uncertainty maps show the latent SD, $\sqrt{\mathrm{Var}[\mu_{\ast} \mid \mathbf{y}]}$.",
        r"- **Scoring.** Against a held-out CLARA cell-month from $n_{\ast}$ overpasses (interval coverage, CRPS, NLPD), "
        r"the noise is added: $\mathrm{Var}[y_{\ast} \mid \mathbf{y}] = \mathrm{Var}[\mu_{\ast} \mid \mathbf{y}] "
        r"+ \sigma_0^2 + \sigma_1^2 / n_{\ast}$.",
        r"- **Plug-in.** $\hat{\boldsymbol\beta}$ and $\hat\theta$ are plugged in; their uncertainty is not "
        "propagated.",
        "",
        "### Estimation: sparse variational GP",
        r"An exact GP costs $O(N^3)$ time and $O(N^2)$ memory, and $N$ reaches " + n_full + " training "
        "cell-months in the full fit. The fit therefore uses the variational approximation of Titsias (2009) with "
        rf"$M = {g['n_inducing']}$ inducing inputs $\mathbf{{Z}}$. These "
        "are k-means centres of the training inputs, fixed during the fit and shared by the three kernels within "
        r"a fold. $\boldsymbol\beta$ and $\theta$ maximise a lower bound on the log marginal likelihood, extended "
        r"here to the heteroscedastic noise $\mathbf{D}$:",
        "",
        "```math",
        r"\log p(\mathbf{y} \mid \boldsymbol\beta, \theta) \;\ge\; \mathcal{F}(\boldsymbol\beta, \theta) = "
        r"\log \mathcal{N}\big(\mathbf{y} \mid \mathbf{H}\boldsymbol\beta,\ \mathbf{Q}_{ff} + \mathbf{D}\big) "
        r"- \tfrac{1}{2}\mathrm{tr}\big(\mathbf{D}^{-1}(\mathbf{K}_{ff} - \mathbf{Q}_{ff})\big), \qquad "
        r"\mathbf{Q}_{ff} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}",
        "```",
        "",
        "- **Optimiser.** L-BFGS (type-II maximum likelihood), started from fixed, data-independent values. "
        f"The starts scale the initial length scales by {starts}, and the best bound wins.",
        r"- **Exactness check.** With $\mathbf{Z}$ equal to the training inputs, $\mathcal{F}$ equals the exact log "
        "marginal likelihood (unit test).",
        r"- **Standardisation.** $y$ and $e$ are standardised with training statistics inside each fit. All "
        "reported values are in physical units.",
        "",
        "### Baselines (2 × 2 ablation)",
        "Same noise model, same step and same folds:",
        "",
        "| model | mean | covariance of y |",
        "|---|---|---|",
        r"| GP + ERA5 (the three kernels) | $\beta_0 + \beta_1 e + \beta_2 s$ | $\mathbf{K}_\theta + \mathbf{D}$ |",
        r"| GP, no ERA5 | $\beta_0 + \beta_2 s$ | $\mathbf{K}_\theta + \mathbf{D}$ |",
        r"| regression on ERA5 (no GP) | $\beta_0 + \beta_1 e + \beta_2 s$ | $\mathbf{D}$ |",
        r"| constant per period (no GP, no ERA5) | $\beta_0 + \beta_2 s$ | $\mathbf{D}$ |",
        "",
        r"Without the GP ($f \equiv 0$) the likelihood is exact. For fixed $(\sigma_0, \sigma_1)$, $\boldsymbol\beta$ "
        r"is the GLS estimate (left), and $(\sigma_0, \sigma_1)$ maximise the profile likelihood. A prediction's "
        "latent variance is on the right:",
        "",
        "```math",
        r"\hat{\boldsymbol\beta} = (\mathbf{H}^\top\mathbf{D}^{-1}\mathbf{H})^{-1}\mathbf{H}^\top\mathbf{D}^{-1}\mathbf{y}, "
        r"\qquad \mathrm{Var}[\mu_{\ast} \mid \mathbf{y}] = "
        r"\mathbf{h}_{\ast}^\top(\mathbf{H}^\top\mathbf{D}^{-1}\mathbf{H})^{-1}\mathbf{h}_{\ast}",
        "```",
        "",
        r"Everything learned from data is learned per CV fold, on its training rows only: $R$, the standardisation, "
        r"$\mathbf{Z}$, $\boldsymbol\beta$ and $\theta$.",
        "",
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    eda = (R / "eda/summary.md").read_text() if (R / "eda/summary.md").exists() else ""
    comp = read(R / "model/kernel_comparison.csv")
    pw = read(R / "model/kernel_pairwise.csv")
    qc = read(R / "preprocessing/qc_counts.csv")
    lines = ["# CLARA OLR from ERA5 and coordinates: spatio-temporal GP kernels compared",
             "",
             f"Generated by `src/report.py` on {dt.datetime.now():%Y-%m-%d %H:%M} from the outputs listed in the "
             f"status table at the end. Grid {cfg['grid']['res_deg']}° × month; target: monthly-mean CLARA "
             "night-side OLR per cell. CLARA's OLR is a radiance (W m⁻² sr⁻¹); ERA5 and CERES are fluxes (W m⁻²), so CLARA is "
             "compared with them only through correlations, ratios and relative values.",
             ""]

    # Q1
    lines += ["## 1. How do CLARA and ERA5 differ?",
              f"Structural and physical comparison: [{rel(R / 'eda/inventory/dataset_comparison.md')}]"
              f"({rel(R / 'eda/inventory/dataset_comparison.md')}). In short:",
              "- **Structure.** CLARA is an instantaneous 30 s radiometer series along one satellite's track, located "
              "by its boresight footprint. ERA5 is a complete 0.25° reanalysis grid of monthly means.",
              "- **Physics.** CLARA measures night-side OLR as a radiance (W m⁻² sr⁻¹). It is not angle-corrected and is "
              "contaminated in daylight. "
              "ERA5 `ttr` is a 24 h accumulation, negative upward; we use its night-time monthly mean in W m⁻².",
              "- **Statistics** (EDA, sampling-matched cell-months):"]
    for line in eda.splitlines():
        if line.startswith(("- **Pooled", "- **By period", "- **Reference check", "- **Spatial structure is")):
            lines.append("  " + line)
    lines += [f"- Figures: [{rel(R / 'eda/physics_sza.png')}]({rel(R / 'eda/physics_sza.png')}), "
              f"[{rel(R / 'eda/physics_vza.png')}]({rel(R / 'eda/physics_vza.png')}), "
              f"[{rel(R / 'eda/temporal_global.png')}]({rel(R / 'eda/temporal_global.png')}), "
              f"[{rel(R / 'eda/spatial_zonal.png')}]({rel(R / 'eda/spatial_zonal.png')}).", ""]

    # Q2
    lines += ["## 2. What preprocessing made them comparable?",
              f"All decisions with reasons: [{rel(R / 'preprocessing/decisions.md')}]({rel(R / 'preprocessing/decisions.md')}).",
              "- **Units and time.** CLARA used as stored (radiance, W m⁻² sr⁻¹). ERA5 converted to a flux in W m⁻², positive "
              "upward, from the hour-of-day product, validated against the monthly product (≤ 0.01 W m⁻²). The "
              "regression slope β₁ carries the conversion between the two. Both datasets aligned on calendar months.",
              "- **QC.** Night-side footprints (SZA > 100°) and VZA ≤ 60° only.",
              f"- **Aggregation.** CLARA binned into {cfg['grid']['res_deg']}° cell-months with overpass counts; never "
              "interpolated.",
              "- **Prediction grid.** Every lat-lon cell of every month. Polar-day cells (no night) use the "
              "darkest-hours ERA5 fallback, are flagged `night_fallback` and are extrapolation (CLARA never observes "
              "them).",
              "- **Pairing.** ERA5 reduced to night-time hours, area-averaged to the same cells, optionally "
              "cap-smoothed to CLARA's effective footprint (radius chosen per CV fold).",
              "- **Missingness.** Kept explicit: an observation table plus a prediction grid with a `clara_observed` flag.",
              "- **Standardisation.** Only inside each CV fold, with training rows."]
    if qc is not None:
        lines += ["", md(qc, "{:.3f}")]
    lines.append("")

    # Q3
    lines += ["## 3. Spatial and temporal structure in CLARA OLR"]
    for line in eda.splitlines():
        if line.startswith(("- **Along track", "- **Spatial residual", "- **Temporal residual", "- **Day to day")):
            lines.append(line)
    lines += [f"- Seasonal cycles by band and period: [{rel(R / 'eda/temporal_seasonal.png')}]({rel(R / 'eda/temporal_seasonal.png')}); "
              f"band series: [{rel(R / 'eda/temporal_bands.png')}]({rel(R / 'eda/temporal_bands.png')}).", ""]

    lines += model_section(cfg)

    if comp is None:
        lines += ["## 4–8. Model results", "", "**Not available: the cross-validation has not been run yet.**", ""]
    else:
        winner = comp.kernel.iloc[0]
        summ = {k: read(R / f"model/{k}/metrics_summary.csv") for k in KERNELS}
        hp_full = {k: read(R / f"model/{k}/hyperparameters_full.csv") for k in KERNELS}
        # Q4
        lines += ["## 4. How much of CLARA's variability do the GPs explain?",
                  "Out-of-sample R² is the share of the held-out cell-months' variance explained (pooled over the "
                  "held-out blocks of each validation type). `R² mean function` is the same fitted GP's mean function "
                  "alone (β₀ + β₁·ERA5 + β₂·post-gap), so the gap between the two is what the spatio-temporal "
                  "covariance adds.", ""]
        rows = []
        for k in KERNELS:
            s = summ[k]
            if s is None:
                continue
            for _, r in s.iterrows():
                rows.append({"kernel": KERNEL_LABELS[k], "validation": r.cv_type, "n": int(r.n), "R² (GP)": r.r2,
                             "R² mean function": r.r2_mean_fn, "RMSE": r.rmse, "RMSE mean function": r.rmse_mean_fn})
            ins = read(R / f"model/{k}/insample_fitted.csv")
            if ins is None and (R / f"model/{k}/insample_fitted_part1.csv").exists():
                ins = pd.concat([pd.read_csv(p) for p in sorted((R / f"model/{k}").glob("insample_fitted_part*.csv"))])
            if ins is not None:
                err = ins.clara_obs - ins.pred_mean
                rows.append({"kernel": KERNEL_LABELS[k], "validation": "in-sample (full fit)", "n": len(ins),
                             "R² (GP)": 1 - (err ** 2).sum() / ((ins.clara_obs - ins.clara_obs.mean()) ** 2).sum(),
                             "R² mean function": 1 - ((ins.clara_obs - ins.mean_fn) ** 2).sum()
                             / ((ins.clara_obs - ins.clara_obs.mean()) ** 2).sum(),
                             "RMSE": np.sqrt((err ** 2).mean()),
                             "RMSE mean function": np.sqrt(((ins.clara_obs - ins.mean_fn) ** 2).mean())})
        lines += [md(pd.DataFrame(rows), "{:.3f}"), ""]

        # Q5
        lines += ["## 5. Which kernel has the best out-of-sample RMSE?",
                  "Winner rule (fixed before running): the lowest mean, over the three validation types, of the mean "
                  "per-fold RMSE. Every kernel was scored on identical folds (SHA-256 checked).", ""]
        cols = ["kernel", "primary_score", "mean_fold_rmse_spatial", "mean_fold_rmse_temporal", "mean_fold_rmse_joint",
                "mean_fold_rmse_area", "mean_fold_mae", "mean_fold_bias", "mean_fold_cover95", "mean_fold_nlpd",
                "mean_fold_crps"]
        t = comp[cols].copy()
        t["kernel"] = t.kernel.map(KERNEL_LABELS)
        lines += [md(t, "{:.3f}"), "",
                  f"**Winner: {KERNEL_LABELS[winner]}.** Paired block-bootstrap differences in per-fold RMSE against "
                  "the winner (positive = worse than the winner):", ""]
        if pw is not None:
            p2 = pw.copy()
            p2["kernel"] = p2.kernel.map(KERNEL_LABELS)
            lines += [md(p2[["cv_type", "kernel", "mean_rmse_diff", "ci95_lo", "ci95_hi", "folds", "distinguishable"]],
                         "{:.3f}"), ""]
            nd = p2[p2.distinguishable == False]  # noqa: E712
            if len(nd):
                lines += [f"Where `distinguishable` is False, the 95 % interval includes zero: across the held-out "
                          f"blocks, {', '.join(sorted(set(nd.kernel)))} cannot be separated from the winner in that "
                          "validation type, so the ranking there is not decisive.", ""]
        lines += [f"Figures: [{rel(R / 'model/figures/rmse_by_fold.png')}]({rel(R / 'model/figures/rmse_by_fold.png')}) "
                  f"(variability across blocks), [{rel(R / 'model/figures/coverage.png')}]"
                  f"({rel(R / 'model/figures/coverage.png')}), [{rel(R / 'model/figures/pit.png')}]"
                  f"({rel(R / 'model/figures/pit.png')}).", ""]
        lines += baseline_section(winner)

        # hyperparameters
        lines += ["### Fitted hyperparameters (full-data fits, physical units)", ""]
        hrows = []
        for k in KERNELS:
            h = hp_full[k]
            if h is None:
                continue
            h = h.iloc[0]
            ins = read(R / f"model/{k}/insample_fitted.csv", usecols=["clara_obs", "era5_night"])
            hrows.append({"kernel": KERNEL_LABELS[k], "β₀": h.beta0_wm2, "β₁ (sr⁻¹)": h.beta1,
                          "β₁ relative": h.beta1 * ins.era5_night.mean() / ins.clara_obs.mean() if ins is not None else np.nan,
                          "β₂ post-gap": h.get("beta2_wm2", np.nan),
                          "σ_A": h.sigma_A_wm2, "ℓ_space,A (km)": h.ell_space_A_km, "ℓ_season (days)": h.ell_season_days,
                          "σ_B": h.sigma_B_wm2, "ℓ_space,B (km)": h.ell_space_B_km, "ℓ_time (years)": h.ell_time_years,
                          "σ₀": h.sigma0_wm2, "σ₁ (per overpass)": h.sigma1_wm2, "ERA5 R (km)": h.era5_radius_km})
        lines += [md(pd.DataFrame(hrows), "{:.2f}"), "",
                  "How to read them:",
                  "β₀, β₂, the σ and the noise scales are in CLARA's units (W m⁻² sr⁻¹).",
                  "- **β₁** is how many W m⁻² sr⁻¹ CLARA changes per W m⁻² of night-time ERA5. It includes the conversion "
                  "from flux to radiance, so only its relative form is comparable across units: **β₁ relative** = β₁ × "
                  "mean ERA5 / mean CLARA over the observed cell-months, the % change of CLARA per 1 % change of ERA5. "
                  "Values well below 1 match the damped relative contrasts seen in the EDA.",
                  "- **β₂** is the calibration step: the change in CLARA's level from the start of the 2021-09 → "
                  "2022-05 gap on, at fixed ERA5. β₀ is the pre-gap intercept and β₀ + β₂ the post-gap one.",
                  "- **Component A** (space × season) is the recurring, seasonal CLARA–ERA5 discrepancy. ℓ_space,A is "
                  "its horizontal scale; ℓ_season is how quickly the pattern changes through the year.",
                  "- **Component B** (space × long-term time) absorbs drift beyond the step, and any regional "
                  "structure of the step. A short ℓ_time and a large σ_B mean the model needs non-recurring temporal "
                  "flexibility.",
                  "- **σ₀ and σ₁** are the noise: σ₀ is irreducible discrepancy (residual geometry effects, weather the "
                  "monthly ERA5 cannot see); σ₁/√n is the sampling error of a cell-month mean from n overpasses.",
                  f"- Per-fold estimates (spread across blocks): `results/model/<kernel>/hyperparameters_<cv_type>.csv`.",
                  ""]

        # Q6 / Q7
        bd = read(R / f"model/{winner}/metrics_breakdown.csv")
        lines += ["## 6. Where and when does the model perform poorly?"]
        if bd is not None:
            for grp in ("band", "season", "period", "year"):
                g = bd[bd.group == grp]
                piv = g.pivot(index="group_value", columns="cv_type", values="rmse")
                lines += ["", f"RMSE (W m⁻² sr⁻¹) of {KERNEL_LABELS[winner]} by {grp} and validation type:", "",
                          md(piv, "{:.1f}", index=True)]
            sp = bd[(bd.group == "band") & (bd.cv_type == "spatial")].sort_values("rmse")
            lines += ["", f"The worst latitude band in spatial CV is {sp.group_value.iloc[-1]} (RMSE {sp.rmse.iloc[-1]:.1f}); "
                          f"the best is {sp.group_value.iloc[0]} (RMSE {sp.rmse.iloc[0]:.1f}). Period and year rows show "
                          "how errors concentrate around the calibration step and in sparse months.", ""]
        lines += ["## 7. Are the predictive uncertainties calibrated?",
                  "Coverage of central intervals for a new noisy cell-month mean (latent variance + noise for its "
                  "number of overpasses), pooled over held-out blocks:", ""]
        crow = []
        for k in KERNELS:
            s = summ[k]
            if s is None:
                continue
            for _, r in s.iterrows():
                crow.append({"kernel": KERNEL_LABELS[k], "validation": r.cv_type, "50 %": r.cover50, "80 %": r.cover80,
                             "95 %": r.cover95, "mean 95 % width (W m⁻² sr⁻¹)": r.width95, "NLPD": r.nlpd, "CRPS": r.crps})
        lines += [md(pd.DataFrame(crow), "{:.3f}"), "",
                  "Coverage close to nominal with a flat PIT histogram means calibrated intervals. Coverage below "
                  "nominal means the intervals are too narrow (overconfident); above nominal, too wide.", ""]

        # Q8
        lines += ["## 8. Where and when is the prediction uncertainty largest?"]
        files = sorted((R / f"model/{winner}").glob("*/[0-9][0-9][0-9][0-9]_[0-9][0-9]_predictions.csv"))
        if files:
            g = pd.concat([pd.read_csv(f, usecols=["year", "month", "lat", "latent_sd", "clara_observed",
                                                   "night_fallback"]) for f in files])
            g = g[~g.night_fallback.astype(bool)]  # polar-day fallback cells are extrapolation; summarised separately
            g["band"] = pd.cut(g.lat, [-90, -60, -30, 0, 30, 60, 90.01], right=False,
                               labels=["90S-60S", "60S-30S", "30S-EQ", "EQ-30N", "30N-60N", "60N-90N"])
            by_band = g.groupby("band", observed=True).latent_sd.median().rename("median latent SD (W m⁻² sr⁻¹)")
            by_month = g.groupby(["year", "month"]).latent_sd.median()
            top = by_month.sort_values(ascending=False).head(8).reset_index()
            lines += ["Median posterior SD of the monthly-mean CLARA OLR (latent; observation noise excluded):", "",
                      md(by_band.to_frame(), "{:.2f}", index=True), "",
                      "Months with the largest median latent SD (typically CLARA gaps and sparse months, where the "
                      "prediction relies on ERA5 and extrapolation in time):", "", md(top, "{:.2f}"), "",
                      f"Monthly maps: `results/model/{winner}/<yyyy>/`; every cell of every month in one table: "
                      f"`results/model/{winner}/grid_predictions.csv`; global mean with full-covariance uncertainty: "
                      f"[{rel(R / f'model/{winner}/global_mean_monthly.png')}]({rel(R / f'model/{winner}/global_mean_monthly.png')}).",
                      ""]
        else:
            lines += ["", "**Not available: the monthly predictions (Phase 6) have not been run yet.**", ""]

    # Q9
    fitted = [json.loads(p.read_text())["gp"] for k in KERNELS if (p := R / f"model/{k}/config.json").exists()]
    m_fit = sorted({g["n_inducing"] for g in fitted}) or [cfg["gp"]["n_inducing"]]
    summ = read(R / "model/fold_summary.csv")
    n_fold = summ.train.median() if summ is not None else np.nan
    ve = [v for k in KERNELS if (v := read(R / f"model/benchmark/vfe_vs_exact_{k}.csv")) is not None]
    if ve:
        ve = pd.concat(ve)
        exact_note = (f"Against the exact GP on a random {int(ve.n_train.iloc[0]):,}-cell-month subset "
                      f"(M = {int(ve.n_inducing.iloc[0])}, same hyperparameters), held-out RMSE differs by at most "
                      f"{(ve.rmse_vfe - ve.rmse_exact).abs().max():.3f} W m⁻² sr⁻¹ and the predictive SD by at most "
                      f"{(ve.median_sd_ratio_vfe_over_exact - 1).abs().max():.1%} "
                      "(`results/model/benchmark/vfe_vs_exact_*.csv`). That test is easy for the approximation "
                      "(one inducing input per 3 observations); ")
    else:
        exact_note = "The approximation was not compared with the exact GP; "
    ms = [v for k in KERNELS if (v := read(R / f"model/benchmark/m_sensitivity_{k}.csv")) is not None]
    m_table = []
    if ms:
        ms = pd.concat(ms)
        rmse = ms.pivot(index="M", columns="kernel", values="rmse")
        bpo = ms.pivot(index="M", columns="kernel", values="bound_per_obs")
        spread = rmse.max() - rmse.min()
        orders = {m: tuple(r.sort_values().index) for m, r in rmse.iterrows()}
        label = lambda o: ", ".join(KERNEL_LABELS[k] for k in o)
        ranking = (f"the kernel ranking is the same at every M (lowest RMSE first: {label(orders[rmse.index[0]])})"
                   if len(set(orders.values())) == 1 else
                   "the kernel ranking changes with M (" + "; ".join(f"M = {m}: {label(o)}" for m, o in orders.items()) + ")")
        m_ref = max(m_fit) if max(m_fit) in bpo.index else bpo.index[0]
        f0 = ms.iloc[0]
        m_note = (f"At that scale, on spatial fold {int(f0.fold)} ({int(f0.n_train):,} training and {int(f0.n_test):,} "
                  f"test cell-months), M = {', '.join(map(str, rmse.index))} changes the out-of-sample RMSE by at most "
                  + ", ".join(f"{spread[k]:.2f} ({KERNEL_LABELS[k]})" for k in rmse.columns) + " W m⁻² sr⁻¹, and "
                  + ranking + ". The bound per observation still rises with M (by up to "
                  f"{(bpo.iloc[-1] - bpo.loc[m_ref]).max():.3f} from M = {m_ref} to {rmse.index.max()}): the likelihood "
                  "approximation has not converged, but the predictions are stable (table below; "
                  "`results/model/benchmark/m_sensitivity_*.csv`). Kernel differences smaller than a kernel's own "
                  "change across M are not decisive.")
        t = pd.concat([rmse.rename(columns=lambda k: f"RMSE {KERNEL_LABELS[k]}"),
                       bpo.rename(columns=lambda k: f"bound/obs {KERNEL_LABELS[k]}")], axis=1).reset_index()
        t["M"] = t.M.astype(int).astype(str)  # md() would print the integer M as a float
        restarts = (fitted[0] if fitted else cfg["gp"])["restarts"]
        m_table = [f"Sensitivity to M on spatial fold {int(f0.fold)} (M = {m_ref} with the {restarts} starts of "
                   "the CV fits, the other M with one start):", "", md(t, "{:.3f}"), ""]
    else:
        m_note = ("The effect of M at that scale was not measured. Kernel differences of a few tenths of a W m⁻² sr⁻¹ may "
                  "therefore partly reflect the approximation.")
    lines += ["## 9. Limitations",
              "- **Calibration step.** The calibration change inside the 2021-09 → 2022-05 gap (CLARA / ERA5 rises by "
              "~23 %) enters the "
              "mean function as one global offset β₂ from 2021-09 on. When in the gap it happened is unknown: by "
              "convention the gap months take the post-gap level, so the maps of those months (never observed) are "
              "extrapolation, and their uncertainty does not include the timing of the step. A step that differs "
              "between regions is left to the long-term GP component.",
              "- **Geometry.** CLARA is not angle-corrected. Within VZA ≤ 60° the residual viewing-angle effect still "
              "changes CLARA / ERA5 by 11 % (post-gap) to 31 % (pre-gap) and is treated as noise because VZA is not "
              "an allowed predictor. The drift of altitude and local time is also absorbed by the time components.",
              "- **Smoothing and scale.** CLARA's footprint is wide (field of view not documented). ERA5 is matched by "
              "spherical-cap smoothing (~10³ km optimum), but CLARA's damped spatial contrasts mean the map's "
              "fine-scale structure is mostly ERA5's, scaled by β₁. It is not information CLARA resolves.",
              "- **Missing observations.** Cell-months are observed only along night-side tracks; polar-day cells "
              "have no night and are predicted from the darkest-hours ERA5 fallback (extrapolation, hatched). "
              "Post-gap cell-months typically rest on a single overpass, so they are much noisier than pre-gap "
              "ones.",
              "- **Temporal meaning.** Monthly means of sparse instantaneous samples are compared with complete "
              "monthly means: day-to-day weather (~34 W m⁻² sr⁻¹ within-cell-month SD of daily means, ~19 % of the mean "
              "CLARA value) is absorbed as noise.",
              f"- **Approximation.** The sparse (VFE) GP uses M = {', '.join(map(str, m_fit))} fixed k-means inducing "
              "inputs, shared by all kernels in a fold. " + exact_note +
              f"in the CV folds there is about one per {n_fold / max(m_fit):,.0f}. " + m_note,
              "- **Mean-coefficient uncertainty.** β₀ and β₁ are point estimates (type-II maximum likelihood); their "
              "uncertainty is not propagated. With ~3×10⁴ training cell-months it is small relative to the other terms.",
              ""] + m_table
    lines += ["## Status of every stage", "", md(status_table()), ""]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    config.save(cfg, OUT / "config_used.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
