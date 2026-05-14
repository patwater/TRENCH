"""
regression.py

OLS regression models and a spatial-lag model for the SGV fragmentation panel.

Models estimated:
  M1  PCI     ~ fragmentation_idx  (bivariate baseline)
  M2  PCI     ~ fragmentation_idx + median_hh_income_z + poverty_rate_z
  M3  PCI     ~ fragmentation_idx + median_hh_income_z + poverty_rate_z
                + is_contract + area_sq_mi_z + log(avg_daily_traffic)
  M4  log(cost_per_lm) ~ fragmentation_idx + median_hh_income_z
                          + is_contract + area_sq_mi_z      (cost frontier)
  M5  PCI (spatial lag) ~ fragmentation_idx + median_hh_income_z
                          + W·PCI  (2SLS)

All OLS models use HC3 heteroskedasticity-robust standard errors.
Spatial lag estimated by 2SLS (instruments: W·X, W²·X).

Outputs → data/output/phase2/:
  ols_results.txt
  regression_coef_plot.png
  spatial_lag_results.txt
"""

import pathlib
import textwrap

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from esda import Moran

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "output" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _prep_covariates(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    df = gdf.copy()
    df["poverty_rate_z"]  = _zscore(df["poverty_rate"])
    df["area_sq_mi_z"]    = _zscore(df["area_sq_mi"])
    df["log_adt"]         = np.log(df["avg_daily_traffic"])
    df["log_adt_z"]       = _zscore(df["log_adt"])
    return df


def _fmt_results(res, title: str) -> str:
    """Format a statsmodels result object into a clean text block."""
    lines = [
        "=" * 72,
        title.center(72),
        "=" * 72,
        f"  Dep. variable : {res.model.endog_names}",
        f"  N             : {int(res.nobs)}",
        f"  R²            : {res.rsquared:.4f}",
        f"  Adj. R²       : {res.rsquared_adj:.4f}",
        f"  F-stat        : {res.fvalue:.3f}   p={res.f_pvalue:.4f}",
        "-" * 72,
        f"  {'Variable':<30} {'Coef':>10} {'Std Err':>10} {'t':>8} {'p':>8}",
        "-" * 72,
    ]
    for name, coef, se, t, p in zip(
        res.model.exog_names,
        res.params, res.bse, res.tvalues, res.pvalues,
    ):
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        lines.append(
            f"  {name:<30} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f} {sig}"
        )
    lines += [
        "-" * 72,
        "  Significance: *** p<0.01  ** p<0.05  * p<0.10",
        "  Standard errors: HC3 heteroskedasticity-robust",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OLS models
# ---------------------------------------------------------------------------

def run_ols_models(gdf: gpd.GeoDataFrame, w) -> dict:
    df = _prep_covariates(gdf)

    specs = {
        "M1_bivariate": (
            "pci_score ~ fragmentation_idx",
            "PCI ~ Fragmentation (bivariate baseline)",
        ),
        "M2_income_controlled": (
            "pci_score ~ fragmentation_idx + median_hh_income_z + poverty_rate_z",
            "PCI ~ Fragmentation + Income controls",
        ),
        "M3_full": (
            "pci_score ~ fragmentation_idx + median_hh_income_z + poverty_rate_z"
            " + is_contract + area_sq_mi_z + log_adt_z",
            "PCI ~ Full specification",
        ),
        "M4_cost": (
            "log_cost_per_lm ~ fragmentation_idx + median_hh_income_z"
            " + is_contract + area_sq_mi_z",
            "log(Cost/LM) ~ Fragmentation + controls",
        ),
    }

    results = {}
    all_text = []

    for key, (formula, title) in specs.items():
        res = smf.ols(formula, data=df).fit(cov_type="HC3")
        results[key] = res
        block = _fmt_results(res, title)
        all_text.append(block)

        # Residual Moran test
        mi = Moran(res.resid.values, w, permutations=999)
        all_text.append(
            f"  Moran's I on residuals: I={mi.I:.4f}  z={mi.z_norm:.3f}  p={mi.p_sim:.3f}"
            + ("  ← spatial autocorrelation in residuals" if mi.p_sim < 0.10 else "")
            + "\n"
        )

        print(f"  {key}: R²={res.rsquared:.3f}  frag coef={res.params['fragmentation_idx']:.3f}"
              f"  p={res.pvalues['fragmentation_idx']:.3f}")

    out_path = OUTPUT_DIR / "ols_results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(all_text))
    print(f"[regression] OLS results → {out_path}")

    _plot_coefficients(results)
    return results


def _plot_coefficients(results: dict) -> None:
    """Forest plot: fragmentation_idx coefficient across all OLS models."""
    model_labels = {
        "M1_bivariate": "M1: bivariate",
        "M2_income_controlled": "M2: + income",
        "M3_full": "M3: full spec",
        "M4_cost": "M4: log(cost)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Fragmentation coefficient across PCI models ----------------------
    ax = axes[0]
    pci_models = ["M1_bivariate", "M2_income_controlled", "M3_full"]
    ys = range(len(pci_models))
    for y, key in zip(ys, pci_models):
        res = results[key]
        coef  = res.params["fragmentation_idx"]
        ci_lo = res.conf_int(alpha=0.10).loc["fragmentation_idx", 0]
        ci_hi = res.conf_int(alpha=0.10).loc["fragmentation_idx", 1]
        color = "#d7191c" if res.pvalues["fragmentation_idx"] < 0.10 else "#aaaaaa"
        ax.errorbar(coef, y, xerr=[[coef - ci_lo], [ci_hi - coef]],
                    fmt="o", color=color, capsize=4, markersize=8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([model_labels[k] for k in pci_models])
    ax.set_xlabel("Coefficient on fragmentation_idx\n(outcome: PCI score)")
    ax.set_title("Fragmentation Effect on PCI\n(90% CI; red = p < 0.10)")

    # ---- Key coefficients in M3 full spec ---------------------------------
    ax2 = axes[1]
    m3  = results["M3_full"]
    vars_to_show = [
        ("fragmentation_idx", "Fragmentation index"),
        ("median_hh_income_z", "Median HH income (z)"),
        ("poverty_rate_z", "Poverty rate (z)"),
        ("is_contract", "Contract city"),
        ("area_sq_mi_z", "City area (z)"),
        ("log_adt_z", "log(ADT) (z)"),
    ]
    for i, (var, label) in enumerate(vars_to_show):
        if var not in m3.params:
            continue
        coef  = m3.params[var]
        ci    = m3.conf_int(alpha=0.10).loc[var]
        color = "#d7191c" if m3.pvalues[var] < 0.10 else "#aaaaaa"
        ax2.errorbar(coef, i, xerr=[[coef - ci[0]], [ci[1] - coef]],
                     fmt="o", color=color, capsize=4, markersize=8)
    ax2.axvline(0, color="black", lw=0.8, ls="--")
    ax2.set_yticks(range(len(vars_to_show)))
    ax2.set_yticklabels([lbl for _, lbl in vars_to_show])
    ax2.set_xlabel("Coefficient (outcome: PCI score)")
    ax2.set_title("M3 Full Specification Coefficients\n(90% CI; red = p < 0.10)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "regression_coef_plot.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Spatial Lag Model (2SLS)
# ---------------------------------------------------------------------------

def run_spatial_lag(gdf: gpd.GeoDataFrame, w) -> dict:
    """
    Estimate PCI = ρ·W·PCI + X·β + ε  via 2SLS.
    Instruments: W·X, W²·X for each covariate.
    """
    df = _prep_covariates(gdf)
    W  = w.sparse.toarray()

    y   = df["pci_score"].values
    X_vars = ["fragmentation_idx", "median_hh_income_z", "poverty_rate_z"]
    X   = sm.add_constant(df[X_vars].values)

    # Endogenous regressor: W*y
    Wy  = W @ y

    # Instruments: W*X, W²*X  (each column of X)
    W2 = W @ W
    instruments = np.hstack([X, W @ X[:, 1:], W2 @ X[:, 1:]])   # excl. constant

    # First stage: regress Wy on instruments
    fs_res = sm.OLS(Wy, instruments).fit()
    Wy_hat = fs_res.fittedvalues

    # Second stage: regress y on [X, Wy_hat]
    X_2sls = np.column_stack([X, Wy_hat])
    ss_res  = sm.OLS(y, X_2sls).fit(cov_type="HC3")

    # Rename params for clarity
    param_names = ["const"] + X_vars + ["rho_W_PCI"]
    params  = dict(zip(param_names, ss_res.params))
    pvals   = dict(zip(param_names, ss_res.pvalues))
    ses     = dict(zip(param_names, ss_res.bse))

    # Moran test on 2SLS residuals
    mi_resid = Moran(ss_res.resid, w, permutations=999)

    lines = [
        "=" * 72,
        "  Spatial Lag Model (2SLS)  — Outcome: PCI score".center(72),
        "=" * 72,
        f"  N            : {len(y)}",
        f"  R² (2nd stg) : {ss_res.rsquared:.4f}",
        "-" * 72,
        f"  {'Variable':<32} {'Coef':>10} {'Std Err':>10} {'p':>8}",
        "-" * 72,
    ]
    for name in param_names:
        sig = "***" if pvals[name] < 0.01 else "**" if pvals[name] < 0.05 else "*" if pvals[name] < 0.10 else ""
        lines.append(f"  {name:<32} {params[name]:>10.4f} {ses[name]:>10.4f} {pvals[name]:>8.4f} {sig}")
    lines += [
        "-" * 72,
        f"  Moran's I (residuals): I={mi_resid.I:.4f}  p={mi_resid.p_sim:.3f}",
        "  Instruments: W·X, W²·X  (each covariate)",
        "  Significance: *** p<0.01  ** p<0.05  * p<0.10",
        "",
        "  Interpretation of ρ (rho_W_PCI):",
        "    Positive ρ → road quality spills over jurisdictional borders",
        "    (neighbours' PCI predicts own PCI, controlling for fragmentation)",
        "",
    ]

    out_path = OUTPUT_DIR / "spatial_lag_results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("[regression] Spatial lag results →", out_path)

    rho = params["rho_W_PCI"]
    frag_coef = params["fragmentation_idx"]
    print(f"  ρ (spatial lag) = {rho:.4f}  p={pvals['rho_W_PCI']:.3f}")
    print(f"  frag coef (2SLS)= {frag_coef:.4f}  p={pvals['fragmentation_idx']:.3f}")

    return {"params": params, "pvalues": pvals, "ses": ses, "r2": ss_res.rsquared}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_regression(gdf: gpd.GeoDataFrame, w) -> dict:
    print("\n── OLS Regression Models ─────────────────────────────────────")
    ols = run_ols_models(gdf, w)
    print("\n── Spatial Lag Model (2SLS) ──────────────────────────────────")
    sl  = run_spatial_lag(gdf, w)
    return {"ols": ols, "spatial_lag": sl}
