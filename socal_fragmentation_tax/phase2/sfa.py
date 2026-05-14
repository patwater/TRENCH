"""
sfa.py

Stochastic Frontier Analysis (SFA) for road maintenance efficiency.

Model A — Cost Frontier (higher cost = more inefficient):
  ln(cost_i) = α + β·X_i + v_i + u_i
  v_i ~ N(0, σ²_v)    symmetric noise
  u_i ~ N⁺(0, σ²_u)   one-sided inefficiency (≥ 0 adds to cost)

  Technical Efficiency: TE_i = exp(−û_i)   ∈ (0, 1]
  TE = 1 → on the frontier (lowest achievable cost given X)
  TE < 1 → inefficient; (1 − TE) fraction of cost is avoidable

Covariates (X_i):
  fragmentation_idx, median_hh_income_z, is_contract, area_sq_mi_z

MLE via scipy.optimize.minimize (Nelder-Mead then L-BFGS-B).
JLMS (1982) point estimates of efficiency used for city rankings.

Outputs → data/output/phase2/:
  sfa_results.txt
  sfa_efficiency_scores.csv
  sfa_efficiency_plot.png
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import minimize
from scipy.special import ndtr   # standard normal CDF
from scipy.stats import norm

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "output" / "phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Log-likelihood for half-normal SFA cost frontier
# ---------------------------------------------------------------------------

def _sfa_loglik(params: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    """
    Negative log-likelihood for the half-normal composed error model.

    params = [β₀, β₁, …, β_k, log_sigma_v, log_sigma_u]
    """
    k     = X.shape[1]
    beta  = params[:k]
    log_sv = params[k]
    log_su = params[k + 1]

    sigma_v = np.exp(log_sv)
    sigma_u = np.exp(log_su)

    if sigma_v < 1e-10 or sigma_u < 1e-10:
        return 1e12

    sigma   = np.sqrt(sigma_v**2 + sigma_u**2)
    lam     = sigma_u / sigma_v           # lambda: signal-to-noise
    eps     = y - X @ beta                # residuals (positive = above frontier)

    # Aigner, Lovell & Schmidt (1977) log-likelihood
    log_sigma = np.log(sigma)
    ll = (
        np.log(2.0)
        - log_sigma
        + norm.logpdf(eps / sigma)
        + norm.logcdf(-lam * eps / sigma)
    )
    return -ll.sum()


# ---------------------------------------------------------------------------
# JLMS efficiency estimator
# ---------------------------------------------------------------------------

def _jlms_efficiency(eps: np.ndarray, sigma_v: float, sigma_u: float) -> np.ndarray:
    """
    Jondrow, Lovell, Materov & Schmidt (1982) point estimates of u_i.
    TE_i = exp(-E[u_i | ε_i])
    """
    sigma2  = sigma_v**2 + sigma_u**2
    sigma   = np.sqrt(sigma2)
    mu_star = -eps * sigma_u**2 / sigma2
    s_star  = sigma_v * sigma_u / sigma

    # E[u | ε] = μ* + σ* · φ(μ*/σ*) / Φ(μ*/σ*)
    ratio   = norm.pdf(mu_star / s_star) / np.maximum(ndtr(mu_star / s_star), 1e-12)
    u_hat   = mu_star + s_star * ratio
    return np.exp(-u_hat)


# ---------------------------------------------------------------------------
# Main SFA routine
# ---------------------------------------------------------------------------

def run_sfa(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    df = gdf.copy()

    # Covariates
    df["area_sq_mi_z"]   = (df["area_sq_mi"] - df["area_sq_mi"].mean()) / df["area_sq_mi"].std()
    df["poverty_rate_z"] = (df["poverty_rate"] - df["poverty_rate"].mean()) / df["poverty_rate"].std()

    covariate_names = [
        "const",
        "fragmentation_idx",
        "median_hh_income_z",
        "is_contract",
        "area_sq_mi_z",
    ]
    X = np.column_stack([
        np.ones(len(df)),
        df["fragmentation_idx"].values,
        df["median_hh_income_z"].values,
        df["is_contract"].values.astype(float),
        df["area_sq_mi_z"].values,
    ])
    y = df["log_cost_per_lm"].values

    k = X.shape[1]

    # ---- Initial values via OLS ------------------------------------------
    import statsmodels.api as sm
    ols = sm.OLS(y, X).fit()
    s   = np.log(ols.resid.std())
    x0  = np.concatenate([ols.params, [s - 0.3, s - 0.3]])

    # ---- Nelder-Mead warm start ------------------------------------------
    res1 = minimize(
        _sfa_loglik, x0, args=(y, X),
        method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-6, "fatol": 1e-6},
    )
    # ---- L-BFGS-B refinement --------------------------------------------
    res2 = minimize(
        _sfa_loglik, res1.x, args=(y, X),
        method="L-BFGS-B",
        options={"maxiter": 10_000, "ftol": 1e-10},
    )
    theta = res2.x

    beta    = theta[:k]
    sigma_v = float(np.exp(theta[k]))
    sigma_u = float(np.exp(theta[k + 1]))
    lam     = sigma_u / sigma_v
    sigma   = np.sqrt(sigma_v**2 + sigma_u**2)

    eps = y - X @ beta
    te  = _jlms_efficiency(eps, sigma_v, sigma_u)

    # ---- Pack results -------------------------------------------------------
    scores = df[["name", "city_class", "fragmentation_idx",
                 "median_hh_income", "pci_score",
                 "cost_per_lane_mile"]].copy()
    scores["te_score"]           = te.round(4)
    scores["inefficiency_pct"]   = ((1 - te) * 100).round(2)
    scores["te_rank"]            = scores["te_score"].rank(ascending=False).astype(int)
    scores = scores.sort_values("te_rank")

    # ---- Write text report -----------------------------------------------
    lines = [
        "=" * 72,
        "  Stochastic Frontier Analysis — Cost Frontier (Half-Normal)".center(72),
        "=" * 72,
        f"  N observations   : {len(df)}",
        f"  Dep. variable    : log(cost_per_lane_mile)",
        f"  Log-likelihood   : {-res2.fun:.4f}",
        f"  σ_v  (noise)     : {sigma_v:.4f}",
        f"  σ_u  (ineffic.)  : {sigma_u:.4f}",
        f"  λ = σ_u/σ_v      : {lam:.4f}  {'(signal >> noise → frontier meaningful)' if lam > 1 else '(noise >> signal → frontier uncertain)'}",
        f"  σ = √(σ²_v+σ²_u): {sigma:.4f}",
        "-" * 72,
        f"  {'Parameter':<28} {'Estimate':>12}",
        "-" * 72,
    ]
    for name, b in zip(covariate_names, beta):
        lines.append(f"  {name:<28} {b:>12.5f}")
    lines += [
        "-" * 72,
        "",
        "  City Efficiency Rankings (TE = 1 → on frontier):",
        "-" * 72,
        f"  {'Rank':<6} {'City':<20} {'TE Score':>10} {'Ineffic %':>11} {'Frag Idx':>10} {'City Class':<14}",
        "-" * 72,
    ]
    for _, row in scores.iterrows():
        lines.append(
            f"  {row.te_rank:<6} {row['name']:<20} {row.te_score:>10.4f}"
            f" {row.inefficiency_pct:>10.1f}%  {row.fragmentation_idx:>9.3f}  {row.city_class:<14}"
        )
    lines += [
        "-" * 72,
        f"  Mean TE: {te.mean():.4f}   (avg inefficiency: {(1-te).mean()*100:.1f}%)",
        f"  Most efficient  : {scores.iloc[0]['name']} (TE={scores.iloc[0].te_score:.4f})",
        f"  Least efficient : {scores.iloc[-1]['name']} (TE={scores.iloc[-1].te_score:.4f})",
        "",
        "  Interpretation:",
        "    fragmentation_idx > 0 → more borders → higher cost per lane-mile",
        "    is_contract coef  > 0 → contract cities pay premium for outsourced services",
        "    TE < 1 means a city could achieve current road maintenance at lower cost",
        "      if it operated at the efficiency of the frontier city.",
    ]

    out_txt = OUTPUT_DIR / "sfa_results.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"[sfa] Results → {out_txt}")

    out_csv = OUTPUT_DIR / "sfa_efficiency_scores.csv"
    scores.to_csv(out_csv, index=False)
    print(f"[sfa] Efficiency scores → {out_csv}")

    # ---- Efficiency chart -----------------------------------------------
    _plot_efficiency(scores, lam, sigma_v, sigma_u)

    print(f"\n  λ = {lam:.3f}  →  "
          + ("frontier is informative (signal dominates noise)" if lam > 1
             else "frontier uncertain (noise dominates) — get more data"))
    print(f"  Mean TE = {te.mean():.3f}  |  range: [{te.min():.3f}, {te.max():.3f}]")
    return scores


def _plot_efficiency(scores: pd.DataFrame, lam: float,
                     sigma_v: float, sigma_u: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Efficiency bar chart -------------------------------------------
    ax = axes[0]
    colors = [
        "#2c7bb6" if c == "Full-Service" else "#d7191c"
        for c in scores["city_class"]
    ]
    bars = ax.barh(
        scores["name"], scores["te_score"],
        color=colors, edgecolor="white", height=0.7,
    )
    ax.axvline(1.0, color="black", lw=1, ls="--")
    ax.axvline(scores["te_score"].mean(), color="grey", lw=1, ls=":")
    ax.set_xlabel("Technical Efficiency (TE) Score")
    ax.set_title("Cost Frontier Efficiency by City\n(blue=Full-Service, red=Contract)")
    ax.set_xlim(0, 1.05)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2c7bb6", label="Full-Service"),
        Patch(facecolor="#d7191c", label="Contract"),
    ], fontsize=8)

    # ---- TE vs fragmentation scatter ------------------------------------
    ax2 = axes[1]
    ax2.scatter(
        scores["fragmentation_idx"], scores["te_score"],
        c=colors, edgecolors="k", linewidth=0.5, s=70,
    )
    for _, row in scores.iterrows():
        ax2.annotate(
            row["name"].split()[0],
            (row["fragmentation_idx"], row["te_score"]),
            fontsize=6.5, xytext=(3, 3), textcoords="offset points",
        )
    m, b = np.polyfit(scores["fragmentation_idx"], scores["te_score"], 1)
    xx = np.linspace(scores["fragmentation_idx"].min(),
                     scores["fragmentation_idx"].max(), 100)
    ax2.plot(xx, m * xx + b, color="darkred", lw=1.5, ls="--")
    ax2.set_xlabel("Fragmentation Index")
    ax2.set_ylabel("Technical Efficiency (TE)")
    ax2.set_title(
        f"Efficiency vs. Fragmentation\n"
        f"λ={lam:.2f}  σ_v={sigma_v:.3f}  σ_u={sigma_u:.3f}"
    )

    plt.suptitle("SGV — Stochastic Frontier Analysis: Cost Efficiency", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sfa_efficiency_plot.png", dpi=150)
    plt.close(fig)
