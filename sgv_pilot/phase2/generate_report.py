"""
generate_report.py

Produces a clean multi-page PDF summary report using matplotlib.PdfPages.
Output → data/output/phase2/SGV_Fragmentation_Report.pdf
"""

import json
import pathlib
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image

BASE       = pathlib.Path(__file__).parent.parent
OUTPUT_DIR = BASE / "data" / "output" / "phase2"

# ── palette ───────────────────────────────────────────────────────────────
NAVY  = "#1e3c72"
TEAL  = "#1b998b"
LIGHT = "#f5f7fa"
WHITE = "#ffffff"
DARK  = "#282828"
MID   = "#666666"
RED   = "#b41e1e"
GREEN = "#1e8246"
GOLD  = "#e8a020"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "text.color":        DARK,
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "figure.facecolor":  WHITE,
})


# ── helpers ───────────────────────────────────────────────────────────────

def _new_page(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    return fig


def _header_band(fig, title: str, subtitle: str = ""):
    ax = fig.add_axes([0, 0.93, 1, 0.07])
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.65, title, color=WHITE, fontsize=13, fontweight="bold", va="center")
    if subtitle:
        ax.text(0.02, 0.15, subtitle, color="#c8dcf0", fontsize=7.5, va="center")
    # teal accent bar
    ax2 = fig.add_axes([0, 0.925, 1, 0.005])
    ax2.set_facecolor(TEAL); ax2.axis("off")


def _footer(fig, page_num: int):
    ax = fig.add_axes([0, 0, 1, 0.025])
    ax.set_facecolor(LIGHT); ax.axis("off")
    ax.text(0.5, 0.5,
            f"SGV Municipal Fragmentation Analysis  |  Phase 2  |  "
            f"{date.today().isoformat()}  |  Page {page_num}",
            color=MID, fontsize=6, ha="center", va="center")


def _section_label(ax_host, fig, y_frac: float, label: str):
    """Draw a navy section header bar at y_frac in figure coordinates."""
    ax = fig.add_axes([0.06, y_frac - 0.007, 0.88, 0.022])
    ax.set_facecolor(NAVY); ax.axis("off")
    ax.text(0.01, 0.5, label, color=WHITE, fontsize=8.5,
            fontweight="bold", va="center")


def _table(ax, headers, rows, col_widths=None, row_height=0.06):
    ax.axis("off")
    n_cols = len(headers)
    n_rows = len(rows)
    if col_widths is None:
        col_widths = [1 / n_cols] * n_cols

    total_h = (n_rows + 1) * row_height
    y_top = 1.0

    # header
    x = 0.0
    for h, w in zip(headers, col_widths):
        rect = FancyBboxPatch((x, y_top - row_height), w, row_height,
                               boxstyle="square,pad=0", linewidth=0,
                               facecolor=NAVY, clip_on=False,
                               transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y_top - row_height / 2, h,
                color=WHITE, fontsize=7, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        x += w

    # rows
    for i, row in enumerate(rows):
        y = y_top - (i + 2) * row_height
        bg = LIGHT if i % 2 == 0 else WHITE
        x = 0.0
        for val, w in zip(row, col_widths):
            rect = FancyBboxPatch((x, y), w, row_height,
                                   boxstyle="square,pad=0", linewidth=0.3,
                                   edgecolor="#dddddd", facecolor=bg,
                                   clip_on=False, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(x + 0.01, y + row_height / 2, str(val),
                    color=DARK, fontsize=6.8, va="center",
                    transform=ax.transAxes)
            x += w

    ax.set_xlim(0, 1)
    ax.set_ylim(y_top - (n_rows + 1) * row_height, y_top)


def _img_axes(fig, image_path, rect):
    """Load a PNG and place it in a figure-coordinate rect [l,b,w,h]."""
    p = pathlib.Path(image_path)
    if not p.exists():
        return
    img = Image.open(p)
    ax = fig.add_axes(rect)
    ax.imshow(np.array(img))
    ax.axis("off")


def _kpi_boxes(fig, items, y_top=0.88):
    """Draw KPI summary boxes. items = list of (label, value, color)."""
    n = len(items)
    w = 0.88 / n
    x0 = 0.06
    for i, (label, value, colour) in enumerate(items):
        ax = fig.add_axes([x0 + i * w + 0.005, y_top - 0.075, w - 0.01, 0.068])
        ax.set_facecolor(LIGHT)
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2)
        ax.axis("off")
        ax.text(0.5, 0.65, value, color=colour, fontsize=16,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        ax.text(0.5, 0.15, label, color=MID, fontsize=6.5,
                ha="center", va="center", transform=ax.transAxes)


def _body_text(fig, text: str, rect, fontsize=8):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.text(0.0, 1.0, text, color=DARK, fontsize=fontsize,
            va="top", ha="left", wrap=True,
            transform=ax.transAxes,
            multialignment="left")


def _save(fig, pdf: PdfPages):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── pages ─────────────────────────────────────────────────────────────────

def page_cover(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))

    # Full navy top band
    ax_top = fig.add_axes([0, 0.60, 1, 0.40])
    ax_top.set_facecolor(NAVY); ax_top.axis("off")
    ax_top.text(0.06, 0.72,
                "SGV Municipal Fragmentation\nAnalysis — Phase 2",
                color=WHITE, fontsize=22, fontweight="bold", va="top",
                linespacing=1.4)
    ax_top.text(0.06, 0.40,
                "Quantifying the implicit cost of jurisdictional fragmentation\n"
                "on road maintenance efficiency",
                color="#c8dcf0", fontsize=10.5, va="top", linespacing=1.5)
    ax_top.text(0.06, 0.14,
                f"San Gabriel Valley Pilot  ·  24 cities  ·  {date.today().isoformat()}",
                color="#8aafd4", fontsize=9)
    # teal accent
    ax_acc = fig.add_axes([0, 0.595, 1, 0.007])
    ax_acc.set_facecolor(TEAL); ax_acc.axis("off")

    # Purpose section
    ax_p = fig.add_axes([0.06, 0.49, 0.88, 0.025])
    ax_p.set_facecolor(NAVY); ax_p.axis("off")
    ax_p.text(0.01, 0.5, "Purpose", color=WHITE, fontsize=9,
              fontweight="bold", va="center")

    ax_body = fig.add_axes([0.06, 0.38, 0.88, 0.11])
    ax_body.axis("off")
    ax_body.text(0, 1,
        "This report summarises Phase 2 of the socal_fragmentation_tax project. "
        "The central hypothesis is that jurisdictional density — measured by the "
        "number of city neighbours, shared border length, and a composite Fragmentation "
        "Index — raises per-lane-mile road maintenance costs and depresses Pavement "
        "Condition Index (PCI) scores, independently of city income and size.",
        color=DARK, fontsize=8.5, va="top", wrap=True, transform=ax_body.transAxes)

    # Methodology table
    ax_m = fig.add_axes([0.06, 0.345, 0.88, 0.025])
    ax_m.set_facecolor(NAVY); ax_m.axis("off")
    ax_m.text(0.01, 0.5, "Methodology", color=WHITE, fontsize=9,
              fontweight="bold", va="center")

    ax_tbl = fig.add_axes([0.06, 0.10, 0.88, 0.245])
    _table(ax_tbl,
           ["Step", "Module", "Description"],
           [
               ["1", "data_prep",        "Queen-contiguity fragmentation metrics; calibrated PCI/cost panel (n=24)"],
               ["2", "spatial_analysis", "Spearman correlation matrix across all variables"],
               ["3", "spatial_analysis", "Global Moran's I + Local LISA cluster maps for PCI and cost"],
               ["4", "regression",       "OLS models M1–M4 with HC3 robust SE + Moran residual test"],
               ["5", "regression",       "Spatial lag model (2SLS) to account for cross-border spillover"],
               ["6", "sfa",             "Stochastic Frontier Analysis — efficiency scores per city"],
           ],
           col_widths=[0.06, 0.20, 0.74],
           row_height=0.13)

    ax_note = fig.add_axes([0.06, 0.04, 0.88, 0.06])
    ax_note.axis("off")
    ax_note.text(0, 0.9,
        "Data note: PCI and cost-per-lane-mile are calibrated synthetic values anchored to published "
        "SGV/SoCal benchmarks. Income from ACS 2023 5-yr estimates. "
        "Pipeline is production-ready — swap in live data with no model code changes.",
        color=MID, fontsize=7, va="top", style="italic", wrap=True,
        transform=ax_note.transAxes)

    _save(fig, pdf)


def page_key_findings(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    _header_band(fig, "Key Findings at a Glance",
                 "San Gabriel Valley — 24 cities — Phase 2 results")
    _footer(fig, 2)

    _kpi_boxes(fig, [
        ("Frag. → PCI\nper SD  (M2)", "–3.0 pts",   RED),
        ("Frag. → Cost\nper SD  (M4)", "+17%",       RED),
        ("Spatial lag frag\n2SLS p=0.009", "–3.02",  RED),
        ("Avg avoidable\ncost  (SFA)", "9.3%",        GOLD),
    ], y_top=0.88)

    # Main findings table
    _section_label(None, fig, 0.805, "The Fragmentation Tax — Across All Models")
    ax1 = fig.add_axes([0.06, 0.58, 0.88, 0.225])
    _table(ax1,
           ["Model", "Outcome", "Frag. Coef.", "p-value", "R²", "Interpretation"],
           [
               ["M1 Bivariate",    "PCI score",    "–0.64",  "0.737", "0.005", "No signal without controls"],
               ["M2 + Income",     "PCI score",    "–2.99",  "0.001", "0.710", "Strong after income control"],
               ["M3 Full spec",    "PCI score",    "–1.46",  "0.548", "0.830", "Area absorbs variance"],
               ["M4 Cost frontier","log(cost/LM)", "+0.157", "0.000", "0.934", "~17% cost premium per SD"],
               ["Spatial Lag 2SLS","PCI score",    "–3.02",  "0.009", "0.837", "Robust to spatial spillover"],
           ],
           col_widths=[0.16, 0.15, 0.12, 0.10, 0.08, 0.39],
           row_height=0.125)

    _section_label(None, fig, 0.570, "Why Income Controls Matter")
    ax_body1 = fig.add_axes([0.06, 0.44, 0.88, 0.125])
    ax_body1.axis("off")
    ax_body1.text(0, 1,
        "Without controlling for income, the fragmentation signal is masked (M1: p=0.74). "
        "Wealthier cities tend to both maintain better roads AND occupy lower-fragmentation "
        "areas — income and fragmentation are correlated. Once income is held constant (M2), "
        "the fragmentation effect becomes highly significant (–3.0 pts/SD, p<0.001). "
        "This confirms income as a confounder, not a mediator, of the fragmentation effect.",
        color=DARK, fontsize=8.5, va="top", wrap=True, transform=ax_body1.transAxes)

    _section_label(None, fig, 0.430, "LISA Spatial Clusters")
    ax2 = fig.add_axes([0.06, 0.24, 0.88, 0.185])
    _table(ax2,
           ["Cluster", "Cities", "Interpretation"],
           [
               ["HH Cost Hotspot", "Azusa, Irwindale",       "High cost surrounded by high-cost neighbours — fragmentation cluster"],
               ["LL PCI Cold spot", "El Monte",              "Low road quality surrounded by similarly poor neighbours"],
               ["HL Outlier",       "Montebello, San Dimas", "Isolated divergence from surrounding pattern"],
           ],
           col_widths=[0.20, 0.22, 0.58],
           row_height=0.19)

    _section_label(None, fig, 0.230, "SFA — Efficiency Leaders & Laggards (selected)")
    ax3 = fig.add_axes([0.06, 0.055, 0.88, 0.170])
    _table(ax3,
           ["Rank", "City", "TE Score", "Avoidable Cost", "Frag. Index", "Class"],
           [
               ["1",  "Arcadia",       "1.000", "0.0%",  "–0.109", "Full-Service"],
               ["1",  "Monterey Park", "1.000", "0.0%",  "–0.377", "Full-Service"],
               ["19", "La Puente",     "0.854", "14.6%", "+1.408", "Contract"],
               ["23", "Irwindale",     "0.766", "23.4%", "+1.408", "Contract"],
               ["24", "Whittier",      "0.721", "27.9%", "–1.598", "Full-Service"],
           ],
           col_widths=[0.08, 0.22, 0.13, 0.17, 0.16, 0.24],
           row_height=0.145)

    _save(fig, pdf)


def page_correlation(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    _header_band(fig, "Correlation Analysis",
                 "Spearman rank correlations — all panel variables")
    _footer(fig, 3)

    _section_label(None, fig, 0.885, "Key Spearman Correlations")
    ax1 = fig.add_axes([0.06, 0.665, 0.88, 0.215])
    _table(ax1,
           ["Variable A", "Variable B", "ρ", "Interpretation"],
           [
               ["PCI score",        "Fragmentation index",  "–0.10", "Negative — higher fragmentation, lower road quality"],
               ["Cost / lane-mile", "Fragmentation index",  "+0.43", "Positive — higher fragmentation, higher cost"],
               ["PCI score",        "Median HH income",     "+0.81", "Strong — wealthier cities maintain better roads"],
               ["Cost / lane-mile", "Median HH income",     "–0.70", "Negative — wealthier cities spend less per lane-mile"],
               ["PCI score",        "Poverty rate",         "–0.72", "Negative — more poverty, worse road quality"],
               ["Cost / lane-mile", "N neighbours",         "+0.42", "More borders → higher cost"],
           ],
           col_widths=[0.22, 0.22, 0.09, 0.47],
           row_height=0.118)

    ax_body = fig.add_axes([0.06, 0.57, 0.88, 0.09])
    ax_body.axis("off")
    ax_body.text(0, 1,
        "Income is the dominant predictor of road quality in the raw correlations. "
        "The fragmentation–PCI correlation appears weak bivariate (ρ = –0.10), but "
        "the cost correlation is materially stronger (ρ = +0.43). Regression analysis "
        "(controlling for income) reveals the fragmentation effect on PCI is suppressed "
        "by the income–fragmentation correlation — income acts as a positive confounder.",
        color=DARK, fontsize=8.5, va="top", wrap=True, transform=ax_body.transAxes)

    _section_label(None, fig, 0.555, "Correlation Heatmap")
    _img_axes(fig, OUTPUT_DIR / "correlation_heatmap.png",
              [0.06, 0.06, 0.88, 0.49])

    ax_cap = fig.add_axes([0.06, 0.04, 0.88, 0.02])
    ax_cap.axis("off")
    ax_cap.text(0.5, 0.5,
        "Figure 1: Spearman correlation matrix across all panel variables (n=24 SGV cities)",
        color=MID, fontsize=7, ha="center", va="center", style="italic")

    _save(fig, pdf)


def page_spatial(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    _header_band(fig, "Spatial Autocorrelation",
                 "Global Moran's I + Local LISA cluster maps")
    _footer(fig, 4)

    with open(OUTPUT_DIR / "moran_global.json") as f:
        moran = json.load(f)

    _section_label(None, fig, 0.885, "Global Moran's I")
    ax1 = fig.add_axes([0.06, 0.74, 0.88, 0.14])
    _table(ax1,
           ["Variable", "Moran's I", "z-score", "p (sim)", "Finding"],
           [
               ["PCI score",         str(moran["pci_score"]["I"]),
                str(moran["pci_score"]["z_score"]),
                str(moran["pci_score"]["p_sim"]),
                moran["pci_score"]["interpretation"]],
               ["Cost / lane-mile",  str(moran["cost_per_lane_mile"]["I"]),
                str(moran["cost_per_lane_mile"]["z_score"]),
                str(moran["cost_per_lane_mile"]["p_sim"]),
                moran["cost_per_lane_mile"]["interpretation"]],
               ["Fragmentation idx", str(moran["fragmentation_idx"]["I"]),
                str(moran["fragmentation_idx"]["z_score"]),
                str(moran["fragmentation_idx"]["p_sim"]),
                moran["fragmentation_idx"]["interpretation"]],
           ],
           col_widths=[0.20, 0.13, 0.13, 0.14, 0.40],
           row_height=0.22)

    ax_body = fig.add_axes([0.06, 0.66, 0.88, 0.075])
    ax_body.axis("off")
    ax_body.text(0, 1,
        "At n=24, the global Moran's I test lacks statistical power to detect moderate "
        "autocorrelation. LISA provides local-level evidence of clustering even where "
        "the global test is not significant.",
        color=DARK, fontsize=8.5, va="top", wrap=True, transform=ax_body.transAxes)

    _section_label(None, fig, 0.650, "LISA Cluster Map — PCI Score")
    _img_axes(fig, OUTPUT_DIR / "lisa_map_pci_score.png",
              [0.03, 0.375, 0.94, 0.275])
    ax_c1 = fig.add_axes([0.06, 0.355, 0.88, 0.02])
    ax_c1.axis("off")
    ax_c1.text(0.5, 0.5,
        "Figure 2: LISA for PCI score. HH=red, LL=blue, HL=orange, LH=light blue, NS=grey. "
        "Right: Moran scatter.",
        color=MID, fontsize=7, ha="center", va="center", style="italic")

    _section_label(None, fig, 0.345, "LISA Cluster Map — Cost per Lane-Mile")
    _img_axes(fig, OUTPUT_DIR / "lisa_map_cost_per_lane_mile.png",
              [0.03, 0.075, 0.94, 0.275])
    ax_c2 = fig.add_axes([0.06, 0.055, 0.88, 0.02])
    ax_c2.axis("off")
    ax_c2.text(0.5, 0.5,
        "Figure 3: LISA for cost/LM. Azusa & Irwindale form a statistically significant "
        "HH cost cluster — consistent with fragmentation friction.",
        color=MID, fontsize=7, ha="center", va="center", style="italic")

    _save(fig, pdf)


def page_regression(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    _header_band(fig, "Regression Analysis",
                 "OLS models M1–M4 (HC3 robust SE) + Spatial Lag 2SLS")
    _footer(fig, 5)

    _section_label(None, fig, 0.885, "Model Specifications")
    ax1 = fig.add_axes([0.06, 0.72, 0.88, 0.16])
    _table(ax1,
           ["Model", "Outcome", "Covariates"],
           [
               ["M1",       "PCI score",     "Fragmentation index only (bivariate baseline)"],
               ["M2",       "PCI score",     "M1 + median HH income (z) + poverty rate (z)"],
               ["M3",       "PCI score",     "M2 + contract dummy + city area (z) + log(ADT)(z)"],
               ["M4",       "log(cost/LM)", "Fragmentation + income + contract + area"],
               ["M5 (2SLS)","PCI score",    "M2 specification + spatial lag W·PCI (instrumented)"],
           ],
           col_widths=[0.12, 0.19, 0.69],
           row_height=0.135)

    ax_note = fig.add_axes([0.06, 0.70, 0.88, 0.02])
    ax_note.axis("off")
    ax_note.text(0, 0.5, "All OLS models: HC3 heteroskedasticity-robust standard errors.",
                 color=MID, fontsize=7, style="italic", va="center")

    _section_label(None, fig, 0.690, "Coefficient Plot")
    _img_axes(fig, OUTPUT_DIR / "regression_coef_plot.png",
              [0.03, 0.415, 0.94, 0.275])
    ax_cap = fig.add_axes([0.06, 0.395, 0.88, 0.02])
    ax_cap.axis("off")
    ax_cap.text(0.5, 0.5,
        "Figure 4: Left — fragmentation coefficient across PCI models (90% CI, red=significant). "
        "Right — M3 full-specification coefficients.",
        color=MID, fontsize=7, ha="center", va="center", style="italic")

    _section_label(None, fig, 0.385, "Interpretation")
    ax_body = fig.add_axes([0.06, 0.06, 0.88, 0.32])
    ax_body.axis("off")
    interp = (
        "M1 → M2 shift:  Adding income controls causes the fragmentation coefficient to jump "
        "from –0.64 (p=0.74, insignificant) to –2.99 (p=0.001, ***). Income is a positive "
        "confounder — wealthier areas have both less fragmentation and better roads, suppressing "
        "the raw fragmentation signal in M1.\n\n"
        "M2 → M3 shift:  The fragmentation coefficient attenuates to –1.46 (p=0.55) when city "
        "area is added. Cities with more neighbours tend to be larger; area independently predicts "
        "PCI (+4.2 pts/SD, p=0.009), so fragmentation and area share variance.\n\n"
        "M4 (cost model):  Fragmentation raises log(cost/LM) by +0.157 per SD (p<0.0001), "
        "equivalent to ~17% cost premium holding income and size fixed. Contract cities pay an "
        "additional ~15% premium (p=0.10).\n\n"
        "Spatial lag (2SLS):  The fragmentation coefficient is –3.02 (p=0.009) after "
        "instrumenting for the spatial lag of PCI using W·X and W²·X. The spatial lag "
        "coefficient ρ is near zero, suggesting limited cross-border spillover at this "
        "sample size — the fragmentation effect appears city-specific rather than diffuse."
    )
    ax_body.text(0, 1, interp, color=DARK, fontsize=8.3, va="top", wrap=True,
                 transform=ax_body.transAxes, linespacing=1.5)

    _save(fig, pdf)


def page_sfa(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    _header_band(fig, "Stochastic Frontier Analysis",
                 "Cost efficiency rankings for all 24 SGV cities")
    _footer(fig, 6)

    _section_label(None, fig, 0.885, "Model")
    ax_body1 = fig.add_axes([0.06, 0.78, 0.88, 0.10])
    ax_body1.axis("off")
    ax_body1.text(0, 1,
        "Cost frontier:  ln(cost_i) = α + β·X_i + v_i + u_i\n"
        "  v_i ~ N(0, σ²_v)   symmetric noise          "
        "  u_i ~ N⁺(0, σ²_u)  one-sided inefficiency term (≥ 0)\n\n"
        "Technical Efficiency  TE_i = exp(–û_i) ∈ (0, 1].  "
        "TE = 1 → city is on the cost frontier (minimum achievable cost).  "
        "JLMS (1982) conditional mean estimator used for city rankings.  "
        "Covariates: fragmentation_idx, median_hh_income_z, is_contract, area_sq_mi_z.",
        color=DARK, fontsize=8.3, va="top", wrap=True, transform=ax_body1.transAxes,
        linespacing=1.6)

    _section_label(None, fig, 0.768, "Efficiency Chart")
    _img_axes(fig, OUTPUT_DIR / "sfa_efficiency_plot.png",
              [0.03, 0.495, 0.94, 0.275])
    ax_cap = fig.add_axes([0.06, 0.475, 0.88, 0.02])
    ax_cap.axis("off")
    ax_cap.text(0.5, 0.5,
        "Figure 5: Left — TE scores by city (blue=Full-Service, red=Contract). "
        "Right — TE vs. fragmentation index with trend line.",
        color=MID, fontsize=7, ha="center", va="center", style="italic")

    _section_label(None, fig, 0.465, "Efficiency Rankings (all 24 cities)")
    sfa_df = pd.read_csv(OUTPUT_DIR / "sfa_efficiency_scores.csv")
    ax2 = fig.add_axes([0.06, 0.065, 0.88, 0.395])
    rows = [
        [str(int(r.te_rank)), r["name"],
         f"{r.te_score:.3f}", f"{r.inefficiency_pct:.1f}%",
         f"{r.fragmentation_idx:+.3f}", r.city_class]
        for _, r in sfa_df.iterrows()
    ]
    _table(ax2,
           ["Rank", "City", "TE", "Avoidable Cost", "Frag. Idx", "Class"],
           rows,
           col_widths=[0.09, 0.26, 0.10, 0.19, 0.15, 0.21],
           row_height=0.0385)

    ax_note = fig.add_axes([0.06, 0.035, 0.88, 0.03])
    ax_note.axis("off")
    mean_te = sfa_df["te_score"].mean()
    by_class = sfa_df.groupby("city_class")["te_score"].mean()
    ax_note.text(0, 0.8,
        f"Mean TE all cities: {mean_te:.3f} ({(1-mean_te)*100:.1f}% avg avoidable cost)  |  "
        f"Full-Service: {by_class.get('Full-Service',0):.3f}  |  "
        f"Contract: {by_class.get('Contract',0):.3f}",
        color=MID, fontsize=7.5, style="italic", va="top")

    _save(fig, pdf)


def page_next_steps(pdf: PdfPages):
    fig = plt.figure(figsize=(8.5, 11))
    _header_band(fig, "Limitations & Next Steps",
                 "Roadmap to production-ready analysis")
    _footer(fig, 7)

    _section_label(None, fig, 0.885, "Data Limitations")
    ax_body1 = fig.add_axes([0.06, 0.785, 0.88, 0.095])
    ax_body1.axis("off")
    ax_body1.text(0, 1,
        "All PCI scores and cost-per-lane-mile figures are calibrated synthetic values "
        "anchored to published benchmarks (SaveCaliforniaStreets 2023, CA SCO Streets & Roads "
        "FY2022). The analytical pipeline is production-ready — replacing the synthetic columns "
        "with live data requires no model code changes.\n\n"
        "Income data is real: ACS 2023 5-year estimates for all 24 SGV cities "
        "(static QuickFacts fallback used in sandbox; Census API call in production).",
        color=DARK, fontsize=8.5, va="top", wrap=True, transform=ax_body1.transAxes,
        linespacing=1.5)

    _section_label(None, fig, 0.775, "Recommended Next Steps")
    ax2 = fig.add_axes([0.06, 0.495, 0.88, 0.275])
    _table(ax2,
           ["Horizon", "Action", "Rationale"],
           [
               ["Short term",   "Connect live APIs",        "Enable Overpass + Census APIs; re-run full pipeline with real PCI and cost data"],
               ["Short term",   "Validate PCI sources",     "Cross-check SaveCaliforniaStreets exports against city-reported survey data"],
               ["Medium term",  "Expand to SCAG region",    "Re-run with ~190 SCAG cities; larger n increases Moran's I power and SFA reliability"],
               ["Medium term",  "Add true ADT",             "Replace synthetic ADT proxy with Caltrans AADT; traffic load is a key omitted variable"],
               ["Longer term",  "DiD quasi-experiment",     "Use JPA formation/dissolution events as natural experiments for causal identification"],
               ["Longer term",  "Procurement microdata",    "Obtain bid tabs to measure HMA/slurry unit costs directly"],
           ],
           col_widths=[0.18, 0.24, 0.58],
           row_height=0.115)

    _section_label(None, fig, 0.485, "Technical Notes")
    ax_body2 = fig.add_axes([0.06, 0.275, 0.88, 0.205])
    ax_body2.axis("off")
    ax_body2.text(0, 1,
        "Fragmentation Index: composite z-score of queen-contiguity neighbour count and "
        "shared border share; computed from Voronoi tessellation (production: TIGER/Line).\n\n"
        "Spatial weights: row-standardised queen contiguity (libpysal 4.14).\n\n"
        "OLS standard errors: HC3 (MacKinnon & White 1985).\n\n"
        "2SLS instruments: W·X and W²·X for each covariate.\n\n"
        "SFA: Aigner-Lovell-Schmidt (1977) half-normal composed error model; "
        "JLMS (1982) conditional mean efficiency estimator. MLE via Nelder-Mead "
        "warm start → L-BFGS-B refinement. σ_v ≈ 0 in this run due to small n=24 "
        "calibrated DGP; expect more balanced λ with live SCAG-wide data.",
        color=DARK, fontsize=8.3, va="top", wrap=True, transform=ax_body2.transAxes,
        linespacing=1.6)

    _section_label(None, fig, 0.265, "Output Files")
    ax3 = fig.add_axes([0.06, 0.055, 0.88, 0.205])
    _table(ax3,
           ["File", "Description"],
           [
               ["panel_data.parquet",               "City-level analytical panel (n=24) with all variables"],
               ["correlation_matrix.csv",            "Spearman ρ matrix across all variables"],
               ["correlation_heatmap.png",           "Visual heatmap of correlations"],
               ["moran_global.json",                 "Global Moran's I for PCI, cost, fragmentation"],
               ["lisa_clusters_pci_score.geojson",   "LISA cluster labels + Ii statistics (PCI)"],
               ["ols_results.txt",                   "M1–M4 OLS tables with Moran residual test"],
               ["spatial_lag_results.txt",           "2SLS spatial lag model results"],
               ["sfa_results.txt + _scores.csv",     "SFA parameters + efficiency rankings per city"],
               ["SGV_Fragmentation_Report.pdf",      "This report"],
           ],
           col_widths=[0.38, 0.62],
           row_height=0.076)

    _save(fig, pdf)


# ── main ──────────────────────────────────────────────────────────────────

def generate():
    out = OUTPUT_DIR / "SGV_Fragmentation_Report.pdf"
    with PdfPages(str(out)) as pdf:
        page_cover(pdf)
        page_key_findings(pdf)
        page_correlation(pdf)
        page_spatial(pdf)
        page_regression(pdf)
        page_sfa(pdf)
        page_next_steps(pdf)

    size_kb = out.stat().st_size / 1024
    print(f"[report] PDF written → {out}  ({size_kb:.0f} KB, 7 pages)")


if __name__ == "__main__":
    generate()
