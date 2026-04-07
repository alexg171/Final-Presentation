"""
Multi-window analysis: Did Musk's acquisition shift Twitter's right_share?

Tests three symmetric windows around the treatment date (Oct 27, 2022):
  6-month  : Apr 27, 2022  – Apr 27, 2023   (primary)
  1-year   : Oct 27, 2021  – Oct 27, 2023
  2-year   : Oct 27, 2020  – Oct 27, 2024

For each window:
  - Classify daily trending topics (lexicon)
  - Compute daily right_share = right / (right + left)
  - Run DiD: right_share ~ post + C(dow) with HC3 SEs
    (single-platform version — treatment dummy IS 'post')
  - Report: pre/post means, DiD coefficient, p-value, CI
  - Plot: smoothed timeseries with treatment vline

Twitter data source: out/twitter_1774668850.csv  (Oct 2020 – Oct 2024)

Output:
  out/multiwindow_results.csv
  out/figures/multiwindow_timeseries.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lexicon import classify_topic

TREATMENT  = pd.Timestamp("2022-10-27")
DATA_FILE  = "out/twitter_1774668850.csv"
FIGURES    = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

WINDOWS = {
    "6-month": (pd.Timestamp("2022-04-27"), pd.Timestamp("2023-04-27")),
    "1-year":  (pd.Timestamp("2021-10-27"), pd.Timestamp("2023-10-27")),
    "2-year":  (pd.Timestamp("2020-10-27"), pd.Timestamp("2024-10-27")),
}


# ── 1. BUILD TWITTER DAILY PANEL ─────────────────────────────────────────────

def build_twitter_panel(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Load the 4-year Twitter file, trim to [start, end], classify topics,
    and return a daily right_share panel.
    """
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()

    df["label"] = df["Topic"].apply(classify_topic)
    pol = df[df["label"].isin(["right", "left"])]

    daily = (pol.groupby([pol["Date"].dt.date, "label"])
               .size().unstack(fill_value=0).rename_axis("date"))
    for c in ["right", "left"]:
        if c not in daily:
            daily[c] = 0
    daily["total"] = daily["right"] + daily["left"]
    daily = daily[daily["total"] >= 1]
    daily["right_share"] = daily["right"] / daily["total"]
    daily.index = pd.to_datetime(daily.index)

    # Add DiD columns
    daily["post"] = (daily.index >= TREATMENT).astype(int)
    daily["dow"]  = daily.index.dayofweek

    return daily.reset_index()


# ── 2. RUN DiD ────────────────────────────────────────────────────────────────

def run_did(panel: pd.DataFrame, label: str) -> dict:
    """
    Single-platform interrupted time series / DiD.
    'post' = the treatment indicator.
    Day-of-week FE absorb weekly seasonality.
    HC3 robust SEs.
    """
    model = smf.ols(
        "right_share ~ post + C(dow)",
        data=panel,
    ).fit(cov_type="HC3")

    pre  = panel.loc[panel["post"] == 0, "right_share"]
    post = panel.loc[panel["post"] == 1, "right_share"]

    coef  = model.params["post"]
    pval  = model.pvalues["post"]
    ci_lo, ci_hi = model.conf_int().loc["post"]
    n_days = len(panel)
    n_pol  = panel["total"].sum()

    result = {
        "window":        label,
        "pre_mean":      pre.mean(),
        "post_mean":     post.mean(),
        "raw_diff":      post.mean() - pre.mean(),
        "did_coef":      coef,
        "ci_lo":         ci_lo,
        "ci_hi":         ci_hi,
        "pval":          pval,
        "stars":         _stars(pval),
        "n_days":        n_days,
        "n_pol_topics":  n_pol,
    }
    return result


def _stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return "n.s."


# ── 3. PRINT SUMMARY ─────────────────────────────────────────────────────────

def print_table(results: list):
    print("\n" + "="*72)
    print("MULTI-WINDOW DiD RESULTS — Twitter right_share post Musk acquisition")
    print("="*72)
    hdr = f"{'Window':<10} {'Pre mean':>9} {'Post mean':>10} {'Raw diff':>9} {'DiD coef':>9} {'p-val':>7} {'95% CI':<20} {'Days':>6}"
    print(hdr)
    print("-"*72)
    for r in results:
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(
            f"{r['window']:<10} {r['pre_mean']:>9.3f} {r['post_mean']:>10.3f} "
            f"{r['raw_diff']:>+9.3f} {r['did_coef']:>+9.3f} {r['pval']:>7.3f}{r['stars']:>4} "
            f"{ci:<20} {r['n_days']:>6}"
        )
    print("="*72)
    print("Note: DiD coef = OLS 'post' coefficient with day-of-week FE and HC3 SEs.")
    print("      right_share = right topics / (right + left topics) per day.")


# ── 4. PLOT ───────────────────────────────────────────────────────────────────

def plot_multiwindow(panels: dict):
    """
    One subplot per window. Each shows 14-day rolling right_share
    with a vertical treatment line.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharey=False)
    colors = {"6-month": "#4e79a7", "1-year": "#f28e2b", "2-year": "#59a14f"}

    for ax, (label, panel) in zip(axes, panels.items()):
        p = panel.set_index("date").sort_index()
        roll = p["right_share"].rolling(14, min_periods=3, center=True).mean()

        ax.plot(p.index, p["right_share"],
                color=colors[label], alpha=0.18, linewidth=0.6)
        ax.plot(roll.index, roll,
                color=colors[label], linewidth=2.0, label="14-day rolling mean")

        ax.axvline(TREATMENT, color="black", linewidth=1.2,
                   linestyle="--", label="Oct 27, 2022 (Musk acquisition)")
        ax.axhline(0.5, color="grey", linewidth=0.7, linestyle=":", alpha=0.6)

        # Shade pre / post
        xlim_l = p.index.min()
        xlim_r = p.index.max()
        ax.axvspan(xlim_l, TREATMENT,   alpha=0.05, color="#4e79a7")
        ax.axvspan(TREATMENT, xlim_r,   alpha=0.05, color="#e15759")

        ax.set_title(f"{label} window", fontsize=11, fontweight="bold")
        ax.set_ylabel("Daily right_share", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if label == "6-month" else 6))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(-0.05, 1.05)
        if label == "6-month":
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Twitter Right-Lean Share Before vs. After Musk Acquisition\n"
        "Three symmetric windows (6-month, 1-year, 2-year)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    out = f"{FIGURES}/multiwindow_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


# ── 5. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []
    all_panels  = {}

    print(f"Loading and classifying topics from {DATA_FILE} ...")
    print("(This takes ~1-2 min for the 2-year window)\n")

    for label, (start, end) in WINDOWS.items():
        print(f"  [{label}]  {start.date()} to {end.date()} ...", end=" ", flush=True)
        panel = build_twitter_panel(start, end)
        n_pol = int(panel["total"].sum())
        avg   = panel["total"].mean()
        print(f"{len(panel)} days with political content, avg {avg:.1f} pol topics/day")

        res = run_did(panel, label)
        all_results.append(res)
        all_panels[label] = panel

    print_table(all_results)

    # Save results table
    out_csv = "out/multiwindow_results.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\nSaved results table: {out_csv}")

    plot_multiwindow(all_panels)
    print("\nDone.")
