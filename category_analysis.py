"""
Categorical composition analysis of Twitter trending topics
Pre vs. Post Musk acquisition (Oct 27, 2022)

Full 4-year window: Oct 27 2020 – Oct 27 2024

Outputs
-------
  out/figures/twitter_category_shift.png         — net pp shift per category
  out/figures/twitter_category_timeseries.png    — monthly share per category
  out/twitter_category_counts.csv               — raw counts
"""

import os, sys, warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from category_lexicon import (
    classify_category,
    DISPLAY_GROUPS,
    CAT_COLORS,
    CATEGORY_ORDER,
)

TREATMENT = pd.Timestamp("2022-10-27")
DATA_FILE = "out/twitter_trending_4yr.csv"
FIGURES = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# All categories in display order
ALL_CATS = [c for c, _ in CATEGORY_ORDER] + ["other"]

# ── 1. LOAD & CLASSIFY ────────────────────────────────────────────────────────


def load_and_classify() -> pd.DataFrame:
    print("Loading and classifying topics …")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    print(f"  {len(df):,} unique date×topic pairs")
    df["category"] = df["Topic"].apply(classify_category)
    df["post"] = (df["Date"] >= TREATMENT).astype(int)
    return df


# ── 2. SUMMARY TABLE ──────────────────────────────────────────────────────────


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["post"] == 0]
    post = df[df["post"] == 1]
    rows = []
    for cat in ALL_CATS:
        n_pre = (pre["category"] == cat).sum()
        n_post = (post["category"] == cat).sum()
        pct_pre = n_pre / len(pre) * 100
        pct_post = n_post / len(post) * 100
        rows.append(
            {
                "category": cat,
                "label": DISPLAY_GROUPS.get(cat, cat),
                "n_pre": n_pre,
                "pct_pre": pct_pre,
                "n_post": n_post,
                "pct_post": pct_post,
                "shift_pp": pct_post - pct_pre,
            }
        )
    return pd.DataFrame(rows)


def print_summary(summ: pd.DataFrame):
    print("\n" + "=" * 82)
    print("CATEGORY COMPOSITION — Pre vs Post Musk Acquisition (Oct 27 2022)")
    print("=" * 82)
    print(
        f"{'Category':<36} {'Pre %':>7} {'Post %':>7} {'Shift':>8}   {'Pre n':>7} {'Post n':>7}"
    )
    print("-" * 82)
    for _, r in summ.iterrows():
        flag = " <--" if abs(r["shift_pp"]) >= 0.5 else ""
        print(
            f"{r['label']:<36} {r['pct_pre']:>7.2f} {r['pct_post']:>7.2f} "
            f"{r['shift_pp']:>+8.2f}pp   {int(r['n_pre']):>7,} {int(r['n_post']):>7,}{flag}"
        )
    print("=" * 82)


# ── 4. PLOT: SHIFT BARS ──────────────────────────────────────────────────────


def plot_shift(summ: pd.DataFrame):
    show = summ[~summ["category"].isin(["other", "religious"])].sort_values("shift_pp")
    colors = [CAT_COLORS.get(c, "#aaaaaa") for c in show["category"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        show["label"], show["shift_pp"], color=colors, edgecolor="white", linewidth=0.4
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage-point shift (Post − Pre)", fontsize=10)
    ax.set_title(
        "Category-Level Shift After Musk Acquisition\n"
        "(positive = more prevalent after Oct 27, 2022)",
        fontsize=12,
        fontweight="bold",
    )

    for bar, val in zip(bars, show["shift_pp"]):
        if abs(val) > 0.02:
            ax.text(
                val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}pp",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=7.5,
            )
    fig.tight_layout()
    out = f"{FIGURES}/twitter_category_shift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 5. PLOT: MONTHLY TIMESERIES ──────────────────────────────────────────────


def plot_timeseries(df: pd.DataFrame):
    df2 = df.copy()
    df2["month"] = df2["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = df2.groupby(["month", "category"]).size().unstack(fill_value=0)
    for cat in ALL_CATS:
        if cat not in monthly:
            monthly[cat] = 0
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    # Show interesting non-filler categories
    show = [
        c
        for c in ALL_CATS
        if c not in ("social_filler", "other", "holidays", "religious")
    ]

    n = len(show)
    fig, axes = plt.subplots(n, 1, figsize=(13, n * 1.5), sharex=True)

    for ax, cat in zip(axes, show):
        color = CAT_COLORS.get(cat, "#aaaaaa")
        ax.fill_between(monthly_pct.index, monthly_pct[cat], color=color, alpha=0.35)
        ax.plot(monthly_pct.index, monthly_pct[cat], color=color, linewidth=1.4)
        ax.axvline(TREATMENT, color="black", linewidth=1.0, linestyle="--", alpha=0.65)
        ax.set_ylabel("%", fontsize=7)
        ax.set_title(
            DISPLAY_GROUPS.get(cat, cat),
            fontsize=8,
            fontweight="bold",
            loc="left",
            pad=2,
        )
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "Monthly Share of Trending Topic Categories\n"
        "Dashed line = Oct 27, 2022 (Musk acquisition)",
        fontsize=12,
        fontweight="bold",
        y=1.002,
    )
    fig.tight_layout()
    out = f"{FIGURES}/twitter_category_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_classify()
    summ = build_summary(df)
    print_summary(summ)

    summ.to_csv("out/twitter_category_counts.csv", index=False)
    print("\nSaved: out/twitter_category_counts.csv")

    print("\nGenerating plots …")
    plot_shift(summ)
    plot_timeseries(df)
    print("\nDone.")
