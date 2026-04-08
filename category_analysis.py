"""
Categorical composition analysis of Twitter trending topics
Pre vs. Post Musk acquisition (Oct 27, 2022)

Full 4-year window: Oct 27 2020 – Oct 27 2024

Outputs
-------
  out/figures/category_composition.png   — stacked % bar pre vs post
  out/figures/category_shift.png         — net pp shift per category
  out/figures/category_timeseries.png    — monthly share per category
  out/figures/bro_shift.png              — male vs female coded over time
  out/category_counts.csv               — raw counts
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from category_lexicon import (
    classify_category, DISPLAY_GROUPS, CAT_COLORS, CATEGORY_ORDER
)

TREATMENT = pd.Timestamp("2022-10-27")
DATA_FILE = "out/twitter_1774668850.csv"
FIGURES   = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# All categories in display order
ALL_CATS = [c for c, _ in CATEGORY_ORDER] + ["other"]

# Gender-coding for the "bro shift" chart
MALE_CODED   = {"wrestling", "combat_sports", "sports_nba", "sports_nfl",
                "sports_mlb", "sports_nhl", "sports_other", "sports_college",
                "sports_soccer", "tech_gaming", "manosphere"}
FEMALE_CODED = {"reality_tv", "taylor_swift", "fandom",
                "sports_womens", "lgbtq_social"}
NEUTRAL_CODED = {"entertainment", "religious", "musk_twitter",
                 "holidays", "true_crime", "news_events",
                 "social_filler", "other"}


# ── 1. LOAD & CLASSIFY ────────────────────────────────────────────────────────

def load_and_classify() -> pd.DataFrame:
    print("Loading and classifying topics …")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    print(f"  {len(df):,} unique date×topic pairs")
    df["category"] = df["Topic"].apply(classify_category)
    df["post"]     = (df["Date"] >= TREATMENT).astype(int)
    return df


# ── 2. SUMMARY TABLE ──────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    pre  = df[df["post"] == 0]
    post = df[df["post"] == 1]
    rows = []
    for cat in ALL_CATS:
        n_pre  = (pre["category"]  == cat).sum()
        n_post = (post["category"] == cat).sum()
        pct_pre  = n_pre  / len(pre)  * 100
        pct_post = n_post / len(post) * 100
        rows.append({
            "category":  cat,
            "label":     DISPLAY_GROUPS.get(cat, cat),
            "n_pre":     n_pre,  "pct_pre":  pct_pre,
            "n_post":    n_post, "pct_post": pct_post,
            "shift_pp":  pct_post - pct_pre,
        })
    return pd.DataFrame(rows)


def print_summary(summ: pd.DataFrame):
    print("\n" + "="*82)
    print("CATEGORY COMPOSITION — Pre vs Post Musk Acquisition (Oct 27 2022)")
    print("="*82)
    print(f"{'Category':<36} {'Pre %':>7} {'Post %':>7} {'Shift':>8}   {'Pre n':>7} {'Post n':>7}")
    print("-"*82)
    for _, r in summ.iterrows():
        flag = " <--" if abs(r["shift_pp"]) >= 0.5 else ""
        print(f"{r['label']:<36} {r['pct_pre']:>7.2f} {r['pct_post']:>7.2f} "
              f"{r['shift_pp']:>+8.2f}pp   {int(r['n_pre']):>7,} {int(r['n_post']):>7,}{flag}")
    print("="*82)


# ── 3. PLOT: STACKED BARS ────────────────────────────────────────────────────

def plot_stacked_bars(summ: pd.DataFrame):
    # Exclude social_filler and other to keep chart readable
    show = summ[~summ["category"].isin(["social_filler", "other"])].copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.array([0, 1])
    bottom = np.zeros(2)

    for _, row in show.iterrows():
        vals   = np.array([row["pct_pre"], row["pct_post"]])
        color  = CAT_COLORS.get(row["category"], "#aaaaaa")
        ax.bar(x, vals, bottom=bottom, color=color,
               label=row["label"], edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Pre-acquisition\n(Oct 2020 – Oct 2022)",
         "Post-acquisition\n(Oct 2022 – Oct 2024)"],
        fontsize=11)
    ax.set_ylabel("Share of trending topics (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Twitter Trending Topic Composition\nPre vs. Post Musk Acquisition"
        "\n(social filler & uncategorised excluded)",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9,
              bbox_to_anchor=(1.01, 1.0), ncol=1)
    fig.tight_layout()
    out = f"{FIGURES}/category_composition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 4. PLOT: SHIFT BARS ──────────────────────────────────────────────────────

def plot_shift(summ: pd.DataFrame):
    show = summ[~summ["category"].isin(["other"])].sort_values("shift_pp")
    colors = [CAT_COLORS.get(c, "#aaaaaa") for c in show["category"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(show["label"], show["shift_pp"],
                   color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage-point shift (Post − Pre)", fontsize=10)
    ax.set_title(
        "Category-Level Shift After Musk Acquisition\n"
        "(positive = more prevalent after Oct 27, 2022)",
        fontsize=12, fontweight="bold")

    for bar, val in zip(bars, show["shift_pp"]):
        if abs(val) > 0.02:
            ax.text(val + (0.02 if val >= 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}pp", va="center",
                    ha="left" if val >= 0 else "right", fontsize=7.5)
    fig.tight_layout()
    out = f"{FIGURES}/category_shift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 5. PLOT: MONTHLY TIMESERIES ──────────────────────────────────────────────

def plot_timeseries(df: pd.DataFrame):
    df2 = df.copy()
    df2["month"] = df2["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = (df2.groupby(["month", "category"])
                  .size().unstack(fill_value=0))
    for cat in ALL_CATS:
        if cat not in monthly:
            monthly[cat] = 0
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    # Show interesting non-filler categories
    show = [c for c in ALL_CATS
            if c not in ("social_filler", "other", "holidays")]

    n = len(show)
    fig, axes = plt.subplots(n, 1, figsize=(13, n * 1.5), sharex=True)

    for ax, cat in zip(axes, show):
        color = CAT_COLORS.get(cat, "#aaaaaa")
        ax.fill_between(monthly_pct.index, monthly_pct[cat],
                        color=color, alpha=0.35)
        ax.plot(monthly_pct.index, monthly_pct[cat],
                color=color, linewidth=1.4)
        ax.axvline(TREATMENT, color="black", linewidth=1.0,
                   linestyle="--", alpha=0.65)
        ax.set_ylabel("%", fontsize=7)
        ax.set_title(DISPLAY_GROUPS.get(cat, cat),
                     fontsize=8, fontweight="bold", loc="left", pad=2)
        ax.tick_params(labelsize=6)
        ax.set_ylim(bottom=0)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "Monthly Share of Trending Topic Categories\n"
        "Dashed line = Oct 27, 2022 (Musk acquisition)",
        fontsize=12, fontweight="bold", y=1.002)
    fig.tight_layout()
    out = f"{FIGURES}/category_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 6. PLOT: BRO SHIFT ───────────────────────────────────────────────────────

def plot_bro_shift(df: pd.DataFrame):
    """
    Aggregate into Male-coded / Female-coded / Neutral buckets monthly.
    Shows gender lean of trending content over time.
    """
    df2 = df.copy()
    df2["month"] = df2["Date"].dt.to_period("M").dt.to_timestamp()

    def gender_bucket(cat):
        if cat in MALE_CODED:   return "male_coded"
        if cat in FEMALE_CODED: return "female_coded"
        return "neutral"

    df2["gender"] = df2["category"].apply(gender_bucket)

    monthly = (df2.groupby(["month", "gender"])
                  .size().unstack(fill_value=0))
    for g in ["male_coded", "female_coded", "neutral"]:
        if g not in monthly:
            monthly[g] = 0
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Top: stacked area
    ax = axes[0]
    ax.stackplot(monthly_pct.index,
                 monthly_pct["male_coded"],
                 monthly_pct["female_coded"],
                 monthly_pct["neutral"],
                 labels=["Male-coded\n(sports, wrestling, UFC, tech, manosphere)",
                         "Female-coded\n(reality TV, Taylor Swift, K-pop, women's sports)",
                         "Neutral\n(news, entertainment, holidays, filler)"],
                 colors=["#4e79a7", "#e15759", "#bab0ac"],
                 alpha=0.8)
    ax.axvline(TREATMENT, color="black", linewidth=1.2,
               linestyle="--", label="Oct 27, 2022")
    ax.set_ylabel("Share of topics (%)", fontsize=9)
    ax.set_title("Gender Lean of Twitter Trending Topics Over Time",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9,
              bbox_to_anchor=(1.01, 1.0))

    # Bottom: male - female gap (rolling 3-month)
    ax2 = axes[1]
    gap = monthly_pct["male_coded"] - monthly_pct["female_coded"]
    gap_roll = gap.rolling(3, center=True, min_periods=1).mean()
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax2.fill_between(gap_roll.index, gap_roll,
                     where=(gap_roll >= 0),
                     color="#4e79a7", alpha=0.5, label="Male > Female")
    ax2.fill_between(gap_roll.index, gap_roll,
                     where=(gap_roll < 0),
                     color="#e15759", alpha=0.5, label="Female > Male")
    ax2.plot(gap_roll.index, gap_roll, color="black", linewidth=1.2)
    ax2.axvline(TREATMENT, color="black", linewidth=1.2,
                linestyle="--", alpha=0.7)
    ax2.set_ylabel("Male% − Female% (3-mo avg)", fontsize=9)
    ax2.set_title("Male-coded minus Female-coded Share (Gap)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    out = f"{FIGURES}/bro_shift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df   = load_and_classify()
    summ = build_summary(df)
    print_summary(summ)

    summ.to_csv("out/category_counts.csv", index=False)
    print("\nSaved: out/category_counts.csv")

    print("\nGenerating plots …")
    plot_stacked_bars(summ)
    plot_shift(summ)
    plot_timeseries(df)
    plot_bro_shift(df)
    print("\nDone.")
