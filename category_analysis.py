"""
Categorical composition analysis of Twitter trending topics
Pre vs. Post Musk acquisition (Oct 27, 2022)

Uses full 4-year window: Oct 27 2020 – Oct 27 2024
Outputs:
  out/figures/category_composition.png   — stacked % bar (pre vs post)
  out/figures/category_timeseries.png    — monthly category share over time
  out/figures/category_shift.png        — net shift (post - pre) per category
  out/category_counts.csv               — raw counts for every category x period
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from category_lexicon import classify_category

TREATMENT = pd.Timestamp("2022-10-27")
DATA_FILE = "out/twitter_1774668850.csv"
FIGURES   = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Display names & colour palette
CAT_META = {
    "wrestling":     {"label": "Wrestling (WWE/AEW)",  "color": "#d62728"},
    "sports":        {"label": "Sports (NBA/NFL/etc)", "color": "#1f77b4"},
    "combat_sports": {"label": "Combat Sports (UFC/Boxing)", "color": "#ff7f0e"},
    "entertainment": {"label": "Entertainment (TV/Music/Film)", "color": "#9467bd"},
    "fandom":        {"label": "Fandom (Anime/K-pop)", "color": "#e377c2"},
    "tech_gaming":   {"label": "Tech & Gaming",        "color": "#17becf"},
    "manosphere":    {"label": "Manosphere-adjacent",  "color": "#8c564b"},
    "news_events":   {"label": "News & Events",        "color": "#7f7f7f"},
    "social_filler": {"label": "Social Filler",        "color": "#bcbd22"},
    "other":         {"label": "Other / Uncategorised","color": "#aec7e8"},
}
CAT_ORDER = list(CAT_META.keys())


# ── 1. LOAD & CLASSIFY ────────────────────────────────────────────────────────

def load_and_classify():
    print("Loading Twitter data and classifying topics …")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    print(f"  {len(df):,} unique date×topic pairs loaded")

    df["category"] = df["Topic"].apply(classify_category)
    df["post"] = (df["Date"] >= TREATMENT).astype(int)
    df["period"] = df["post"].map({0: "Pre (before Oct 27 2022)",
                                   1: "Post (after Oct 27 2022)"})
    return df


# ── 2. SUMMARY TABLE ──────────────────────────────────────────────────────────

def build_summary(df):
    pre  = df[df["post"] == 0]
    post = df[df["post"] == 1]

    rows = []
    for cat in CAT_ORDER:
        n_pre  = (pre["category"]  == cat).sum()
        n_post = (post["category"] == cat).sum()
        pct_pre  = n_pre  / len(pre)  * 100
        pct_post = n_post / len(post) * 100
        rows.append({
            "category": cat,
            "label":    CAT_META[cat]["label"],
            "n_pre":    n_pre,  "pct_pre":  pct_pre,
            "n_post":   n_post, "pct_post": pct_post,
            "shift_pp": pct_post - pct_pre,
        })
    return pd.DataFrame(rows)


def print_summary(summ):
    print("\n" + "="*78)
    print("CATEGORY COMPOSITION  — Pre vs Post Musk Acquisition")
    print("="*78)
    hdr = f"{'Category':<32} {'Pre %':>7} {'Post %':>7} {'Shift':>7}  {'Pre n':>7} {'Post n':>7}"
    print(hdr)
    print("-"*78)
    for _, r in summ.iterrows():
        flag = " <--" if abs(r["shift_pp"]) >= 1.0 else ""
        print(f"{r['label']:<32} {r['pct_pre']:>7.2f} {r['pct_post']:>7.2f} "
              f"{r['shift_pp']:>+7.2f}pp  {int(r['n_pre']):>7,} {int(r['n_post']):>7,}{flag}")
    print("="*78)


# ── 3. PLOTS ──────────────────────────────────────────────────────────────────

def plot_stacked_bars(summ):
    """Side-by-side 100% stacked bar: Pre | Post"""
    fig, ax = plt.subplots(figsize=(10, 7))

    periods = ["Pre (before Oct 27 2022)", "Post (after Oct 27 2022)"]
    pcts = {
        "Pre (before Oct 27 2022)":  summ["pct_pre"].values,
        "Post (after Oct 27 2022)":  summ["pct_post"].values,
    }

    x = np.arange(len(periods))
    bottoms = {p: np.zeros(len(periods)) for p in periods}
    bottoms_arr = np.zeros(2)

    bars = []
    for i, cat in enumerate(CAT_ORDER):
        vals = np.array([summ.loc[summ["category"]==cat, "pct_pre"].values[0],
                         summ.loc[summ["category"]==cat, "pct_post"].values[0]])
        b = ax.bar(x, vals, bottom=bottoms_arr,
                   color=CAT_META[cat]["color"],
                   label=CAT_META[cat]["label"],
                   edgecolor="white", linewidth=0.4)
        bars.append(b)
        bottoms_arr += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-acquisition\n(Oct 2020 – Oct 2022)",
                         "Post-acquisition\n(Oct 2022 – Oct 2024)"],
                        fontsize=11)
    ax.set_ylabel("Share of trending topics (%)", fontsize=10)
    ax.set_ylim(0, 103)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Twitter Trending Topic Composition\nPre vs. Post Musk Acquisition",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
              bbox_to_anchor=(1.38, 1.0))

    fig.tight_layout()
    out = f"{FIGURES}/category_composition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_shift(summ):
    """Horizontal bar: net shift in percentage points per category"""
    s = summ.sort_values("shift_pp")
    colors = [CAT_META[c]["color"] for c in s["category"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(s["label"], s["shift_pp"], color=colors,
                   edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Percentage-point shift (Post − Pre)", fontsize=10)
    ax.set_title("Category-Level Shift Post Musk Acquisition\n"
                 "(positive = more prevalent after Oct 27, 2022)",
                 fontsize=12, fontweight="bold")

    for bar, val in zip(bars, s["shift_pp"]):
        ax.text(val + (0.05 if val >= 0 else -0.05),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}pp", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)

    fig.tight_layout()
    out = f"{FIGURES}/category_shift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_timeseries(df):
    """Monthly share of each category over time"""
    df2 = df.copy()
    df2["month"] = df2["Date"].dt.to_period("M").dt.to_timestamp()

    monthly = (df2.groupby(["month", "category"])
                  .size().unstack(fill_value=0))
    for cat in CAT_ORDER:
        if cat not in monthly: monthly[cat] = 0
    monthly = monthly[CAT_ORDER]
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

    # Show only the more interesting categories (drop social_filler & other)
    show = [c for c in CAT_ORDER if c not in ("social_filler", "other")]

    fig, axes = plt.subplots(len(show), 1, figsize=(13, 14), sharex=True)

    for ax, cat in zip(axes, show):
        ax.fill_between(monthly_pct.index,
                         monthly_pct[cat],
                         color=CAT_META[cat]["color"], alpha=0.4)
        ax.plot(monthly_pct.index, monthly_pct[cat],
                color=CAT_META[cat]["color"], linewidth=1.5)
        ax.axvline(TREATMENT, color="black", linewidth=1.0,
                   linestyle="--", alpha=0.7)
        ax.set_ylabel("%", fontsize=8)
        ax.set_title(CAT_META[cat]["label"], fontsize=9, fontweight="bold",
                     loc="left", pad=2)
        ax.tick_params(labelsize=7)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.suptitle("Monthly Share of Each Category in Twitter Trending Topics\n"
                 "Dashed line = Oct 27, 2022 (Musk acquisition)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = f"{FIGURES}/category_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 4. MAIN ───────────────────────────────────────────────────────────────────

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
    print("\nDone.")
