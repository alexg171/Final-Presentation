"""
Category-level DiD: Twitter trending share vs Reddit post volume

Design
------
For each Twitter content category:
  - Treated unit : Twitter — daily share of trending topics in that category
  - Control unit : Reddit  — daily post volume in the MATCHED category subreddit(s)
                             from reddit_category.tsv, normalized to pre-treatment baseline
                             Falls back to generic political subreddit baseline for
                             categories with no matched subreddit.

Both series are expressed as % deviation from their own pre-treatment mean:
  deviation_it = (y_it - mean_i_pre) / mean_i_pre * 100

DiD model (HC3 robust SEs):
  deviation_it = α + β1·twitter_i + β2·post_t + β3·(twitter_i × post_t) + ε_it

  β3 = "did Twitter's category share deviate from its own baseline MORE than
        Reddit's organic activity deviated from its baseline after Oct 27, 2022?"

Positive β3  → Twitter algorithmically amplified this category beyond organic interest
Negative β3  → Twitter suppressed this category below what organic interest would predict

Output
------
  out/category_did_results.csv
  out/figures/category_did.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from category_lexicon import classify_category, DISPLAY_GROUPS, CAT_COLORS

TREATMENT  = pd.Timestamp("2022-10-27")
PRE_START  = pd.Timestamp("2020-10-27")
POST_END   = pd.Timestamp("2024-10-27")
FIGURES    = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

# Categories to test (skip tiny ones)
TEST_CATS = [
    "wrestling", "combat_sports", "sports_nba", "sports_nfl",
    "sports_mlb", "sports_nhl", "sports_soccer", "sports_college",
    "sports_womens", "sports_other", "reality_tv", "entertainment",
    "taylor_swift", "fandom", "tech_gaming", "lgbtq_social",
    "religious", "musk_twitter", "politics", "news_events",
]


# ── 1. TWITTER: daily category shares ────────────────────────────────────────

def build_twitter_panel() -> pd.DataFrame:
    print("Building Twitter daily category panel …")
    df = pd.read_csv("out/twitter_trending_4yr.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df = df[(df["Date"] >= PRE_START) & (df["Date"] <= POST_END)]
    df["category"] = df["Topic"].apply(classify_category)

    # Daily topic counts per category
    daily_cat = (df.groupby([df["Date"].dt.date, "category"])
                   .size().unstack(fill_value=0).rename_axis("date"))
    daily_cat.index = pd.to_datetime(daily_cat.index)
    daily_total = daily_cat.sum(axis=1)

    for cat in TEST_CATS:
        if cat not in daily_cat:
            daily_cat[cat] = 0

    # Share = category count / total topics that day
    share = daily_cat[TEST_CATS].div(daily_total, axis=0) * 100
    share = share.reindex(pd.date_range(PRE_START, POST_END, freq="D"), fill_value=np.nan)
    return share


# Matched subreddit(s) per Twitter category
CAT_TO_SUBREDDIT = {
    "wrestling":      ["SquaredCircle"],
    "combat_sports":  ["MMA"],
    "sports_nba":     ["nba"],
    "sports_nfl":     ["nfl"],
    "sports_mlb":     ["baseball"],
    "sports_nhl":     ["hockey"],
    "sports_soccer":  ["soccer"],
    "sports_college": ["CFB"],
    "reality_tv":     ["BravoRealHousewives", "LoveIslandTV", "thebachelor",
                       "survivor", "BigBrother", "Vanderpumprules",
                       "MAFS_TV", "90DayFiance"],
    "entertainment":  ["television"],
    "taylor_swift":   ["TaylorSwift"],
    "fandom":         ["anime", "kpop"],
    "tech_gaming":    ["gaming"],
    "lgbtq_social":   ["lgbt"],
    "news_events":    ["worldnews"],
    "musk_twitter":   ["worldnews"],
    "sports_womens":  ["wnba", "NWSL"],
    # No matched sub — will fall back to generic political baseline:
    # sports_other, manosphere, religious, true_crime, politics
}


# ── 2. REDDIT: per-category and generic baselines ────────────────────────────

def build_reddit_controls() -> dict:
    """
    Returns a dict: category -> daily n_posts pd.Series (full 4yr date range).
    Uses matched subreddits where available; falls back to generic political baseline.
    """
    date_idx = pd.date_range(PRE_START, POST_END, freq="D")

    # Load matched category subreddits
    cat_path = "out/reddit_category.tsv"
    cat_controls = {}
    if os.path.exists(cat_path):
        rc = pd.read_csv(cat_path, sep="\t", parse_dates=["date"])
        rc = rc[(rc["date"] >= PRE_START) & (rc["date"] <= POST_END)]
        for cat, subs in CAT_TO_SUBREDDIT.items():
            sub_data = rc[rc["subreddit"].isin(subs)]
            if sub_data.empty:
                continue
            daily = (sub_data.groupby(sub_data["date"].dt.date)["n_posts"]
                     .sum().rename_axis("date"))
            daily.index = pd.to_datetime(daily.index)
            daily = daily.reindex(date_idx, fill_value=0).astype(float)
            cat_controls[cat] = daily

    # Generic political baseline (fallback)
    print("Building Reddit controls …")
    rd = pd.read_csv("out/reddit_trending.tsv", sep="\t", parse_dates=["date"])
    rd = rd[(rd["date"] >= PRE_START) & (rd["date"] <= POST_END)]
    generic = rd.groupby(rd["date"].dt.date).size().rename_axis("date")
    generic.index = pd.to_datetime(generic.index)
    generic = generic.reindex(date_idx, fill_value=0).astype(float)

    matched = len(cat_controls)
    total   = len(TEST_CATS)
    print(f"  Matched subreddits: {matched}/{total} categories")
    print(f"  Fallback (generic political): {total - matched} categories")

    return cat_controls, generic


# ── 3. NORMALIZE to % deviation from pre-treatment mean ──────────────────────

def normalize(series: pd.Series, pre_mask: np.ndarray) -> pd.Series:
    pre_mean = series[pre_mask].mean()
    if pre_mean == 0 or np.isnan(pre_mean):
        return pd.Series(np.nan, index=series.index)
    return (series - pre_mean) / pre_mean * 100


# ── 4. RUN DiD PER CATEGORY ──────────────────────────────────────────────────

def run_category_did(tw_share: pd.DataFrame,
                     cat_controls: dict,
                     generic_baseline: pd.Series) -> pd.DataFrame:

    all_dates = tw_share.index
    pre_mask  = np.array(all_dates < TREATMENT)
    post_mask = np.array(all_dates >= TREATMENT)

    generic_norm = normalize(generic_baseline, pre_mask)

    results = []
    for cat in TEST_CATS:
        tw_series = tw_share[cat]
        tw_norm   = normalize(tw_series, pre_mask)

        # Use matched subreddit if available, else generic political baseline
        if cat in cat_controls:
            rd_norm   = normalize(cat_controls[cat], pre_mask)
            ctrl_label = CAT_TO_SUBREDDIT[cat]
        else:
            rd_norm   = generic_norm
            ctrl_label = ["generic_political"]

        # Stack into panel: one row per (date, unit)
        tw_df = pd.DataFrame({
            "date":      all_dates,
            "deviation": tw_norm.values,
            "twitter":   1,
            "post":      post_mask.astype(int),
        })
        rd_df = pd.DataFrame({
            "date":      all_dates,
            "deviation": rd_norm.values,
            "twitter":   0,
            "post":      post_mask.astype(int),
        })
        panel = pd.concat([tw_df, rd_df], ignore_index=True)
        panel["did"] = panel["twitter"] * panel["post"]
        panel = panel.dropna(subset=["deviation"])

        if panel["deviation"].std() == 0 or len(panel) < 50:
            continue

        try:
            model = smf.ols(
                "deviation ~ twitter + post + did", data=panel
            ).fit(cov_type="HC3")

            coef  = model.params["did"]
            pval  = model.pvalues["did"]
            ci_lo, ci_hi = model.conf_int().loc["did"]

            # Raw pre/post means
            tw_pre  = tw_norm[pre_mask].mean()
            tw_post = tw_norm[post_mask].mean()
            rd_pre  = rd_norm[pre_mask].mean()
            rd_post = rd_norm[post_mask].mean()

            results.append({
                "category":    cat,
                "label":       DISPLAY_GROUPS.get(cat, cat),
                "control":     "+".join(ctrl_label),
                "tw_pre_dev":  tw_pre,
                "tw_post_dev": tw_post,
                "rd_pre_dev":  rd_pre,
                "rd_post_dev": rd_post,
                "did_coef":    coef,
                "ci_lo":       ci_lo,
                "ci_hi":       ci_hi,
                "pval":        pval,
                "stars":       _stars(pval),
                "n_obs":       len(panel),
            })
        except Exception as e:
            print(f"  [{cat}] model failed: {e}")

    return pd.DataFrame(results)


def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "n.s."


# ── 5. PRINT TABLE ───────────────────────────────────────────────────────────

def print_results(res: pd.DataFrame):
    print("\n" + "="*90)
    print("CATEGORY DiD RESULTS")
    print("b3 = Twitter category deviation ABOVE/BELOW what Reddit organic baseline predicts")
    print("Positive b3 = Twitter algorithmically amplified; Negative = suppressed")
    print("="*90)
    hdr = (f"{'Category':<36} {'TW pre':>7} {'TW post':>7} "
           f"{'DiD b3':>9} {'p':>7}      {'95% CI'}")
    print(hdr)
    print("-"*90)
    for _, r in res.sort_values("did_coef", ascending=False).iterrows():
        ci = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
        print(f"{r['label']:<36} {r['tw_pre_dev']:>+7.1f} {r['tw_post_dev']:>+7.1f} "
              f"{r['did_coef']:>+9.2f} {r['pval']:>7.3f}{r['stars']:>5}   {ci}")
    print("="*90)
    print("Note: deviations expressed as % from own pre-treatment mean.")
    print("      Reddit control = matched category subreddit (or generic political fallback).")


# ── 6. PLOT ──────────────────────────────────────────────────────────────────

def plot_results(res: pd.DataFrame):
    df = res.sort_values("did_coef")
    colors = [CAT_COLORS.get(c, "#aaaaaa") for c in df["category"]]
    sig    = df["pval"] < 0.10

    fig, ax = plt.subplots(figsize=(11, 8))

    # Error bars
    xerr = np.array([
        df["did_coef"] - df["ci_lo"],
        df["ci_hi"]    - df["did_coef"]
    ])
    bars = ax.barh(df["label"], df["did_coef"],
                   color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.4)
    ax.errorbar(df["did_coef"], df["label"],
                xerr=xerr, fmt="none",
                color="black", linewidth=1.0, capsize=3)

    # Star labels
    for _, row in df.iterrows():
        if row["stars"] != "n.s.":
            ax.text(row["ci_hi"] + 1, row["label"],
                    row["stars"], va="center", fontsize=9,
                    color="black", fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel(
        "DiD coefficient β3 (% deviation above/below Reddit baseline)",
        fontsize=10)
    ax.set_title(
        "Category DiD: Did Twitter Amplify or Suppress Each Content Type?\n"
        "Positive = amplified beyond organic interest  |  Negative = suppressed\n"
        "(Matched Reddit subreddit as per-category control)",
        fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = f"{FIGURES}/category_did.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 7. TIMESERIES PLOT: Twitter vs Reddit deviations per category ─────────────

def plot_category_timeseries(tw_share: pd.DataFrame,
                              rd_baseline: pd.Series,
                              res: pd.DataFrame):
    """
    For the top 6 most interesting categories (by abs DiD coef),
    show Twitter deviation vs Reddit baseline deviation over time.
    """
    all_dates = tw_share.index
    pre_mask  = np.array(all_dates < TREATMENT)

    rd_norm = normalize(rd_baseline, pre_mask)
    rd_roll = rd_norm.rolling(30, center=True, min_periods=5).mean()

    top_cats = res.reindex(
        res["did_coef"].abs().sort_values(ascending=False).index
    ).head(6)["category"].tolist()

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharey=False)
    axes = axes.flatten()

    for ax, cat in zip(axes, top_cats):
        tw_norm = normalize(tw_share[cat], pre_mask)
        tw_roll = tw_norm.rolling(30, center=True, min_periods=5).mean()

        color = CAT_COLORS.get(cat, "#4e79a7")

        ax.plot(all_dates, tw_roll, color=color, linewidth=2,
                label="Twitter (treated)")
        ax.plot(all_dates, rd_roll, color="grey", linewidth=1.5,
                linestyle="--", label="Reddit baseline (control)")
        ax.axvline(TREATMENT, color="black", linewidth=1.0,
                   linestyle=":", alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.fill_between(all_dates,
                         tw_roll.fillna(0), rd_roll.fillna(0),
                         where=(tw_roll.fillna(0) > rd_roll.fillna(0)),
                         alpha=0.15, color=color)
        ax.fill_between(all_dates,
                         tw_roll.fillna(0), rd_roll.fillna(0),
                         where=(tw_roll.fillna(0) <= rd_roll.fillna(0)),
                         alpha=0.15, color="red")

        row = res[res["category"] == cat].iloc[0]
        ax.set_title(
            f"{DISPLAY_GROUPS.get(cat, cat)}\n"
            f"DiD β3 = {row['did_coef']:+.1f}pp  {row['stars']}",
            fontsize=9, fontweight="bold")
        ax.set_ylabel("% dev. from pre mean", fontsize=8)
        ax.tick_params(labelsize=7)
        if cat == top_cats[0]:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Twitter vs Reddit Deviation from Pre-Treatment Baseline\n"
        "30-day rolling average | Blue shading = Twitter above Reddit | "
        "Red shading = Twitter below Reddit",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = f"{FIGURES}/category_did_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tw_share                    = build_twitter_panel()
    cat_controls, generic_baseline = build_reddit_controls()

    res = run_category_did(tw_share, cat_controls, generic_baseline)
    print_results(res)

    res.to_csv("out/category_did_results.csv", index=False)
    print("\nSaved: out/category_did_results.csv")

    plot_results(res)
    # Pass generic baseline for the timeseries overlay (consistent visual reference)
    plot_category_timeseries(tw_share, generic_baseline, res)
    print("\nDone.")
