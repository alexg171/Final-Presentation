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
from category_subreddit_mapping import CAT_TO_SUBREDDIT

TREATMENT = pd.Timestamp("2022-10-27")
PRE_START = pd.Timestamp("2020-10-27")
POST_END = pd.Timestamp("2024-10-27")
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

# Categories to test
# Dropped: lgbtq_social (event-driven spikes, no common rhythm with control)
#          religious   (no matched subreddit, generic fallback, not a headline result)
# "sports_other", "tech_gaming",
TEST_CATS = [
    "wrestling",
    "combat_sports",
    "sports_nba",
    "sports_nfl",
    "sports_mlb",
    "sports_nhl",
    "sports_soccer",
    "sports_college",
    "sports_womens",
    "reality_tv",
    "entertainment",
    "taylor_swift",
    "fandom",
    "news_politics",
]

# Merged categories: combine Twitter shares and Reddit controls before DiD
MERGED_CATS = {
    "news_politics": {
        "sources": ["news_events", "politics"],
        "label": "News & Politics",
        "controls": [
            "worldnews",
            "news",
            "politics",
            "PoliticalDiscussion",
            "conservative",
            "republican",
            "democrats",
            "Liberal",
            "progressive",
            "libertarian",
            "NeutralPolitics",
        ],
    },
}


# ── 1. TWITTER: daily category shares ────────────────────────────────────────


def build_twitter_panel() -> pd.DataFrame:
    print("Building Twitter daily category panel …")
    df = pd.read_csv("out/twitter_trending_4yr.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df = df[(df["Date"] >= PRE_START) & (df["Date"] <= POST_END)]
    df["category"] = df["Topic"].apply(classify_category)

    # Daily topic counts per category
    daily_cat = (
        df.groupby([df["Date"].dt.date, "category"])
        .size()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    daily_cat.index = pd.to_datetime(daily_cat.index)
    daily_total = daily_cat.sum(axis=1)

    for cat in TEST_CATS:
        # For merged cats, sum their source categories
        if cat in MERGED_CATS:
            sources = MERGED_CATS[cat]["sources"]
            daily_cat[cat] = sum(daily_cat[s] if s in daily_cat else 0 for s in sources)
        elif cat not in daily_cat:
            daily_cat[cat] = 0

    # Share = category count / total topics that day
    share = daily_cat[TEST_CATS].div(daily_total, axis=0) * 100
    share = share.reindex(
        pd.date_range(PRE_START, POST_END, freq="D"), fill_value=np.nan
    )
    return share


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
        # Add merged category controls
        for cat, info in MERGED_CATS.items():
            subs = info["controls"]
            sub_data = rc[rc["subreddit"].isin(subs)]
            if sub_data.empty:
                continue
            daily = (
                sub_data.groupby(sub_data["date"].dt.date)["n_posts"]
                .sum()
                .rename_axis("date")
            )
            daily.index = pd.to_datetime(daily.index)
            daily = daily.reindex(date_idx, fill_value=0).astype(float)
            cat_controls[cat] = daily

        for cat, subs in CAT_TO_SUBREDDIT.items():
            sub_data = rc[rc["subreddit"].isin(subs)]
            if sub_data.empty:
                continue
            daily = (
                sub_data.groupby(sub_data["date"].dt.date)["n_posts"]
                .sum()
                .rename_axis("date")
            )
            daily.index = pd.to_datetime(daily.index)
            daily = daily.reindex(date_idx, fill_value=0).astype(float)
            cat_controls[cat] = daily

    matched = len(cat_controls)
    total = len(TEST_CATS)
    print(f"  Matched subreddits: {matched}/{total} categories")

    return cat_controls


# ── 3. LOG-DEVIATION NORMALIZATION ───────────────────────────────────────────
# Log-transform each series, then demean by its own pre-treatment log-mean.
# Result: both Twitter and Reddit start at 0 and are expressed in log-deviation
# units (≈ % change for small values, better behaved for large swings).
# This keeps both series on the same scale before stacking into the DiD panel.


def log_normalize(series: pd.Series, pre_mask: np.ndarray) -> pd.Series:
    """log(y + small_constant) demeaned by pre-period log mean."""
    # Choose constant relative to series magnitude
    const = 1e-4 if series.max() <= 1 else 1.0  # share vs count
    log_s = np.log(series.clip(lower=0) + const)
    log_pre_mean = log_s[pre_mask].mean()
    if np.isnan(log_pre_mean):
        return pd.Series(np.nan, index=series.index)
    return log_s - log_pre_mean


# ── 4. RUN DiD PER CATEGORY ──────────────────────────────────────────────────


def run_category_did(
    tw_share: pd.DataFrame, cat_controls: dict, generic_baseline: pd.Series
) -> pd.DataFrame:

    all_dates = tw_share.index
    pre_mask = np.array(all_dates < TREATMENT)
    post_mask = np.array(all_dates >= TREATMENT)

    generic_lnorm = log_normalize(generic_baseline, pre_mask)

    results = []
    for cat in TEST_CATS:
        tw_lnorm = log_normalize(tw_share[cat], pre_mask)

        # Use matched subreddit if available, else generic political baseline
        if cat in cat_controls:
            rd_lnorm = log_normalize(cat_controls[cat], pre_mask)
            ctrl_label = (
                MERGED_CATS[cat]["controls"]
                if cat in MERGED_CATS
                else CAT_TO_SUBREDDIT.get(cat, ["unknown"])
            )
        else:
            rd_lnorm = generic_lnorm
            ctrl_label = ["generic_political"]

        # Stack into panel: one row per (date, unit)
        tw_df = pd.DataFrame(
            {
                "date": all_dates,
                "log_dev": tw_lnorm.values,
                "twitter": 1,
                "post": post_mask.astype(int),
            }
        )
        rd_df = pd.DataFrame(
            {
                "date": all_dates,
                "log_dev": rd_lnorm.values,
                "twitter": 0,
                "post": post_mask.astype(int),
            }
        )
        panel = pd.concat([tw_df, rd_df], ignore_index=True)
        panel["did"] = panel["twitter"] * panel["post"]
        panel = panel.dropna(subset=["log_dev"])

        if panel["log_dev"].std() == 0 or len(panel) < 50:
            continue

        try:
            model = smf.ols("log_dev ~ twitter + post + did", data=panel).fit(
                cov_type="HC3"
            )

            coef = model.params["did"]
            pval = model.pvalues["did"]
            ci_lo, ci_hi = model.conf_int().loc["did"]

            # Pre/post log-deviation means per platform
            tw_pre = tw_lnorm[pre_mask].mean()
            tw_post = tw_lnorm[post_mask].mean()
            rd_pre = rd_lnorm[pre_mask].mean()
            rd_post = rd_lnorm[post_mask].mean()

            results.append(
                {
                    "category": cat,
                    "label": (
                        MERGED_CATS[cat]["label"]
                        if cat in MERGED_CATS
                        else DISPLAY_GROUPS.get(cat, cat)
                    ),
                    "control": "+".join(ctrl_label),
                    "tw_pre_ldev": tw_pre,
                    "tw_post_ldev": tw_post,
                    "rd_pre_ldev": rd_pre,
                    "rd_post_ldev": rd_post,
                    "did_coef": coef,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "pval": pval,
                    "stars": _stars(pval),
                    "n_obs": len(panel),
                }
            )
        except Exception as e:
            print(f"  [{cat}] model failed: {e}")

    return pd.DataFrame(results)


def _stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "n.s."


# ── 5. PRINT TABLE ───────────────────────────────────────────────────────────


def print_results(res: pd.DataFrame):
    print("\n" + "=" * 90)
    print("CATEGORY DiD RESULTS  (log-deviation model)")
    print("outcome = log(y) − log_pre_mean  →  both series start at 0, same scale")
    print(
        "b3 = log-deviation DiD ≈ % effect; positive = amplified, negative = suppressed"
    )
    print("=" * 90)
    hdr = (
        f"{'Category':<36} {'TW pre':>8} {'TW post':>8} "
        f"{'DiD b3':>9} {'p':>7}      {'95% CI'}"
    )
    print(hdr)
    print("-" * 90)
    for _, r in res.sort_values("did_coef", ascending=False).iterrows():
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(
            f"{r['label']:<36} {r['tw_pre_ldev']:>+8.3f} {r['tw_post_ldev']:>+8.3f} "
            f"{r['did_coef']:>+9.3f} {r['pval']:>7.3f}{r['stars']:>5}   {ci}"
        )
    print("=" * 90)
    print(
        "Note: outcome = log(y + c) − pre-period log mean (c=1e-4 for shares, 1 for counts)."
    )
    print("      β3 ≈ log-point change; approx % for small values.")
    print("      Reddit control = matched category subreddit (or generic fallback).")


# ── 6. PLOT ──────────────────────────────────────────────────────────────────


def plot_results(res: pd.DataFrame):
    df = res.sort_values("did_coef")
    colors = [CAT_COLORS.get(c, "#aaaaaa") for c in df["category"]]
    sig = df["pval"] < 0.10

    fig, ax = plt.subplots(figsize=(11, 8))

    # Error bars
    xerr = np.array([df["did_coef"] - df["ci_lo"], df["ci_hi"] - df["did_coef"]])
    bars = ax.barh(
        df["label"],
        df["did_coef"],
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.errorbar(
        df["did_coef"],
        df["label"],
        xerr=xerr,
        fmt="none",
        color="black",
        linewidth=1.0,
        capsize=3,
    )

    # Star labels
    for _, row in df.iterrows():
        if row["stars"] != "n.s.":
            ax.text(
                row["ci_hi"] + 1,
                row["label"],
                row["stars"],
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
            )

    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel(
        "DiD coefficient β₃  (log-points; ≈ % change for small values)", fontsize=10
    )
    ax.set_title(
        "Category DiD: Did Twitter Amplify or Suppress Each Content Type?\n"
        "Outcome = log(share)  |  Control = log(Reddit posts)  |  HC3 robust SEs\n"
        "Positive = amplified beyond organic interest  |  Negative = suppressed",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout()
    out = f"{FIGURES}/category_did.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 7. TIMESERIES PLOT: Twitter vs Reddit deviations per category ─────────────


def plot_category_timeseries(
    tw_share: pd.DataFrame, rd_baseline: pd.Series, res: pd.DataFrame
):
    """
    For the top 6 most interesting categories (by abs DiD coef),
    show Twitter deviation vs Reddit baseline deviation over time.
    """
    all_dates = tw_share.index
    pre_mask = np.array(all_dates < TREATMENT)

    rd_norm = log_normalize(rd_baseline, pre_mask)
    rd_roll = rd_norm.rolling(30, center=True, min_periods=5).mean()

    top_cats = (
        res.reindex(res["did_coef"].abs().sort_values(ascending=False).index)
        .head(6)["category"]
        .tolist()
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharey=False)
    axes = axes.flatten()

    for ax, cat in zip(axes, top_cats):
        tw_norm = log_normalize(tw_share[cat], pre_mask)
        tw_roll = tw_norm.rolling(30, center=True, min_periods=5).mean()

        color = CAT_COLORS.get(cat, "#4e79a7")

        ax.plot(all_dates, tw_roll, color=color, linewidth=2, label="Twitter (treated)")
        ax.plot(
            all_dates,
            rd_roll,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            label="Reddit baseline (control)",
        )
        ax.axvline(TREATMENT, color="black", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.fill_between(
            all_dates,
            tw_roll.fillna(0),
            rd_roll.fillna(0),
            where=(tw_roll.fillna(0) > rd_roll.fillna(0)),
            alpha=0.15,
            color=color,
        )
        ax.fill_between(
            all_dates,
            tw_roll.fillna(0),
            rd_roll.fillna(0),
            where=(tw_roll.fillna(0) <= rd_roll.fillna(0)),
            alpha=0.15,
            color="red",
        )

        row = res[res["category"] == cat].iloc[0]
        ax.set_title(
            f"{DISPLAY_GROUPS.get(cat, cat)}\n"
            f"DiD β3 = {row['did_coef']:+.1f}pp  {row['stars']}",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_ylabel("% dev. from pre mean", fontsize=8)
        ax.tick_params(labelsize=7)
        if cat == top_cats[0]:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Twitter vs Reddit Deviation from Pre-Treatment Baseline\n"
        "30-day rolling average | Blue shading = Twitter above Reddit | "
        "Red shading = Twitter below Reddit",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    out = f"{FIGURES}/category_did_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── EXPORT STATA PANEL ───────────────────────────────────────────────────────


def export_stata_panel(
    tw_share: pd.DataFrame, cat_controls: dict, generic_baseline: pd.Series
):
    """
    Export the stacked DiD panel to out/stata_panel.csv for Stata analysis.
    One row per (date, category, platform).
    Columns: date, category, twitter, post, log_dev, did, quarter
    """
    all_dates = tw_share.index
    pre_mask = np.array(all_dates < TREATMENT)
    post_mask = np.array(all_dates >= TREATMENT)

    def quarter_bin(d):
        delta = (pd.Timestamp(d) - TREATMENT).days
        return int(np.floor(delta / 91.25))

    quarters = np.array([quarter_bin(d) for d in all_dates])

    rows = []
    for cat in TEST_CATS:
        tw_lnorm = log_normalize(tw_share[cat], pre_mask)
        rd_series = cat_controls[cat] if cat in cat_controls else generic_baseline
        rd_lnorm = log_normalize(rd_series.reindex(all_dates), pre_mask)

        for i, date in enumerate(all_dates):
            post_val = int(post_mask[i])
            q = int(np.clip(quarters[i], -8, 8))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "category": cat,
                    "twitter": 1,
                    "post": post_val,
                    "log_dev": (
                        tw_lnorm.iloc[i] if not np.isnan(tw_lnorm.iloc[i]) else np.nan
                    ),
                    "did": post_val,
                    "quarter": q,
                }
            )
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "category": cat,
                    "twitter": 0,
                    "post": post_val,
                    "log_dev": (
                        rd_lnorm.iloc[i] if not np.isnan(rd_lnorm.iloc[i]) else np.nan
                    ),
                    "did": 0,
                    "quarter": q,
                }
            )

    panel = pd.DataFrame(rows)
    panel.to_csv("out/stata_panel.csv", index=False)
    print(f"Saved: out/stata_panel.csv  ({len(panel):,} rows)")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tw_share = build_twitter_panel()
    cat_controls = build_reddit_controls()

    # Build generic baseline = aggregate of all matched subreddit controls
    date_idx = pd.date_range(PRE_START, POST_END, freq="D")
    if cat_controls:
        generic_baseline = pd.concat(cat_controls.values(), axis=1).sum(axis=1)
        generic_baseline = generic_baseline.reindex(date_idx, fill_value=0).astype(
            float
        )
    else:
        generic_baseline = pd.Series(1.0, index=date_idx)  # flat fallback

    res = run_category_did(tw_share, cat_controls, generic_baseline)
    print_results(res)

    res.to_csv("out/category_did_results.csv", index=False)
    print("\nSaved: out/category_did_results.csv")

    # ── Export stacked panel for Stata ───────────────────────────────────────
    export_stata_panel(tw_share, cat_controls, generic_baseline)

    plot_results(res)
    plot_category_timeseries(tw_share, generic_baseline, res)
    print("\nDone.")
