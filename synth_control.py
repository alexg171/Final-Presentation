"""
Synthetic Control: Twitter political bias post-Musk acquisition

Design
------
Treated unit : Twitter daily right_share (lexicon-classified trending topics)
Donor pool   : 6 political Reddit subreddits (Lai et al. 2024 labels)
               Right: r/conservative, r/republican, r/libertarian
               Left:  r/liberal, r/progressive, r/politics

Each subreddit contributes a daily post-count time series.
We find weights W that minimize the pre-treatment distance between
Twitter's right_share and a synthetic version computed as:

  Synthetic_t(W) = Σ w_right_i * posts_right_i_t
                   ─────────────────────────────────────────────────
                   Σ w_right_i * posts_right_i_t + Σ w_left_j * posts_left_j_t

Treatment effect = actual Twitter - Synthetic Twitter (post-period)

Inference: permutation test — run synthetic control on each subreddit
as if it were the treated unit, compare post/pre RMSPE ratios.

Output
------
  out/figures/synthetic_control.png
  out/figures/synthetic_placebo.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lexicon import classify_topic

# ── constants ────────────────────────────────────────────────────────────────
TREATMENT     = pd.Timestamp("2022-10-27")
PRE_START     = pd.Timestamp("2022-04-27")
POST_END      = pd.Timestamp("2023-04-27")
FIGURES       = "out/figures"
RIGHT_SUBS    = ["conservative", "republican", "libertarian"]
LEFT_SUBS     = ["liberal", "progressive", "politics"]
POLITICAL_SUBS = RIGHT_SUBS + LEFT_SUBS   # column order kept throughout

os.makedirs(FIGURES, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── 1. DATA LOADING ──────────────────────────────────────────────────────────

def build_twitter_series() -> pd.Series:
    df = pd.read_csv("out/twitter_1774593752.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df = df[(df["Date"] >= PRE_START) & (df["Date"] <= POST_END)]
    df["label"] = df["Topic"].apply(classify_topic)

    pol = df[df["label"].isin(["right", "left"])]
    daily = (pol.groupby([pol["Date"].dt.date, "label"])
               .size().unstack(fill_value=0).rename_axis("date"))
    for c in ["right", "left"]:
        if c not in daily: daily[c] = 0
    daily["total"] = daily["right"] + daily["left"]
    daily = daily[daily["total"] >= 1]
    daily["right_share"] = daily["right"] / daily["total"]
    daily.index = pd.to_datetime(daily.index)

    series = daily["right_share"].reindex(
        pd.date_range(PRE_START, POST_END, freq="D"))
    print(f"Twitter: {daily['total'].count()} days with political content  "
          f"(avg {daily['total'].mean():.1f} topics/day)")
    return series


def build_reddit_counts() -> pd.DataFrame:
    df = pd.read_csv("out/reddit_trending.tsv", sep="\t", parse_dates=["date"])
    df = df[(df["date"] >= PRE_START) & (df["date"] <= POST_END)]
    df = df[df["subreddit"].isin(POLITICAL_SUBS)]

    counts = (df.groupby([df["date"].dt.date, "subreddit"])
                .size().unstack(fill_value=0).rename_axis("date"))
    counts.index = pd.to_datetime(counts.index)
    for s in POLITICAL_SUBS:
        if s not in counts: counts[s] = 0

    full_idx = pd.date_range(PRE_START, POST_END, freq="D")
    counts = counts[POLITICAL_SUBS].reindex(full_idx, fill_value=0)
    print(f"Reddit:  {(counts.sum(axis=1) > 0).sum()} days with post data  "
          f"(avg {counts.sum(axis=1).mean():.0f} posts/day)")
    return counts


# ── 2. SYNTHETIC CONTROL MATH ────────────────────────────────────────────────

def synth_series(W: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    W      : (6,) — first 3 weights for right subs, last 3 for left subs
    counts : (T, 6) — daily post counts per subreddit
    Returns: (T,) synthetic right_share
    """
    right = counts[:, :3] @ W[:3]   # weighted right-sub posts
    left  = counts[:, 3:] @ W[3:]   # weighted left-sub posts
    total = right + left
    return np.where(total > 0, right / total, np.nan)


def fit_weights(twitter_pre: np.ndarray,
                counts_pre: np.ndarray,
                n_starts: int = 80) -> np.ndarray:
    """
    Find W* minimising pre-treatment RMSPE via constrained optimisation.
    Runs n_starts random initialisations to escape local minima.
    """
    valid = ~np.isnan(twitter_pre)
    y, X  = twitter_pre[valid], counts_pre[valid]

    def obj(W):
        s = synth_series(W, X)
        ok = ~np.isnan(s)
        return np.mean((y[ok] - s[ok]) ** 2) if ok.any() else 1e6

    rng = np.random.default_rng(42)
    best_W, best_val = np.ones(6) / 6, np.inf
    for _ in range(n_starts):
        W0  = rng.dirichlet(np.ones(6))
        res = minimize(obj, W0, method="SLSQP",
                       bounds=[(0, 1)] * 6,
                       constraints={"type": "eq", "fun": lambda W: W.sum() - 1},
                       options={"ftol": 1e-12, "maxiter": 2000})
        if res.fun < best_val:
            best_val, best_W = res.fun, res.x

    return best_W


# ── 3. PERMUTATION INFERENCE ─────────────────────────────────────────────────

def rmspe(actual, synthetic, mask):
    diff = actual[mask] - synthetic[mask]
    return np.sqrt(np.nanmean(diff ** 2))


def permutation_test(tw: pd.Series, rd: pd.DataFrame):
    """
    For each subreddit, treat it as the 'treated unit'.
    Compute post/pre RMSPE ratio and compare to Twitter's ratio.
    """
    all_dates  = tw.index
    pre_mask   = np.array(all_dates < TREATMENT)
    post_mask  = np.array(all_dates >= TREATMENT)
    counts_arr = rd.values.astype(float)

    results = {}

    # Twitter (real treated unit)
    W = fit_weights(tw.values[pre_mask], counts_arr[pre_mask])
    s = synth_series(W, counts_arr)
    pre_r  = rmspe(tw.values, s, pre_mask)
    post_r = rmspe(tw.values, s, post_mask)
    results["Twitter"] = {
        "W": W, "synth": s,
        "pre_rmspe": pre_r, "post_rmspe": post_r,
        "ratio": post_r / pre_r if pre_r > 0 else np.nan,
        "avg_gap": np.nanmean((tw.values - s)[post_mask]),
    }

    # Placebos — each subreddit as treated unit
    for i, sub in enumerate(POLITICAL_SUBS):
        # Actual series for this sub: right_share = 1 if right sub, 0 if left
        sub_right_share = np.full(len(all_dates),
                                  1.0 if sub in RIGHT_SUBS else 0.0)
        sub_right_share = sub_right_share + np.random.default_rng(i).normal(
            0, 0.02, len(sub_right_share))   # tiny jitter so optimiser isn't trivial
        sub_right_share = np.clip(sub_right_share, 0, 1)

        # Donor pool excludes this subreddit's column
        donor_cols = [c for c in POLITICAL_SUBS if c != sub]
        donor_idx  = [POLITICAL_SUBS.index(c) for c in donor_cols]
        donor_arr  = counts_arr[:, donor_idx]

        # Fit with 5-weight vector
        valid = pre_mask
        y_pre = sub_right_share[valid]
        X_pre = donor_arr[valid]

        def obj5(W5):
            # Map back to right/left structure for the 5-subreddit donor pool
            right_idx_in_donor = [j for j, c in enumerate(donor_cols) if c in RIGHT_SUBS]
            left_idx_in_donor  = [j for j, c in enumerate(donor_cols) if c in LEFT_SUBS]
            r = X_pre[:, right_idx_in_donor] @ W5[right_idx_in_donor] if right_idx_in_donor else np.zeros(len(X_pre))
            l = X_pre[:, left_idx_in_donor]  @ W5[left_idx_in_donor]  if left_idx_in_donor  else np.zeros(len(X_pre))
            total = r + l
            s5 = np.where(total > 0, r / total, np.nan)
            ok = ~np.isnan(s5)
            return np.mean((y_pre[ok] - s5[ok]) ** 2) if ok.any() else 1e6

        rng = np.random.default_rng(i * 7)
        best_W5, best_v = np.ones(5) / 5, np.inf
        for _ in range(30):
            W0 = rng.dirichlet(np.ones(5))
            res = minimize(obj5, W0, method="SLSQP",
                           bounds=[(0, 1)] * 5,
                           constraints={"type": "eq", "fun": lambda W: W.sum() - 1},
                           options={"ftol": 1e-10, "maxiter": 1000})
            if res.fun < best_v:
                best_v, best_W5 = res.fun, res.x

        right_in_d = [j for j, c in enumerate(donor_cols) if c in RIGHT_SUBS]
        left_in_d  = [j for j, c in enumerate(donor_cols) if c in LEFT_SUBS]
        r_full = donor_arr[:, right_in_d] @ best_W5[right_in_d] if right_in_d else np.zeros(len(all_dates))
        l_full = donor_arr[:, left_in_d]  @ best_W5[left_in_d]  if left_in_d  else np.zeros(len(all_dates))
        tot    = r_full + l_full
        s_sub  = np.where(tot > 0, r_full / tot, np.nan)

        pre_r2  = rmspe(sub_right_share, s_sub, pre_mask)
        post_r2 = rmspe(sub_right_share, s_sub, post_mask)
        results[f"r/{sub}"] = {
            "pre_rmspe": pre_r2, "post_rmspe": post_r2,
            "ratio": post_r2 / pre_r2 if pre_r2 > 0 else np.nan,
            "avg_gap": np.nanmean((sub_right_share - s_sub)[post_mask]),
        }

    return results


# ── 4. PLOTS ─────────────────────────────────────────────────────────────────

def plot_main(tw: pd.Series, synth: np.ndarray, W: np.ndarray, avg_gap: float):
    all_dates  = tw.index
    pre_mask   = all_dates < TREATMENT
    post_mask  = all_dates >= TREATMENT

    synth_s = pd.Series(synth, index=all_dates)
    tw_sm   = tw.rolling(14, min_periods=3, center=True).mean()
    sy_sm   = synth_s.rolling(14, min_periods=3, center=True).mean()
    gap_sm  = (tw - synth_s).rolling(14, min_periods=3, center=True).mean()

    fig, axes = plt.subplots(3, 1, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [3, 2, 1.5]})

    # Panel A — actual vs synthetic
    ax = axes[0]
    ax.plot(all_dates, tw.values,    color="#c0392b", alpha=0.12, lw=0.6)
    ax.plot(all_dates, synth,        color="#2980b9", alpha=0.12, lw=0.6)
    ax.plot(tw_sm.index, tw_sm,      color="#c0392b", lw=2.2,
            label="Twitter (actual)")
    ax.plot(sy_sm.index, sy_sm,      color="#2980b9", lw=2.2, ls="--",
            label="Synthetic Twitter")
    ax.axvline(TREATMENT, color="black", lw=1.3, ls="--")
    ax.axhline(0.5, color="grey", lw=0.7, ls=":", alpha=0.5, label="Parity")
    ax.fill_between(all_dates[post_mask],
                    sy_sm[post_mask].fillna(method="ffill"),
                    tw_sm[post_mask].fillna(method="ffill"),
                    alpha=0.15, color="#c0392b",
                    label=f"Gap ≈ +{avg_gap*100:.1f}pp")
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("Right share of political content")
    ax.set_title("Synthetic Control: Actual vs Synthetic Twitter (14-day rolling avg)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(TREATMENT + pd.Timedelta(days=4), 0.97,
            "Musk acquisition\nOct 27, 2022", fontsize=8, va="top")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Panel B — gap
    ax = axes[1]
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.axvline(TREATMENT, color="black", lw=1.3, ls="--")
    ax.plot(gap_sm.index, gap_sm.values, color="#c0392b", lw=2.0)
    ax.fill_between(gap_sm.index, 0, gap_sm.values,
                    where=(gap_sm.index >= TREATMENT),
                    alpha=0.2, color="#c0392b", label="Post-treatment gap")
    ax.fill_between(gap_sm.index, 0, gap_sm.values,
                    where=(gap_sm.index < TREATMENT),
                    alpha=0.1, color="grey", label="Pre-treatment fit")
    ax.set_ylabel("Actual − Synthetic\n(treatment effect)")
    ax.set_title("Treatment Effect Gap")
    ax.legend(frameon=False, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Panel C — weights
    ax = axes[2]
    colors = ["#c0392b"] * 3 + ["#2980b9"] * 3
    bars = ax.bar(POLITICAL_SUBS, W, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for bar, w in zip(bars, W):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{w:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Donor weight")
    ax.set_title("Optimal Subreddit Weights  (red = right-coded, blue = left-coded)")
    ax.set_ylim(0, max(W) * 1.35)

    fig.suptitle(
        "Synthetic Control: Effect of Musk Acquisition on Twitter Political Right Share",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/synthetic_control.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/synthetic_control.png")


def plot_placebo(perm_results: dict):
    names  = list(perm_results.keys())
    ratios = [perm_results[n]["ratio"] for n in names]
    gaps   = [perm_results[n]["avg_gap"] for n in names]
    colors = ["#c0392b" if n == "Twitter" else "#aaa" for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSPE ratios
    ax = axes[0]
    bars = ax.bar(names, ratios, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(1, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Post/Pre RMSPE ratio")
    ax.set_title("Permutation Test: Post/Pre RMSPE Ratio\n(Twitter should rank highest)")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)

    # Average gaps
    ax = axes[1]
    bar_colors = ["#c0392b" if g > 0 else "#2980b9" for g in gaps]
    bar_colors[names.index("Twitter")] = "#c0392b"
    ax.bar(names, [g * 100 for g in gaps], color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Avg post-treatment gap (pp)")
    ax.set_title("Average Post-Treatment Gap\n(red = Twitter, grey = placebo subreddits)")
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)

    fig.suptitle("Synthetic Control — Permutation Inference", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/synthetic_placebo.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/synthetic_placebo.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tw = build_twitter_series()
    rd = build_reddit_counts()

    all_dates = tw.index
    pre_mask  = np.array(all_dates < TREATMENT)
    post_mask = np.array(all_dates >= TREATMENT)

    print("\nFitting synthetic control weights (80 random starts)...")
    W_opt = fit_weights(tw.values[pre_mask], rd.values[pre_mask])

    print("\nOptimal weights:")
    for sub, w in zip(POLITICAL_SUBS, W_opt):
        print(f"  r/{sub:<20s} {w:.4f}")

    synth = synth_series(W_opt, rd.values)
    synth_s = pd.Series(synth, index=all_dates)

    pre_rmspe  = rmspe(tw.values, synth, pre_mask)
    post_rmspe = rmspe(tw.values, synth, post_mask)
    avg_gap    = float(np.nanmean((tw.values - synth)[post_mask]))

    print(f"\n  Pre-treatment RMSPE  : {pre_rmspe:.4f}")
    print(f"  Post-treatment RMSPE : {post_rmspe:.4f}")
    print(f"  RMSPE ratio          : {post_rmspe/pre_rmspe:.2f}x")
    print(f"  Avg treatment effect : {avg_gap:+.4f}  ({avg_gap*100:+.2f}pp)")

    plot_main(tw, synth, W_opt, avg_gap)

    print("\nRunning permutation inference (this takes ~1 min)...")
    perm = permutation_test(tw, rd)

    print("\nPermutation results:")
    print(f"  {'Unit':<25s} {'pre_RMSPE':>10} {'post_RMSPE':>11} {'ratio':>7} {'avg_gap':>9}")
    for name, r in perm.items():
        print(f"  {name:<25s} {r['pre_rmspe']:>10.4f} {r['post_rmspe']:>11.4f} "
              f"{r['ratio']:>7.2f} {r['avg_gap']*100:>+9.2f}pp")

    tw_ratio = perm["Twitter"]["ratio"]
    all_ratios = [v["ratio"] for v in perm.values() if not np.isnan(v["ratio"])]
    rank = sorted(all_ratios, reverse=True).index(tw_ratio) + 1
    pval = rank / len(all_ratios)
    print(f"\n  Twitter RMSPE ratio rank: {rank}/{len(all_ratios)}  =>  p ~ {pval:.3f}")

    plot_placebo(perm)
    print("\nDone. Figures saved to out/figures/")
