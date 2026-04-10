"""
DiD Analysis: Did Musk's Twitter acquisition shift political trending content rightward?

Design
------
  Treated  : Twitter  (Musk acquisition Oct 27, 2022)
  Control  : Reddit political subreddits (Lai et al. 2024 ideology scores)
  Outcome  : right_share = right / (right + left) political content per day
             -- Twitter:  keyword-classified trending topics (lexicon.py)
             -- Reddit:   subreddit label from Lai et al. (right vs left only;
                          center subs excluded from denominator for comparability)

Windows
-------
  6-month  : Apr 27, 2022 -- Apr 27, 2023   (primary)
  1-year   : Oct 27, 2021 -- Oct 27, 2023
  2-year   : Oct 27, 2020 -- Oct 27, 2024

Primary metric : topic-count-based right_share
Robustness     : volume-weighted right_share (Twitter) / score-weighted (Reddit)

Model
-----
  right_share ~ twitter + post + did + C(dow)
    twitter = 1 if Twitter, 0 if Reddit
    post    = 1 if date >= Oct 27, 2022
    did     = twitter x post   <- TREATMENT EFFECT (DiD coefficient)

Outputs
-------
  out/figures/did_timeseries.png      -- raw right_share, both platforms (6-month)
  out/figures/did_event_study.png     -- event study (week-relative, 6-month)
  out/figures/did_placebo.png         -- placebo DiD at 3 fake dates (6-month)
  out/figures/did_bar.png             -- DiD coefficient with 95% CI (6-month)
  out/figures/did_multiwindow.png     -- DiD coefs across all 3 windows
  out/did_results.csv                 -- regression tables for all 3 windows
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lexicon import classify_topic

# ── constants ────────────────────────────────────────────────────────────────
TREATMENT = pd.Timestamp("2022-10-27")
FIGURES   = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

WINDOWS = {
    "6-month": (pd.Timestamp("2022-04-27"), pd.Timestamp("2023-04-27")),
    "1-year":  (pd.Timestamp("2021-10-27"), pd.Timestamp("2023-10-27")),
    "2-year":  (pd.Timestamp("2020-10-27"), pd.Timestamp("2024-10-27")),
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# ── 1. BUILD TWITTER DAILY PANEL ─────────────────────────────────────────────

def build_twitter_panel(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv("out/twitter_trending_4yr.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    df["label"] = df["Topic"].apply(classify_topic)

    pol = df[df["label"].isin(["right", "left"])].copy()

    # Primary: topic-count right_share
    daily = (
        pol.groupby([pol["Date"].dt.date, "label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    for col in ["right", "left"]:
        if col not in daily:
            daily[col] = 0

    daily["total"]       = daily["right"] + daily["left"]
    daily                = daily[daily["total"] >= 1]
    daily["right_share"] = daily["right"] / daily["total"]
    daily["platform"]    = "twitter"

    # Robustness: volume-weighted right_share
    # Volume is stored as European dot-separated string (e.g. "1.156.767") -- parse to numeric
    if "Volume" in pol.columns:
        pol = pol.copy()
        pol["Volume"] = pd.to_numeric(
            pol["Volume"].astype(str).str.replace(".", "", regex=False), errors="coerce"
        )
    pol_vol = pol[pol["Volume"] > 0].copy() if "Volume" in pol.columns else pol.copy()
    vol_daily = (
        pol_vol.groupby([pol_vol["Date"].dt.date, "label"])["Volume"]
        .sum()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    for col in ["right", "left"]:
        if col not in vol_daily:
            vol_daily[col] = 0
    vol_daily["total_vol"] = vol_daily["right"] + vol_daily["left"]
    vol_daily = vol_daily[vol_daily["total_vol"] > 0]
    vol_daily["right_share_vol"] = vol_daily["right"] / vol_daily["total_vol"]

    daily = daily.join(vol_daily[["right_share_vol"]], how="left")

    print(f"  Twitter [{start.date()} - {end.date()}]: {len(daily)} days, "
          f"avg {daily['total'].mean():.1f} pol topics/day, "
          f"right_share={daily['right_share'].mean():.3f}")
    return daily[["right_share", "right_share_vol", "platform"]].reset_index()


# ── 2. BUILD REDDIT DAILY PANEL ──────────────────────────────────────────────

def build_reddit_panel(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = "out/reddit_trending.tsv"
    if not os.path.exists(path):
        print("Reddit data not found -- skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df[(df["date"] >= start.date()) & (df["date"] <= end.date())]

    # right_share by post count (primary)
    pol = df[df["label"].isin(["right", "left"])].copy()
    daily = (
        pol.groupby(["date", "label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    for col in ["right", "left"]:
        if col not in daily:
            daily[col] = 0

    daily["total"]       = daily["right"] + daily["left"]
    daily                = daily[daily["total"] >= 1]
    daily["right_share"] = daily["right"] / daily["total"]
    daily["platform"]    = "reddit"

    # Robustness: score-weighted right_share
    score_daily = (
        pol.groupby(["date", "label"])["score"]
        .sum()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    for col in ["right", "left"]:
        if col not in score_daily:
            score_daily[col] = 0
    score_daily["total_score"] = score_daily["right"] + score_daily["left"]
    score_daily = score_daily[score_daily["total_score"] > 0]
    score_daily["right_share_vol"] = score_daily["right"] / score_daily["total_score"]

    daily = daily.join(score_daily[["right_share_vol"]], how="left")

    print(f"  Reddit  [{start.date()} - {end.date()}]: {len(daily)} days, "
          f"avg {daily['total'].mean():.1f} pol posts/day, "
          f"right_share={daily['right_share'].mean():.3f}")
    return daily[["right_share", "right_share_vol", "platform"]].reset_index()


# ── 3. BUILD DiD PANEL ───────────────────────────────────────────────────────

def build_did_panel(tw: pd.DataFrame, rd: pd.DataFrame) -> pd.DataFrame:
    panel = pd.concat([tw, rd], ignore_index=True)
    panel["date"]    = pd.to_datetime(panel["date"])
    panel["twitter"] = (panel["platform"] == "twitter").astype(int)
    panel["post"]    = (panel["date"] >= TREATMENT).astype(int)
    panel["did"]     = panel["twitter"] * panel["post"]
    panel["dow"]     = panel["date"].dt.dayofweek
    panel["week"]    = ((panel["date"] - TREATMENT).dt.days // 7).clip(-26, 26)
    return panel.dropna(subset=["right_share"])


# ── 4. DiD REGRESSION ────────────────────────────────────────────────────────

def run_did(panel: pd.DataFrame, label: str, verbose: bool = True) -> dict:
    model = smf.ols(
        "right_share ~ twitter + post + did + C(dow)",
        data=panel
    ).fit(cov_type="HC3")

    coef = model.params["did"]
    pval = model.pvalues["did"]
    ci   = model.conf_int().loc["did"]

    # Robustness: volume/score-weighted
    rob_coef, rob_pval = np.nan, np.nan
    rob_panel = panel.dropna(subset=["right_share_vol"])
    if len(rob_panel) >= 50:
        rob_model = smf.ols(
            "right_share_vol ~ twitter + post + did + C(dow)",
            data=rob_panel
        ).fit(cov_type="HC3")
        rob_coef = rob_model.params["did"]
        rob_pval = rob_model.pvalues["did"]

    pre_tw  = panel[(panel["platform"]=="twitter") & (panel["post"]==0)]["right_share"].mean()
    post_tw = panel[(panel["platform"]=="twitter") & (panel["post"]==1)]["right_share"].mean()
    pre_rd  = panel[(panel["platform"]=="reddit")  & (panel["post"]==0)]["right_share"].mean()
    post_rd = panel[(panel["platform"]=="reddit")  & (panel["post"]==1)]["right_share"].mean()

    if verbose:
        print(f"\n  [{label}] DiD coef = {coef:+.4f}  p={pval:.3f}  "
              f"CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"          Pre  : Twitter={pre_tw:.3f}  Reddit={pre_rd:.3f}")
        print(f"          Post : Twitter={post_tw:.3f}  Reddit={post_rd:.3f}")
        if not np.isnan(rob_coef):
            print(f"          Robustness (vol/score-weighted): {rob_coef:+.4f}  p={rob_pval:.3f}")

    stars = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."

    return {
        "window":        label,
        "pre_twitter":   pre_tw,
        "post_twitter":  post_tw,
        "pre_reddit":    pre_rd,
        "post_reddit":   post_rd,
        "raw_diff":      (post_tw - pre_tw) - (post_rd - pre_rd),
        "did_coef":      coef,
        "ci_lo":         ci[0],
        "ci_hi":         ci[1],
        "pval":          pval,
        "stars":         stars(pval),
        "rob_coef":      rob_coef,
        "rob_pval":      rob_pval,
        "rob_stars":     stars(rob_pval),
        "n_obs":         len(panel),
        "_model":        model,
    }


# ── 5. EVENT STUDY ───────────────────────────────────────────────────────────

def run_event_study(panel: pd.DataFrame):
    df = panel[panel["week"] != -1].copy()
    all_weeks = sorted(df["week"].unique())

    X = pd.DataFrame({
        "const":   1,
        "twitter": df["twitter"].values,
        "dow_1":   (df["dow"] == 1).astype(int).values,
        "dow_2":   (df["dow"] == 2).astype(int).values,
        "dow_3":   (df["dow"] == 3).astype(int).values,
        "dow_4":   (df["dow"] == 4).astype(int).values,
        "dow_5":   (df["dow"] == 5).astype(int).values,
        "dow_6":   (df["dow"] == 6).astype(int).values,
    }, index=df.index)

    for w in all_weeks:
        X[f"tw_w{w}"] = df["twitter"].values * (df["week"] == w).astype(int).values

    model = sm.OLS(df["right_share"].values, X).fit(cov_type="HC3")

    interact_cols = [c for c in X.columns if c.startswith("tw_w")]
    weeks, betas, cis = [], [], []
    for c in interact_cols:
        w = int(c.replace("tw_w", ""))
        idx = list(X.columns).index(c)
        weeks.append(w)
        betas.append(model.params[idx])
        cis.append(1.96 * model.bse[idx])

    order = np.argsort(weeks)
    return [weeks[i] for i in order], [betas[i] for i in order], [cis[i] for i in order]


# ── 6. PLACEBO TEST ──────────────────────────────────────────────────────────

def run_placebo(panel: pd.DataFrame) -> pd.DataFrame:
    fake_dates = [
        pd.Timestamp("2022-06-01"),
        pd.Timestamp("2022-08-01"),
        pd.Timestamp("2022-10-01"),
        TREATMENT,
    ]
    results = []
    for d in fake_dates:
        p = panel.copy()
        p["post_fake"] = (p["date"] >= d).astype(int)
        p["did_fake"]  = p["twitter"] * p["post_fake"]
        try:
            m = smf.ols("right_share ~ twitter + post_fake + did_fake + C(dow)",
                        data=p).fit(cov_type="HC3")
            results.append({
                "date": d,
                "coef": m.params["did_fake"],
                "ci":   1.96 * m.bse["did_fake"],
                "pval": m.pvalues["did_fake"],
                "real": (d == TREATMENT),
            })
        except Exception as e:
            print(f"  Placebo {d.date()} failed: {e}")
    return pd.DataFrame(results)


# ── 7. PLOTS ─────────────────────────────────────────────────────────────────

def plot_timeseries(panel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4))
    for platform, color, label in [
        ("twitter", "#c0392b", "Twitter (treated)"),
        ("reddit",  "#2980b9", "Reddit (control)"),
    ]:
        sub = panel[panel["platform"] == platform].set_index("date").sort_index()
        smooth = sub["right_share"].rolling(14, min_periods=3, center=True).mean()
        ax.plot(sub.index, sub["right_share"], color=color, lw=0.5, alpha=0.25)
        ax.plot(smooth.index, smooth, color=color, lw=2.2, label=label)

    ax.axvline(TREATMENT, color="black", lw=1.4, ls="--",
               label="Musk acquisition (Oct 27, 2022)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.set_ylabel("Right share of political content")
    ax.set_title("Right-Coded Share of Political Content: Twitter vs Reddit (14-day rolling avg)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_timeseries.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_timeseries.png")


def plot_event_study(weeks, betas, cis):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=1.2, ls="--", label="Musk acquisition")
    ax.axvspan(-26, -1, alpha=0.04, color="blue")
    ax.axvspan(0,   26, alpha=0.04, color="red")

    pre  = [(w, b, e) for w, b, e in zip(weeks, betas, cis) if w < 0]
    post = [(w, b, e) for w, b, e in zip(weeks, betas, cis) if w >= 0]

    for subset, color in [(pre, "#2980b9"), (post, "#c0392b")]:
        if not subset:
            continue
        ws, bs, es = zip(*subset)
        ax.errorbar(ws, bs, yerr=es, fmt="o", color=color,
                    capsize=3, markersize=4, lw=1.2)

    ax.set_title("Event Study: Twitter vs Reddit Right Share (week relative to acquisition)")
    ax.set_xlabel("Weeks relative to Musk acquisition")
    ax.set_ylabel("Coeff: Twitter minus Reddit right_share difference")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_event_study.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_event_study.png")


def plot_did_bar(result: dict):
    coef = result["did_coef"]
    ci   = (result["ci_hi"] - result["ci_lo"]) / 2
    pval = result["pval"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    color = "#c0392b" if coef > 0 else "#2980b9"
    ax.bar(["Twitter x Post\n(DiD estimate)"], [coef],
           color=color, alpha=0.85, width=0.45)
    ax.errorbar(["Twitter x Post\n(DiD estimate)"], [coef],
                yerr=ci, fmt="none", color="black", capsize=8, lw=2)
    ax.axhline(0, color="black", lw=0.8)
    sig = result["stars"]
    ax.text(0, coef + (ci + 0.005) * np.sign(coef) if coef != 0 else ci + 0.005,
            f"{coef:+.3f} {sig}", ha="center", va="bottom" if coef > 0 else "top",
            fontsize=11, fontweight="bold")
    ax.set_ylabel("Effect on right_share")
    ax.set_title(f"DiD Coefficient (6-month window)\n(p = {pval:.3f})")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_bar.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_bar.png")


def plot_placebo(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#aaa", "#aaa", "#aaa", "#c0392b"]
    labels = [f"{r['date'].strftime('%b %d, %Y')}{' <- REAL' if r['real'] else ' (placebo)'}"
              for _, r in results.iterrows()]

    for i, (_, r) in enumerate(results.iterrows()):
        ax.bar(i, r["coef"], color=colors[i], alpha=0.85, width=0.55)
        ax.errorbar(i, r["coef"], yerr=r["ci"],
                    fmt="none", color="black", capsize=6, lw=1.5)
        sig = "***" if r["pval"] < 0.01 else "**" if r["pval"] < 0.05 \
              else "*" if r["pval"] < 0.1 else "n.s."
        ax.text(i, r["coef"] + r["ci"] * np.sign(r["coef"]) + 0.003,
                sig, ha="center", fontsize=11)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("DiD Coefficient")
    ax.set_title("Placebo Test: DiD Coefficient at Fake vs Real Treatment Dates")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_placebo.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_placebo.png")


def plot_multiwindow(all_results: list):
    """
    Horizontal bar chart comparing DiD estimates across all 3 windows.
    Primary (topic-count) and robustness (volume/score-weighted) side by side.
    Style: matches category_shift.png bar format.
    """
    labels    = [r["window"] for r in all_results]
    coefs     = [r["did_coef"] for r in all_results]
    ci_lo     = [r["ci_lo"] for r in all_results]
    ci_hi     = [r["ci_hi"] for r in all_results]
    stars     = [r["stars"] for r in all_results]
    rob_coefs = [r["rob_coef"] for r in all_results]
    rob_pvs   = [r["rob_pval"] for r in all_results]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))

    # Primary bars
    colors_primary = ["#c0392b" if c > 0 else "#2980b9" for c in coefs]
    ax.barh(y + 0.18, coefs, height=0.32,
            color=colors_primary, alpha=0.85, label="Primary (topic count / post count)")
    xerr = np.array([
        [c - lo for c, lo in zip(coefs, ci_lo)],
        [hi - c for c, hi in zip(coefs, ci_hi)],
    ])
    ax.errorbar(coefs, y + 0.18, xerr=xerr,
                fmt="none", color="black", linewidth=1.2, capsize=4)

    # Robustness bars (skip NaN)
    for i, (rc, rp) in enumerate(zip(rob_coefs, rob_pvs)):
        if not np.isnan(rc):
            rc_color = "#e74c3c" if rc > 0 else "#5dade2"
            ax.barh(y[i] - 0.18, rc, height=0.32,
                    color=rc_color, alpha=0.55,
                    label="Robustness (vol/score-weighted)" if i == 0 else "")

    # Star labels
    for i, (c, hi, st) in enumerate(zip(coefs, ci_hi, stars)):
        if st != "n.s.":
            ax.text(hi + 0.003, y[i] + 0.18, st, va="center", fontsize=10, fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['window']}" for r in all_results], fontsize=11)
    ax.set_xlabel("DiD coefficient: Twitter right_share shift vs Reddit baseline", fontsize=10)
    ax.set_title(
        "Right-Lean DiD Estimates Across Windows\n"
        "Twitter (treated) vs Reddit political subreddits (control)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = f"{FIGURES}/did_multiwindow.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 8. SUMMARY TABLE ─────────────────────────────────────────────────────────

def print_summary(all_results: list):
    print("\n" + "="*90)
    print("MULTI-WINDOW DiD RESULTS -- Twitter right_share post Musk acquisition")
    print("Control: Reddit political subreddits (Lai et al. 2024)")
    print("="*90)
    hdr = (f"{'Window':<10} {'TW pre':>7} {'TW post':>8} {'RD pre':>7} {'RD post':>8} "
           f"{'DiD coef':>9} {'p':>7}      {'95% CI':<22} {'Rob coef':>9} {'Rob p':>7}")
    print(hdr)
    print("-"*90)
    for r in all_results:
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        rob = f"{r['rob_coef']:+.3f}" if not np.isnan(r['rob_coef']) else "  n/a "
        robp = f"{r['rob_pval']:.3f}{r['rob_stars']:>4}" if not np.isnan(r['rob_pval']) else "      "
        print(
            f"{r['window']:<10} {r['pre_twitter']:>7.3f} {r['post_twitter']:>8.3f} "
            f"{r['pre_reddit']:>7.3f} {r['post_reddit']:>8.3f} "
            f"{r['did_coef']:>+9.3f} {r['pval']:>7.3f}{r['stars']:>4} "
            f"{ci:<22} {rob:>9} {robp}"
        )
    print("="*90)
    print("Note: Primary = topic-count (Twitter) / post-count (Reddit) right_share.")
    print("      Robustness = volume-weighted (Twitter) / score-weighted (Reddit).")
    print("      Model: right_share ~ twitter + post + did + C(dow), HC3 SEs.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading Twitter and Reddit data...")

    all_results = []
    primary_panel = None

    for label, (start, end) in WINDOWS.items():
        print(f"\n[Window: {label}]")
        tw = build_twitter_panel(start, end)
        rd = build_reddit_panel(start, end)

        if rd.empty:
            print("  ERROR: Reddit data missing.")
            continue

        panel = build_did_panel(tw, rd)
        result = run_did(panel, label)
        all_results.append(result)

        if label == "6-month":
            primary_panel = panel

    if not all_results:
        print("No results -- check data files.")
        sys.exit(1)

    print_summary(all_results)

    # Save results CSV (drop internal _model key)
    csv_rows = [{k: v for k, v in r.items() if k != "_model"} for r in all_results]
    pd.DataFrame(csv_rows).to_csv("out/did_results.csv", index=False)
    print("\nSaved: out/did_results.csv")

    # Multi-window comparison figure
    plot_multiwindow(all_results)

    # 6-month detail figures
    if primary_panel is not None:
        print("\nGenerating 6-month detail figures...")
        plot_timeseries(primary_panel)
        primary_result = next(r for r in all_results if r["window"] == "6-month")
        plot_did_bar(primary_result)

        print("Running event study (6-month)...")
        weeks, betas, cis = run_event_study(primary_panel)
        plot_event_study(weeks, betas, cis)

        print("Running placebo tests (6-month)...")
        placebo_df = run_placebo(primary_panel)
        print(placebo_df[["date", "coef", "ci", "pval"]].to_string(index=False))
        plot_placebo(placebo_df)

    print("\nDone. Figures saved to out/figures/")
