"""
category_demographics.py
Generates demographic profile visualizations for each Twitter trending category.
Data sourced from Morning Consult, Nielsen, Statista, Pew Research (2022-2023).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("out/figures/demographics", exist_ok=True)

# ── Demographic Data ──────────────────────────────────────────────────────────
# Each entry: (label, pct_male, age_label, age_median, political_score, political_label, source)
# political_score: 0 = strongly liberal, 50 = mixed/neutral, 100 = strongly conservative

CATEGORIES = [
    {
        "key":        "wrestling",
        "label":      "Wrestling (WWE/AEW)",
        "pct_male":   65,
        "age_label":  "18–49",
        "age_median": 38,
        "pol_score":  68,
        "pol_label":  "Conservative-lean",
        "source":     "Wrestlenomics 2023; Polling.com",
    },
    {
        "key":        "combat_sports",
        "label":      "Combat Sports (UFC/Boxing)",
        "pct_male":   75,
        "age_label":  "25–44",
        "age_median": 34,
        "pol_score":  65,
        "pol_label":  "Conservative-lean",
        "source":     "Morning Consult / Statista 2023",
    },
    {
        "key":        "sports_nfl",
        "label":      "Sports — NFL",
        "pct_male":   66,
        "age_label":  "18–54",
        "age_median": 43,
        "pol_score":  58,
        "pol_label":  "Slight Conservative",
        "source":     "Morning Consult 2023; Statista",
    },
    {
        "key":        "sports_nba",
        "label":      "Sports — NBA",
        "pct_male":   56,
        "age_label":  "18–49",
        "age_median": 37,
        "pol_score":  35,
        "pol_label":  "Liberal-lean",
        "source":     "Morning Consult 2023; Nielsen",
    },
    {
        "key":        "sports_mlb",
        "label":      "Sports — MLB",
        "pct_male":   62,
        "age_label":  "35–64",
        "age_median": 53,
        "pol_score":  60,
        "pol_label":  "Slight Conservative",
        "source":     "Morning Consult 2023; Nielsen",
    },
    {
        "key":        "sports_nhl",
        "label":      "Sports — NHL",
        "pct_male":   67,
        "age_label":  "35–54",
        "age_median": 45,
        "pol_score":  65,
        "pol_label":  "Conservative-lean",
        "source":     "Statista 2020; Morning Consult",
    },
    {
        "key":        "sports_soccer",
        "label":      "Sports — Soccer/International",
        "pct_male":   57,
        "age_label":  "18–34",
        "age_median": 30,
        "pol_score":  38,
        "pol_label":  "Slight Liberal",
        "source":     "MLS/Nielsen 2023; YouGov",
    },
    {
        "key":        "sports_college",
        "label":      "Sports — College",
        "pct_male":   62,
        "age_label":  "25–54",
        "age_median": 41,
        "pol_score":  62,
        "pol_label":  "Conservative-lean",
        "source":     "Morning Consult 2023",
    },
    {
        "key":        "sports_womens",
        "label":      "Sports — Women's (WNBA/NWSL)",
        "pct_male":   42,
        "age_label":  "18–44",
        "age_median": 33,
        "pol_score":  30,
        "pol_label":  "Liberal-lean",
        "source":     "Nielsen 2023; NWSL Fan Survey",
    },
    {
        "key":        "sports_other",
        "label":      "Sports — Other (F1/Golf/Tennis)",
        "pct_male":   66,
        "age_label":  "25–54",
        "age_median": 42,
        "pol_score":  58,
        "pol_label":  "Slight Conservative",
        "source":     "F1 Fan Survey 2023; Golf Channel data",
    },
    {
        "key":        "reality_tv",
        "label":      "Reality TV (Bravo/RH)",
        "pct_male":   33,
        "age_label":  "25–54",
        "age_median": 44,
        "pol_score":  40,
        "pol_label":  "Slight Liberal",
        "source":     "Nielsen 2023; Bravo Media demographics",
    },
    {
        "key":        "entertainment",
        "label":      "Entertainment (TV/Film/Music)",
        "pct_male":   44,
        "age_label":  "18–49",
        "age_median": 36,
        "pol_score":  38,
        "pol_label":  "Slight Liberal",
        "source":     "Nielsen 2023",
    },
    {
        "key":        "taylor_swift",
        "label":      "Taylor Swift",
        "pct_male":   48,
        "age_label":  "18–49 (Millennial-heavy)",
        "age_median": 38,
        "pol_score":  25,
        "pol_label":  "Liberal (55% Dem)",
        "source":     "Morning Consult Mar 2023 (n=356)",
    },
    {
        "key":        "fandom",
        "label":      "Fandom (Anime/K-pop)",
        "pct_male":   38,
        "age_label":  "13–24",
        "age_median": 20,
        "pol_score":  22,
        "pol_label":  "Strongly Liberal",
        "source":     "Anime Survey 2022; K-pop GMU Survey 2023",
    },
    {
        "key":        "tech_gaming",
        "label":      "Tech & Gaming",
        "pct_male":   58,
        "age_label":  "18–34",
        "age_median": 31,
        "pol_score":  42,
        "pol_label":  "Slight Liberal",
        "source":     "Pew Research 2023; Statista",
    },
    {
        "key":        "lgbtq_social",
        "label":      "Social Justice",
        "pct_male":   32,
        "age_label":  "18–34",
        "age_median": 26,
        "pol_score":  10,
        "pol_label":  "Strongly Liberal",
        "source":     "GLAAD 2023; Pew Research",
    },
    {
        "key":        "news_politics",
        "label":      "News & Politics",
        "pct_male":   57,
        "age_label":  "35–64",
        "age_median": 47,
        "pol_score":  50,
        "pol_label":  "Mixed",
        "source":     "Pew Research 2022; Reuters Inst.",
    },
]

# ── Colors ────────────────────────────────────────────────────────────────────
MALE_COLOR   = "#2166ac"   # blue
FEMALE_COLOR = "#d6604d"   # rose-red

def pol_color(score):
    """Interpolate between blue (0=liberal) and red (100=conservative)."""
    blue = np.array([33, 102, 172]) / 255
    red  = np.array([178, 24, 43]) / 255
    t = score / 100
    return tuple((1 - t) * blue + t * red)

def age_color(median):
    """Young = green, old = purple."""
    young = np.array([77, 175, 74]) / 255
    old   = np.array([152, 78, 163]) / 255
    t = np.clip((median - 18) / (60 - 18), 0, 1)
    return tuple((1 - t) * young + t * old)

# ── Figure 1: Summary grid (all categories, 3 metrics) ───────────────────────
def plot_summary_grid():
    n = len(CATEGORIES)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 0.72 + 1.5))
    fig.suptitle("Content Category Audience Demographics\n(Gender · Age · Political Lean)",
                 fontsize=14, fontweight="bold", y=1.01)

    col_titles = ["Gender Split", "Age Skew", "Political Lean"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=10, fontweight="bold", pad=8)

    for i, cat in enumerate(CATEGORIES):
        pct_m  = cat["pct_male"]
        pct_f  = 100 - pct_m
        pol    = cat["pol_score"]
        median = cat["age_median"]

        # Row label
        axes[i, 0].annotate(
            cat["label"], xy=(-0.55, 0.5),
            xycoords="axes fraction", fontsize=8,
            ha="right", va="center", fontweight="bold"
        )

        # ── Col 0: Gender bar ──
        ax = axes[i, 0]
        ax.barh(0, pct_m, color=MALE_COLOR,   height=0.5, label="Male")
        ax.barh(0, pct_f, left=pct_m, color=FEMALE_COLOR, height=0.5, label="Female")
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        if pct_m >= 15:
            ax.text(pct_m / 2, 0, f"{pct_m}% M", ha="center", va="center",
                    color="white", fontsize=7.5, fontweight="bold")
        if pct_f >= 15:
            ax.text(pct_m + pct_f / 2, 0, f"{pct_f}% F", ha="center", va="center",
                    color="white", fontsize=7.5, fontweight="bold")

        # ── Col 1: Age bar ──
        ax = axes[i, 1]
        ax.barh(0, 1, color=age_color(median), height=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.text(0.5, 0, f"{cat['age_label']}\n(med. {median})",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

        # ── Col 2: Political bar ──
        ax = axes[i, 2]
        ax.barh(0, 1, color=pol_color(pol), height=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.text(0.5, 0, cat["pol_label"],
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(color=MALE_COLOR,   label="Male"),
        mpatches.Patch(color=FEMALE_COLOR, label="Female"),
        mpatches.Patch(color=pol_color(10),  label="Liberal"),
        mpatches.Patch(color=pol_color(50),  label="Mixed"),
        mpatches.Patch(color=pol_color(90),  label="Conservative"),
        mpatches.Patch(color=age_color(20),  label="Younger audience"),
        mpatches.Patch(color=age_color(50),  label="Older audience"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=7,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout(rect=[0.18, 0.02, 1, 0.99])
    out = "out/figures/demographics/summary_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Individual card per category ────────────────────────────────────
def plot_individual_card(cat):
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.2))
    fig.suptitle(cat["label"], fontsize=12, fontweight="bold")

    pct_m  = cat["pct_male"]
    pct_f  = 100 - pct_m
    pol    = cat["pol_score"]
    median = cat["age_median"]

    # Gender
    ax = axes[0]
    ax.barh(0, pct_m, color=MALE_COLOR, height=0.55)
    ax.barh(0, pct_f, left=pct_m, color=FEMALE_COLOR, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("Gender Split", fontsize=9, pad=4)
    ax.axis("off")
    if pct_m >= 12:
        ax.text(pct_m / 2, 0, f"{pct_m}%\nMale",
                ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    if pct_f >= 12:
        ax.text(pct_m + pct_f / 2, 0, f"{pct_f}%\nFemale",
                ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    # Age
    ax = axes[1]
    # Draw age spectrum bar
    for x in range(100):
        t = x / 100
        c = age_color(18 + t * (65 - 18))
        ax.barh(0, 1, left=x, color=c, height=0.55)
    # Mark median
    median_x = (median - 18) / (65 - 18) * 100
    ax.axvline(median_x, color="white", lw=2.5)
    ax.text(median_x, 0.38, f"med. {median}", ha="center", va="bottom",
            fontsize=8, color="white", fontweight="bold")
    ax.text(median_x, -0.38, cat["age_label"], ha="center", va="top",
            fontsize=7.5, color="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("Age Skew", fontsize=9, pad=4)
    ax.text(0, -0.6, "18", ha="center", fontsize=7, color="gray")
    ax.text(100, -0.6, "65+", ha="center", fontsize=7, color="gray")
    ax.axis("off")

    # Political
    ax = axes[2]
    for x in range(100):
        ax.barh(0, 1, left=x, color=pol_color(x), height=0.55)
    ax.axvline(pol, color="white", lw=2.5)
    ax.text(pol, 0.38, cat["pol_label"], ha="center", va="bottom",
            fontsize=8, color="white", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("Political Lean", fontsize=9, pad=4)
    ax.text(0,  -0.6, "Liberal",      ha="left",   fontsize=7, color="gray")
    ax.text(100, -0.6, "Conservative", ha="right", fontsize=7, color="gray")
    ax.axis("off")

    # Source footnote
    fig.text(0.5, -0.05, f"Source: {cat['source']}", ha="center",
             fontsize=6.5, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = f"out/figures/demographics/{cat['key']}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: "Bro index" scatter — male% vs conservative% ───────────────────
def plot_bro_scatter():
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for cat in CATEGORIES:
        x = cat["pol_score"]
        y = cat["pct_male"]
        ax.scatter(x, y, s=110, zorder=3,
                   color=pol_color(cat["pol_score"]), edgecolors="white", lw=0.8)
        # offset labels to avoid overlap
        offsets = {
            "wrestling":    ( 2, -3),
            "combat_sports":( 2,  1),
            "sports_nfl":   ( 2,  1),
            "sports_nba":   (-2,  2),
            "sports_mlb":   ( 2, -3),
            "sports_nhl":   ( 2,  1),
            "sports_soccer":(-2,  2),
            "sports_college":( 2, 1),
            "sports_womens":(-2, -3),
            "sports_other": ( 2,  1),
            "reality_tv":   (-2, -3),
            "entertainment":(-16,  1),
            "taylor_swift": (-2, -3),
            "fandom":       (-2,  2),
            "tech_gaming":  ( 2, -3),
            "lgbtq_social": (-2,  2),
            "news_politics":( 2,  1),
        }
        dx, dy = offsets.get(cat["key"], (2, 1))
        short = cat["label"].replace("Sports — ", "").replace(" (WWE/AEW)", "")\
                            .replace(" (UFC/Boxing)", "").replace("/International", "")\
                            .replace(" (Anime/K-pop)", "").replace(" (TV/Film/Music)", "")\
                            .replace(" (F1/Golf/Tennis)", "").replace(" (WNBA/NWSL)", "")\
                            .replace(" (Bravo/RH)", "")
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(dx * 4, dy * 4), fontsize=7.5, va="center")

    # Quadrant lines
    ax.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(50, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Quadrant labels
    ax.text( 5, 95, "Male + Liberal",   fontsize=8, color="gray", alpha=0.7)
    ax.text(72, 95, "Male + Conservative", fontsize=8, color="gray", alpha=0.7)
    ax.text( 5,  5, "Female + Liberal", fontsize=8, color="gray", alpha=0.7)
    ax.text(65,  5, "Female + Conservative", fontsize=8, color="gray", alpha=0.7)

    ax.set_xlabel("Political Lean  ←  Liberal (0) · · · Conservative (100)  →",
                  fontsize=9)
    ax.set_ylabel("% Male Audience", fontsize=9)
    ax.set_title("Audience Profile by Category:\nGender vs. Political Lean",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.15)

    # Color bar legend
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("polcmap",
            [pol_color(0), pol_color(50), pol_color(100)])
    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.12,
                      fraction=0.03, aspect=40)
    cb.set_label("Liberal ←                               → Conservative",
                 fontsize=8)
    cb.set_ticks([])

    plt.tight_layout()
    out = "out/figures/demographics/bro_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Age × Gender bubble chart ──────────────────────────────────────
def plot_age_gender_bubble():
    fig, ax = plt.subplots(figsize=(9, 6))

    for cat in CATEGORIES:
        x = cat["age_median"]
        y = cat["pct_male"]
        ax.scatter(x, y, s=200, zorder=3,
                   color=pol_color(cat["pol_score"]),
                   edgecolors="white", lw=0.8, alpha=0.9)
        short = cat["label"].replace("Sports — ", "").replace(" (WWE/AEW)", "")\
                            .replace(" (UFC/Boxing)", "").replace("/International", "")\
                            .replace(" (Anime/K-pop)", "").replace(" (TV/Film/Music)", "")\
                            .replace(" (F1/Golf/Tennis)", "").replace(" (WNBA/NWSL)", "")\
                            .replace(" (Bravo/RH)", "")
        ax.annotate(short, (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=7.5)

    ax.axhline(50, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.set_xlabel("Median Audience Age", fontsize=9)
    ax.set_ylabel("% Male Audience", fontsize=9)
    ax.set_title("Audience Profile by Category:\nMedian Age vs. Gender Split\n"
                 "(dot color = political lean: blue=liberal, red=conservative)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(15, 58)
    ax.set_ylim(20, 85)
    ax.grid(True, alpha=0.15)

    # Y-axis helper lines
    ax.text(15.5, 51, "← More Male", fontsize=7.5, color="gray", va="bottom")
    ax.text(15.5, 49, "← More Female", fontsize=7.5, color="gray", va="top")

    plt.tight_layout()
    out = "out/figures/demographics/age_gender_bubble.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating demographic visualizations...")
    plot_summary_grid()
    for cat in CATEGORIES:
        plot_individual_card(cat)
    plot_bro_scatter()
    plot_age_gender_bubble()
    print("\nAll demographic figures saved to out/figures/demographics/")
