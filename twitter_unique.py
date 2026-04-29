import pandas as pd
from category_lexicon import classify_category

UNIQUE_PATH = "out/unique_topics.csv"

# ── Shorthand → official category name ───────────────────────────────────────
LABEL_ALIASES = {
    # sports
    "soccer": "sports_soccer",
    "nfl": "sports_nfl",
    "nba": "sports_nba",
    "mlb": "sports_mlb",
    "nhl": "sports_nhl",
    "college sports": "sports_college",
    "college": "sports_college",
    "f1": "sports_other",
    "golf": "sports_other",
    "tennis": "sports_other",
    "sports other": "sports_other",
    "womens sports": "sports_womens",
    "wnba": "sports_womens",
    "boxing sports": "combat_sports",
    "boxing": "combat_sports",
    "mma": "combat_sports",
    "ufc": "combat_sports",
    
    # entertainment
    "music": "entertainment",
    "tv": "entertainment",
    "movies": "entertainment",
    "film": "entertainment",
    "movie": "entertainment",
    
    # other shorthands
    "kpop": "fandom",
    "anime": "fandom",
    "reality": "reality_tv",
    "reality tv": "reality_tv",
    "news": "news_events",
    "news events": "news_events",
    "taylor swift": "taylor_swift",
    "taylor": "taylor_swift",
    "politics": "politics",
    "political": "politics",
    "lgbtq": "lgbtq_social",
    "social justice": "lgbtq_social",
    "religion": "religious",
    "gaming": "tech_gaming",
    "tech": "tech_gaming",
    "crypto": "tech_gaming",
    "manosphere": "manosphere",
    "true crime": "true_crime",
    "musk": "musk_twitter",
    "twitter": "musk_twitter",
    "holidays": "holidays",
    "holiday": "holidays",
    "wrestling": "wrestling",
    "wwe": "wrestling",
    "social": "social_filler",
    "filler": "social_filler",
    "social filler": "social_filler",
    "ads": "social_filler",
    "sponsored": "social_filler",
    "nascar": "sports_other",
    "sports": "sports_other",
    "woke": "lgbtq_social",
    "woke maybe": "lgbtq_social",
    "news politics": "news_events",
    "late night": "entertainment",
    "lindsey graham?": "politics",
    "zukerberg facebook idk": "news_events",
}


def normalize_label(raw: str) -> str:
    """Map shorthand user label to official category name."""
    if pd.isna(raw) or str(raw).strip() == "":
        return raw
    key = str(raw).strip().lower()
    return LABEL_ALIASES.get(key, key)  # fall back to whatever they typed if no match


# ── Load twitter trending data ────────────────────────────────────────────────
df = pd.read_csv("out/twitter_trending_4yr.csv")
freq = df["Topic"].value_counts()

# ── Load existing unique_topics to preserve manual labels ────────────────────
try:
    existing = pd.read_csv(UNIQUE_PATH, dtype=str)
    # Build lookup: topic -> manually_labeled value
    # Normalize shorthand labels to official names on load
    existing["manually_labeled"] = existing["manually_labeled"].apply(normalize_label)
    manual_map = (
        existing[
            existing["manually_labeled"].notna() & (existing["manually_labeled"] != "")
        ]
        .set_index("topic")["manually_labeled"]
        .to_dict()
    )
    print(f"  Loaded {len(manual_map)} existing manual labels to preserve.")
except FileNotFoundError:
    manual_map = {}
    print("  No existing unique_topics.csv found — starting fresh.")

# ── Build fresh DataFrame with current lexicon labels ────────────────────────
unique_df = pd.DataFrame(
    {
        "topic": freq.index,
        "frequency": freq.values,
    }
)

unique_df["label"] = unique_df["topic"].apply(classify_category)

# ── Overlay manual labels where present ──────────────────────────────────────
unique_df["manually_labeled"] = unique_df["topic"].map(manual_map)

# Where a manual label exists, use it as the effective label
mask = unique_df["manually_labeled"].notna() & (unique_df["manually_labeled"] != "")
unique_df.loc[mask, "label"] = unique_df.loc[mask, "manually_labeled"]

# ── Save ──────────────────────────────────────────────────────────────────────
unique_df.to_csv(UNIQUE_PATH, index=False)

n_manual = mask.sum()
n_total = len(unique_df)
n_other = (unique_df["label"] == "other").sum()
pct_other = 100 * n_other / n_total

print(f"  {n_total:,} unique topics written to {UNIQUE_PATH}")
print(f"  {n_manual} manual label overrides applied")
print(f"  {n_other:,} still 'other'  ({pct_other:.1f}% of unique topics)")
print("Done.")
