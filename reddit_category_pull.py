"""
Pull daily post counts for category-specific subreddits via Arctic Shift API.
No ideology labels needed — just volume per subreddit per day.

Subreddits mapped to Twitter categories:
  wrestling     -> SquaredCircle
  sports_nba    -> nba
  sports_nfl    -> nfl
  sports_mlb    -> baseball
  sports_nhl    -> hockey
  sports_soccer -> soccer
  sports_college-> CFB
  combat_sports -> MMA
  reality_tv    -> BravoRealHousewives
  entertainment -> television
  taylor_swift  -> TaylorSwift
  fandom        -> anime, kpop
  tech_gaming   -> gaming
  lgbtq_social  -> lgbt
  musk_twitter  -> Twitter (subreddit)
  news_events   -> news

Output: out/reddit_category.tsv
  date | subreddit | category | n_posts | total_score
"""

import os, time, requests, json
from datetime import datetime, timedelta
import pandas as pd

START    = datetime(2020, 10, 27)
END      = datetime(2024, 10, 27)
OUT_FILE = "out/reddit_category.tsv"
API_URL  = "https://arctic-shift.photon-reddit.com/api/posts/search"

SUBREDDIT_CATEGORY = {
    "SquaredCircle":      "wrestling",
    "nba":                "sports_nba",
    "nfl":                "sports_nfl",
    "baseball":           "sports_mlb",
    "hockey":             "sports_nhl",
    "soccer":             "sports_soccer",
    "CFB":                "sports_college",
    "MMA":                "combat_sports",
    "BravoRealHousewives":"reality_tv",
    "television":         "entertainment",
    "TaylorSwift":        "taylor_swift",
    "anime":              "fandom",
    "kpop":               "fandom",
    "gaming":             "tech_gaming",
    "lgbt":               "lgbtq_social",
    "worldnews":          "news_events",
}

os.makedirs("out", exist_ok=True)

# ── resume logic ──────────────────────────────────────────────────────────────
done = set()
if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE, sep="\t", parse_dates=["date"])
    for _, row in existing.iterrows():
        done.add((str(row["date"].date()), row["subreddit"]))
    print(f"Resuming — {len(done)} (date, subreddit) pairs already collected.")
else:
    with open(OUT_FILE, "w") as f:
        f.write("date\tsubreddit\tcategory\tn_posts\ttotal_score\n")
    print("Starting fresh.")

# ── pull ──────────────────────────────────────────────────────────────────────
all_dates = []
d = START
while d <= END:
    all_dates.append(d)
    d += timedelta(days=1)

total_pairs = len(all_dates) * len(SUBREDDIT_CATEGORY)
done_count  = len(done)
idx = 0

for current_date in all_dates:
    date_str   = current_date.strftime("%Y-%m-%d")
    after_ts   = int(current_date.timestamp())
    before_ts  = int((current_date + timedelta(days=1)).timestamp())

    for sub, cat in SUBREDDIT_CATEGORY.items():
        idx += 1
        key = (date_str, sub)
        if key in done:
            continue

        params = {
            "subreddit": sub,
            "after":     after_ts,
            "before":    before_ts,
            "limit":     100,
            "fields":    "score",
        }

        for attempt in range(4):
            try:
                r = requests.get(API_URL, params=params, timeout=20)
                if r.status_code == 200:
                    posts = r.json().get("data", [])
                    n_posts     = len(posts)
                    total_score = sum(p.get("score", 0) for p in posts)
                    with open(OUT_FILE, "a") as f:
                        f.write(f"{date_str}\t{sub}\t{cat}\t{n_posts}\t{total_score}\n")
                    done.add(key)
                    done_count += 1
                    break
                elif r.status_code == 429:
                    time.sleep(10)
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(5)

        time.sleep(0.25)

    if idx % (len(SUBREDDIT_CATEGORY) * 30) == 0:
        pct = done_count / total_pairs * 100
        print(f"  {date_str} — {done_count:,}/{total_pairs:,} pairs ({pct:.1f}%)")

print(f"\nDone. {done_count:,} pairs written to {OUT_FILE}")
