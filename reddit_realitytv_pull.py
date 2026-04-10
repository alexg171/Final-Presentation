"""
Pull daily post counts for reality TV subreddits to build a synthetic control.

Subreddits:
  LoveIsBlind, LoveIslandTV, thebachelor, BravoRealHousewives (already have),
  survivor, BigBrother, VPumpRules, MAFS_TV, 90DayFiance

Output: out/reddit_realitytv.tsv
  date | subreddit | n_posts | total_score
"""

import os, time, requests
from datetime import datetime, timedelta
import pandas as pd

START    = datetime(2020, 10, 27)
END      = datetime(2024, 10, 27)
OUT_FILE = "out/reddit_realitytv.tsv"
API_URL  = "https://arctic-shift.photon-reddit.com/api/posts/search"

SUBS = [
    "LoveIsBlind",
    "LoveIslandTV",
    "thebachelor",
    "BravoRealHousewives",
    "survivor",
    "BigBrother",
    "Vanderpumprules",
    "MAFS_TV",
    "90DayFiance",
]

os.makedirs("out", exist_ok=True)

# ── resume logic ──────────────────────────────────────────────────────────────
done = set()
if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE, sep="\t", parse_dates=["date"])
    for _, row in existing.iterrows():
        done.add((str(row["date"].date()), row["subreddit"]))
    print(f"Resuming -- {len(done)} (date, subreddit) pairs already collected.")
else:
    with open(OUT_FILE, "w") as f:
        f.write("date\tsubreddit\tn_posts\ttotal_score\n")
    print("Starting fresh.")

# ── pull ──────────────────────────────────────────────────────────────────────
all_dates = []
d = START
while d <= END:
    all_dates.append(d)
    d += timedelta(days=1)

total_pairs = len(all_dates) * len(SUBS)
done_count  = len(done)

for current_date in all_dates:
    date_str  = current_date.strftime("%Y-%m-%d")
    after_ts  = int(current_date.timestamp())
    before_ts = int((current_date + timedelta(days=1)).timestamp())

    for sub in SUBS:
        key = (date_str, sub)
        if key in done:
            done_count += 1
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
                        f.write(f"{date_str}\t{sub}\t{n_posts}\t{total_score}\n")
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

    if done_count % (len(SUBS) * 30) == 0 and done_count > 0:
        pct = done_count / total_pairs * 100
        print(f"  {date_str} -- {done_count:,}/{total_pairs:,} pairs ({pct:.1f}%)")

print(f"\nDone. {done_count:,} pairs written to {OUT_FILE}")
