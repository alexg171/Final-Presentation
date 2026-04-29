"""
Pull daily post counts for category-specific subreddits via Arctic Shift time series API.
One API call per subreddit (count) + one for score = 2 calls per subreddit total.
Far faster than the old per-day approach (was 1,462 calls per subreddit).

Output: out/reddit_category.tsv
  date | subreddit | category | n_posts | total_score
"""

import os, time, requests
import pandas as pd
import sys
sys.path.insert(0, ".")
from category_subreddit_mapping import SUBREDDIT_CATEGORY

START    = "2020-10-27"
END      = "2024-10-27"
OUT_FILE = "out/reddit_category.tsv"
API_URL  = "https://arctic-shift.photon-reddit.com/api/time_series"

os.makedirs("out", exist_ok=True)

# ── check what's already done ─────────────────────────────────────────────────
done_subs = set()
if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE, sep="\t", parse_dates=["date"])
    # A sub is "done" if it has data covering the full date range
    for sub in existing["subreddit"].unique():
        sub_data = existing[existing["subreddit"] == sub]
        if len(sub_data) >= 1400:  # ~1462 days, allow small gaps
            done_subs.add(sub)
    print(f"Already complete: {sorted(done_subs)}")
else:
    with open(OUT_FILE, "w") as f:
        f.write("date\tsubreddit\tcategory\tn_posts\ttotal_score\n")
    print("Starting fresh.")

# ── fetch time series for one subreddit ──────────────────────────────────────

def fetch_series(sub: str, metric: str) -> dict:
    """Returns {date_str: value} for the given metric (posts/count or posts/sum_score)."""
    for attempt in range(4):
        try:
            r = requests.get(API_URL, params={
                "key":       f"r/{sub}/{metric}",
                "precision": "day",
                "after":     START,
                "before":    END,
            }, timeout=30)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return {
                    pd.Timestamp(row["date"], unit="s").strftime("%Y-%m-%d"): row["value"]
                    for row in data
                }
            elif r.status_code == 429:
                print(f"  Rate limited — waiting 15s")
                time.sleep(15)
            else:
                print(f"  HTTP {r.status_code} for {sub}/{metric}")
                time.sleep(2)
        except Exception as e:
            print(f"  Error {sub}/{metric}: {e}")
            time.sleep(5)
    return {}


# ── pull all subreddits ───────────────────────────────────────────────────────

to_pull = [s for s in SUBREDDIT_CATEGORY if s not in done_subs]
print(f"\nPulling {len(to_pull)} subreddits: {to_pull}\n")

for sub in to_pull:
    cat = SUBREDDIT_CATEGORY[sub]
    print(f"  {sub} ({cat}) ...", end=" ", flush=True)

    counts = fetch_series(sub, "posts/count")
    time.sleep(0.5)
    scores = fetch_series(sub, "posts/sum_score")
    time.sleep(0.5)

    if not counts:
        print("NO DATA")
        continue

    rows = []
    for date_str, n in counts.items():
        rows.append({
            "date":        date_str,
            "subreddit":   sub,
            "category":    cat,
            "n_posts":     n,
            "total_score": scores.get(date_str, 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, sep="\t", mode="a", header=False, index=False)
    print(f"{len(rows)} days collected")

print(f"\nDone. Output: {OUT_FILE}")
