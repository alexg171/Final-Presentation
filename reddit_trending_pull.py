"""
Pull historical top posts from ideologically validated political subreddits
via Arctic Shift Reddit archive (free pushshift replacement).

Subreddit selection based on empirically derived ideology scores from:
  Lai et al. (2024) "Estimating the Ideology of Political YouTube Videos"
  Political Analysis. Figure 2: subreddit liberal→conservative spectrum.

Used as synthetic parallel trend / counterfactual for Twitter DiD study.
Reddit was not acquired by Musk — any post-Oct 2022 divergence from Twitter
in right/left share is attributable to the Twitter-specific treatment.

No setup required — public API, no auth needed.

Outputs: out/reddit_trending.csv
Columns:  date, subreddit, label, title, score, rank
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

START    = datetime(2020, 10, 27)
END      = datetime(2024, 10, 27)
OUT_FILE = "out/reddit_trending.tsv"
API_URL  = "https://arctic-shift.photon-reddit.com/api/posts/search"

# Ideologically validated subreddits — Lai et al. (2024) Figure 2
SUBREDDIT_LABELS = {
    # Right-leaning
    "conservative": "right",
    "republican":   "right",
    "libertarian":  "right",
    # Left-leaning
    "liberal":      "left",
    "progressive":  "left",
    "politics":     "left",
    # Center anchor
    "news":         "center",
    "worldnews":    "center",
}


def fetch_sub_day(subreddit: str, date: datetime) -> list:
    """Fetch posts from a subreddit for a single day (sorted desc by time)."""
    after  = int(date.timestamp())
    before = int((date + timedelta(days=1)).timestamp())

    params = {
        "subreddit": subreddit,
        "after":     str(after),
        "before":    str(before),
        "sort":      "desc",
        "limit":     "100",
        "fields":    "subreddit,title,score,created_utc",
    }

    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def fetch_day(date: datetime) -> pd.DataFrame:
    """Aggregate top posts across all subreddits for one day, tagged by label."""
    rows = []
    for sub, label in SUBREDDIT_LABELS.items():
        try:
            posts = fetch_sub_day(sub, date)
            for p in posts:
                rows.append({
                    "date":      date.date(),
                    "subreddit": p.get("subreddit", sub),
                    "label":     label,
                    "title":     p.get("title", "").replace("\n", " ").replace("\r", " "),
                    "score":     p.get("score", 0),
                })
            time.sleep(0.3)
        except Exception as e:
            print(f"  (warn: r/{sub} failed — {e})")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def pull_reddit():
    os.makedirs("out", exist_ok=True)

    # Resume from where we left off
    already_done = set()
    header_written = False
    if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
        try:
            existing = pd.read_csv(OUT_FILE, sep="\t", parse_dates=["date"])
            already_done = set(existing["date"].dt.date.unique())
            header_written = True
            print(f"Resuming — {len(already_done)} days already collected.")
        except Exception:
            print("Existing file unreadable — starting fresh.")
            os.remove(OUT_FILE)
    total_days = (END - START).days + 1
    current    = START
    day_num    = 0

    while current <= END:
        day_num += 1
        d = current.date()

        if d in already_done:
            print(f"[{day_num}/{total_days}] {d} ... skipped.")
            current += timedelta(days=1)
            continue

        print(f"[{day_num}/{total_days}] {d} ...", end=" ", flush=True)

        for attempt in range(3):
            try:
                df = fetch_day(current)
                if not df.empty:
                    df.to_csv(OUT_FILE, mode="a", index=False, header=not header_written, sep="\t")
                    header_written = True
                    print(f"{len(df)} posts saved.")
                else:
                    print("no data.")
                break
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"ERROR: {e} — retrying in {wait}s...")
                time.sleep(wait)
        else:
            print("FAILED after 3 attempts, skipping.")

        time.sleep(0.5)
        current += timedelta(days=1)

    print(f"\nDone. Output: {OUT_FILE}")


if __name__ == "__main__":
    pull_reddit()
