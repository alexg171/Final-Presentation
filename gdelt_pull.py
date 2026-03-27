"""
Pull GDELT Doc API v2 article counts by day for a set of topics,
bracketing Elon Musk's Twitter acquisition (Oct 27, 2022).

GDELT free API — no key needed.
Outputs: out/gdelt_coverage.csv
Columns: date, topic, article_count
"""

import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------------------------------------
# Same topics as Google Trends for direct comparison
# -----------------------------------------------------------------
TOPICS = [
    # Political
    "Republican", "Democrat", "Biden", "Trump", "abortion",
    "gun control", "immigration", "climate change", "free speech", "censorship",
    # Neutral controls
    "NFL", "NBA", "Netflix", "Taylor Swift", "hurricane",
]

START_DATE = datetime(2022, 4, 1)
END_DATE   = datetime(2023, 4, 30)
OUT_FILE   = "out/gdelt_coverage.csv"

GDELT_URL  = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_article_count(topic: str, date: datetime) -> int:
    """Return the number of English news articles mentioning `topic` on `date`."""
    day_str   = date.strftime("%Y%m%d")
    start_str = day_str + "000000"
    end_str   = day_str + "235959"

    params = {
        "query":         f'"{topic}" sourcelang:english',
        "mode":          "artlist",
        "maxrecords":    "250",
        "startdatetime": start_str,
        "enddatetime":   end_str,
        "format":        "json",
    }

    try:
        r = requests.get(GDELT_URL, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            return len(articles)
        else:
            return -1
    except Exception:
        return -1


def pull_gdelt():
    records = []
    dates = pd.date_range(START_DATE, END_DATE, freq="W")   # weekly to keep runtime reasonable

    total = len(TOPICS) * len(dates)
    done  = 0

    for topic in TOPICS:
        print(f"\nTopic: {topic}")
        for date in dates:
            count = fetch_article_count(topic, date.to_pydatetime())
            records.append({"date": date.date(), "topic": topic, "article_count": count})
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{total} done...")
            time.sleep(0.5)   # avoid rate-limiting

    df = pd.DataFrame(records)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_FILE}")
    print(df.head(10))


if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    pull_gdelt()
