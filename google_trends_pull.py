"""
Pull Google Trends interest-over-time data for a set of topics,
bracketing Elon Musk's Twitter acquisition (Oct 27, 2022).

Outputs: out/google_trends.csv
Columns: date, topic, interest (0-100 index, relative per batch)
"""

import time
import pandas as pd
from pytrends.request import TrendReq

# -----------------------------------------------------------------
# Topics: mix of political (may show bias) and neutral (controls)
# pytrends max 5 per request
# -----------------------------------------------------------------
TOPIC_BATCHES = [
    # Political / potentially biased
    ["Republican", "Democrat", "Biden", "Trump", "abortion"],
    ["gun control", "immigration", "climate change", "free speech", "censorship"],
    # Neutral controls
    ["NFL", "NBA", "Netflix", "Taylor Swift", "hurricane"],
]

TIMEFRAME = "2022-04-01 2023-04-30"   # 6 months pre + 6 months post acquisition
GEO = "US"
OUT_FILE = "out/google_trends.csv"


def pull_trends():
    pytrends = TrendReq(hl="en-US", tz=360)
    all_frames = []

    for batch in TOPIC_BATCHES:
        print(f"Fetching: {batch}")
        try:
            pytrends.build_payload(batch, timeframe=TIMEFRAME, geo=GEO)
            df = pytrends.interest_over_time()
            if df.empty:
                print(f"  No data returned for {batch}")
                continue
            df = df.drop(columns=["isPartial"], errors="ignore")
            df = df.reset_index().melt(id_vars="date", var_name="topic", value_name="interest")
            all_frames.append(df)
            time.sleep(2)   # be polite to the API
        except Exception as e:
            print(f"  Error on {batch}: {e}")
            time.sleep(10)

    if not all_frames:
        print("No data collected.")
        return

    result = pd.concat(all_frames, ignore_index=True)
    result.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(result)} rows to {OUT_FILE}")
    print(result.head())


if __name__ == "__main__":
    import os; os.makedirs("out", exist_ok=True)
    pull_trends()
