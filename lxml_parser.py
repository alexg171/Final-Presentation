import pandas as pd
from bs4 import BeautifulSoup
import os

# To use this, ensure you have lxml installed: pip install lxml
# It is significantly faster than the default html.parser

def parse_trending_file(html_content, date="2019-01-01"):
    # Using 'lxml' for high-speed parsing
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 1. Process Main Trending Table
    # Using CSS selectors (select) instead of find_all for speed
    main_data = []
    blocks = soup.select("#all_table .tek_tablo")
    
    for block in blocks:
        # Extract hour once per block
        hour_tag = block.select_one(".trend_baslik611")
        hour = hour_tag.get_text(strip=True) if hour_tag else "00:00"
        
        # Select all rows at once
        rows = block.select(".tr_table")
        vols = block.select(".tr_table1")
        
        for row, vol in zip(rows, vols):
            main_data.append({
                "Date": date,
                "Hour": hour,
                "Rank": getattr(row.select_one(".sira611"), 'text', 'N/A').strip(),
                "Topic": getattr(row.select_one(".word_ars"), 'text', 'N/A').strip(),
                "Volume": getattr(vol.select_one(".volume61"), 'text', '0').replace(' tweet', '').strip()
            })

    # 2. Process Summary Boxes (Most Tweeted & Longest)
    # Mapping configuration: {ID: (Value_Class, Column_Name)}
    summaries = {
        "en_volume_b": ("table_bbiv", "Peak_Volume"),
        "en_hour_b": ("table_bbivv", "Duration")
    }
    
    summary_results = {"en_volume_b": [], "en_hour_b": []}
    
    for div_id, (val_class, col_name) in summaries.items():
        container = soup.select_one(f"#{div_id}")
        if container:
            # Finding all spans with the target classes within this specific ID
            topics = container.select(".table_bbk")
            values = container.select(f".{val_class}")
            
            for i, (t, v) in enumerate(zip(topics, values), 1):
                summary_results[div_id].append({
                    "Date": date,
                    "Rank": i,
                    "Topic": t.get_text(strip=True),
                    col_name: v.get_text(strip=True)
                })

    return main_data, summary_results["en_volume_b"], summary_results["en_hour_b"]

# --- Execution ---
file_path = 'trending.html'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    
    main_list, most_tweeted, longest_trending = parse_trending_file(html_data)
    
    # Fast export to CSV
    pd.DataFrame(main_list).to_csv("trending_table_data.csv", index=False)
    pd.DataFrame(most_tweeted).to_csv("most_tweeted_data.csv", index=False)
    pd.DataFrame(longest_trending).to_csv("longest_trending_data.csv", index=False)
    
    print("Scraping complete. Data files generated.")
else:
    print(f"File {file_path} not found. Are you sure it exists, or is it just a figment of your imagination?")