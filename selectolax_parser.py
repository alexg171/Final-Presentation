from selectolax.lexbor import LexborHTMLParser

def parse_with_selectolax(html_content, date="2019-01-01"):
    parser = LexborHTMLParser(html_content)
    
    # 1. Main Trending Table
    main_data = []
    # .css() is correct for multiple elements
    for block in parser.css("#all_table .tek_tablo"):
        # Use .css_first() for a single element
        hour_node = block.css_first(".trend_baslik611")
        hour = hour_node.text(strip=True) if hour_node else "00:00"
        
        rows = block.css(".tr_table")
        vols = block.css(".tr_table1")
        
        for row, vol in zip(rows, vols):
            rank_node = row.css_first(".sira611")
            topic_node = row.css_first(".word_ars")
            volume_node = vol.css_first(".volume61")
            
            main_data.append({
                "Date": date,
                "Hour": hour,
                "Rank": rank_node.text(strip=True) if rank_node else "N/A",
                "Topic": topic_node.text(strip=True) if topic_node else "N/A",
                "Volume": volume_node.text(strip=True).replace(' tweet', '').strip() if volume_node else "0"
            })

    # 2. Summary Boxes
    summaries = {
        "en_volume_b": ("table_bbiv", "Peak_Volume"),
        "en_hour_b": ("table_bbivv", "Duration")
    }
    
    summary_results = {"en_volume_b": [], "en_hour_b": []}
    
    for div_id, (val_class, col_name) in summaries.items():
        container = parser.css_first(f"#{div_id}")
        if container:
            topics = container.css(".table_bbk")
            values = container.css(f".{val_class}")
            
            for i, (t, v) in enumerate(zip(topics, values), 1):
                summary_results[div_id].append({
                    "Date": date,
                    "Rank": i,
                    "Topic": t.text(strip=True),
                    col_name: v.text(strip=True)
                })

    return main_data, summary_results["en_volume_b"], summary_results["en_hour_b"]