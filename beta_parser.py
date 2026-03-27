import pandas as pd
from bs4 import BeautifulSoup
import re

with open('trending.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')

dummy_date = '2019-01-01'

def get_trending_table_data(soup):
    entries = []
    all_table = soup.find("div", id="all_table")
    trending_tables = all_table.find_all("div", "tek_tablo") if all_table else []

    for trending_table in trending_tables:
        hour = trending_table.find("div", "trend_baslik611").get_text(strip=True) if trending_table.find("div", "trend_baslik611") else "00:00"
        trending_items = trending_table.find_all("tr", class_="tr_table")
        volume_items = trending_table.find_all("tr", class_="tr_table1")
        for item, volume_item in zip(trending_items, volume_items):
            rank = item.find("td", class_="sira611").get_text(strip=True) if item.find("td", class_="sira611") else "N/A"
            topic = item.find("span", class_="word_ars").get_text(strip=True) if item.find("span", class_="word_ars") else "N/A"
            volume = volume_item.find("span", "volume61").get_text(strip=True).replace(' tweet', '') if volume_item.find("span", "volume61") else "0"
            entries.append({
                "Date": dummy_date,
                "Hour": hour,
                "Rank": rank,
                "Topic": topic,
                "Volume": volume
            })
    return entries

def most_tweeted(soup):
    most_tweeted_list = []
    head = soup.find("span", id="en_volume_b")
    if head:
        topics = head.find_all("span", "table_bbk")
        volumes = head.find_all("span", "table_bbiv")
        for rank, topic, volume in zip(range(1, 6), topics, volumes):
            most_tweeted_list.append({
                "Date": dummy_date,
                "Rank": rank,
                "Topic": topic.get_text(strip=True),
                "Peak_Volume": volume.get_text(strip=True)
            })
    return most_tweeted_list

def longest_trending(soup):
    longest_trending_list = []
    head = soup.find("span", id="en_hour_b")
    if head:
        topics = head.find_all("span", "table_bbk")
        durations = head.find_all("span", "table_bbivv")

        for rank, topic, duration in zip(range(1, 6), topics, durations):
            longest_trending_list.append({
                "Date": dummy_date,
                "Rank": rank,
                "Topic": topic.get_text(strip=True),
                "Duration": duration.get_text(strip=True)
            })
    return longest_trending_list


trending_data = get_trending_table_data(soup)
df_trending = pd.DataFrame(trending_data)
df_trending.to_csv("trending_table_data.csv", index=False)

most_tweeted_data = most_tweeted(soup)
df_most_tweeted = pd.DataFrame(most_tweeted_data)
df_most_tweeted.to_csv("most_tweeted_data.csv", index=False)

longest_trending_data = longest_trending(soup)
df_longest_trending = pd.DataFrame(longest_trending_data)
df_longest_trending.to_csv("longest_trending_data.csv", index=False)

