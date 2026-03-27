import os
import pandas as pd
import time
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selectolax_parser import parse_with_selectolax # Ensure this filename is correct

def get_headful_driver():
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1200,800")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    
    return webdriver.Chrome(options=chrome_options)

def scrape_trending_page(url, date, driver, timestamp):
    driver.get(url)
    
    try:
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.ID, "all_table")))
        
        time.sleep(1) 
        
        rendered_html = driver.page_source
        main_list, most_tweeted, longest_trending = parse_with_selectolax(rendered_html, date)
        
        if main_list and main_list[0]['Topic'] != '-':
            os.makedirs("out", exist_ok=True)
            pd.DataFrame(main_list).to_csv(f"out/trending_{timestamp}.csv", index=False, mode='a', header=not os.path.exists(f"out/trending_{timestamp}.csv"))
            pd.DataFrame(most_tweeted).to_csv(f"out/most_{timestamp}.csv", index=False, mode='a', header=not os.path.exists(f"out/most_{timestamp}.csv"))
            pd.DataFrame(longest_trending).to_csv(f"out/longest_{timestamp}.csv", index=False, mode='a', header=not os.path.exists(f"out/longest_{timestamp}.csv"))
            print(f"Success for {date}")
        else:
            print(f"Data was still dashes for {date}. You might need a longer wait or a scroll.")

    except Exception as e:
        print(f"Error on {date}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="01-01-2019")
    parser.add_argument("--end_date", type=str, default="01-02-2019")
    args = parser.parse_args()

    date_range = pd.date_range(start=args.start_date, end=args.end_date)
    timestamp = int(time.time())
    print(f"Starting scraping from {args.start_date} to {args.end_date} at timestamp {timestamp}")
    
    driver = get_headful_driver()
    
    try:
        for date in date_range:
            formatted_date = date.strftime("%d-%m-%Y")
            url = f"https://archive.twitter-trending.com/united-states/{formatted_date}"
            scrape_trending_page(url, date, driver, timestamp)
    finally:
        timestop = int(time.time())
        print(f"Scraping completed in {timestop - timestamp} seconds.")
        driver.quit()