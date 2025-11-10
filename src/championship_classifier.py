import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def scrape_standings(year):
    """
    Scrape AFC and NFC standings for a given year from Pro Football Reference.
    Returns a DataFrame with relevant columns.
    """
    url = f"https://www.pro-football-reference.com/years/{year}/"
    print(f"Scraping {year} data from {url} ...")

    try:
        tables = pd.read_html(url)
    except Exception as e:
        print(f"❌ Failed to read {url}: {e}")
        return None

    afc_table, nfc_table = None, None

    # Detect AFC and NFC tables based on known labels in the table
    for df in tables:
        if any("AFC East" in str(x) for x in df.columns) or "AFC East" in str(df.values):
            afc_table = df
        elif any("NFC East" in str(x) for x in df.columns) or "NFC East" in str(df.values):
            nfc_table = df

    if afc_table is None or nfc_table is None:
        print(f"⚠️  No standings tables found for {year}, skipping.")
        return None

    combined = pd.concat([afc_table, nfc_table], ignore_index=True)
    combined.columns = combined.columns.str.strip()

    # Map possible column names to standardized names
    rename_map = {
        'Tm': 'Tm',
        'W-L%': 'Win_Pct',
        'W-L %': 'Win_Pct',
        'Pts': 'Pts_For',
        'PF': 'Pts_For',
        'Pts For': 'Pts_For',
        'Pts.1': 'Pts_Against',
        'PA': 'Pts_Against',
        'Pts Against': 'Pts_Against',
        'PD': 'PD',
        'SOS': 'SOS'
    }

    # Keep only columns that exist in the table
    cols_to_keep = [c for c in rename_map.keys() if c in combined.columns]
    filtered = combined[cols_to_keep].rename(columns=rename_map)

    # Add missing columns if necessary
    for col in ['Pts_For', 'Pts_Against', 'PD', 'SOS', 'Win_Pct']:
        if col not in filtered.columns:
            filtered[col] = None

    # Add Year column
    filtered["Year"] = year

    # Drop rows with empty team names
    filtered = filtered.dropna(subset=["Tm"])
    filtered = filtered[filtered["Tm"].str.len() > 2]

    return filtered

def scrape_all_years(start=2005, end=2024):
    """
    Scrape standings for multiple years and save to CSV.
    """
    all_data = []

    for year in range(start, end + 1):
        df = scrape_standings(year)
        if df is not None:
            all_data.append(df)
        time.sleep(1.5)  # polite delay to avoid hammering the website

    if not all_data:
        raise ValueError("No data scraped — check if site structure changed.")

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Scraped data for {len(all_data)} seasons ({start}–{end})")

    # Ensure the 'data' folder exists
    os.makedirs("../data", exist_ok=True)

    # Save combined CSV
    full_df.to_csv("../data/nfl_standings_2005_2024.csv", index=False)
    print("💾 Saved to data/nfl_standings_2005_2024.csv")

    return full_df

if __name__ == "__main__":
    data = scrape_all_years(2005, 2024)
    print("\nSample of scraped data:")
    print(data.head())