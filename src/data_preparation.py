"""
Data Preparation Module for NFL Championship Prediction
Cleans scraped data and adds Super Bowl winner labels
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure we're working from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Super Bowl winners by season (season = regular season year)
SUPER_BOWL_WINNERS = {
    2005: "Pittsburgh Steelers",
    2006: "Indianapolis Colts",
    2007: "New York Giants",
    2008: "Pittsburgh Steelers",
    2009: "New Orleans Saints",
    2010: "Green Bay Packers",
    2011: "New York Giants",
    2012: "Baltimore Ravens",
    2013: "Seattle Seahawks",
    2014: "New England Patriots",
    2015: "Denver Broncos",
    2016: "New England Patriots",
    2017: "Philadelphia Eagles",
    2018: "New England Patriots",
    2019: "Kansas City Chiefs",
    2020: "Tampa Bay Buccaneers",
    2021: "Los Angeles Rams",
    2022: "Kansas City Chiefs",
    2023: "Kansas City Chiefs",
    # 2024: TBD (as of February 2026, this would be known)
}


def clean_team_name(name):
    """
    Standardize team names by removing asterisks and extra characters.
    """
    if pd.isna(name):
        return None
    
    name = str(name).strip()
    # Remove playoff indicators (*, +)
    name = name.replace('*', '').replace('+', '').strip()
    
    # Filter out division headers
    divisions = ['AFC East', 'AFC West', 'AFC North', 'AFC South',
                 'NFC East', 'NFC West', 'NFC North', 'NFC South']
    
    if name in divisions or 'AFC' in name or 'NFC' in name:
        return None
    
    return name


def prepare_data(input_path='data/nfl_standings_2005_2024.csv', 
                 output_path='data/nfl_ml_ready.csv'):
    """
    Load, clean, and prepare NFL data for machine learning.
    
    Returns:
        DataFrame with cleaned data and target variable (Won_SB)
    """
    print("🔄 Loading scraped data...")
    df = pd.read_csv(input_path)
    
    print(f"   Raw data shape: {df.shape}")
    
    # Clean team names
    print("🧹 Cleaning team names...")
    df['Tm'] = df['Tm'].apply(clean_team_name)
    
    # Drop rows with no valid team name
    df = df.dropna(subset=['Tm'])
    
    # Convert numeric columns
    print("🔢 Converting numeric columns...")
    numeric_cols = ['Win_Pct', 'Pts_For', 'Pts_Against', 'PD', 'SOS']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate additional features
    print("⚙️  Engineering features...")
    
    # Points per game (assuming 16-17 game seasons)
    df['PPG'] = df['Pts_For'] / 16  # Approximate
    df['PA_PG'] = df['Pts_Against'] / 16
    
    # Win percentage (if missing, calculate from other data if available)
    if df['Win_Pct'].isna().any():
        print("   ⚠️  Some Win_Pct values missing")
    
    # Add target variable: Did this team win the Super Bowl?
    print("🏆 Adding Super Bowl winner labels...")
    
    def is_champion(row):
        year = int(row['Year'])
        team = row['Tm']
        
        if year in SUPER_BOWL_WINNERS:
            return 1 if SUPER_BOWL_WINNERS[year] == team else 0
        else:
            return np.nan  # Unknown (e.g., 2024 season)
    
    df['Won_SB'] = df.apply(is_champion, axis=1)
    
    # Drop rows where we don't know the outcome yet
    df_labeled = df.dropna(subset=['Won_SB'])
    
    print(f"\n✅ Cleaned data shape: {df_labeled.shape}")
    print(f"   Champions in dataset: {df_labeled['Won_SB'].sum()}")
    print(f"   Years covered: {df_labeled['Year'].min()} - {df_labeled['Year'].max()}")
    
    # Save cleaned data
    df_labeled.to_csv(output_path, index=False)
    print(f"💾 Saved cleaned data to {output_path}")
    
    return df_labeled


def get_feature_columns():
    """
    Returns list of feature columns to use for modeling.
    """
    return [
        'Win_Pct',
        'Pts_For',
        'Pts_Against',
        'PD',
        'PPG',
        'PA_PG',
        # SOS excluded if mostly null
    ]


def create_team_season_features(df):
    """
    Advanced feature engineering: rolling stats, momentum, etc.
    
    This is where you'd add:
    - Last N games performance
    - Year-over-year improvement
    - Historical playoff appearances
    - Division strength
    """
    df = df.copy()
    
    # Sort by team and year
    df = df.sort_values(['Tm', 'Year'])
    
    # Previous season performance (lag features)
    df['Prev_Win_Pct'] = df.groupby('Tm')['Win_Pct'].shift(1)
    df['Prev_PD'] = df.groupby('Tm')['PD'].shift(1)
    
    # Year-over-year improvement
    df['Win_Pct_Change'] = df['Win_Pct'] - df['Prev_Win_Pct']
    df['PD_Change'] = df['PD'] - df['Prev_PD']
    
    return df


if __name__ == "__main__":
    # Run data preparation
    df = prepare_data()
    
    print("\n📊 Sample of prepared data:")
    print(df[['Year', 'Tm', 'Win_Pct', 'Pts_For', 'Pts_Against', 'Won_SB']].head(10))
    
    print("\n🏆 Super Bowl winners in dataset:")
    champions = df[df['Won_SB'] == 1][['Year', 'Tm', 'Win_Pct', 'PD']]
    print(champions)
