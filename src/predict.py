"""
Prediction Module - Predict Future NFL Champions
Uses trained XGBoost model to predict championship probabilities
"""

import pandas as pd
import numpy as np
import os
import sys

# Ensure we're working from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from train_model import NFLChampionshipPredictor
from data_preparation import create_team_season_features


def load_current_season_data(csv_path='data/nfl_standings_2005_2024.csv', year=2024):
    """
    Load data for a specific season to make predictions.
    """
    df = pd.read_csv(csv_path)
    df_year = df[df['Year'] == year].copy()
    
    return df_year


def predict_championship_probabilities(predictor, df, year=2024):
    """
    Predict championship probability for each team in the given year.
    
    Args:
        predictor: Trained NFLChampionshipPredictor instance
        df: DataFrame with team stats for all years
        year: Year to predict
    
    Returns:
        DataFrame with teams ranked by championship probability
    """
    # Get data for prediction year
    df_year = df[df['Year'] == year].copy()
    
    if len(df_year) == 0:
        print(f"⚠️  No data found for year {year}")
        return None
    
    # Add advanced features
    df_full = create_team_season_features(df)
    df_predict = df_full[df_full['Year'] == year].copy()
    
    # Prepare features (same as training)
    X_predict = df_predict[predictor.features].fillna(0)
    
    # Make predictions
    probabilities = predictor.model.predict_proba(X_predict)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'Team': df_predict['Tm'].values,
        'Win_Pct': df_predict['Win_Pct'].values,
        'Point_Diff': df_predict['PD'].values,
        'Championship_Prob': probabilities
    })
    
    # Rank by probability
    results = results.sort_values('Championship_Prob', ascending=False)
    results['Rank'] = range(1, len(results) + 1)
    
    return results[['Rank', 'Team', 'Championship_Prob', 'Win_Pct', 'Point_Diff']]


def predict_future_champion(year=2025):
    """
    Main function to predict the champion for a future/current season.
    """
    print("=" * 70)
    print(f"NFL CHAMPIONSHIP PREDICTION - {year} Season")
    print("=" * 70)
    
    # Load trained model
    print("\n📥 Loading trained model...")
    try:
        predictor = NFLChampionshipPredictor.load_model()
    except FileNotFoundError:
        print("❌ Model not found! Please run train_model.py first.")
        return
    
    # Load all data (to get lag features)
    print("📊 Loading historical data...")
    from data_preparation import prepare_data
    df = prepare_data()
    
    # Make predictions
    print(f"\n🔮 Predicting {year} Championship Probabilities...\n")
    
    results = predict_championship_probabilities(predictor, df, year=year)
    
    if results is None:
        return
    
    # Display results
    print("🏆 Championship Probability Rankings:")
    print("=" * 70)
    
    # Format for display
    results_display = results.copy()
    results_display['Championship_Prob'] = results_display['Championship_Prob'].apply(lambda x: f"{x*100:.2f}%")
    results_display['Win_Pct'] = results_display['Win_Pct'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    results_display['Point_Diff'] = results_display['Point_Diff'].apply(lambda x: f"{int(x):+d}" if pd.notna(x) else "N/A")
    
    print(results_display.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"🎯 PREDICTED {year} CHAMPION: {results.iloc[0]['Team']}")
    print(f"   Probability: {results.iloc[0]['Championship_Prob']*100:.2f}%")
    print("=" * 70)
    
    # Save predictions
    import os
    os.makedirs('results', exist_ok=True)
    results.to_csv(f'results/predictions_{year}.csv', index=False)
    print(f"\n💾 Predictions saved to results/predictions_{year}.csv")
    
    return results


def compare_predictions_to_actual():
    """
    Compare model predictions to actual outcomes for validation.
    """
    print("=" * 70)
    print("PREDICTION VALIDATION - Historical Accuracy")
    print("=" * 70)
    
    # Load model
    predictor = NFLChampionshipPredictor.load_model()
    
    # Load data
    from data_preparation import prepare_data, SUPER_BOWL_WINNERS
    df = prepare_data()
    
    print("\n📊 Comparing Predictions vs Actual Winners:\n")
    
    validation_years = [2018, 2019, 2020, 2021, 2022, 2023]
    
    results = []
    
    for year in validation_years:
        predictions = predict_championship_probabilities(predictor, df, year=year)
        
        if predictions is None:
            continue
        
        predicted_champion = predictions.iloc[0]['Team']
        actual_champion = SUPER_BOWL_WINNERS.get(year, "Unknown")
        
        # Find rank of actual champion
        actual_rank = predictions[predictions['Team'] == actual_champion]['Rank'].values
        actual_rank = actual_rank[0] if len(actual_rank) > 0 else None
        
        actual_prob = predictions[predictions['Team'] == actual_champion]['Championship_Prob'].values
        actual_prob = actual_prob[0] if len(actual_prob) > 0 else None
        
        results.append({
            'Year': year,
            'Predicted': predicted_champion,
            'Actual': actual_champion,
            'Correct': '✅' if predicted_champion == actual_champion else '❌',
            'Actual_Rank': actual_rank,
            'Actual_Prob': f"{actual_prob*100:.2f}%" if actual_prob else "N/A"
        })
        
        print(f"{year}: Predicted {predicted_champion:25s} | Actual: {actual_champion:25s} | {results[-1]['Correct']}")
    
    results_df = pd.DataFrame(results)
    
    accuracy = (results_df['Correct'] == '✅').sum() / len(results_df)
    
    print("\n" + "=" * 70)
    print(f"📈 Prediction Accuracy: {accuracy*100:.1f}%")
    print(f"   Correct: {(results_df['Correct'] == '✅').sum()} / {len(results_df)}")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    # Example: Predict 2024 champion
    predict_future_champion(year=2024)
    
    print("\n\n")
    
    # Validate historical predictions
    compare_predictions_to_actual()
