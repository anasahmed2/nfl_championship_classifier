"""
Interactive Demo - NFL Championship Prediction System
Quick start guide and examples
"""

import pandas as pd
from train_model import NFLChampionshipPredictor
from predict import predict_championship_probabilities, compare_predictions_to_actual
from data_preparation import prepare_data

print("=" * 70)
print("🏈 NFL CHAMPIONSHIP PREDICTION - INTERACTIVE DEMO")
print("=" * 70)

# ============================================================================
# DEMO 1: Load Trained Model and Check Performance
# ============================================================================
print("\n📊 DEMO 1: Model Performance Summary")
print("-" * 70)

# Load trained model
predictor = NFLChampionshipPredictor.load_model('../models/championship_model.pkl')

print("\n✅ Model Details:")
print(f"   Features used: {len(predictor.features)}")
print(f"   Best hyperparameters: {predictor.best_params}")

# Load CV results
cv_results = pd.read_csv('../results/cv_results.csv')

print("\n📈 Cross-Validation Performance:")
print(f"   Average Log Loss:    {cv_results['log_loss'].mean():.4f}")
print(f"   Average ROC-AUC:     {cv_results['roc_auc'].mean():.4f}")
print(f"   Average Brier Score: {cv_results['brier_score'].mean():.4f}")

# ============================================================================
# DEMO 2: Predict Championship Probabilities for a Specific Year
# ============================================================================
print("\n" + "=" * 70)
print("🔮 DEMO 2: Championship Predictions for 2023")
print("-" * 70)

df = prepare_data()
predictions_2023 = predict_championship_probabilities(predictor, df, year=2023)

print("\n🏆 Top 10 Teams (2023 Season):")
print(predictions_2023.head(10).to_string(index=False))

print(f"\n⭐ Predicted Champion: {predictions_2023.iloc[0]['Team']}")
print(f"   Probability: {predictions_2023.iloc[0]['Championship_Prob']*100:.2f}%")

# ============================================================================
# DEMO 3: Historical Accuracy Check
# ============================================================================
print("\n" + "=" * 70)
print("📊 DEMO 3: Historical Prediction Accuracy")
print("-" * 70)

validation_results = compare_predictions_to_actual()

# ============================================================================
# DEMO 4: Top Features Driving Predictions
# ============================================================================
print("\n" + "=" * 70)
print(" 🧠 DEMO 4: Most Important Features")
print("-" * 70)

importance_df = pd.DataFrame({
    'Feature': predictor.features,
    'Importance': predictor.model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance Rankings:")
print(importance_df.to_string(index=False))

# ============================================================================
# DEMO 5: Make Your Own Predictions
# ============================================================================
print("\n" + "=" * 70)
print("🎯 DEMO 5: How to Predict Any Season")
print("-" * 70)

print("""
To predict championship probabilities for any year:

```python
from train_model import NFLChampionshipPredictor
from predict import predict_championship_probabilities
from data_preparation import prepare_data

# Load model
predictor = NFLChampionshipPredictor.load_model('../models/championship_model.pkl')

# Load data
df = prepare_data()

# Make predictions for desired year (e.g., 2022)
predictions = predict_championship_probabilities(predictor, df, year=2022)

# Show top teams
print(predictions.head(10))
```

To retrain the model with new data:

```python
from train_model import main as train_main

# Update data/nfl_standings_2005_2024.csv with new season data
# Then retrain:
train_main()
```
""")

print("\n" + "=" * 70)
print("✅ DEMO COMPLETE!")
print("=" * 70)
print("\nℹ️  For detailed analysis, check these visualizations:")
print("   • results/win_pct_vs_championship.png - Champion characteristics")
print("   • results/prob_distribution.png - Team probability rankings")
print("   • results/cv_performance.png - Model performance over time")
print("   • results/feature_importance_detailed.png - What drives predictions")
print("   • results/champion_stats.png - Champions vs non-champions")
print("   • results/prediction_accuracy.png - Historical accuracy")
print("   • results/correlation_heatmap.png - Feature relationships")
