"""
Master Pipeline - Run Complete NFL Championship Prediction System
Executes: Data Prep → Model Training → Predictions → Visualizations
"""

import sys
import os

print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     NFL CHAMPIONSHIP PREDICTION SYSTEM - XGBoost ML Pipeline       ║
║                                                                    ║
║     🏈 Predicting Super Bowl Champions Using Machine Learning     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

def run_full_pipeline():
    """
    Execute the complete machine learning pipeline.
    """
    
    # Step 1: Data Preparation
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    from data_preparation import prepare_data
    df = prepare_data()
    print(f"✅ Data prepared: {len(df)} team-seasons, {df['Won_SB'].sum()} champions")
    
    # Step 2: Model Training
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING (XGBoost)")
    print("="*70)
    print("⏱️  This may take a few minutes...")
    
    from train_model import main as train_main
    train_main()
    
    # Step 3: Make Predictions
    print("\n" + "="*70)
    print("STEP 3: GENERATE PREDICTIONS")
    print("="*70)
    
    from predict import predict_future_champion, compare_predictions_to_actual
    
    # Predict current/recent season
    predict_future_champion(year=2024)
    
    print("\n")
    
    # Validate on historical data
    compare_predictions_to_actual()
    
    # Step 4: Create Visualizations
    print("\n" + "="*70)
    print("STEP 4: GENERATE VISUALIZATIONS")
    print("="*70)
    
    from visualize import create_all_visualizations
    create_all_visualizations()
    
    # Summary
    print("\n\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                    ✅ PIPELINE COMPLETE! ✅                        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n📂 Output Files Generated:")
    print("   📊 Data:          data/nfl_ml_ready.csv")
    print("   🤖 Model:         models/championship_model.pkl")
    print("   📈 Predictions:   results/predictions_2024.csv")
    print("   📉 CV Results:    results/cv_results.csv")
    print("   🎨 Plots:         results/*.png (7 visualizations)")
    
    print("\n🔮 Next Steps:")
    print("   • Review visualizations in results/ folder")
    print("   • Check model performance in CV results")
    print("   • Use predict.py to forecast future seasons")
    print("   • Update data with new seasons and retrain")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
