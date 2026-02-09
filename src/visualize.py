"""
Visualization Module - Create Plots and Graphs
Comprehensive visualization suite for NFL championship prediction analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure we're working from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from train_model import NFLChampionshipPredictor
from data_preparation import prepare_data, SUPER_BOWL_WINNERS
from predict import predict_championship_probabilities
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create results directory
os.makedirs('results', exist_ok=True)


def plot_win_pct_vs_championship(df, save_path='results/win_pct_vs_championship.png'):
    """
    Scatter plot: Win Percentage vs Championship Success
    """
    plt.figure(figsize=(12, 7))
    
    champions = df[df['Won_SB'] == 1]
    non_champions = df[df['Won_SB'] == 0]
    
    plt.scatter(non_champions['Win_Pct'], non_champions['PD'], 
                alpha=0.3, s=50, c='gray', label='Non-Champions')
    plt.scatter(champions['Win_Pct'], champions['PD'], 
                alpha=0.9, s=200, c='gold', edgecolors='black', 
                linewidths=2, marker='*', label='Champions')
    
    plt.xlabel('Win Percentage', fontsize=12, weight='bold')
    plt.ylabel('Point Differential', fontsize=12, weight='bold')
    plt.title('Win Percentage vs Point Differential\n(Champions vs Non-Champions)', 
              fontsize=14, weight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_championship_probability_distribution(predictor, df, year=2023, 
                                                 save_path='results/prob_distribution.png'):
    """
    Bar chart: Championship probability distribution for a specific year
    """
    predictions = predict_championship_probabilities(predictor, df, year=year)
    
    if predictions is None:
        print(f"⚠️  No predictions for year {year}")
        return
    
    # Get actual champion
    actual_champion = SUPER_BOWL_WINNERS.get(year, None)
    
    plt.figure(figsize=(14, 8))
    
    # Create color map: highlight actual champion
    colors = ['gold' if team == actual_champion else 'steelblue' 
              for team in predictions['Team']]
    
    bars = plt.barh(predictions['Team'], predictions['Championship_Prob'] * 100, 
                    color=colors, edgecolor='black', linewidth=0.7)
    
    plt.xlabel('Championship Probability (%)', fontsize=12, weight='bold')
    plt.ylabel('Team', fontsize=12, weight='bold')
    plt.title(f'{year} Championship Probability Predictions\n(Actual Winner: {actual_champion})', 
              fontsize=14, weight='bold')
    plt.gca().invert_yaxis()
    
    # Add probability labels
    for i, (prob, team) in enumerate(zip(predictions['Championship_Prob'], predictions['Team'])):
        plt.text(prob * 100 + 0.5, i, f'{prob*100:.2f}%', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_time_series_cv_results(cv_results, save_path='results/cv_performance.png'):
    """
    Line plot: Cross-validation performance over time
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Log Loss
    axes[0].plot(cv_results['test_year'], cv_results['log_loss'], 
                marker='o', linewidth=2, markersize=8, color='crimson')
    axes[0].axhline(cv_results['log_loss'].mean(), color='gray', 
                   linestyle='--', alpha=0.7, label=f'Mean: {cv_results["log_loss"].mean():.4f}')
    axes[0].set_ylabel('Log Loss', fontsize=11, weight='bold')
    axes[0].set_title('Time-Based Cross-Validation Performance', fontsize=13, weight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1].plot(cv_results['test_year'], cv_results['roc_auc'], 
                marker='s', linewidth=2, markersize=8, color='green')
    axes[1].axhline(cv_results['roc_auc'].mean(), color='gray', 
                   linestyle='--', alpha=0.7, label=f'Mean: {cv_results["roc_auc"].mean():.4f}')
    axes[1].set_ylabel('ROC-AUC', fontsize=11, weight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Brier Score
    axes[2].plot(cv_results['test_year'], cv_results['brier_score'], 
                marker='^', linewidth=2, markersize=8, color='purple')
    axes[2].axhline(cv_results['brier_score'].mean(), color='gray', 
                   linestyle='--', alpha=0.7, label=f'Mean: {cv_results["brier_score"].mean():.4f}')
    axes[2].set_xlabel('Test Year', fontsize=11, weight='bold')
    axes[2].set_ylabel('Brier Score', fontsize=11, weight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_feature_importance_detailed(predictor, save_path='results/feature_importance_detailed.png'):
    """
    Enhanced feature importance visualization
    """
    importance_df = pd.DataFrame({
        'Feature': predictor.features,
        'Importance': predictor.model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    
    plt.barh(importance_df['Feature'], importance_df['Importance'], 
            color=colors, edgecolor='black', linewidth=0.7)
    
    plt.xlabel('Importance Score', fontsize=12, weight='bold')
    plt.ylabel('Feature', fontsize=12, weight='bold')
    plt.title('XGBoost Feature Importance\n(Championship Prediction Model)', 
              fontsize=14, weight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (feat, imp) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
        plt.text(imp + 0.005, i, f'{imp:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_champion_characteristics(df, save_path='results/champion_stats.png'):
    """
    Box plots comparing champions vs non-champions across key metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('Win_Pct', 'Win Percentage'),
        ('PD', 'Point Differential'),
        ('Pts_For', 'Points Scored'),
        ('Pts_Against', 'Points Allowed')
    ]
    
    for ax, (col, title) in zip(axes.flat, metrics):
        df_plot = df[[col, 'Won_SB']].dropna()
        df_plot['Status'] = df_plot['Won_SB'].map({1: 'Champion', 0: 'Non-Champion'})
        
        sns.boxplot(data=df_plot, x='Status', y=col, ax=ax, 
                   palette={'Champion': 'gold', 'Non-Champion': 'lightgray'},
                   linewidth=1.5)
        
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Statistical Characteristics: Champions vs Non-Champions', 
                fontsize=15, weight='bold', y=1.00)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_prediction_accuracy_timeline(save_path='results/prediction_accuracy.png'):
    """
    Timeline showing prediction accuracy for historical years
    """
    from predict import predict_championship_probabilities
    
    predictor = NFLChampionshipPredictor.load_model()
    df = prepare_data()
    
    validation_years = list(range(2010, 2024))
    accuracy_data = []
    
    for year in validation_years:
        predictions = predict_championship_probabilities(predictor, df, year=year)
        
        if predictions is None:
            continue
        
        predicted_champion = predictions.iloc[0]['Team']
        actual_champion = SUPER_BOWL_WINNERS.get(year, "Unknown")
        
        correct = 1 if predicted_champion == actual_champion else 0
        
        # Get actual champion rank
        actual_rank = predictions[predictions['Team'] == actual_champion]['Rank'].values
        actual_rank = actual_rank[0] if len(actual_rank) > 0 else None
        
        accuracy_data.append({
            'Year': year,
            'Correct': correct,
            'Actual_Rank': actual_rank,
            'Predicted': predicted_champion,
            'Actual': actual_champion
        })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Accuracy timeline
    colors = ['green' if c == 1 else 'red' for c in accuracy_df['Correct']]
    ax1.bar(accuracy_df['Year'], accuracy_df['Correct'], color=colors, 
           edgecolor='black', linewidth=1.2, alpha=0.8)
    ax1.set_ylabel('Correct Prediction (1=Yes, 0=No)', fontsize=11, weight='bold')
    ax1.set_title('Year-by-Year Prediction Accuracy', fontsize=13, weight='bold')
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add labels
    for year, correct, pred, actual in zip(accuracy_df['Year'], accuracy_df['Correct'], 
                                           accuracy_df['Predicted'], accuracy_df['Actual']):
        if correct == 0:
            ax1.text(year, 0.05, f'{pred[:3]}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Actual champion rank
    ax2.plot(accuracy_df['Year'], accuracy_df['Actual_Rank'], 
            marker='o', linewidth=2, markersize=8, color='navy')
    ax2.axhline(1, color='gold', linestyle='--', linewidth=2, 
               label='Perfect Prediction (Rank 1)', alpha=0.7)
    ax2.set_xlabel('Year', fontsize=11, weight='bold')
    ax2.set_ylabel('Actual Champion Predicted Rank', fontsize=11, weight='bold')
    ax2.set_title('How Well Did We Rank the Actual Champion?', fontsize=13, weight='bold')
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()
    
    return accuracy_df


def plot_correlation_heatmap(df, save_path='results/correlation_heatmap.png'):
    """
    Correlation heatmap of key features
    """
    features_to_plot = ['Win_Pct', 'Pts_For', 'Pts_Against', 'PD', 'PPG', 'PA_PG', 'Won_SB']
    correlation_df = df[features_to_plot].corr()
    
    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(correlation_df, dtype=bool))
    
    sns.heatmap(correlation_df, annot=True, fmt='.3f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
               mask=mask, vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Heatmap\n(Lower Triangle)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def create_all_visualizations():
    """
    Generate all visualization plots.
    """
    print("=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    print("\n📥 Loading data...")
    df = prepare_data()
    
    # Load model
    print("📥 Loading model...")
    predictor = NFLChampionshipPredictor.load_model()
    
    print("\n🎨 Generating plots...\n")
    
    # 1. Win % vs Championship
    plot_win_pct_vs_championship(df)
    
    # 2. Probability distribution for recent year
    plot_championship_probability_distribution(predictor, df, year=2023)
    
    # 3. CV performance
    try:
        cv_results = pd.read_csv('results/cv_results.csv')
        plot_time_series_cv_results(cv_results)
    except FileNotFoundError:
        print("⚠️  CV results not found, skipping CV plot")
    
    # 4. Feature importance
    plot_feature_importance_detailed(predictor)
    
    # 5. Champion characteristics
    plot_champion_characteristics(df)
    
    # 6. Prediction accuracy timeline
    plot_prediction_accuracy_timeline()
    
    # 7. Correlation heatmap
    plot_correlation_heatmap(df)
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("   Check the results/ folder for plots")
    print("=" * 70)


if __name__ == "__main__":
    create_all_visualizations()
