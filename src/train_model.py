"""
XGBoost Championship Prediction Model
Includes time-based cross-validation and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    log_loss, roc_auc_score, brier_score_loss, 
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preparation import prepare_data, get_feature_columns, create_team_season_features


class NFLChampionshipPredictor:
    """
    XGBoost-based model to predict NFL championship probability.
    """
    
    def __init__(self, features=None):
        self.features = features or get_feature_columns()
        self.model = None
        self.best_params = None
        self.cv_results = None
        
    def prepare_training_data(self, df):
        """
        Prepare features and target for training.
        """
        # Add advanced features
        df = create_team_season_features(df)
        
        # Update feature list with lag features
        extended_features = self.features + [
            'Prev_Win_Pct', 
            'Prev_PD',
            'Win_Pct_Change',
            'PD_Change'
        ]
        
        # Keep only rows where we have all features
        available_features = [f for f in extended_features if f in df.columns]
        
        X = df[available_features].copy()
        y = df['Won_SB'].copy()
        
        # Handle missing values (from lag features in first year)
        X = X.fillna(0)
        
        return X, y, df[['Year', 'Tm']], available_features
    
    def time_based_train_test_split(self, X, y, metadata, test_year):
        """
        Split data using time-based logic: train on past, test on one year.
        """
        train_mask = metadata['Year'] < test_year
        test_mask = metadata['Year'] == test_year
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        return X_train, X_test, y_train, y_test
    
    def time_based_cross_validation(self, X, y, metadata, start_year=2010):
        """
        Perform time-based cross-validation.
        Train on years before test year, test on one year at a time.
        """
        results = []
        years = sorted(metadata['Year'].unique())
        test_years = [yr for yr in years if yr >= start_year]
        
        print(f"\n⏱️  Time-Based Cross-Validation")
        print(f"   Testing years: {test_years}")
        print("=" * 70)
        
        for test_year in test_years:
            X_train, X_test, y_train, y_test = self.time_based_train_test_split(
                X, y, metadata, test_year
            )
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Train model
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                verbosity=0
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Metrics
            ll = log_loss(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else np.nan
            brier = brier_score_loss(y_test, y_pred_proba)
            
            results.append({
                'test_year': test_year,
                'log_loss': ll,
                'roc_auc': auc,
                'brier_score': brier,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            print(f"   {test_year}: LogLoss={ll:.4f}, AUC={auc:.4f}, Brier={brier:.4f}")
        
        print("=" * 70)
        
        results_df = pd.DataFrame(results)
        
        print(f"\n📊 Average CV Metrics:")
        print(f"   Log Loss:    {results_df['log_loss'].mean():.4f} ± {results_df['log_loss'].std():.4f}")
        print(f"   ROC-AUC:     {results_df['roc_auc'].mean():.4f} ± {results_df['roc_auc'].std():.4f}")
        print(f"   Brier Score: {results_df['brier_score'].mean():.4f} ± {results_df['brier_score'].std():.4f}")
        
        return results_df
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Use GridSearchCV to find best hyperparameters.
        """
        print("\n🔍 Starting Hyperparameter Tuning...")
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        
        # Use 3-fold CV (not time-based for hyperparameter search)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_log_loss',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n✅ Best Parameters: {grid_search.best_params_}")
        print(f"   Best CV Log Loss: {-grid_search.best_score_:.4f}")
        
        self.best_params = grid_search.best_params_
        
        return grid_search.best_estimator_
    
    def train_final_model(self, X, y, use_tuning=False):
        """
        Train the final model on all available data.
        """
        print("\n🚀 Training Final Model...")
        
        if use_tuning and self.best_params:
            print("   Using tuned hyperparameters")
            self.model = xgb.XGBClassifier(
                **self.best_params,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=0
            )
        else:
            print("   Using default hyperparameters")
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                verbosity=0
            )
        
        self.model.fit(X, y)
        
        print("✅ Model training complete!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, metadata_test):
        """
        Evaluate model on test set.
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        print("\n📈 Model Evaluation")
        print("=" * 70)
        
        # Metrics
        ll = log_loss(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba) if len(y_test.unique()) > 1 else np.nan
        brier = brier_score_loss(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Log Loss:      {ll:.4f}")
        print(f"ROC-AUC:       {auc:.4f}")
        print(f"Brier Score:   {brier:.4f}")
        print(f"Accuracy:      {acc:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Champion', 'Champion']))
        
        # Show top predicted teams
        results = metadata_test.copy()
        results['Probability'] = y_pred_proba
        results['Actual'] = y_test.values
        results = results.sort_values('Probability', ascending=False)
        
        print("\n🏆 Top 10 Predicted Championship Probabilities:")
        print(results[['Year', 'Tm', 'Probability', 'Actual']].head(10))
        
        return {
            'log_loss': ll,
            'roc_auc': auc,
            'brier_score': brier,
            'accuracy': acc
        }
    
    def plot_feature_importance(self, save_path='../results/feature_importance.png'):
        """
        Plot feature importance from trained model.
        """
        import os
        os.makedirs('../results', exist_ok=True)
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title('XGBoost Feature Importance', fontsize=14, weight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Feature importance plot saved to {save_path}")
        plt.close()
        
        return importance_df
    
    def save_model(self, path='../models/championship_model.pkl'):
        """
        Save trained model to disk.
        """
        import os
        os.makedirs('../models', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'features': self.features,
            'best_params': self.best_params,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, path)
        print(f"💾 Model saved to {path}")
    
    @classmethod
    def load_model(cls, path='../models/championship_model.pkl'):
        """
        Load trained model from disk.
        """
        model_data = joblib.load(path)
        
        predictor = cls(features=model_data['features'])
        predictor.model = model_data['model']
        predictor.best_params = model_data.get('best_params')
        
        print(f"✅ Model loaded from {path}")
        print(f"   Trained on: {model_data.get('trained_date', 'Unknown')}")
        
        return predictor


def main():
    """
    Main training pipeline.
    """
    print("=" * 70)
    print("NFL CHAMPIONSHIP PREDICTION - XGBoost Training Pipeline")
    print("=" * 70)
    
    # Load and prepare data
    print("\n📥 Loading data...")
    df = prepare_data()
    
    # Initialize predictor
    predictor = NFLChampionshipPredictor()
    
    # Prepare training data
    X, y, metadata, features = predictor.prepare_training_data(df)
    predictor.features = features
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(features)}")
    print(f"   Champions: {y.sum()}")
    print(f"   Years: {metadata['Year'].min()} - {metadata['Year'].max()}")
    
    # Time-based cross-validation
    cv_results = predictor.time_based_cross_validation(X, y, metadata, start_year=2010)
    predictor.cv_results = cv_results
    
    # Hyperparameter tuning on training data
    train_mask = metadata['Year'] < 2020
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    best_model = predictor.hyperparameter_tuning(X_train, y_train)
    
    # Train final model on all data
    predictor.train_final_model(X, y, use_tuning=True)
    
    # Evaluate on recent years
    test_mask = metadata['Year'] >= 2020
    if test_mask.sum() > 0:
        X_test = X[test_mask]
        y_test = y[test_mask]
        metadata_test = metadata[test_mask]
        
        metrics = predictor.evaluate_model(X_test, y_test, metadata_test)
    
    # Feature importance
    importance_df = predictor.plot_feature_importance()
    print("\n📊 Feature Importance:")
    print(importance_df)
    
    # Save model
    predictor.save_model()
    
    # Save CV results
    cv_results.to_csv('../results/cv_results.csv', index=False)
    print("💾 CV results saved to ../results/cv_results.csv")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
