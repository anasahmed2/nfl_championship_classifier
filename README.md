# NFL Championship Classifier 🏈

Machine Learning system to predict NFL Super Bowl champions using **XGBoost** and historical team performance data.

---

## 🎯 Project Overview

This project implements a **supervised classification** model that predicts the probability of each NFL team winning the Super Bowl based on season statistics.

### Key Features
- ✅ **XGBoost** gradient boosting classifier
- ✅ **Time-based cross-validation** (prevents data leakage)
- ✅ **Hyperparameter tuning** with GridSearchCV
- ✅ **Multiple evaluation metrics** (Log Loss, ROC-AUC, Brier Score)
- ✅ **Feature engineering** with lag features and derived metrics
- ✅ **Comprehensive visualizations** (7 different plot types)
- ✅ **Historical validation** against actual Super Bowl winners

---

## 📊 Machine Learning Approach

### Problem Type
**Binary Classification (per team)**
- Target: `Won_SB` (1 = Champion, 0 = Not Champion)
- Output: Championship probability for each team

### Model Architecture
- **Algorithm**: XGBoost (eXtreme Gradient Boosting)
- **Reason**: Best for structured/tabular sports data with non-linear relationships
- **Loss Function**: Binary cross-entropy (log loss)

### Validation Strategy
**Time-Based Cross-Validation**
```
Train: 2005-2015 → Test: 2016
Train: 2005-2016 → Test: 2017
Train: 2005-2017 → Test: 2018
...
```
This simulates real-world prediction where we only have past data.

### Features Used
1. **Basic Stats**: Win %, Points For/Against, Point Differential
2. **Derived Features**: Points per game, scoring efficiency
3. **Lag Features**: Previous season performance, year-over-year change
4. **Advanced Metrics**: Strength of schedule (when available)

---

## 🚀 Quick Start

### Installation
```powershell
# Clone/navigate to project
cd nfl_championship_classifier

# Install dependencies
pip install pandas requests beautifulsoup4 lxml matplotlib seaborn scikit-learn xgboost
```

### Run Complete Pipeline
```powershell
# Scrape data + Train model + Generate predictions + Create plots
python src/run_pipeline.py
```

### Individual Components
```powershell
# 1. Scrape NFL data
python src/championship_classifier.py

# 2. Prepare data and add labels
python src/data_preparation.py

# 3. Train XGBoost model
python src/train_model.py

# 4. Make predictions
python src/predict.py

# 5. Generate visualizations
python src/visualize.py
```

---

## 📁 Project Structure

```
nfl_championship_classifier/
│
├── src/
│   ├── championship_classifier.py    # Web scraper (Pro Football Reference)
│   ├── data_preparation.py           # Data cleaning + feature engineering
│   ├── train_model.py                # XGBoost training pipeline
│   ├── predict.py                    # Championship predictions
│   ├── visualize.py                  # Plot generation
│   └── run_pipeline.py               # Master execution script
│
├── data/
│   ├── nfl_standings_2005_2024.csv   # Raw scraped data
│   └── nfl_ml_ready.csv              # Cleaned ML-ready dataset
│
├── models/
│   └── championship_model.pkl         # Trained XGBoost model
│
├── results/
│   ├── predictions_2024.csv           # Championship probabilities
│   ├── cv_results.csv                 # Time-based CV metrics
│   └── *.png                          # 7 visualization plots
│
└── README.md
```

---

## 📈 Evaluation Metrics

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Log Loss** | Probability calibration quality | Lower = better probability estimates |
| **ROC-AUC** | Ranking ability (0-1 scale) | Higher = better at ranking teams |
| **Brier Score** | Mean squared error of probabilities | Lower = more accurate predictions |
| **Accuracy** | Direct prediction correctness | Simple but imbalanced (only 1 winner) |

---

## 🔧 Hyperparameter Tuning

GridSearchCV optimizes:
- `n_estimators`: Number of boosting trees (50-150)
- `max_depth`: Tree depth (2-4)
- `learning_rate`: Step size (0.01-0.1)
- `subsample`: Row sampling fraction (0.7-1.0)
- `colsample_bytree`: Feature sampling fraction (0.7-1.0)
- `min_child_weight`: Regularization (1-5)

---

## 📊 Generated Visualizations

1. **Win % vs Championship** - Scatter plot showing champion characteristics
2. **Probability Distribution** - Bar chart of championship probabilities by team
3. **CV Performance Timeline** - Model performance across test years
4. **Feature Importance** - XGBoost feature contribution rankings
5. **Champion Statistics** - Box plots comparing champions vs non-champions
6. **Prediction Accuracy** - Historical validation results
7. **Correlation Heatmap** - Feature relationships

---

## 🎯 Example Output

```
🏆 Championship Probability Rankings (2024):
═══════════════════════════════════════════════════════════════════
Rank  Team                      Championship_Prob  Win_Pct  Point_Diff
  1   Kansas City Chiefs              23.45%        0.688      +123
  2   San Francisco 49ers             18.32%        0.750      +156
  3   Baltimore Ravens                15.67%        0.813      +209
  ...
```

---

## 🧠 Design Rationale

### Why XGBoost?
- ✅ Handles **non-linear relationships** (e.g., great offense + bad defense ≠ champion)
- ✅ Robust to **feature interactions**
- ✅ Built-in **regularization** prevents overfitting
- ✅ Industry standard for **tabular data**

### Why Time-Based CV?
- ❌ Random split would **leak future data** into training
- ✅ Mimics **real prediction scenario** (only past data available)
- ✅ More **realistic performance estimates**

### Why Not Other Models?
| Model | Issue |
|-------|-------|
| Linear Regression | Can't capture complex interactions |
| KNN | Poor with high-dimensional sparse data |
| Neural Networks | Needs way more data, harder to interpret |

---

## 🔮 Future Enhancements

- [ ] Add **player-level features** (QB rating, injury data)
- [ ] Include **playoff-specific stats** (not just regular season)
- [ ] Implement **Monte Carlo simulation** for playoff paths
- [ ] Add **DVOA** and **advanced analytics** (if scraping expanded)
- [ ] Ensemble with **LightGBM + Random Forest**
- [ ] Real-time **mid-season predictions** as games are played

---

## 📚 Data Source

- **Website**: [Pro Football Reference](https://www.pro-football-reference.com/)
- **Years**: 2005-2024 (20 seasons)
- **Teams**: All 32 NFL teams (per season)
- **Update Method**: Re-run `championship_classifier.py` scraper

---

## ⚠️ Limitations

1. **Limited features**: Only basic team stats (no QB/player-specific data)
2. **Small dataset**: ~640 team-seasons, only ~19 champions
3. **Class imbalance**: 1 champion vs 31 non-champions per year
4. **Injuries not captured**: Major player injuries can drastically change outcomes
5. **Playoff randomness**: Single-elimination bracket has inherent variance

---

## 📜 License

This project is for **educational purposes** only. NFL data is property of the NFL and Pro Football Reference.

---

## 👤 Author

**Your Name**
- Project: NFL Championship Prediction System
- Technology Stack: Python, XGBoost, Scikit-Learn, Pandas, Matplotlib

---

## 🙏 Acknowledgments

- Pro Football Reference for historical data
- XGBoost development team
- Scikit-learn community

---

**Last Updated**: February 2026
