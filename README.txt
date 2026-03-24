# NFL Game Outcome Prediction
### Final Project — CSIC 4170/6170 (Intro to Computational Investing)

This project implements an end-to-end machine learning pipeline that predicts NFL game outcomes using historical pre-game data from 2017–2025. The system performs data exploration, cleaning, feature engineering, model training, evaluation, and automated predictions for both single games and entire weeks. The pipeline is designed to be reproducible, modular, and deployable.

---

## 1. Project Structure

```
.
├── code/
│   ├── config.py
│   ├── main.py
│   ├── phase1_exploration.py
│   ├── phase2_preprocessing.py
│   ├── phase3_models.py
│   ├── phase4_evaluation.py
│   ├── predict_game.py
│   ├── utils_features.py
│   └── utils_io.py
├── data/
│   ├── raw/                    # raw CSVs
│   ├── cleaned_games.csv
│   ├── label_encoders.pkl
│   ├── feature_names.json
├── code/models/
│   ├── random_forest_best.pkl
│   └── xgboost_best.pkl
├── plots/
│   ├── RandomForest_confusion_matrix.png
│   ├── XGBoost_confusion_matrix.png
│   ├── RandomForest_feature_importances.png
│   └── XGBoost_feature_importances.png
└── results/
    ├── model_comparison.csv
    └── predictions_week_<season>_<week>_pretty.csv
```

---

## 2. Environment Setup

### Create environment & install dependencies
```bash
# Option A: Conda
conda create -n nflpred python=3.10 -y
conda activate nflpred
pip install -r requirements.txt

# Option B: Virtual environment (pip)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### requirements.txt (recommended)
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
xgboost>=2.0
matplotlib>=3.8
joblib>=1.3
```

---

## 3. Configure the Project

Open `code/config.py` and set:

```python
GAME_CSV_PATHS = ["data/raw/games_2017_2025.csv"]
PBP_CSV_PATHS  = []  # optional play-by-play

MODELS_DIR = "code/models"
OUTPUT_DIR = "results"
CLEANED_DATA_PATH = "data/cleaned_games.csv"

SEASON_COL = "Season"
WEEK_COL   = "Week"
HOME_TEAM_COL = "HomeTeam"
AWAY_TEAM_COL = "AwayTeam"
HOME_SCORE_COL = "HomeScore"
AWAY_SCORE_COL = "AwayScore"

TARGET_COL = "HomeWin"

TIME_BASED_SPLIT = False
TEST_SIZE = 0.2
RANDOM_STATE = 42

FILTER_REGULAR_SEASON_ONLY = True
USE_PLAY_BY_PLAY = False
```

Place your raw CSV data inside:
**`data/raw/`**

---

## 4. Run the Full Pipeline (Training + Evaluation)

Run the main script:

```bash
python code/main.py
```

This performs:

1. Raw data validation + exploratory summary
2. Cleaning, feature engineering, team encodings
3. Train Random Forest & XGBoost (GridSearchCV)
4. Save models, encoders, feature list
5. Generate evaluation visualizations and comparison tables

Outputs include:

- `code/models/*.pkl`
- `data/label_encoders.pkl`
- `data/feature_names.json`
- `plots/*.png`
- `results/model_comparison.csv`

---

## 5. Making Predictions

### A. Predict a single game
```bash
python code/predict_game.py \
  --home "Lions" \
  --away "Cowboys" \
  --season 2025 \
  --week 14
```

Example output:
```
Predicted Winner: Lions
Confidence: 54.9%
Lions Win: 54.9%
Cowboys Win: 45.1%
```

---

### B. Predict an entire week
```bash
python code/predict_game.py --weekly 2025 14
```

Outputs a formatted table plus a CSV at:
```
results/predictions_week_2025_14_pretty.csv
```

Table columns:
```
home_team | away_team | predicted_winner | home_win_probability | away_win_probability
```

---

## 6. Reproducibility

To ensure training and prediction use identical logic:

- The pipeline **saves label encoders** → `data/label_encoders.pkl`
- It saves **feature ordering** → `data/feature_names.json`
- Prediction script loads both to avoid mismatched feature orders

This prevents the common ML issue of “model expects X features but got Y”.

---

## 7. Troubleshooting

### Error: “X has N features, model expects M”
Cause: mismatched columns during prediction.
Fix: ensure `predict_game.py` loads `feature_names.json` and aligns columns before inference.

### Warning: “X has feature names, but model fitted without feature names”
Harmless. Fixed by passing `X.values` into `.predict_proba()`.

### FutureWarning: pandas groupby apply
Caused by rolling form features in `utils_features.py`.
Can be silenced by adding `include_groups=False` if using pandas ≥ 2.2.

### Unexpectedly high accuracy (>95%)
This indicates leakage.
Ensure game results or correlated columns (e.g., AwayWin, point_diff) are excluded.

---

## 8. Updating the Dataset

To train on new seasons:

1. Add new rows to the raw CSV in `data/raw/`.
2. Run `python code/main.py` again.
3. New models/encoders overwrite previous artifacts.

---

## 9. License

Academic use for educational purposes.
All datasets originally sourced from publicly available NFL records.

---

## 10. Summary

This repository contains a full machine-learning workflow for NFL win prediction, built to be modular, reproducible, and deployable. It supports full-season training, ensemble evaluation, and both single-game and multi-game predictions with confidence scores. The project’s structure and design make it easy to extend with additional features such as injury reports, betting lines, rolling performance metrics, or advanced play-by-play analytics.

