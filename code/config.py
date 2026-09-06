"""
Configuration for NFL Game Prediction Pipeline
Data comes from the nflverse project via the nfl_data_py package.
"""
import os

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

SCHEDULES_CACHE = os.path.join(CACHE_DIR, 'schedules.parquet')
PBP_CACHE = os.path.join(CACHE_DIR, 'pbp_team_game.parquet')
INJURIES_CACHE = os.path.join(CACHE_DIR, 'injuries.parquet')
FEATURES_PATH = os.path.join(DATA_DIR, 'game_features.parquet')
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, 'feature_names.json')
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, 'model_comparison.csv')
ATS_BACKTEST_PATH = os.path.join(RESULTS_DIR, 'ats_backtest.csv')
CALIBRATION_PATH = os.path.join(RESULTS_DIR, 'calibration.csv')

# DATA RANGE
# The nflverse PBP dataset with EPA/success is reliable from 1999.
# Model on the modern era only: 2002 is the current 32-team realignment.
TRAIN_SEASON_START = 2002
CURRENT_SEASON = 2026  # nfl_data_py fills this in as games are played

# MODELING
TARGET_COL = 'home_win'
# Time-based split — always test on the most recent full seasons
TEST_SEASONS = [2024, 2025]  # holdout for honest accuracy
# Held out from training and used to fit the isotonic calibrator only.
# Choose a season that's recent (distribution close to test) but not in TEST_SEASONS.
CALIBRATION_SEASONS = [2023]
# 2026 games (as they play) are for LIVE prediction, not test evaluation

RANDOM_STATE = 42
CV_FOLDS = 5
CV_SCORING = 'neg_log_loss'  # log-loss = calibration-aware for betting

# Feature engineering
ROLLING_WINDOW = 8  # last N games per team for rolling averages
ELO_K = 20.0
ELO_HFA = 55.0   # home-field advantage in Elo points
ELO_INIT = 1500.0
ELO_SEASON_REGRESS = 0.25  # regress 25% toward 1500 between seasons

# Hyperparameter grids — trimmed for practicality
RF_PARAM_GRID = {
    'n_estimators': [300, 500],
    'max_depth': [6, 10, None],
    'min_samples_leaf': [5, 10, 20],
    'max_features': ['sqrt'],
}

XGB_PARAM_GRID = {
    'n_estimators': [200, 400],
    'max_depth': [3, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1.0, 3.0],
}

# BETTING ANALYSIS
# Only flag bets where model edge over the market implied prob exceeds this.
# 5 pp is a starting threshold; the pipeline reports ATS at multiple thresholds.
BET_EDGE_THRESHOLD = 0.05
ATS_EDGE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]  # backtest sweep
STANDARD_JUICE = -110  # for ATS backtest ROI

# PLOTTING
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 6)
TOP_N_FEATURES = 20

# LOGGING
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(BASE_DIR, 'pipeline.log')
