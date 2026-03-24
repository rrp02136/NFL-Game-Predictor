"""
Configuration file for NFL Game Prediction Project
All paths, column names, and hyperparameters are centralized here
"""

import os
import glob

# PATHS
# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Input Data Paths
GAME_CSV_PATHS = [
    os.path.join(DATA_DIR, 'Season_Scores', '2017_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2018_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2019_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2020_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2021_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2022_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2023_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2024_scores.csv'),
    os.path.join(DATA_DIR, 'Season_Scores', '2025_scores.csv')
]
# Build PBP paths dynamically for weeks 1-17 for each year
PBP_CSV_PATHS = []
for year in range(2017, 2026):  # 2017 to 2025
    for week in range(1, 18):  # Weeks 1-17 (regular season only)
        path = os.path.join(DATA_DIR, 'Season_plays', 'new_format', f'{year}_plays_by_week', f'{year}_Week_{week}_plays.csv')
        PBP_CSV_PATHS.append(path)

# Output Paths
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'cleaned_games.csv')
RAW_GAMES_COMBINED_PATH = os.path.join(DATA_DIR, 'raw_games_combined.csv')
RAW_PLAYS_COMBINED_PATH = os.path.join(DATA_DIR, 'raw_plays_combined.csv')
MODEL_COMPARISON_PATH = os.path.join(DATA_DIR, 'model_comparison.csv')
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, 'feature_names.txt')
LABEL_ENCODERS_PATH = os.path.join(DATA_DIR, 'label_encoders.pkl')

# Model and Plot Directories
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# COLUMN NAME MAPPING
# Game-level Scores Columns
SEASON_COL = 'Season'
WEEK_COL = 'Week'
GAME_ID_COL = 'game_id'
HOME_TEAM_COL = 'HomeTeam'
AWAY_TEAM_COL = 'AwayTeam'
HOME_SCORE_COL = 'HomeScore'
AWAY_SCORE_COL = 'AwayScore'
HOME_WIN_COL = 'HomeWin'
POSTSEASON_COL = 'PostSeason'

# Advanced Stats Column (Optional)
HOME_TOTAL_EPA_COL = None
AWAY_TOTAL_EPA_COL = None
HOME_PASS_EPA_COL = None
AWAY_PASS_EPA_COL = None
HOME_RUSH_EPA_COL = None
AWAY_RUSH_EPA_COL = None
HOME_OFF_EPA_COL = None
AWAY_OFF_EPA_COL = None
HOME_DEF_EPA_COL = None
AWAY_DEF_EPA_COL = None
SPREAD_LINE_COL = None
TOTAL_LINE_COL = None
HOME_TURNOVERS_COL = None
AWAY_TURNOVERS_COL = None

# Play-by-Play Columns
PBP_GAME_ID_COL = 'game_id'
PBP_SEASON_COL = 'Season'
PBP_WEEK_COL = 'Week'
PBP_HOME_TEAM_COL = 'HomeTeam'
PBP_AWAY_TEAM_COL = 'AwayTeam'
PBP_POSTEAM_COL = 'TeamWithPossession'
PBP_QUARTER_COL = 'Quarter'
PBP_PLAY_TYPE_COL = 'PlayOutcome'
PBP_IS_SCORING_COL = 'IsScoringPlay'
PBP_IS_SCORING_DRIVE_COL = 'IsScoringDrive'
PBP_PLAY_DESC_COL = 'PlayDescription'

# TARGET VARIABLE
TARGET_COL = 'home_team_win'

# MODELING PARAMETERS
# Data Split
SPLIT_STRATEGY = 'time_based'
TEST_SIZE = 0.2
RANDOM_STATE = 42
TIME_SPLIT_SEASON = 2024

# Random Forest Hyperparameter Grid
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# XGBoost Hyperparameter Grid
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# Cross-Validation
CV_FOLDS = 5
CV_SCORING = 'accuracy'

# FEATURE ENGINEERING SETTINGS
USE_PBP_FEATURES = True  # Set to False if not using play-by-play
# Rolling Window for Team Averages
ROLLING_WINDOW = 3
# Categorical Columns to Encode
CATEGORICAL_COLS = [HOME_TEAM_COL, AWAY_TEAM_COL]

# PLOTTING SETTINGS
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 6)
TOP_N_FEATURES = 20

# LOGGING
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE = os.path.join(BASE_DIR, 'pipeline.log')