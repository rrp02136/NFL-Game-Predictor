"""
Phase 2: Data Preprocessing and Feature Engineering
Clean data, create target variable, and engineer predictive features.
"""
import logging
import pandas as pd
import numpy as np
import joblib
from utils_features import (
    compute_point_differential, compute_epa_differentials,
    aggregate_pbp_to_game, encode_teams, handle_missing_values,
    create_game_id
)
from utils_io import save_dataframe

def clean_games_data(df_games: pd.DataFrame, config) -> pd.DataFrame:
    # Clean and standardize game-level data
    logging.info("Cleaning game data...")
    df = df_games.copy()

    # Remove rows with missing scores
    score_cols = [config.HOME_SCORE_COL, config.AWAY_SCORE_COL]
    initial_len = len(df)
    df = df.dropna(subset=score_cols)
    if len(df) < initial_len:
        logging.info(f"Removed {initial_len - len(df)} rows with missing scores")

    # Convert scores to numeric
    df[config.HOME_SCORE_COL] = pd.to_numeric(df[config.HOME_SCORE_COL], errors='coerce')
    df[config.AWAY_SCORE_COL] = pd.to_numeric(df[config.AWAY_SCORE_COL], errors='coerce')

    # Remove any remaining rows with NaN scores after conversion
    df = df.dropna(subset=score_cols)

    # Standardize team names (remove extra spaces, etc.)
    team_cols = [config.HOME_TEAM_COL, config.AWAY_TEAM_COL]
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Create game_id if not exists
    if config.GAME_ID_COL not in df.columns:
        df = create_game_id(df, config.SEASON_COL, config.WEEK_COL,
                            config.AWAY_TEAM_COL, config.HOME_TEAM_COL)

    # Remove obvious outliers (scores > 100 are likely data errors)
    outliers = (df[config.HOME_SCORE_COL] > 100) | (df[config.AWAY_SCORE_COL] > 100)
    if outliers.sum() > 0:
        logging.warning(f"Removing {outliers.sum()} outlier games with scores > 100")
        df = df[~outliers]

    # Convert week to numeric where possible (handle "Hall Of Fame", etc.)
    if config.WEEK_COL in df.columns:
        # Create a numeric week column
        week_mapping = {
            'Hall Of Fame': 0,
            'Wild Card': 19,
            'Wildcard': 19,
            'Divisional': 20,
            'Conference': 21,
            'Conference Championships': 21,
            'Super Bowl': 22,
            'Superbowl': 22
        }

        df['week_numeric'] = df[config.WEEK_COL].apply(
            lambda x: week_mapping.get(x, x) if isinstance(x, str) else x
        )
        df['week_numeric'] = pd.to_numeric(df['week_numeric'], errors='coerce')

    logging.info(f"Cleaned data shape: {df.shape}")
    return df

def create_target_variable(df: pd.DataFrame, config) -> pd.DataFrame:
    # Create binary target variable: 1 if home team wins, 0 otherwise
    logging.info("Creating target variable...")
    df = df.copy()

    # Create target: home team wins
    df[config.TARGET_COL] = (df[config.HOME_SCORE_COL] > df[config.AWAY_SCORE_COL]).astype(int)

    # Check for ties
    ties = (df[config.HOME_SCORE_COL] == df[config.AWAY_SCORE_COL]).sum()
    if ties > 0:
        logging.warning(f"Found {ties} tied games (will be labeled as home loss)")

    # Log target distribution
    target_dist = df[config.TARGET_COL].value_counts()
    logging.info(f"Target distribution:\n  Home wins: {target_dist.get(1, 0)}\n  Away wins: {target_dist.get(0, 0)}")
    logging.info(f"Home win rate: {df[config.TARGET_COL].mean():.2%}")

    return df

def engineer_game_features(df_games: pd.DataFrame, config) -> pd.DataFrame:
    # Engineer features from game-level data
    logging.info("Engineering game-level features...")
    df = df_games.copy()

    # Basic score features (for analysis, not model training)
    df = compute_point_differential(df, config.HOME_SCORE_COL, config.AWAY_SCORE_COL)

    # Total score
    df['total_score'] = df[config.HOME_SCORE_COL] + df[config.AWAY_SCORE_COL]

    # Compute EPA differentials if available
    df = compute_epa_differentials(df, config)

    # Create playoff indicator
    if config.POSTSEASON_COL in df.columns:
        df['is_playoff'] = df[config.POSTSEASON_COL].astype(int)
    elif 'week_numeric' in df.columns:
        df['is_playoff'] = (df['week_numeric'] >= 19).astype(int)
    else:
        df['is_playoff'] = 0

    # Week-based features
    if 'week_numeric' in df.columns:
        df['week_of_season'] = df['week_numeric']
        # Early season (weeks 1-6), mid season (7-12), late season (13+)
        df['season_period'] = pd.cut(df['week_numeric'], bins=[0, 6, 12, 25], labels=[0, 1, 2])
        df['season_period'] = df['season_period'].astype(float)

    logging.info(f"Engineered features. New shape: {df.shape}")
    return df

def engineer_pbp_features(df_pbp: pd.DataFrame, df_games: pd.DataFrame, config) -> pd.DataFrame:
    # Engineer features from play-by-play data and merge with games data
    if df_pbp is None or df_pbp.empty:
        logging.info("No play-by-play data available for feature engineering")
        return df_games

    logging.info("Engineering play-by-play features...")

    # Aggregate play-by-play to game level
    pbp_agg = aggregate_pbp_to_game(df_pbp, config)

    if pbp_agg.empty:
        logging.warning("Play-by-play aggregation resulted in empty DataFrame")
        return df_games

    # Merge with games data
    df = df_games.merge(pbp_agg, on='game_id', how='left')

    # Create differential features from play-by-play
    pbp_stat_cols = [col for col in pbp_agg.columns if col != 'game_id']

    for col in pbp_stat_cols:
        if col.endswith('_home') and col.replace('_home', '_away') in df.columns:
            base_name = col.replace('_home', '')
            df[f'{base_name}_diff'] = df[col] - df[col.replace('_home', '_away')]

    # Fill NaN values in play-by-play features with 0 (games without PBP data)
    pbp_feature_cols = [col for col in df.columns if col in pbp_agg.columns and col != 'game_id']
    df[pbp_feature_cols] = df[pbp_feature_cols].fillna(0)

    logging.info(f"Merged play-by-play features. New shape: {df.shape}")
    return df

def build_modeling_dataset(df_games: pd.DataFrame, df_pbp: pd.DataFrame, config) -> pd.DataFrame:
    # Orchestrate the full preprocessing pipeline
    logging.info("Building modeling dataset...")

    # Step 1: Clean data
    df = clean_games_data(df_games, config)

    # Step 2: Create target variable
    df = create_target_variable(df, config)

    # Step 3: Engineer game-level features
    df = engineer_game_features(df, config)

    # Step 4: Engineer play-by-play features
    if config.USE_PBP_FEATURES:
        df = engineer_pbp_features(df_pbp, df, config)

    # Step 5: Encode categorical variables
    df, encoders = encode_teams(df, config.CATEGORICAL_COLS, fit=True)

    # Save encoders
    joblib.dump(encoders, config.LABEL_ENCODERS_PATH)
    logging.info(f"Saved label encoders to {config.LABEL_ENCODERS_PATH}")

    # Step 6: Handle missing values
    # Define strategy for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != config.TARGET_COL]

    missing_strategy = {col: 'median' for col in numeric_cols if df[col].isna().sum() > 0}
    if missing_strategy:
        df = handle_missing_values(df, missing_strategy)

    # Step 7: Select features for modeling
    # Exclude identification columns and leakage features
    exclude_cols = [
        config.HOME_TEAM_COL, config.AWAY_TEAM_COL, config.GAME_ID_COL,
        config.HOME_SCORE_COL, config.AWAY_SCORE_COL, 'point_diff',
        'Date', 'Day', 'GameStatus', 'AwayRecord', 'HomeRecord',
        'AwaySeeding', 'HomeSeeding', config.HOME_WIN_COL, 'AwayWin', 'HomeWin'
    ]

    # Keep only relevant columns
    feature_cols = [col for col in df.columns if col not in exclude_cols or col == config.TARGET_COL]
    df = df[feature_cols]

    # Remove any remaining non-numeric columns (except target)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    non_numeric = [col for col in non_numeric if col != config.TARGET_COL]
    if len(non_numeric) > 0:
        logging.warning(f"Dropping non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # Final cleanup: remove any rows with remaining NaN values
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        logging.warning(f"Dropped {initial_len - len(df)} rows with NaN values")

    # Save cleaned data
    save_dataframe(df, config.CLEANED_DATA_PATH)

    # Save feature names
    feature_names = [col for col in df.columns if col != config.TARGET_COL]
    with open(config.FEATURE_NAMES_PATH, 'w') as f:
        f.write('\n'.join(feature_names))
    logging.info(f"Saved {len(feature_names)} feature names to {config.FEATURE_NAMES_PATH}")

    logging.info(f"Final modeling dataset shape: {df.shape}")
    logging.info(f"Features: {len(feature_names)}, Target: {config.TARGET_COL}")

    return df