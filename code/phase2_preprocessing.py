"""
Phase 2: Preprocessing & feature engineering.
Produces the game-level modeling table with pre-game-only features.
"""
import logging
import json
import pandas as pd
from utils_features import build_game_features, select_feature_columns, drop_early_season_rows


def build_modeling_dataset(schedules: pd.DataFrame, pbp_agg: pd.DataFrame, config) -> pd.DataFrame:
    logging.info("Building modeling dataset...")
    games = build_game_features(schedules, pbp_agg, config)
    games = drop_early_season_rows(games, config)

    feature_cols = select_feature_columns(games, config)
    logging.info(f"Selected {len(feature_cols)} model features:")
    for c in feature_cols:
        logging.info(f"  - {c}")

    # Persist for prediction time
    with open(config.FEATURE_NAMES_PATH, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logging.info(f"Wrote {config.FEATURE_NAMES_PATH}")

    # Save the full engineered frame (features + target + context for ATS)
    games.to_parquet(config.FEATURES_PATH, index=False)
    logging.info(f"Wrote {config.FEATURES_PATH} shape={games.shape}")

    # Sanity check: no NaNs in the feature matrix
    X = games[feature_cols]
    nan_counts = X.isna().sum()
    if nan_counts.any():
        logging.warning(f"Features contain NaN, filling with column median:\n{nan_counts[nan_counts > 0]}")
        for col in feature_cols:
            if X[col].isna().any():
                games[col] = games[col].fillna(games[col].median())

    home_wr = games['home_win'].mean()
    logging.info(f"Home win rate in modeling set: {home_wr:.2%}  (baseline to beat)")

    return games
