"""
End-to-end NFL win-prediction pipeline.
Loads schedules + PBP via nfl_data_py, engineers pre-game features (no leakage),
trains RF + XGBoost with time-series CV, and reports honest accuracy, calibration,
and an against-the-spread backtest.
"""
import json
import logging
import sys

import config
from utils_io import setup_logging, ensure_directory
from phase1_exploration import load_schedules, load_team_game_pbp_aggregates, summarize
from phase2_preprocessing import build_modeling_dataset
from phase3_models import split_train_test, train_all_models
from phase4_evaluation import evaluate_all_models


def main(refresh_data: bool = False):
    setup_logging(config.LOG_LEVEL, config.LOG_FILE)
    logging.info("=== NFL GAME PREDICTION PIPELINE ===")
    ensure_directory(config.MODELS_DIR)
    ensure_directory(config.PLOTS_DIR)
    ensure_directory(config.RESULTS_DIR)
    ensure_directory(config.CACHE_DIR)

    # Phase 1: pull data
    logging.info("\n--- PHASE 1: DATA ---")
    schedules = load_schedules(config, refresh=refresh_data)
    pbp_agg = load_team_game_pbp_aggregates(config, refresh=refresh_data)
    summarize(schedules, pbp_agg)

    # Phase 2: features
    logging.info("\n--- PHASE 2: FEATURES ---")
    games = build_modeling_dataset(schedules, pbp_agg, config)
    with open(config.FEATURE_NAMES_PATH) as f:
        feature_cols = json.load(f)

    # Phase 3: split + train
    logging.info("\n--- PHASE 3: TRAINING ---")
    X_train, X_test, y_train, y_test, test_context = split_train_test(games, feature_cols, config)
    if len(X_test) == 0:
        logging.error("Test set is empty. Check TEST_SEASONS in config.")
        sys.exit(1)
    models = train_all_models(X_train, y_train, config)

    # Phase 4: evaluate
    logging.info("\n--- PHASE 4: EVALUATION ---")
    evaluate_all_models(models, X_test, y_test, feature_cols, test_context, config)

    logging.info("\n=== PIPELINE COMPLETE ===")


if __name__ == '__main__':
    refresh = '--refresh-data' in sys.argv
    main(refresh_data=refresh)
