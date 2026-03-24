"""
Main File for NFL Game Prediction Pipeline
All phases operate here: exploration, preprocessing, modeling, and evaluation
"""
import logging
import sys
import config
from utils_io import setup_logging, ensure_directory
from phase1_exploration import load_raw_games, load_raw_pbp, explore_data
from phase2_preprocessing import build_modeling_dataset
from phase3_models import split_data, train_all_models
from phase4_evaluation import evaluate_all_models

def main():
    # Execute complete NFL game prediction pipeline
    try:
        # Step 1: Setup logging and directories
        setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        logging.info("\nNFL GAME OUTCOME PREDICTION PIPELINE")
        logging.info(f"Configuration loaded from: {config.__file__}")

        ensure_directory(config.MODELS_DIR)
        ensure_directory(config.PLOTS_DIR)
        ensure_directory(config.DATA_DIR)

        # Step 2: Phase 1 - Data Exploration
        logging.info("\nPHASE 1: DATA EXPLORATION")

        df_games = load_raw_games(config)
        df_pbp = load_raw_pbp(config) if config.USE_PBP_FEATURES else None
        explore_data(df_games, df_pbp)

        # Step 3: Phase 2 - Preprocessing & Feature Engineering
        logging.info("\nPHASE 2: PREPROCESSING & FEATURE ENGINEERING")

        df_model = build_modeling_dataset(df_games, df_pbp, config)

        if len(df_model) == 0:
            logging.error("No data available after preprocessing. Exiting.")
            sys.exit(1)

        # Step 4: Split data
        logging.info("\nDATA SPLITTING")

        X_train, X_test, y_train, y_test = split_data(df_model, config)

        # Step 5: Phase 3 - Model Training
        logging.info("\nPHASE 3: MODEL TRAINING")

        models_dict = train_all_models(X_train, y_train, config)

        # Step 6: Phase 4 - Model Evaluation
        logging.info("\nPHASE 4: MODEL EVALUATION")

        feature_names = [col for col in df_model.columns if col != config.TARGET_COL]
        comparison = evaluate_all_models(models_dict, X_test, y_test, feature_names, config)

        # Final summary
        logging.info("\nPIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("\nFinal Model Comparison:")
        logging.info(f"\n{comparison.to_string()}")
        logging.info("\nOutput Locations:")
        logging.info(f" - Models: {config.MODELS_DIR}")
        logging.info(f" - Plots: {config.PLOTS_DIR}")
        logging.info(f" - Data: {config.DATA_DIR}")
        logging.info(f" - Log file: {config.LOG_FILE}")

    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()