"""
Phase 3: Model Training
Train Random Forest and XGBoost models with hyperparameter tuning.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils_io import save_model

def split_data(df: pd.DataFrame, config) -> tuple:
    # Split data into training and testing sets
    logging.info(f"Splitting data using strategy: {config.SPLIT_STRATEGY}")

    # Separate features and target
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Features: {X.shape[1]}")

    if config.SPLIT_STRATEGY == 'random':
        # Random split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        logging.info("Performed random stratified split")

    elif config.SPLIT_STRATEGY == 'time_based':
        # Time-based split
        if config.SEASON_COL in df.columns:
            train_mask = df[config.SEASON_COL] < config.TIME_SPLIT_SEASON
            test_mask = df[config.SEASON_COL] >= config.TIME_SPLIT_SEASON

            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

            logging.info(f"Time-based split: train on seasons < {config.TIME_SPLIT_SEASON}")
        else:
            logging.warning("Season column not found, falling back to random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=y
            )
    else:
        raise ValueError(f"Unknown split strategy: {config.SPLIT_STRATEGY}")

    logging.info(f"Training set: {len(X_train)} samples ({len(X_train) / len(df):.1%})")
    logging.info(f"Test set: {len(X_test)} samples ({len(X_test) / len(df):.1%})")
    logging.info(f"Training target distribution: {y_train.value_counts().to_dict()}")
    logging.info(f"Test target distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, config) -> tuple:
    # Train Random Forest classifier with hyperparameter tuning
    logging.info("\nTRAINING RANDOM FOREST CLASSIFIER")

    # Initialize base model
    rf_base = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)

    # Perform grid search with cross-validation
    logging.info(f"Performing GridSearchCV with {config.CV_FOLDS}-fold cross-validation")
    logging.info(f"Parameter grid: {config.RF_PARAM_GRID}")

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=config.RF_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search
    logging.info("Starting grid search (this may take several minutes)...")
    grid_search.fit(X_train, y_train)

    # Extract best model and results
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    # Log results
    logging.info("\nRANDOM FOREST RESULTS")
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best CV score: {grid_search.best_score_:.4f}")

    # Show top 5 configurations
    results_df = pd.DataFrame(cv_results)
    results_df = results_df.sort_values('rank_test_score')
    logging.info("\nTop 5 configurations:")
    cols_to_show = ['params', 'mean_test_score', 'std_test_score']
    logging.info(f"\n{results_df[cols_to_show].head().to_string()}")

    return best_model, cv_results

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, config) -> tuple:
    # Train XGBoost classifier with hyperparameter tuning
    logging.info("\nTRAINING XGBOOST CLASSIFIER")

    # Initialize base model
    xgb_base = XGBClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Perform grid search with cross-validation
    logging.info(f"Performing GridSearchCV with {config.CV_FOLDS}-fold cross-validation")
    logging.info(f"Parameter grid: {config.XGB_PARAM_GRID}")

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=config.XGB_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search
    logging.info("Starting grid search (this may take several minutes)...")
    grid_search.fit(X_train, y_train)

    # Extract best model and results
    best_model = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    # Log results
    logging.info("\nXGBOOST RESULTS")
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best CV score: {grid_search.best_score_:.4f}")

    # Show top 5 configurations
    results_df = pd.DataFrame(cv_results)
    results_df = results_df.sort_values('rank_test_score')
    logging.info("\nTop 5 configurations:")
    cols_to_show = ['params', 'mean_test_score', 'std_test_score']
    logging.info(f"\n{results_df[cols_to_show].head().to_string()}")

    return best_model, cv_results

def train_all_models(X_train: pd.DataFrame, y_train: pd.Series, config) -> dict:
    # Train both Random Forest and XGBoost models
    models_dict = {}

    # Train Random Forest
    rf_model, rf_cv_results = train_random_forest(X_train, y_train, config)
    save_model(rf_model, f"{config.MODELS_DIR}/random_forest_best.pkl")
    models_dict['random_forest'] = {
        'model': rf_model,
        'cv_results': rf_cv_results
    }

    # Train XGBoost
    xgb_model, xgb_cv_results = train_xgboost(X_train, y_train, config)
    save_model(xgb_model, f"{config.MODELS_DIR}/xgboost_best.pkl")
    models_dict['xgboost'] = {
        'model': xgb_model,
        'cv_results': xgb_cv_results
    }

    logging.info("=" * 80)
    logging.info("ALL MODELS TRAINED AND SAVED")
    logging.info("=" * 80)

    return models_dict