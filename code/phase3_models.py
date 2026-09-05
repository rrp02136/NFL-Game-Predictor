"""
Phase 3: Model training with time-based CV and hyperparameter search.
"""
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils_io import save_model


def split_train_test(games: pd.DataFrame, feature_cols: list, config) -> tuple:
    """
    Train on everything prior to config.TEST_SEASONS.
    Test on config.TEST_SEASONS.
    2026 (current in-season) is neither trained nor tested — used only for live prediction.
    """
    games = games.sort_values(['season', 'week']).reset_index(drop=True)
    train_mask = ~games['season'].isin(config.TEST_SEASONS) & (games['season'] < config.CURRENT_SEASON)
    test_mask = games['season'].isin(config.TEST_SEASONS)

    X_train = games.loc[train_mask, feature_cols]
    y_train = games.loc[train_mask, config.TARGET_COL]
    X_test = games.loc[test_mask, feature_cols]
    y_test = games.loc[test_mask, config.TARGET_COL]

    logging.info(f"Train: {len(X_train)} games (seasons {games.loc[train_mask,'season'].min()}-{games.loc[train_mask,'season'].max()})")
    logging.info(f"Test:  {len(X_test)} games (seasons {sorted(config.TEST_SEASONS)})")
    logging.info(f"Train home win rate: {y_train.mean():.2%}")
    logging.info(f"Test  home win rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test, games.loc[test_mask].reset_index(drop=True)


def _grid_search(estimator, grid: dict, X_train, y_train, config, name: str):
    cv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    logging.info(f"[{name}] GridSearchCV: {sum(1 for _ in _iter_grid(grid))} configs x {config.CV_FOLDS} folds")
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=cv,
        scoring=config.CV_SCORING,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logging.info(f"[{name}] best params: {gs.best_params_}")
    logging.info(f"[{name}] best CV {config.CV_SCORING}: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.cv_results_


def _iter_grid(grid: dict):
    from itertools import product
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def train_random_forest(X_train, y_train, config):
    logging.info("Training Random Forest...")
    base = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
    return _grid_search(base, config.RF_PARAM_GRID, X_train, y_train, config, 'RF')


def train_xgboost(X_train, y_train, config):
    logging.info("Training XGBoost...")
    base = XGBClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method='hist',
    )
    return _grid_search(base, config.XGB_PARAM_GRID, X_train, y_train, config, 'XGB')


def train_all_models(X_train, y_train, config) -> dict:
    models = {}
    rf_model, rf_cv = train_random_forest(X_train, y_train, config)
    save_model(rf_model, f"{config.MODELS_DIR}/random_forest_best.pkl")
    models['random_forest'] = {'model': rf_model, 'cv_results': rf_cv}

    xgb_model, xgb_cv = train_xgboost(X_train, y_train, config)
    save_model(xgb_model, f"{config.MODELS_DIR}/xgboost_best.pkl")
    models['xgboost'] = {'model': xgb_model, 'cv_results': xgb_cv}

    return models
