"""
Phase 3: Model training with time-based CV, hyperparameter search, and isotonic
calibration. The calibrator maps raw model output to actual frequencies so that
"70% predicted" really means ~70% win rate — essential for betting-edge use.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier
from utils_io import save_model


def split_train_calib_test(games: pd.DataFrame, feature_cols: list, config) -> tuple:
    """
    Three-way time split:
      train_fit  = everything before CALIBRATION_SEASONS
      train_cal  = CALIBRATION_SEASONS (isotonic fit only)
      test       = TEST_SEASONS (held out, never seen by any component)
    Games in CURRENT_SEASON are excluded from all three (used only for live prediction).
    """
    games = games.sort_values(['season', 'week']).reset_index(drop=True)
    cal = set(config.CALIBRATION_SEASONS)
    test = set(config.TEST_SEASONS)
    current = {config.CURRENT_SEASON}

    fit_mask = ~games['season'].isin(cal | test | current) & (games['season'] < min(cal | test))
    cal_mask = games['season'].isin(cal)
    test_mask = games['season'].isin(test)

    X_fit = games.loc[fit_mask, feature_cols]; y_fit = games.loc[fit_mask, config.TARGET_COL]
    X_cal = games.loc[cal_mask, feature_cols]; y_cal = games.loc[cal_mask, config.TARGET_COL]
    X_test = games.loc[test_mask, feature_cols]; y_test = games.loc[test_mask, config.TARGET_COL]
    test_ctx = games.loc[test_mask].reset_index(drop=True)

    logging.info(f"Train (fit):        {len(X_fit)} games ({sorted(games.loc[fit_mask,'season'].unique())})")
    logging.info(f"Train (calibrate):  {len(X_cal)} games ({sorted(config.CALIBRATION_SEASONS)})")
    logging.info(f"Test (holdout):     {len(X_test)} games ({sorted(config.TEST_SEASONS)})")
    logging.info(f"Home-win baseline (test): {y_test.mean():.2%}")

    return X_fit, y_fit, X_cal, y_cal, X_test, y_test, test_ctx


def _iter_grid(grid: dict):
    from itertools import product
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def _grid_search(estimator, grid, X_fit, y_fit, config, name: str):
    cv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    n_configs = sum(1 for _ in _iter_grid(grid))
    logging.info(f"[{name}] GridSearchCV: {n_configs} configs x {config.CV_FOLDS} folds")
    gs = GridSearchCV(estimator, grid, cv=cv, scoring=config.CV_SCORING,
                      n_jobs=-1, verbose=1, refit=True)
    gs.fit(X_fit, y_fit)
    logging.info(f"[{name}] best params: {gs.best_params_}")
    logging.info(f"[{name}] best CV {config.CV_SCORING}: {gs.best_score_:.4f}")
    return gs.best_estimator_


def _calibrate(fitted_model, X_cal, y_cal, name: str):
    logging.info(f"[{name}] fitting isotonic calibrator on {len(X_cal)} games...")
    cal = CalibratedClassifierCV(FrozenEstimator(fitted_model), method='isotonic')
    cal.fit(X_cal, y_cal)
    return cal


def train_random_forest(X_fit, y_fit, X_cal, y_cal, config):
    logging.info("Training Random Forest...")
    base = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)
    fitted = _grid_search(base, config.RF_PARAM_GRID, X_fit, y_fit, config, 'RF')
    return _calibrate(fitted, X_cal, y_cal, 'RF')


def train_xgboost(X_fit, y_fit, X_cal, y_cal, config):
    logging.info("Training XGBoost...")
    base = XGBClassifier(random_state=config.RANDOM_STATE, n_jobs=-1,
                         eval_metric='logloss', tree_method='hist')
    fitted = _grid_search(base, config.XGB_PARAM_GRID, X_fit, y_fit, config, 'XGB')
    return _calibrate(fitted, X_cal, y_cal, 'XGB')


def train_all_models(X_fit, y_fit, X_cal, y_cal, config) -> dict:
    models = {}
    rf = train_random_forest(X_fit, y_fit, X_cal, y_cal, config)
    save_model(rf, f"{config.MODELS_DIR}/random_forest_best.pkl")
    models['random_forest'] = {'model': rf}

    xgb = train_xgboost(X_fit, y_fit, X_cal, y_cal, config)
    save_model(xgb, f"{config.MODELS_DIR}/xgboost_best.pkl")
    models['xgboost'] = {'model': xgb}

    return models
