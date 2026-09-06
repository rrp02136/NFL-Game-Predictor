"""
Phase 4: Evaluation + betting-relevant diagnostics.
Reports accuracy, log-loss, Brier, ROC-AUC, reliability diagram, and an
against-the-spread backtest across multiple edge thresholds.
"""
import logging
import os
from math import erf, sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, roc_auc_score, confusion_matrix
)
from utils_io import ensure_directory


def _american_to_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    return (-odds) / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _underlying_estimator(model):
    """Extract the raw fitted RF/XGB from CalibratedClassifierCV(FrozenEstimator(base))."""
    if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
        est = model.calibrated_classifiers_[0].estimator
        # est may itself be a FrozenEstimator wrapping the real classifier
        if hasattr(est, 'estimator'):
            est = est.estimator
        return est
    return model


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_prob),
        'brier': brier_score_loss(y_test, y_prob),
        'roc_auc': roc_auc_score(y_test, y_prob),
    }
    logging.info(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != 'model'))
    return metrics


def plot_confusion(y_true, y_pred, name: str, config):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=config.PLOT_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Home Win'], yticklabels=['Away Win', 'Home Win'])
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f'{name}_confusion_matrix.png')
    plt.savefig(path, dpi=config.PLOT_DPI); plt.close()
    logging.info(f"Saved {path}")


def plot_reliability(y_true, y_prob, name: str, config, n_bins: int = 10):
    df = pd.DataFrame({'p': y_prob, 'y': y_true})
    df['bin'] = pd.cut(df['p'], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    agg = df.groupby('bin', observed=True).agg(mean_pred=('p', 'mean'),
                                                 actual=('y', 'mean'),
                                                 n=('y', 'size')).reset_index(drop=True)
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    plt.plot(agg['mean_pred'], agg['actual'], 'o-', label=name)
    for _, row in agg.iterrows():
        plt.annotate(f"n={int(row['n'])}", (row['mean_pred'], row['actual']),
                     fontsize=8, textcoords='offset points', xytext=(5, 5))
    plt.xlabel('Predicted probability'); plt.ylabel('Actual win rate')
    plt.title(f'Reliability Diagram — {name}')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f'{name}_reliability.png')
    plt.savefig(path, dpi=config.PLOT_DPI); plt.close()
    logging.info(f"Saved {path}")


def plot_feature_importance(model, feature_names, name: str, config):
    est = _underlying_estimator(model)
    if not hasattr(est, 'feature_importances_'):
        return
    imp = pd.DataFrame({'feature': feature_names,
                        'importance': est.feature_importances_}).sort_values('importance', ascending=False)
    top = imp.head(config.TOP_N_FEATURES)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top)), top['importance'])
    plt.yticks(range(len(top)), top['feature'])
    plt.gca().invert_yaxis()
    plt.title(f'Top {config.TOP_N_FEATURES} Features — {name}')
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f'{name}_feature_importance.png')
    plt.savefig(path, dpi=config.PLOT_DPI); plt.close()
    logging.info(f"Saved {path}")
    logging.info(f"Top 10 features for {name}:")
    for _, row in imp.head(10).iterrows():
        logging.info(f"  {row['feature']:40s}  {row['importance']:.4f}")


def _implied_home_prob(row) -> float:
    hml, aml = row.get('home_moneyline', np.nan), row.get('away_moneyline', np.nan)
    if not pd.isna(hml) and not pd.isna(aml):
        h, a = _american_to_prob(hml), _american_to_prob(aml)
        s = h + a
        if s > 0:
            return h / s
    spread = row.get('spread_line', np.nan)
    if not pd.isna(spread):
        sigma = 13.86
        return 0.5 * (1 + erf(spread / (sigma * sqrt(2))))
    return np.nan


def ats_backtest_sweep(test_context: pd.DataFrame, y_prob: np.ndarray,
                       name: str, config) -> pd.DataFrame:
    """Report ATS bets/hit/ROI at multiple edge thresholds; save the 5% bet log."""
    ctx = test_context.copy()
    ctx['model_home_prob'] = y_prob
    ctx['implied_home_prob'] = ctx.apply(_implied_home_prob, axis=1)
    ctx['edge'] = ctx['model_home_prob'] - ctx['implied_home_prob']
    ctx['home_actual'] = ctx['home_win']

    rows = []
    for thr in config.ATS_EDGE_THRESHOLDS:
        bets = ctx[ctx['edge'].abs() >= thr].copy()
        if bets.empty:
            rows.append({'model': name, 'edge_threshold': thr,
                         'n_bets': 0, 'hit_rate': np.nan, 'roi': np.nan})
            continue
        bets['bet_home'] = (bets['edge'] > 0).astype(int)
        bets['won_bet'] = ((bets['bet_home'] == 1) & (bets['home_actual'] == 1)) | \
                         ((bets['bet_home'] == 0) & (bets['home_actual'] == 0))
        payout = np.where(bets['won_bet'], 100 / 110, -1.0)  # -110 juice
        rows.append({'model': name, 'edge_threshold': thr,
                     'n_bets': int(len(bets)),
                     'hit_rate': float(bets['won_bet'].mean()),
                     'roi': float(payout.mean())})

        if abs(thr - config.BET_EDGE_THRESHOLD) < 1e-9:
            out = os.path.join(config.RESULTS_DIR, f'{name}_ats_bets.csv')
            bets[['game_id', 'season', 'week', 'home_team', 'away_team', 'spread_line',
                  'model_home_prob', 'implied_home_prob', 'edge', 'bet_home',
                  'home_actual', 'won_bet']].to_csv(out, index=False)
            logging.info(f"Saved {out}")

    sweep = pd.DataFrame(rows)
    logging.info(f"[{name}] ATS sweep:")
    for _, r in sweep.iterrows():
        thr_pct = f"{r['edge_threshold']:.0%}"
        if r['n_bets'] == 0:
            logging.info(f"  edge>={thr_pct}: no bets")
        else:
            logging.info(f"  edge>={thr_pct}: bets={r['n_bets']:4d}  hit={r['hit_rate']:.2%}  "
                         f"ROI={r['roi']:+.2%}  (break-even = 52.4%)")
    return sweep


def evaluate_all_models(models: dict, X_test, y_test, feature_names: list,
                        test_context: pd.DataFrame, config) -> pd.DataFrame:
    ensure_directory(config.PLOTS_DIR)
    ensure_directory(config.RESULTS_DIR)

    metric_rows = []
    sweep_frames = []
    for name, entry in models.items():
        model = entry['model']
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metric_rows.append(evaluate_model(model, X_test, y_test, name))
        plot_confusion(y_test, y_pred, name, config)
        plot_reliability(y_test.values, y_prob, name, config)
        plot_feature_importance(model, feature_names, name, config)

        sweep_frames.append(ats_backtest_sweep(test_context, y_prob, name, config))

    metrics_df = pd.DataFrame(metric_rows).set_index('model')
    metrics_df.to_csv(config.MODEL_COMPARISON_PATH)
    logging.info(f"\n=== MODEL METRICS ===\n{metrics_df.to_string()}")
    logging.info(f"Saved {config.MODEL_COMPARISON_PATH}")

    sweep = pd.concat(sweep_frames, ignore_index=True)
    sweep.to_csv(config.ATS_BACKTEST_PATH, index=False)
    logging.info(f"Saved {config.ATS_BACKTEST_PATH}")

    return metrics_df
