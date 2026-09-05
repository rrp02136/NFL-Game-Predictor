"""
Phase 4: Evaluation + betting-relevant diagnostics.
Beyond accuracy/AUC we report:
  - Brier score & log-loss (calibration-sensitive)
  - Reliability diagram (are 60% predictions winning 60%?)
  - ATS backtest: does betting when |model_prob - implied_prob| >= threshold profit?
"""
import logging
import os
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


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
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
    return agg


def plot_feature_importance(model, feature_names, name: str, config):
    if not hasattr(model, 'feature_importances_'):
        return
    imp = pd.DataFrame({'feature': feature_names,
                        'importance': model.feature_importances_}).sort_values('importance', ascending=False)
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


def ats_backtest(test_context: pd.DataFrame, y_prob: np.ndarray, name: str, config) -> dict:
    """
    Convert spread_line to an implied home win probability using a normal-approx trick:
    P(home covers ~= wins) ~ implied from moneyline; if moneyline missing, fall back to
    approximating from spread with a 13.86-point sigma (historical NFL scoring stdev).
    Bet home when model_prob - implied_prob > threshold; bet away otherwise (mirror).
    ROI computed at -110 juice on 1-unit stakes.
    """
    ctx = test_context.copy()
    ctx['model_home_prob'] = y_prob

    # Prefer explicit home_moneyline if present; otherwise convert spread
    home_ml = ctx['home_moneyline'].astype(float)
    away_ml = ctx['away_moneyline'].astype(float)
    implied_from_ml = home_ml.apply(_american_to_prob)
    # Devig by symmetric normalization when both sides available
    both = implied_from_ml + away_ml.apply(_american_to_prob)
    both = both.where(both > 0, np.nan)
    ctx['implied_home_prob'] = implied_from_ml / both

    # Fallback: use spread_line if moneyline is missing (spread convention: negative = home favored)
    missing = ctx['implied_home_prob'].isna() & ctx['spread_line'].notna()
    from math import erf, sqrt
    sigma = 13.86
    if missing.any():
        ctx.loc[missing, 'implied_home_prob'] = ctx.loc[missing, 'spread_line'].apply(
            lambda s: 0.5 * (1 + erf(s / (sigma * (2 ** 0.5))))
        )

    ctx['edge'] = ctx['model_home_prob'] - ctx['implied_home_prob']
    ctx['home_actual'] = ctx['home_win']

    thr = config.BET_EDGE_THRESHOLD
    bets = ctx[ctx['edge'].abs() >= thr].copy()
    if bets.empty:
        logging.info(f"[{name}] No bets clear the {thr:.0%} edge threshold")
        return {'model': name, 'n_bets': 0, 'hit_rate': np.nan, 'roi': np.nan}

    # Bet home when edge > thr, bet away when edge < -thr
    bets['bet_home'] = (bets['edge'] > 0).astype(int)
    bets['won_bet'] = ((bets['bet_home'] == 1) & (bets['home_actual'] == 1)) | \
                     ((bets['bet_home'] == 0) & (bets['home_actual'] == 0))
    # -110 juice: risk 110 to win 100
    payout = np.where(bets['won_bet'], 100 / 110, -1.0)
    roi = payout.mean()
    hit = bets['won_bet'].mean()

    logging.info(f"[{name}] ATS backtest: bets={len(bets)}, hit={hit:.2%}, ROI={roi:+.2%} "
                 f"(break-even hit = 52.4%)")

    out_path = os.path.join(config.RESULTS_DIR, f'{name}_ats_bets.csv')
    ensure_directory(config.RESULTS_DIR)
    bets[['game_id', 'season', 'week', 'home_team', 'away_team', 'spread_line',
          'model_home_prob', 'implied_home_prob', 'edge', 'bet_home',
          'home_actual', 'won_bet']].to_csv(out_path, index=False)
    logging.info(f"Saved {out_path}")

    return {'model': name, 'n_bets': int(len(bets)), 'hit_rate': float(hit), 'roi': float(roi)}


def evaluate_all_models(models: dict, X_test, y_test, feature_names: list,
                        test_context: pd.DataFrame, config) -> pd.DataFrame:
    ensure_directory(config.PLOTS_DIR)
    ensure_directory(config.RESULTS_DIR)

    metric_rows = []
    ats_rows = []
    for name, entry in models.items():
        model = entry['model']
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metric_rows.append(evaluate_model(model, X_test, y_test, name))
        plot_confusion(y_test, y_pred, name, config)
        plot_reliability(y_test.values, y_prob, name, config)
        plot_feature_importance(model, feature_names, name, config)

        ats_rows.append(ats_backtest(test_context, y_prob, name, config))

    metrics_df = pd.DataFrame(metric_rows).set_index('model')
    ats_df = pd.DataFrame(ats_rows).set_index('model')
    combined = metrics_df.join(ats_df)

    combined.to_csv(config.MODEL_COMPARISON_PATH)
    logging.info(f"\n=== MODEL COMPARISON ===\n{combined.to_string()}")
    logging.info(f"Saved {config.MODEL_COMPARISON_PATH}")
    return combined
