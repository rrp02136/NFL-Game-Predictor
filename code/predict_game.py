"""
Live prediction CLI.

Single game:
  python code/predict_game.py --home KC --away BAL --season 2026 --week 5

Full week (uses the actual NFL schedule from nfl_data_py):
  python code/predict_game.py --weekly 2026 5

Both modes print the model's home win probability, the market-implied win
probability from the closing (or current) line, the edge, and a suggested
fractional-Kelly stake for the recommended side.
"""
import argparse
import json
import logging
import os
import sys
from math import erf

import joblib
import numpy as np
import pandas as pd

import config
from utils_io import setup_logging
from phase1_exploration import load_schedules, load_team_game_pbp_aggregates
from utils_features import build_game_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_best.pkl')


def _american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    return (-odds) / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _implied_home_prob(row) -> float:
    hml = row.get('home_moneyline', np.nan)
    aml = row.get('away_moneyline', np.nan)
    if not pd.isna(hml) and not pd.isna(aml):
        h = _american_to_prob(hml)
        a = _american_to_prob(aml)
        s = h + a
        if s > 0:
            return h / s
    spread = row.get('spread_line', np.nan)
    if not pd.isna(spread):
        sigma = 13.86
        return 0.5 * (1 + erf(spread / (sigma * (2 ** 0.5))))
    return np.nan


def _kelly_fraction(p: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly (default: quarter-Kelly). Returns 0 if no edge."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - p
    f_full = (b * p - q) / b
    return max(0.0, f_full * fraction)


def _decimal_from_american(odds: float) -> float:
    if pd.isna(odds):
        return 1.909  # -110 default
    return 1 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run `python code/main.py` first.")
    model = joblib.load(MODEL_PATH)
    with open(config.FEATURE_NAMES_PATH) as f:
        feature_cols = json.load(f)
    return model, feature_cols


def build_prediction_frame(season: int, week: int, home: str = None, away: str = None) -> pd.DataFrame:
    """
    Rebuild game features from cached schedules + PBP. This is exactly the same
    engineering used at training time (so pre-game features come from PRIOR games).
    Returns the row(s) matching the requested week (and optionally single matchup).
    """
    schedules = load_schedules(config)
    pbp_agg = load_team_game_pbp_aggregates(config)

    # Include the target week's rows even though scores are NaN, by temporarily
    # marking them with dummy scores so build_team_game_long keeps them.
    upcoming = schedules[(schedules['season'] == season) & (schedules['week'] == week)].copy()
    if home and away:
        upcoming = upcoming[(upcoming['home_team'] == home) & (upcoming['away_team'] == away)]
    if upcoming.empty:
        raise ValueError(f"No scheduled game(s) found for season={season} week={week} "
                         f"home={home} away={away}")

    # Give unplayed games dummy scores so they pass through feature building.
    # They'll be dropped from the training target but retained here for feature lookup.
    stub = schedules.copy()
    mask = (stub['season'] == season) & (stub['week'] == week) & stub['home_score'].isna()
    stub.loc[mask, 'home_score'] = 0
    stub.loc[mask, 'away_score'] = 0

    games = build_game_features(stub, pbp_agg, config)
    from utils_features import drop_early_season_rows
    games = drop_early_season_rows(games, config)

    subset = games[(games['season'] == season) & (games['week'] == week)]
    if home and away:
        subset = subset[(subset['home_team'] == home) & (subset['away_team'] == away)]
    if subset.empty:
        raise ValueError("Games matched schedule but were dropped by feature engineering "
                         "(insufficient rolling history). Try a later week.")
    return subset.reset_index(drop=True)


def format_prediction(row: pd.Series, model_prob: float) -> dict:
    implied = _implied_home_prob(row)
    edge = model_prob - implied if not pd.isna(implied) else np.nan

    bet_side, kelly = None, 0.0
    if not pd.isna(edge):
        if edge >= config.BET_EDGE_THRESHOLD:
            bet_side = row['home_team']
            kelly = _kelly_fraction(model_prob, _decimal_from_american(row.get('home_moneyline', np.nan)))
        elif edge <= -config.BET_EDGE_THRESHOLD:
            bet_side = row['away_team']
            kelly = _kelly_fraction(1 - model_prob, _decimal_from_american(row.get('away_moneyline', np.nan)))

    return {
        'season': int(row['season']),
        'week': int(row['week']),
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'spread_line': None if pd.isna(row.get('spread_line')) else float(row['spread_line']),
        'model_home_prob': round(float(model_prob), 4),
        'implied_home_prob': None if pd.isna(implied) else round(float(implied), 4),
        'edge_home': None if pd.isna(edge) else round(float(edge), 4),
        'suggested_side': bet_side,
        'suggested_kelly_frac': round(float(kelly), 4),
    }


def predict(season: int, week: int, home: str = None, away: str = None) -> pd.DataFrame:
    model, feature_cols = load_model_and_features()
    games = build_prediction_frame(season, week, home, away)
    X = games[feature_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    rows = [format_prediction(games.iloc[i], probs[i]) for i in range(len(games))]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="NFL game prediction with betting-edge analysis.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--home', type=str, help="Home team code (e.g. KC)")
    group.add_argument('--weekly', nargs=2, metavar=('SEASON', 'WEEK'),
                       help="Predict every game in a week: --weekly 2026 5")
    parser.add_argument('--away', type=str, help="Away team code (with --home)")
    parser.add_argument('--season', type=int, help="Season year (with --home)")
    parser.add_argument('--week', type=int, help="Week number (with --home)")
    parser.add_argument('--output', type=str, help="Optional CSV output path")
    args = parser.parse_args()

    setup_logging(config.LOG_LEVEL, config.LOG_FILE)

    if args.weekly:
        season, week = int(args.weekly[0]), int(args.weekly[1])
        df = predict(season, week)
        default_out = os.path.join(config.RESULTS_DIR, f'predictions_week_{season}_{week}.csv')
    else:
        if not (args.away and args.season and args.week):
            parser.error("--home requires --away, --season, and --week")
        df = predict(args.season, args.week, args.home, args.away)
        default_out = os.path.join(config.RESULTS_DIR,
                                    f'prediction_{args.season}_{args.week}_{args.away}_at_{args.home}.csv')

    print(df.to_string(index=False))
    out = args.output or default_out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
