"""
Prediction Script for Future NFL Games
Load trained model and make predictions on new games
"""
import logging
import pandas as pd
import joblib
import sys
import os

def setup_logging():
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_model_and_encoders(model_path, encoder_path):
    # Load trained model and label encoders
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logging.info(f"Loading encoders from {encoder_path}")
    encoders = joblib.load(encoder_path)

    return model, encoders

def load_feature_names(feature_names_path):
    # Load feature names used during training
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    logging.info(f"Loaded {len(feature_names)} feature names")
    return feature_names

def encode_team_name(team_name, encoder, column_name):
    # Encode team name using trained encoder
    if team_name in encoder.classes_:
        return encoder.transform([team_name])[0]
    else:
        logging.warning(f"Unknown team: {team_name}. Using -1")
        return -1

def prepare_game_input(home_team, away_team, season, week,
                       encoders, feature_names,
                       total_score=None, is_playoff=0):
    # Prepare single game for prediction
    # Use default total score if not provided (league average ~45)
    if total_score is None:
        total_score = 45.0

    # Encode teams
    home_encoded = encode_team_name(home_team, encoders['HomeTeam'], 'HomeTeam')
    away_encoded = encode_team_name(away_team, encoders['AwayTeam'], 'AwayTeam')

    # Create feature dictionary
    features = {
        'Season': season,
        'HomeTeam_encoded': home_encoded,
        'AwayTeam_encoded': away_encoded,
        'total_score': total_score,
        'PostSeason': is_playoff,
        'is_playoff': is_playoff,
        'week_numeric': week,
        'week_of_season': week,
        'season_period': 0 if week <= 6 else (1 if week <= 12 else 2)
    }

    # Create DataFrame with all required features
    df = pd.DataFrame([features])

    # Ensure all features from training are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0  # Add missing features with default value

    # Reorder columns to match training order
    df = df[feature_names]

    return df

def predict_game(model, game_df):
    # Make prediction for game
    prediction = model.predict(game_df)[0]
    probabilities = model.predict_proba(game_df)[0]
    home_win_prob = probabilities[1]

    return prediction, home_win_prob

def main():
    # Main prediction function
    setup_logging()

    # Paths
    MODEL_PATH = "code/models/xgboost_best.pkl"  # Best model
    ENCODER_PATH = "data/label_encoders.pkl"
    FEATURE_NAMES_PATH = "data/feature_names.txt"

    # Load model and encoders
    model, encoders = load_model_and_encoders(MODEL_PATH, ENCODER_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)

    # Example prediction
    logging.info("\nMAKING PREDICTION")

    # Input game details
    home_team = "Lions"
    away_team = "Cowboys"
    season = 2025
    week = 14
    total_score = 54.5

    logging.info(f"Game: {away_team} @ {home_team}")
    logging.info(f"Season: {season}, Week: {week}")

    # Prepare input
    game_input = prepare_game_input(
        home_team=home_team,
        away_team=away_team,
        season=season,
        week=week,
        encoders=encoders,
        feature_names=feature_names,
        total_score=total_score,  # Estimated total score
        is_playoff=0
    )

    # Make prediction
    prediction, home_win_prob = predict_game(model, game_input)

    # Display results
    logging.info("\nPREDICTION RESULTS")
    if prediction == 1:
        logging.info(f"Predicted Winner: {home_team}")
        logging.info(f"Confidence: {home_win_prob * 100:.1f}%")
    else:
        logging.info(f"Predicted Winner: {away_team}")
        logging.info(f"Confidence: {(1 - home_win_prob) * 100:.1f}%")

    logging.info(f"\nProbabilities:")
    logging.info(f"  {home_team} Win: {home_win_prob * 100:.1f}%")
    logging.info(f"  {away_team} Win: {(1 - home_win_prob) * 100:.1f}%")

def predict_multiple_games(games_list):
    # Predict multiple games at once
    setup_logging()

    # Load model and encoders
    MODEL_PATH = "code/models/xgboost_best.pkl"
    ENCODER_PATH = "data/label_encoders.pkl"
    FEATURE_NAMES_PATH = "data/feature_names.txt"

    model, encoders = load_model_and_encoders(MODEL_PATH, ENCODER_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)

    results = []

    for game in games_list:
        # Prepare input
        game_input = prepare_game_input(
            home_team=game['home_team'],
            away_team=game['away_team'],
            season=game['season'],
            week=game['week'],
            encoders=encoders,
            feature_names=feature_names,
            total_score=game.get('total_score', 45.0),
            is_playoff=game.get('is_playoff', 0)
        )

        # Make prediction
        prediction, home_win_prob = predict_game(model, game_input)

        results.append({
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'predicted_winner': game['home_team'] if prediction == 1 else game['away_team'],
            'home_win_probability': home_win_prob,
            'away_win_probability': 1 - home_win_prob
        })

    # Display results
    results_df = pd.DataFrame(results)
    logging.info("\nPREDICTIONS FOR MULTIPLE GAMES")
    logging.info(f"\n{results_df.to_string(index=False)}")

    return results_df

if __name__ == "__main__":
    main()
    games = [
        {'home_team': 'Texans', 'away_team': 'Bills', 'season': 2024, 'week': 12, 'total_score': 43.5},
        {'home_team': 'Lions', 'away_team': 'Giants', 'season': 2024, 'week': 12, 'total_score': 50.5},
        {'home_team': 'Titans', 'away_team': 'Seahawks', 'season': 2024, 'week': 12, 'total_score': 40.5},
        {'home_team': 'Bengals', 'away_team': 'Patriots', 'season': 2024, 'week': 12, 'total_score': 49.5},
        {'home_team': 'Ravens', 'away_team': 'Jets', 'season': 2024, 'week': 12, 'total_score': 44.5},
        {'home_team': 'Chiefs', 'away_team': 'Colts', 'season': 2024, 'week': 12, 'total_score': 50.5},
        {'home_team': 'Packers', 'away_team': 'Vikings', 'season': 2024, 'week': 12, 'total_score': 41.5},
        {'home_team': 'Bears', 'away_team': 'Steelers', 'season': 2024, 'week': 12, 'total_score': 45.5},
        {'home_team': 'Cardinals', 'away_team': 'Jaguars', 'season': 2024, 'week': 12, 'total_score': 47.5},
        {'home_team': 'Raiders', 'away_team': 'Browns', 'season': 2024, 'week': 12, 'total_score': 36.5},
        {'home_team': 'Cowboys', 'away_team': 'Eagles', 'season': 2024, 'week': 12, 'total_score': 47.5},
        {'home_team': 'Saints', 'away_team': 'Falcons', 'season': 2024, 'week': 12, 'total_score': 40.5},
        {'home_team': 'Rams', 'away_team': 'Buccaneers', 'season': 2024, 'week': 12, 'total_score': 49.5},
        {'home_team': '49ers', 'away_team': 'Panthers', 'season': 2024, 'week': 12, 'total_score': 49.5},
        {'home_team': 'Lions', 'away_team': 'Packers', 'season': 2025, 'week': 13, 'total_score': 48.5},
        {'home_team': 'Cowboys', 'away_team': 'Chiefs', 'season': 2025, 'week': 13, 'total_score': 52.5},
        {'home_team': 'Bengals', 'away_team': 'Ravens', 'season': 2025, 'week': 13, 'total_score': 52.5},
        {'home_team': 'Bears', 'away_team': 'Eagles', 'season': 2025, 'week': 13, 'total_score': 44.5},
        {'home_team': 'Texans', 'away_team': 'Colts', 'season': 2025, 'week': 13, 'total_score': 44.5},
        {'home_team': 'Rams', 'away_team': 'Panthers', 'season': 2025, 'week': 13, 'total_score': 45.5},
        {'home_team': '49ers', 'away_team': 'Browns', 'season': 2025, 'week': 13, 'total_score': 35.5},
        {'home_team': 'Jaguars', 'away_team': 'Titans', 'season': 2025, 'week': 13, 'total_score': 41.5},
        {'home_team': 'Cardinals', 'away_team': 'Buccaneers', 'season': 2025, 'week': 13, 'total_score': 44.5},
        {'home_team': 'Saints', 'away_team': 'Dolphins', 'season': 2025, 'week': 13, 'total_score': 42.5},
        {'home_team': 'Falcons', 'away_team': 'Jets', 'season': 2025, 'week': 13, 'total_score': 38.5},
        {'home_team': 'Vikings', 'away_team': 'Seahawks', 'season': 2025, 'week': 13, 'total_score': 41.5},
        {'home_team': 'Raiders', 'away_team': 'Chargers', 'season': 2025, 'week': 13, 'total_score': 40.5},
        {'home_team': 'Bills', 'away_team': 'Steelers', 'season': 2025, 'week': 13, 'total_score': 45.5},
        {'home_team': 'Broncos', 'away_team': 'Commanders', 'season': 2025, 'week': 13, 'total_score': 43.5},
        {'home_team': 'Giants', 'away_team': 'Patriots', 'season': 2025, 'week': 13, 'total_score': 46.5},
        {'home_team': 'Lions', 'away_team': 'Cowboys', 'season': 2025, 'week': 14, 'total_score': 54.5},
        {'home_team': 'Falcons', 'away_team': 'Seahawks', 'season': 2025, 'week': 14, 'total_score': 44.5},
        {'home_team': 'Bills', 'away_team': 'Bengals', 'season': 2025, 'week': 14, 'total_score': 52.5},
        {'home_team': 'Packers', 'away_team': 'Bears', 'season': 2025, 'week': 14, 'total_score': 44.5},
        {'home_team': 'Browns', 'away_team': 'Titans', 'season': 2025, 'week': 14, 'total_score': 33.5},
        {'home_team': 'Jaguars', 'away_team': 'Colts', 'season': 2025, 'week': 14, 'total_score': 47.5},
        {'home_team': 'Vikings', 'away_team': 'Commanders', 'season': 2025, 'week': 14, 'total_score': 41.5},
        {'home_team': 'Jets', 'away_team': 'Dolphins', 'season': 2025, 'week': 14, 'total_score': 41.5},
        {'home_team': 'Ravens', 'away_team': 'Steelers', 'season': 2025, 'week': 14, 'total_score': 42.5},
        {'home_team': 'Raiders', 'away_team': 'Broncos', 'season': 2025, 'week': 14, 'total_score': 40.5},
        {'home_team': 'Cardinals', 'away_team': 'Rams', 'season': 2025, 'week': 14, 'total_score': 48.5},
        {'home_team': 'Chiefs', 'away_team': 'Texans', 'season': 2025, 'week': 14, 'total_score': 41.5},
        {'home_team': 'Chargers', 'away_team': 'Eagles', 'season': 2025, 'week': 14, 'total_score': 40.5}
    ]
    predict_multiple_games(games)


