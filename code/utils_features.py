"""
Feature Engineering Utility Functions for NFL Prediction Project
Contains helper functions for creating predictive features from game data.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def compute_point_differential(df: pd.DataFrame, home_score_col: str,
                               away_score_col: str) -> pd.DataFrame:
    # Calculate point differential (home - away) (Analysis Only)
    df = df.copy()
    if home_score_col in df.columns and away_score_col in df.columns:
        df['point_diff'] = df[home_score_col] - df[away_score_col]
        logging.debug("Point differential computed")
    else:
        logging.warning(f"Could not compute point differential: missing columns")
    return df

def compute_epa_differentials(df: pd.DataFrame, config) -> pd.DataFrame:
    # Calculate EPA differentials (home - away) for various metrics
    df = df.copy()

    epa_pairs = [
        (config.HOME_TOTAL_EPA_COL, config.AWAY_TOTAL_EPA_COL, 'epa_diff'),
        (config.HOME_PASS_EPA_COL, config.AWAY_PASS_EPA_COL, 'pass_epa_diff'),
        (config.HOME_RUSH_EPA_COL, config.AWAY_RUSH_EPA_COL, 'rush_epa_diff'),
        (config.HOME_OFF_EPA_COL, config.AWAY_OFF_EPA_COL, 'off_epa_diff'),
        (config.HOME_DEF_EPA_COL, config.AWAY_DEF_EPA_COL, 'def_epa_diff'),
    ]

    for home_col, away_col, diff_col in epa_pairs:
        if home_col and away_col and home_col in df.columns and away_col in df.columns:
            df[diff_col] = df[home_col] - df[away_col]
            logging.debug(f"Computed {diff_col}")
        else:
            logging.debug(f"Skipping {diff_col}: columns not available")

    return df

def aggregate_pbp_to_game(df_pbp: pd.DataFrame, config) -> pd.DataFrame:
    # Aggregate play-by-play data to game-level features
    if df_pbp is None or df_pbp.empty:
        logging.warning("No play-by-play data to aggregate")
        return pd.DataFrame()

    logging.info(f"Aggregating play-by-play data: {len(df_pbp)} plays")

    # Create game_id if not exists
    if config.PBP_GAME_ID_COL not in df_pbp.columns:
        df_pbp[config.PBP_GAME_ID_COL] = (
                df_pbp[config.PBP_SEASON_COL].astype(str) + '_' +
                df_pbp[config.PBP_WEEK_COL].astype(str) + '_' +
                df_pbp[config.PBP_AWAY_TEAM_COL] + '_' +
                df_pbp[config.PBP_HOME_TEAM_COL]
        )

    # Initialize aggregation dictionary
    agg_data = []

    # Group by game
    for game_id, game_plays in df_pbp.groupby(config.PBP_GAME_ID_COL):
        game_dict = {'game_id': game_id}

        # Get home and away teams
        home_team = game_plays[config.PBP_HOME_TEAM_COL].iloc[0]
        away_team = game_plays[config.PBP_AWAY_TEAM_COL].iloc[0]

        # Aggregate for home team
        home_plays = game_plays[game_plays[config.PBP_POSTEAM_COL].str.contains(home_team, na=False, case=False)]
        away_plays = game_plays[game_plays[config.PBP_POSTEAM_COL].str.contains(away_team, na=False, case=False)]

        # Count total plays
        game_dict['total_plays_home'] = len(home_plays)
        game_dict['total_plays_away'] = len(away_plays)

        # Count scoring plays
        if config.PBP_IS_SCORING_COL in game_plays.columns:
            game_dict['scoring_plays_home'] = home_plays[config.PBP_IS_SCORING_COL].sum()
            game_dict['scoring_plays_away'] = away_plays[config.PBP_IS_SCORING_COL].sum()

        # Count scoring drives (unique drives that scored)
        if config.PBP_IS_SCORING_DRIVE_COL in game_plays.columns:
            game_dict['scoring_drives_home'] = home_plays[config.PBP_IS_SCORING_DRIVE_COL].sum()
            game_dict['scoring_drives_away'] = away_plays[config.PBP_IS_SCORING_DRIVE_COL].sum()

        # Analyze play types
        if config.PBP_PLAY_TYPE_COL in game_plays.columns:
            for team_name, team_plays in [('home', home_plays), ('away', away_plays)]:
                play_types = team_plays[config.PBP_PLAY_TYPE_COL].value_counts()
                game_dict[f'pass_plays_{team_name}'] = play_types.get('Pass', 0)
                game_dict[f'rush_plays_{team_name}'] = play_types.get('Rush', 0)
                game_dict[f'kickoff_plays_{team_name}'] = play_types.get('Kickoff', 0)
                game_dict[f'punt_plays_{team_name}'] = play_types.get('Punt', 0)

        agg_data.append(game_dict)

    result_df = pd.DataFrame(agg_data)
    logging.info(f"Aggregated to {len(result_df)} games with {len(result_df.columns)} features")

    return result_df

def calculate_rolling_averages(df: pd.DataFrame, team_col: str, stat_cols: list,
                               window: int = 3, season_col: str = None) -> pd.DataFrame:
    # Calculate rolling averages for team statistics
    df = df.copy()
    df = df.sort_values(['Season', 'Week'] if 'Season' in df.columns else ['Week'])

    for stat_col in stat_cols:
        if stat_col not in df.columns:
            logging.debug(f"Skipping rolling average for missing column: {stat_col}")
            continue

        rolling_col_name = f'{stat_col}_rolling_{window}'

        if season_col:
            # Calculate within each season separately
            df[rolling_col_name] = df.groupby([team_col, season_col])[stat_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        else:
            # Calculate across all games
            df[rolling_col_name] = df.groupby(team_col)[stat_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

        logging.debug(f"Created rolling average: {rolling_col_name}")

    return df

def encode_teams(df: pd.DataFrame, team_cols: list, fit: bool = True,
                encoders: dict = None) -> tuple:
    # Encode team names to numeric values using LabelEncoder
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in team_cols:
        if col not in df.columns:
            logging.warning(f"Column {col} not found for encoding")
            continue

        encoded_col_name = f'{col}_encoded'

        if fit:
            # Fit new encoder
            encoder = LabelEncoder()
            df[encoded_col_name] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            logging.debug(f"Fitted encoder for {col}: {len(encoder.classes_)} unique values")
        else:
            # Use existing encoder
            if col not in encoders:
                logging.error(f"No encoder found for {col}")
                continue
            encoder = encoders[col]
            # Handle unseen categories
            df[encoded_col_name] = df[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
            logging.debug(f"Applied existing encoder for {col}")

    return df, encoders

def handle_missing_values(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    # Handle missing values according to specified strategy
    df = df.copy()

    for col, method in strategy.items():
        if col not in df.columns:
            continue

        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        logging.info(f"Handling {missing_count} missing values in {col} using {method}")

        if method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'mode':
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
        elif method == 'zero':
            df[col].fillna(0, inplace=True)
        elif method == 'drop':
            df.dropna(subset=[col], inplace=True)
        else:
            logging.warning(f"Unknown strategy '{method}' for column {col}")

    return df

def create_game_id(df: pd.DataFrame, season_col: str, week_col: str,
                   away_col: str, home_col: str) -> pd.DataFrame:
    # Create unique game identifier from season, week, and team info
    df = df.copy()
    df['game_id'] = (
            df[season_col].astype(str) + '_' +
            df[week_col].astype(str) + '_' +
            df[away_col].astype(str) + '_' +
            df[home_col].astype(str)
    )
    logging.debug("Created game_id column")
    return df

