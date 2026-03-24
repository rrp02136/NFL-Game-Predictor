"""
Phase 1: Data Exploration
Load and explore raw game and play-by-play data
"""
import logging
import pandas as pd
import numpy as np
from utils_io import save_dataframe

def load_raw_games(config) -> pd.DataFrame:
    # Load and combine game-level scores data from multiple sources
    logging.info("Loading game-level data...")

    dfs = []
    for path in config.GAME_CSV_PATHS:
        try:
            df = pd.read_csv(path)
            logging.info(f"Loaded {len(df)} games from {path}")
            dfs.append(df)
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
        except Exception as e:
            logging.error(f"Error loading {path}: {str(e)}")

    if not dfs:
        raise ValueError("No game data files could be loaded")

    # Combine all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates if any
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates()
    if len(df_combined) < initial_len:
        logging.warning(f"Removed {initial_len - len(df_combined)} duplicate rows")

    logging.info(f"Total games loaded: {len(df_combined)}")

    # Save combined data
    save_dataframe(df_combined, config.RAW_GAMES_COMBINED_PATH)

    return df_combined

def load_raw_pbp(config) -> pd.DataFrame:
    # Load and combine play-by-play data from multiple sources
    if not config.USE_PBP_FEATURES or not config.PBP_CSV_PATHS:
        logging.info("Play-by-play data loading skipped (disabled in config)")
        return pd.DataFrame()

    logging.info("Loading play-by-play data...")

    dfs = []
    for path in config.PBP_CSV_PATHS:
        try:
            df = pd.read_csv(path)
            logging.info(f"Loaded {len(df)} plays from {path}")
            dfs.append(df)
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
        except Exception as e:
            logging.error(f"Error loading {path}: {str(e)}")

    if not dfs:
        logging.warning("No play-by-play data files could be loaded")
        return pd.DataFrame()

    # Combine all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)

    logging.info(f"Total plays loaded: {len(df_combined)}")

    # Save combined data
    save_dataframe(df_combined, config.RAW_PLAYS_COMBINED_PATH)

    return df_combined

def explore_data(df_games: pd.DataFrame, df_pbp: pd.DataFrame = None) -> None:
    # Perform exploratory data analysis on game and play-by-play data
    logging.info("\nEXPLORATORY DATA ANALYSIS")

    # Game-level data exploration
    logging.info("\nGAME-LEVEL DATA")
    logging.info(f"Shape: {df_games.shape}")
    logging.info(f"Columns: {list(df_games.columns)}")

    # Display data types and missing values
    logging.info("\nData Types and Missing Values:")
    info_df = pd.DataFrame({
        'Column': df_games.columns,
        'Type': df_games.dtypes.values,
        'Missing': df_games.isna().sum().values,
        'Missing %': (df_games.isna().sum().values / len(df_games) * 100).round(2)
    })
    logging.info(f"\n{info_df.to_string()}")

    # Numerical columns summary
    logging.info("\nNumerical Columns Summary:")
    numeric_cols = df_games.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logging.info(f"\n{df_games[numeric_cols].describe().to_string()}")

    # Check for unique values in key columns
    key_columns = ['Season', 'Week', 'HomeTeam', 'AwayTeam']
    existing_key_cols = [col for col in key_columns if col in df_games.columns]

    if existing_key_cols:
        logging.info("\nUnique Values in Key Columns:")
        for col in existing_key_cols:
            unique_count = df_games[col].nunique()
            logging.info(f"  {col}: {unique_count} unique values")
            if col in ['Season', 'Week']:
                # Handle mixed types in Week column (strings and numbers)
                try:
                    if col == 'Season':
                        logging.info(f"    Range: {df_games[col].min()} to {df_games[col].max()}")
                    else:
                        # For Week, just show unique values without sorting due to mixed types
                        unique_vals = df_games[col].dropna().unique()
                        logging.info(f"    Unique weeks: {len(unique_vals)}")
                except:
                    logging.info(f"    Range: Cannot determine (mixed types)")
            elif unique_count < 50:  # Show values if not too many
                try:
                    # Drop NaN values before sorting to avoid comparison issues
                    unique_vals = df_games[col].dropna().unique()
                    # Try to sort, but handle if mixed types exist
                    try:
                        sorted_vals = sorted(unique_vals)
                        logging.info(f"Values: {sorted_vals}")
                    except:
                        logging.info(f"Values: {list(unique_vals)}")
                except:
                    logging.info(f"Values: Cannot display (mixed types)")

    # Games per season
    if 'Season' in df_games.columns:
        games_per_season = df_games.groupby('Season').size()
        logging.info("\nGames per Season:")
        logging.info(f"\n{games_per_season.to_string()}")

    # Check for data quality issues
    logging.info("\nData Quality Checks:")

    # Check for negative scores
    if 'HomeScore' in df_games.columns and 'AwayScore' in df_games.columns:
        neg_scores = ((df_games['HomeScore'] < 0) | (df_games['AwayScore'] < 0)).sum()
        logging.info(f"Negative scores: {neg_scores}")

        # Check for ties
        ties = (df_games['HomeScore'] == df_games['AwayScore']).sum()
        logging.info(f"Tied games: {ties}")

        # Check for outliers (very high scores)
        high_scores = ((df_games['HomeScore'] > 70) | (df_games['AwayScore'] > 70)).sum()
        logging.info(f"Games with scores > 70: {high_scores}")

    # Play-by-play data exploration
    if df_pbp is not None and not df_pbp.empty:
        logging.info("\nPLAY-BY-PLAY DATA")
        logging.info(f"Shape: {df_pbp.shape}")
        logging.info(f"Columns: {list(df_pbp.columns)}")

        # Play-by-play missing values
        logging.info("\nPlay-by-Play Missing Values:")
        pbp_missing = df_pbp.isna().sum()
        pbp_missing_pct = (pbp_missing / len(df_pbp) * 100).round(2)
        pbp_info = pd.DataFrame({
            'Column': pbp_missing.index,
            'Missing': pbp_missing.values,
            'Missing %': pbp_missing_pct.values
        })
        logging.info(f"\n{pbp_info[pbp_info['Missing'] > 0].to_string()}")

        # Play type distribution
        if 'PlayOutcome' in df_pbp.columns:
            play_types = df_pbp['PlayOutcome'].value_counts().head(10)
            logging.info("\nTop 10 Play Types:")
            logging.info(f"\n{play_types.to_string()}")

    logging.info("\nEXPLORATION COMPLETE")

