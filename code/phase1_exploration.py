"""
Phase 1: Data ingestion via nfl_data_py.
Pulls schedules (with betting lines, weather, rest, div flag) and play-by-play
(for per-team per-game EPA aggregates). Parquet-caches to avoid re-downloading.
"""
import logging
import os
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from utils_io import ensure_directory


PBP_COLS = [
    'game_id', 'season', 'week', 'posteam', 'defteam',
    'home_team', 'away_team', 'play_type', 'epa', 'success',
    'pass', 'rush', 'yards_gained', 'season_type',
]


def load_schedules(config, refresh: bool = False) -> pd.DataFrame:
    """Fetch schedules for every season in range, cache to parquet."""
    ensure_directory(config.CACHE_DIR)
    if not refresh and os.path.exists(config.SCHEDULES_CACHE):
        df = pd.read_parquet(config.SCHEDULES_CACHE)
        logging.info(f"Loaded cached schedules: {df.shape}")
        return df

    seasons = list(range(config.TRAIN_SEASON_START, config.CURRENT_SEASON + 1))
    logging.info(f"Fetching schedules from nflverse for seasons {seasons[0]}-{seasons[-1]}...")
    df = nfl.import_schedules(seasons)
    df.to_parquet(config.SCHEDULES_CACHE, index=False)
    logging.info(f"Schedules fetched and cached: {df.shape} -> {config.SCHEDULES_CACHE}")
    return df


def load_team_game_pbp_aggregates(config, refresh: bool = False) -> pd.DataFrame:
    """
    Return per (game_id, team) offensive aggregates from play-by-play:
    epa_per_play, success_rate, pass_epa, rush_epa, plays, yards_per_play.
    Defensive stats are derived by joining on the opponent's row.
    """
    ensure_directory(config.CACHE_DIR)
    if not refresh and os.path.exists(config.PBP_CACHE):
        df = pd.read_parquet(config.PBP_CACHE)
        logging.info(f"Loaded cached PBP aggregates: {df.shape}")
        return df

    seasons = list(range(config.TRAIN_SEASON_START, config.CURRENT_SEASON + 1))
    logging.info(f"Fetching PBP for seasons {seasons[0]}-{seasons[-1]} (this can take a few minutes)...")
    pbp = nfl.import_pbp_data(seasons, columns=PBP_COLS, downcast=True)

    # Only offensive plays with a valid posteam and epa
    off = pbp[pbp['posteam'].notna() & pbp['epa'].notna()].copy()
    off = off[off['play_type'].isin(['pass', 'run'])]

    grp = off.groupby(['game_id', 'season', 'week', 'posteam'], as_index=False)
    agg = grp.agg(
        off_epa=('epa', 'mean'),
        off_success=('success', 'mean'),
        off_plays=('epa', 'size'),
        off_ypp=('yards_gained', 'mean'),
        off_pass_epa=('epa', lambda s: s[off.loc[s.index, 'pass'] == 1].mean()),
        off_rush_epa=('epa', lambda s: s[off.loc[s.index, 'rush'] == 1].mean()),
    )
    agg = agg.rename(columns={'posteam': 'team'})
    agg.to_parquet(config.PBP_CACHE, index=False)
    logging.info(f"PBP aggregates cached: {agg.shape} -> {config.PBP_CACHE}")
    return agg


def summarize(schedules: pd.DataFrame, pbp_agg: pd.DataFrame) -> None:
    """Brief EDA summary logged for sanity checks."""
    logging.info("=== DATA SUMMARY ===")
    played = schedules.dropna(subset=['home_score', 'away_score'])
    logging.info(f"Schedules total rows: {len(schedules):,}")
    logging.info(f"Games with results:   {len(played):,}")
    logging.info(f"Season range:         {schedules['season'].min()}-{schedules['season'].max()}")
    logging.info(f"Games per season (played):")
    per_season = played.groupby('season').size()
    for season, n in per_season.items():
        logging.info(f"  {season}: {n}")
    home_wr = (played['home_score'] > played['away_score']).mean()
    logging.info(f"Home win rate: {home_wr:.2%}")
    logging.info(f"PBP team-game rows: {len(pbp_agg):,}")
