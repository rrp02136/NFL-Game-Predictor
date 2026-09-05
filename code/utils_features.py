"""
Feature-engineering helpers.
All rolling features are computed from a team's PRIOR games only (shift(1))
so nothing from the game being predicted leaks in.
"""
import logging
import numpy as np
import pandas as pd


def build_team_game_long(schedules: pd.DataFrame, pbp_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide (home/away per row) schedule into a long team-game frame:
    one row per (game_id, team). Attach that team's offensive PBP aggregates
    and the OPPONENT's offensive aggregates (which are that team's defensive stats).
    """
    played = schedules.dropna(subset=['home_score', 'away_score']).copy()

    home = played[['game_id', 'season', 'week', 'gameday',
                   'home_team', 'away_team',
                   'home_score', 'away_score',
                   'home_rest']].rename(columns={
        'home_team': 'team', 'away_team': 'opp',
        'home_score': 'pts_for', 'away_score': 'pts_against',
        'home_rest': 'rest',
    })
    home['is_home'] = 1

    away = played[['game_id', 'season', 'week', 'gameday',
                   'away_team', 'home_team',
                   'away_score', 'home_score',
                   'away_rest']].rename(columns={
        'away_team': 'team', 'home_team': 'opp',
        'away_score': 'pts_for', 'home_score': 'pts_against',
        'away_rest': 'rest',
    })
    away['is_home'] = 0

    long = pd.concat([home, away], ignore_index=True)
    long['won'] = (long['pts_for'] > long['pts_against']).astype(int)

    # Attach the team's own offensive PBP aggregates
    long = long.merge(pbp_agg, on=['game_id', 'season', 'week', 'team'], how='left')

    # Attach opponent's offensive aggregates -> these become team's defensive-allowed stats
    opp_agg = pbp_agg.rename(columns={
        'team': 'opp',
        'off_epa': 'def_epa_allowed',
        'off_success': 'def_success_allowed',
        'off_plays': 'def_plays_faced',
        'off_ypp': 'def_ypp_allowed',
        'off_pass_epa': 'def_pass_epa_allowed',
        'off_rush_epa': 'def_rush_epa_allowed',
    })
    long = long.merge(opp_agg, on=['game_id', 'season', 'week', 'opp'], how='left')

    long['gameday'] = pd.to_datetime(long['gameday'], errors='coerce')
    long = long.sort_values(['team', 'gameday', 'week']).reset_index(drop=True)
    return long


def add_rolling_features(long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling averages of prior N games per team, computed as shift(1).rolling(N)."""
    stat_cols = [
        'off_epa', 'off_success', 'off_ypp', 'off_pass_epa', 'off_rush_epa',
        'def_epa_allowed', 'def_success_allowed', 'def_ypp_allowed',
        'def_pass_epa_allowed', 'def_rush_epa_allowed',
        'pts_for', 'pts_against', 'won',
    ]
    long = long.copy()
    grp = long.groupby('team', sort=False)
    for col in stat_cols:
        long[f'{col}_r{window}'] = (
            grp[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=3).mean())
        )
    return long


def compute_elo(schedules: pd.DataFrame,
                k: float, hfa: float, init: float, season_regress: float) -> pd.DataFrame:
    """
    Compute pre-game Elo for home & away team for every played game.
    Season regression: at the start of each new season, rating = init + (1-regress) * (rating - init).
    Returns DataFrame with columns: game_id, home_elo_pre, away_elo_pre.
    """
    played = schedules.dropna(subset=['home_score', 'away_score']).copy()
    played['gameday'] = pd.to_datetime(played['gameday'], errors='coerce')
    played = played.sort_values(['season', 'gameday', 'week']).reset_index(drop=True)

    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []

    for _, g in played.iterrows():
        h, a, s = g['home_team'], g['away_team'], int(g['season'])

        # Season regression for each team when they first appear in a new season
        for t in (h, a):
            if t not in ratings:
                ratings[t] = init
                last_season[t] = s
            elif last_season[t] != s:
                ratings[t] = init + (1.0 - season_regress) * (ratings[t] - init)
                last_season[t] = s

        h_elo = ratings[h]
        a_elo = ratings[a]

        # Expected home win prob (Elo with home-field advantage)
        expected_home = 1.0 / (1.0 + 10.0 ** (-((h_elo + hfa) - a_elo) / 400.0))
        actual_home = 1.0 if g['home_score'] > g['away_score'] else (0.5 if g['home_score'] == g['away_score'] else 0.0)

        rows.append({'game_id': g['game_id'], 'home_elo_pre': h_elo, 'away_elo_pre': a_elo})

        # Update ratings for next game
        delta = k * (actual_home - expected_home)
        ratings[h] = h_elo + delta
        ratings[a] = a_elo - delta

    return pd.DataFrame(rows)


def build_game_features(schedules: pd.DataFrame, pbp_agg: pd.DataFrame, config) -> pd.DataFrame:
    """
    Return one row per played game with pre-game features and the target `home_win`.
    Betting lines are kept for later ATS analysis but NOT used as model features.
    """
    logging.info("Building team-game long table...")
    long = build_team_game_long(schedules, pbp_agg)

    logging.info(f"Adding rolling features (window={config.ROLLING_WINDOW})...")
    long = add_rolling_features(long, window=config.ROLLING_WINDOW)

    # Pivot to home/away
    home_cols = long[long['is_home'] == 1].copy()
    away_cols = long[long['is_home'] == 0].copy()

    r = config.ROLLING_WINDOW
    rolling_stats = [
        f'off_epa_r{r}', f'off_success_r{r}', f'off_ypp_r{r}',
        f'off_pass_epa_r{r}', f'off_rush_epa_r{r}',
        f'def_epa_allowed_r{r}', f'def_success_allowed_r{r}', f'def_ypp_allowed_r{r}',
        f'def_pass_epa_allowed_r{r}', f'def_rush_epa_allowed_r{r}',
        f'pts_for_r{r}', f'pts_against_r{r}', f'won_r{r}',
    ]
    keep = ['game_id', 'season', 'week', 'gameday', 'team', 'rest'] + rolling_stats

    h = home_cols[keep].rename(columns={'team': 'home_team', 'rest': 'home_rest',
                                        **{c: f'home_{c}' for c in rolling_stats}})
    a = away_cols[keep].rename(columns={'team': 'away_team', 'rest': 'away_rest',
                                        **{c: f'away_{c}' for c in rolling_stats}})

    games = h.merge(a, on=['game_id', 'season', 'week', 'gameday'])

    # Differential features (home - away)
    for stat in rolling_stats:
        games[f'{stat}_diff'] = games[f'home_{stat}'] - games[f'away_{stat}']

    games['rest_diff'] = games['home_rest'] - games['away_rest']

    # Merge schedule context: weather, roof, div_game, betting lines, scores
    sched = schedules[['game_id', 'game_type', 'div_game', 'roof',
                       'temp', 'wind', 'spread_line', 'total_line',
                       'home_score', 'away_score', 'home_moneyline', 'away_moneyline']].copy()
    games = games.merge(sched, on='game_id', how='left')

    # Roof: outdoor games get real temp/wind; indoor treat as neutral
    games['roof_outdoor'] = (games['roof'] == 'outdoors').astype(int)
    games['temp'] = np.where(games['roof_outdoor'] == 1, games['temp'], 70.0)
    games['wind'] = np.where(games['roof_outdoor'] == 1, games['wind'], 0.0)
    games['temp'] = games['temp'].fillna(70.0)
    games['wind'] = games['wind'].fillna(0.0)

    games['is_playoff'] = (games['game_type'] != 'REG').astype(int)
    games['div_game'] = games['div_game'].fillna(0).astype(int)

    # Elo
    logging.info("Computing Elo ratings...")
    elo = compute_elo(schedules, config.ELO_K, config.ELO_HFA,
                      config.ELO_INIT, config.ELO_SEASON_REGRESS)
    games = games.merge(elo, on='game_id', how='left')
    games['elo_diff'] = games['home_elo_pre'] - games['away_elo_pre']

    # Target
    games = games.dropna(subset=['home_score', 'away_score'])
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)

    return games


def select_feature_columns(games: pd.DataFrame, config) -> list[str]:
    """Whitelist of pre-game features passed into the model."""
    r = config.ROLLING_WINDOW
    diffs = [
        f'off_epa_r{r}_diff', f'off_success_r{r}_diff', f'off_ypp_r{r}_diff',
        f'off_pass_epa_r{r}_diff', f'off_rush_epa_r{r}_diff',
        f'def_epa_allowed_r{r}_diff', f'def_success_allowed_r{r}_diff',
        f'def_ypp_allowed_r{r}_diff',
        f'def_pass_epa_allowed_r{r}_diff', f'def_rush_epa_allowed_r{r}_diff',
        f'pts_for_r{r}_diff', f'pts_against_r{r}_diff', f'won_r{r}_diff',
    ]
    other = [
        'elo_diff', 'rest_diff',
        'div_game', 'is_playoff', 'roof_outdoor', 'temp', 'wind',
        'home_elo_pre', 'away_elo_pre',
    ]
    cols = [c for c in diffs + other if c in games.columns]
    return cols


def drop_early_season_rows(games: pd.DataFrame, config) -> pd.DataFrame:
    """
    Rolling features need at least 3 prior games; drop rows where they're NaN.
    Keeps Week 4+ of the first season and all subsequent seasons roughly intact.
    """
    r = config.ROLLING_WINDOW
    key = f'off_epa_r{r}_diff'
    before = len(games)
    games = games.dropna(subset=[key]).reset_index(drop=True)
    logging.info(f"Dropped {before - len(games)} rows lacking rolling history "
                 f"({len(games)} games remain)")
    return games
