"""
Feature-engineering helpers.
All rolling features are computed from a team's PRIOR games only (shift(1))
so nothing from the game being predicted leaks in.
"""
import logging
import math
import numpy as np
import pandas as pd


# Approximate lat/lon of each team's home stadium (nflverse team codes).
# Used for travel-distance feature. Shared venues use the same coords.
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    'ARI': (33.5276, -112.2626),  # State Farm, Glendale
    'ATL': (33.7554, -84.4008),   # Mercedes-Benz
    'BAL': (39.2780, -76.6227),   # M&T Bank
    'BUF': (42.7738, -78.7870),   # Highmark
    'CAR': (35.2258, -80.8528),   # Bank of America
    'CHI': (41.8623, -87.6167),   # Soldier Field
    'CIN': (39.0954, -84.5160),   # Paycor
    'CLE': (41.5061, -81.6996),   # Cleveland Browns Stadium
    'DAL': (32.7473, -97.0945),   # AT&T
    'DEN': (39.7439, -105.0201),  # Empower
    'DET': (42.3400, -83.0456),   # Ford Field
    'GB':  (44.5013, -88.0622),   # Lambeau
    'HOU': (29.6847, -95.4107),   # NRG
    'IND': (39.7601, -86.1639),   # Lucas Oil
    'JAX': (30.3239, -81.6373),   # EverBank
    'KC':  (39.0489, -94.4839),   # Arrowhead
    'LA':  (33.9535, -118.3392),  # SoFi (legacy code for Rams)
    'LAC': (33.9535, -118.3392),  # SoFi (Chargers)
    'LAR': (33.9535, -118.3392),  # SoFi (Rams)
    'LV':  (36.0909, -115.1830),  # Allegiant
    'MIA': (25.9580, -80.2389),   # Hard Rock
    'MIN': (44.9738, -93.2581),   # U.S. Bank
    'NE':  (42.0910, -71.2643),   # Gillette
    'NO':  (29.9508, -90.0812),   # Caesars Superdome
    'NYG': (40.8135, -74.0745),   # MetLife
    'NYJ': (40.8135, -74.0745),   # MetLife
    'OAK': (37.7516, -122.2005),  # Legacy Raiders (pre-2020)
    'PHI': (39.9008, -75.1675),   # Lincoln Financial
    'PIT': (40.4468, -80.0158),   # Acrisure
    'SD':  (32.7831, -117.1196),  # Legacy Chargers (pre-2017)
    'SEA': (47.5952, -122.3316),  # Lumen
    'SF':  (37.4032, -121.9700),  # Levi's
    'STL': (38.6329, -90.1885),   # Legacy Rams (pre-2016)
    'TB':  (27.9759, -82.5033),   # Raymond James
    'TEN': (36.1665, -86.7713),   # Nissan
    'WAS': (38.9078, -76.8645),   # Northwest / FedEx
}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _american_to_prob(odds) -> float:
    if pd.isna(odds):
        return np.nan
    return (-odds) / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


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

    # Travel distance: how far the AWAY team traveled to the home stadium
    def _travel(row):
        h = STADIUM_COORDS.get(row['home_team'])
        a = STADIUM_COORDS.get(row['away_team'])
        if h is None or a is None:
            return np.nan
        return _haversine_km(h[0], h[1], a[0], a[1])
    games['away_travel_km'] = games.apply(_travel, axis=1)
    games['away_travel_km'] = games['away_travel_km'].fillna(games['away_travel_km'].median())

    # Market features: convert moneylines to devigged implied home win prob
    h_prob = games['home_moneyline'].apply(_american_to_prob)
    a_prob = games['away_moneyline'].apply(_american_to_prob)
    both = h_prob + a_prob
    games['market_home_prob'] = h_prob / both.where(both > 0)
    # Fallback: derive from spread if moneyline missing (13.86 = historical NFL score sigma)
    sigma = 13.86
    missing = games['market_home_prob'].isna() & games['spread_line'].notna()
    games.loc[missing, 'market_home_prob'] = games.loc[missing, 'spread_line'].apply(
        lambda s: 0.5 * (1 + math.erf(s / (sigma * math.sqrt(2))))
    )
    # Absolute spread magnitude (favorite strength) and total (game pace)
    games['spread_line'] = games['spread_line'].astype(float)
    games['total_line'] = games['total_line'].astype(float)

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
        'elo_diff', 'rest_diff', 'away_travel_km',
        'div_game', 'is_playoff', 'roof_outdoor', 'temp', 'wind',
        'home_elo_pre', 'away_elo_pre',
        # Market features — the single most predictive input for a betting model.
        # Included deliberately so the model defers to Vegas by default and only
        # overrides when its own signals strongly disagree.
        'spread_line', 'total_line', 'market_home_prob',
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
