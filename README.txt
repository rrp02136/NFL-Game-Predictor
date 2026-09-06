# NFL Game Outcome Prediction

Machine-learning pipeline for predicting NFL regular-season and playoff game
winners, plus a betting-edge analysis that compares model probabilities to
sportsbook implied probabilities.

Data source: [nflverse](https://nflverse.nflverse.com/) via the `nfl_data_py`
package (schedules 2002–present including betting lines, rest days, weather,
and division flags; play-by-play with EPA/success back to 2002).

---

## 1. Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS users training XGBoost need OpenMP:

```bash
brew install libomp
```

---

## 2. Run the pipeline

```bash
python code/main.py
```

On first run this downloads ~2 GB of play-by-play (2002–current) and caches
it as parquet under `data/cache/`. Subsequent runs skip the download.
Force a refresh (e.g. to pull the latest 2026 weekly results) with:

```bash
python code/main.py --refresh-data
```

The pipeline:

1. Pulls schedules + PBP from nflverse
2. Builds per-team per-game EPA aggregates
3. Engineers **pre-game-only** features: rolling (window=8) offensive/defensive
   EPA, success rate, yards/play; Elo rating with season regression; rest
   differential; div_game, weather (outdoor games), playoff flag
4. Trains RandomForest + XGBoost with time-series CV
5. Reports accuracy, log-loss, Brier score, reliability diagram, and an
   against-the-spread backtest at the configured edge threshold

Outputs:

- `code/models/random_forest_best.pkl`, `code/models/xgboost_best.pkl`
- `code/plots/*.png` (confusion, reliability, feature importance)
- `results/model_comparison.csv`
- `results/{random_forest,xgboost}_ats_bets.csv`
- `data/game_features.parquet`, `data/feature_names.json`

---

## 3. Make predictions

Single game:

```bash
python code/predict_game.py --home KC --away BAL --season 2026 --week 5
```

Full week:

```bash
python code/predict_game.py --weekly 2026 5
```

Output columns:

```
season week home_team away_team spread_line model_home_prob
implied_home_prob edge_home suggested_side suggested_kelly_frac
```

`suggested_side` is populated only when |model_prob − implied_prob| >=
`BET_EDGE_THRESHOLD` (default 5 percentage points). `suggested_kelly_frac` is
quarter-Kelly on the recommended side.

Team codes follow the standard nflverse 2–3 letter codes: `KC`, `BAL`, `SF`,
`LAR`, `NYG`, `NYJ`, `JAX`, `LV`, `LAC`, `WAS`, etc.

---

## 4. Configuration

All knobs live in `code/config.py`:

- `TRAIN_SEASON_START = 2002`, `CURRENT_SEASON = 2026`
- `TEST_SEASONS = [2024, 2025]` — held out for honest test metrics
- `ROLLING_WINDOW = 8`
- `ELO_K`, `ELO_HFA`, `ELO_SEASON_REGRESS` — Elo hyperparameters
- `BET_EDGE_THRESHOLD = 0.05` — minimum probability edge before flagging a bet
- Hyperparameter grids for RF/XGB

---

## 5. Honest expectations & betting disclaimer

**Current test-set results (2024–2025 holdout, 570 games):**

| Model | Accuracy | AUC | Brier | ATS hit @ 10% edge | ATS ROI @ 10% |
|---|---|---|---|---|---|
| Random Forest (calibrated) | 68.1% | 0.734 | 0.210 | 49.3% (67 bets) | −6.0% |
| XGBoost (calibrated)       | 66.5% | 0.710 | 0.214 | 39.7% (116 bets) | −24.3% |

**What this means honestly:**

- Straight-up NFL win prediction ceilings around **~65–70%** with strong
  features. This model sits at 68.1%. Anything above ~72% almost certainly
  means data leakage.
- **Beating the closing spread is much harder than picking winners.** Pro
  sharps hit **~53–55% ATS** long-term (break-even at -110 juice is 52.4%).
  This model does NOT beat the closing NFL side spread — the ATS backtest is
  negative at every edge threshold tested. That's the expected finding: full-
  game NFL sides are the most efficient market in sports.
- **How to actually use this responsibly:**
  1. Straight-up predictions are useful for research and non-market decisions
     (survivor pools, weekly picks against family, team-total prop research).
  2. If you're going to bet, only use it as one input among many, and
     preferentially on markets where the line is stale or thin (early-week
     openers, less-followed props, live in-game lines).
  3. The Kelly fraction shown is **quarter-Kelly** — small on purpose. Even
     the best models blow up bankrolls at full Kelly because measured edges
     are optimistic vs. real-world edges (variance, line movement, juice).
- The backtest is retrospective and assumes the closing line — real execution
  is worse (you'll get worse lines than the closer).
- Sports betting involves real risk of loss. Bet only what you can afford to
  lose. This project is educational, not financial advice.

---

## 6. Project structure

```
code/
  config.py               # paths, feature params, grids, betting knobs
  main.py                 # end-to-end pipeline
  phase1_exploration.py   # nfl_data_py loaders + parquet caching
  phase2_preprocessing.py # feature build orchestration
  phase3_models.py        # time-series CV grid search
  phase4_evaluation.py    # metrics, calibration, ATS backtest
  predict_game.py         # CLI: single-game or weekly prediction
  utils_features.py       # team-game long, rolling stats, Elo
  utils_io.py             # logging + persistence helpers
data/
  cache/                  # parquet caches of nflverse pulls (gitignored)
  game_features.parquet   # final engineered table (gitignored)
  feature_names.json      # feature order used at inference
results/                  # metrics + ATS bet logs (gitignored)
code/models/              # trained model pickles
code/plots/               # confusion, reliability, importance PNGs
```

---

## 7. Retraining as the 2026 season progresses

`nfl_data_py` publishes new PBP roughly 24–48 hours after each game. To pull
the latest results and retrain:

```bash
python code/main.py --refresh-data
```

Early-season predictions (weeks 1–3) are always weaker because the rolling
window is filled mostly with prior-season carryover. Trust the model more
from week 4 onward.
