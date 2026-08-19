"""Central configuration: paths, data sources, model and strategy constants.

Single source of truth for values that used to be duplicated across modules:
the LightGBM hyper-parameters were duplicated across the training scripts,
the squad quotas lived in both the optimiser and the advisor, and the dataset
URLs in several places. Import from here instead of redefining things locally.
"""
import os
from pathlib import Path

# ------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent          # repository root
MODELS_DIR = ROOT / "models"
MODEL_PATH = str(MODELS_DIR / "two_stage_v3.pkl")       # current (v3) model
LEGACY_MODEL_PATH = str(MODELS_DIR / "points_model.txt")  # old single-stage

# ------------------------------------------------------------- data sources
FPL_API = "https://fantasy.premierleague.com/api"
VAASTAV_BASE = ("https://raw.githubusercontent.com/vaastav/"
                "Fantasy-Premier-League/master/data")
MERGED_GW_URL = VAASTAV_BASE + "/{season}/gws/merged_gw.csv"
TEAMS_URL = VAASTAV_BASE + "/{season}/teams.csv"
PLAYERS_RAW_URL = VAASTAV_BASE + "/{season}/players_raw.csv"

# ----------------------------------------------------------------- seasons
CURRENT_SEASON = "2025-26"          # the finished season the hosted demo runs on
LIVE_SEASON = "2026-27"             # the live season (used when DATA_SOURCE=live)
TEST_SEASON = "2024-25"                                  # out-of-time backtest
SEED_SEASON = "2025-26"          # last completed season, used to seed pre-season/GW1 cold-start

# Stack-awareness: how hard to discourage starting two players who face each other
# with opposed point sources (attacker vs defender/keeper). 0 disables it.
CONFLICT_PENALTY = 0.35

# Bench players score no XI points, so the optimiser would otherwise fill the
# bench with the cheapest bodies regardless of whether they actually play. This
# small weight on squad-but-not-starting players' projections makes it prefer
# nailed, playing cheap options (which can auto-sub in) over non-playing
# reserves — small enough not to distort the XI or pull budget off the starters.
BENCH_WEIGHT = 0.1

# Projections blend the next few gameweeks so upcoming fixture difficulty counts,
# not just the immediate one. Weights decay (this gameweek heaviest); the length
# also sets how many gameweeks are blended. Used by the cold-start builder and by
# in-season transfer decisions (the XI you field is still scored on this GW only).
FIXTURE_WEIGHTS = [1.0, 0.6, 0.36, 0.22]

# ------------------------------------------------------------- data source
# Where predict_all / the fixture ticker read per-gameweek history from:
#   "db"   -> local Postgres (default for local development)
#   "csv"  -> the vaastav CSVs directly, no database (used for the hosted demo)
#   "live" -> the live FPL API (the 2026-27 in-season mode; see ingestion/live.py)
# Override with the FPL_DATA_SOURCE environment variable (e.g. on deploy).
DATA_SOURCE = os.getenv("FPL_DATA_SOURCE", "db")

# The season the app is actually working on: the live season in live mode,
# otherwise the demo season. Everything in-season (predictions, fixtures,
# advice) keys off this so the demo (csv/db) and live (2026-27) never mix.
ACTIVE_SEASON = LIVE_SEASON if DATA_SOURCE == "live" else CURRENT_SEASON

# ------------------------------------------------------------------- model
LGBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
    random_state=42, n_jobs=-1,
)
CLF_N_ESTIMATORS = 500                                   # P(plays) stage: fewer trees
EXTRA_FEATURES = ["opp_defence", "opp_attack"]           # fixture-difficulty features

# --------------------------------------------------------- squad / optimiser
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}          # 15-man composition
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}         # valid formation bounds
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
DEFAULT_BUDGET = 1000                                    # £100.0m; prices in tenths
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}

# ----------------------------------------------------------- advisor / chips
HIT = 4                          # points cost per transfer beyond the free ones
MAX_FT = 5                       # cap on banked free transfers
BANK_THRESHOLD = 1.0             # don't burn a free transfer for a smaller gain
HALF_DEADLINE = {1: 19, 2: 38}   # last gameweek each chip-set stays valid
HALF_LABEL = {1: "First half of the season", 2: "Second half of the season"}
CHIP_LABEL = {"wildcard": "Wildcard", "freehit": "Free Hit",
              "bboost": "Bench Boost", "3xc": "Triple Captain"}
