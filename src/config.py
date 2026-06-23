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

# ----------------------------------------------------------------- seasons
CURRENT_SEASON = "2025-26"
TEST_SEASON = "2024-25"                                  # out-of-time backtest

# ------------------------------------------------------------- data source
# Where predict_all / the fixture ticker read per-gameweek history from:
#   "db"   -> local Postgres (the default demo: historical vaastav data)
#   "live" -> the live FPL API (the 2026-27 in-season mode; see ingestion/live.py)
#   "csv"  -> reserved: read the vaastav CSVs directly (no database)
# Override with the FPL_DATA_SOURCE environment variable (e.g. on deploy).
DATA_SOURCE = os.getenv("FPL_DATA_SOURCE", "db")

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
