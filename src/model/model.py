"""Production expected-points model: two-stage + fixture difficulty.

  - loads the historical gameweek table from Postgres,
  - attaches each opponent's attack/defence strength (via the shared team
    loader) as fixture-difficulty features,
  - builds the leakage-free lagged features (features.py) plus the two fixture
    features,
  - trains the two-stage model  P(plays) x E[points | plays],
  - reports against FPL's own xP and a form baseline, and saves the model.

Run:  python -m src.model.model
"""
import os

import joblib
import lightgbm as lgb
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.model.features import build_feature_frame, feature_columns, TARGET
from src.data.history import load_history_from_db
from src.data.teams import team_strength
from src.config import (MODEL_PATH, EXTRA_FEATURES, TEST_SEASON,
                        LGBM_PARAMS as PARAMS, CLF_N_ESTIMATORS)


def metrics(y_true, y_pred) -> dict:
    """MAE / RMSE / Spearman for a set of predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "Spearman": spearmanr(y_true, y_pred).correlation,
    }


def print_table(results: dict) -> None:
    """Pretty-print a {model_name: metrics} comparison table."""
    print(f"\n{'model':<22}{'MAE':>8}{'RMSE':>8}{'Spearman':>10}")
    print("-" * 48)
    for name, m in results.items():
        print(f"{name:<22}{m['MAE']:>8.3f}{m['RMSE']:>8.3f}{m['Spearman']:>10.3f}")


def attach_opponent_strength(history: pd.DataFrame) -> pd.DataFrame:
    """Merge each row's opponent attack/defence strength (home/away aware)."""
    strength = team_strength(sorted(history["season"].unique()))
    df = history.merge(strength, on=["season", "opponent_team"], how="left")
    # player at home -> opponent is the away side, and vice versa
    df["opp_defence"] = df["strength_defence_away"].where(
        df["was_home"], df["strength_defence_home"]
    )
    df["opp_attack"] = df["strength_attack_away"].where(
        df["was_home"], df["strength_attack_home"]
    )
    return df


def run(df: pd.DataFrame, test_season: str = TEST_SEASON) -> None:
    df = attach_opponent_strength(df)
    feats = build_feature_frame(df)
    cols = feature_columns() + EXTRA_FEATURES

    train = feats[feats["season"] != test_season]
    test = feats[feats["season"] == test_season]
    print(f"Train rows: {len(train):,}  |  Test rows ({test_season}): {len(test):,}")

    played = train["minutes"] > 0
    clf = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": CLF_N_ESTIMATORS})
    clf.fit(train[cols], played.astype(int))
    p_play = clf.predict_proba(test[cols])[:, 1]

    reg = lgb.LGBMRegressor(**PARAMS)
    reg.fit(train.loc[played, cols], train.loc[played, TARGET])
    pts_if_play = reg.predict(test[cols])

    combined = p_play * pts_if_play

    y = test[TARGET]
    results = {
        "two-stage + fixtures": metrics(y, combined),
        "baseline: form(3)": metrics(y, test["roll_total_points_3"]),
    }
    if "xp" in test and test["xp"].notna().any():
        m = test["xp"].notna()
        results["baseline: FPL xP"] = metrics(y[m], test.loc[m, "xp"])
    print_table(results)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"clf": clf, "reg": reg, "cols": cols}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    run(load_history_from_db())
