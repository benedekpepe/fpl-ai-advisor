"""Baseline single-stage expected-points model — kept for the backtest.

One LightGBM regressor on the leakage-free lagged features, with no two-stage
split and no fixture-strength features. The production model lives in
``model.py``; this is the honest baseline its results are compared against
(form(3) and FPL's own xP are the other reference points).

Run:  python -m src.model.baseline
"""
import os

import lightgbm as lgb
import pandas as pd

from src.model.features import build_feature_frame, feature_columns, TARGET
from src.model.model import metrics, print_table
from src.data.history import load_history_from_db
from src.config import LEGACY_MODEL_PATH as MODEL_PATH, TEST_SEASON, LGBM_PARAMS


def run_training(df: pd.DataFrame, test_season: str = TEST_SEASON) -> None:
    feats = build_feature_frame(df)
    cols = feature_columns()

    train = feats[feats["season"] != test_season]
    test = feats[feats["season"] == test_season]
    print(f"Train rows: {len(train):,}  |  Test rows ({test_season}): {len(test):,}")

    X_train, y_train = train[cols], train[TARGET]
    X_test, y_test = test[cols], test[TARGET]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results = {"baseline single-stage": metrics(y_test, pred)}
    # Baseline: predict the last-3-game average points (a naive "form" model).
    results["baseline: form(3)"] = metrics(y_test, test["roll_total_points_3"])
    # Reference: FPL's own expected points for that gameweek.
    if "xp" in test and test["xp"].notna().any():
        mask = test["xp"].notna()
        results["baseline: FPL xP"] = metrics(y_test[mask], test.loc[mask, "xp"])

    print_table(results)

    os.makedirs("models", exist_ok=True)
    model.booster_.save_model(MODEL_PATH)
    imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:8]
    print(f"\nModel saved to {MODEL_PATH}")
    print("Top features:", ", ".join(f"{c}({v})" for c, v in imp))


if __name__ == "__main__":
    run_training(load_history_from_db())
