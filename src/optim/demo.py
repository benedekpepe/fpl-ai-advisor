"""Demo the optimiser on a historical gameweek.

Loads the saved two-stage model, predicts every player's points for one
gameweek, runs the squad/XI/captain optimiser, and prints the recommended
team alongside what those players *actually* scored.

Run:  python -m src.optim.demo                 # default season + GW
      python -m src.optim.demo 2024-25 20       # specific season and GW
"""
import sys

import joblib
import pandas as pd

from src.model.features import build_feature_frame
from src.data.history import load_history_from_db
from src.model.model import attach_opponent_strength
from src.optim.optimizer import optimize_squad
from src.config import MODEL_PATH

DEFAULT_SEASON = "2024-25"
DEFAULT_GW = 20


def predict_gameweek(season: str, gw: int) -> pd.DataFrame:
    bundle = joblib.load(MODEL_PATH)
    clf, reg, cols = bundle["clf"], bundle["reg"], bundle["cols"]

    df = attach_opponent_strength(load_history_from_db())
    feats = build_feature_frame(df)
    rows = feats[(feats["season"] == season) & (feats["gw"] == gw)].copy()

    p_play = clf.predict_proba(rows[cols])[:, 1]
    rows["pred"] = p_play * reg.predict(rows[cols])

    # Aggregate to one row per player (sums across double-gameweek fixtures).
    agg = rows.groupby("element").agg(
        name=("name", "first"), position=("position", "first"),
        team=("team", "first"), price=("value", "first"),
        pred=("pred", "sum"), actual=("total_points", "sum"),
    ).reset_index().rename(columns={"element": "id"})
    return agg


def show(season: str, gw: int) -> None:
    players = predict_gameweek(season, gw)
    squad = optimize_squad(players)

    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.sort_values(
        ["starting", "position"], key=lambda s: s.map(order) if s.name == "position" else s,
        ascending=[False, True],
    )

    xi = squad[squad["starting"] == 1]
    bench = squad[squad["starting"] == 0]
    cap = squad[squad["captain"] == 1].iloc[0]

    print(f"\n=== Recommended team for {season} GW{gw} ===")
    print(f"{'pos':<5}{'player':<20}{'team':<14}{'£':>6}{'pred':>7}{'actual':>8}")
    print("-" * 60)
    for _, r in xi.iterrows():
        star = "  (C)" if r["captain"] else ""
        print(f"{r['position']:<5}{(r['name']+star):<20}{r['team']:<14}"
              f"{r['price']/10:>6.1f}{r['pred']:>7.2f}{r['actual']:>8.0f}")
    print("-- bench --")
    for _, r in bench.iterrows():
        print(f"{r['position']:<5}{r['name']:<20}{r['team']:<14}"
              f"{r['price']/10:>6.1f}{r['pred']:>7.2f}{r['actual']:>8.0f}")

    xi_actual = xi["actual"].sum() + cap["actual"]   # captain doubled
    xi_pred = xi["pred"].sum() + cap["pred"]
    print("-" * 60)
    print(f"Squad cost: £{squad['price'].sum()/10:.1f}m   "
          f"XI predicted (incl. C): {xi_pred:.1f}   "
          f"XI actual (incl. C): {xi_actual:.0f}")


if __name__ == "__main__":
    season = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEASON
    gw = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GW
    show(season, gw)
