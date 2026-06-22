"""Unit tests for the squad / starting-XI / captain optimiser.

These run with no database, model or network — the optimiser takes a plain
DataFrame, so we feed it a small synthetic pool and assert the FPL rules hold.
"""
import pandas as pd

from src.optim.optimizer import optimize_squad
from src.config import SQUAD, XI_MIN, XI_MAX, MAX_PER_CLUB, DEFAULT_BUDGET


def make_pool() -> pd.DataFrame:
    """A small, feasible pool spread across many clubs (prices in tenths)."""
    counts = {"GK": 4, "DEF": 10, "MID": 10, "FWD": 8}
    rows, pid = [], 0
    for pos, n in counts.items():
        for j in range(n):
            pid += 1
            rows.append({
                "id": pid,
                "name": f"{pos}{j}",
                "position": pos,
                "team": f"club{pid % 12}",        # 12 clubs -> max-per-club easy
                "price": 40 + j * 4,              # 4.0m upward
                "pred": float(j + 1 + (0 if pos == "GK" else 2)),
            })
    return pd.DataFrame(rows)


def test_squad_has_correct_quotas():
    sq = optimize_squad(make_pool())
    assert len(sq) == 15
    assert sq["position"].value_counts().to_dict() == SQUAD


def test_squad_within_budget():
    sq = optimize_squad(make_pool(), budget=DEFAULT_BUDGET)
    assert sq["price"].sum() <= DEFAULT_BUDGET


def test_squad_respects_max_per_club():
    sq = optimize_squad(make_pool())
    assert sq["team"].value_counts().max() <= MAX_PER_CLUB


def test_starting_xi_is_a_valid_formation():
    xi = optimize_squad(make_pool()).query("starting == 1")
    assert len(xi) == 11
    by_pos = xi["position"].value_counts().to_dict()
    assert by_pos.get("GK", 0) == 1
    for pos in ("DEF", "MID", "FWD"):
        assert XI_MIN[pos] <= by_pos.get(pos, 0) <= XI_MAX[pos]


def test_exactly_one_captain_and_it_is_the_top_starter():
    sq = optimize_squad(make_pool())
    caps = sq.query("captain == 1")
    assert len(caps) == 1
    starters = sq.query("starting == 1")
    assert caps.iloc[0]["pred"] == starters["pred"].max()


def test_max_per_club_binds_when_one_club_is_stacked():
    """Even if one club has five elite players, at most three can be picked."""
    pool = make_pool()
    stacked = pd.DataFrame([{
        "id": 1000 + k, "name": f"star{k}", "position": "MID",
        "team": "stacked_fc", "price": 50, "pred": 50.0,
    } for k in range(5)])
    sq = optimize_squad(pd.concat([pool, stacked], ignore_index=True))
    assert (sq["team"] == "stacked_fc").sum() <= MAX_PER_CLUB
