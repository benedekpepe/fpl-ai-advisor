"""Squad / starting-XI / captain optimisation via integer programming.

Given a table of players with a predicted-points column, price, position and
club, this picks — in a single mixed-integer program — the 15-man squad, the
starting XI and the captain that maximise predicted XI points (captain counted
twice), subject to the real FPL rules.

Pure and dependency-light: no DB, no model. Feed it a DataFrame, get a
selection back. That makes it easy to test and reuse.
"""
import pandas as pd
import pulp

from src.config import SQUAD, XI_MIN, XI_MAX, MAX_PER_CLUB, DEFAULT_BUDGET


def optimize_squad(players: pd.DataFrame, budget: int = DEFAULT_BUDGET) -> pd.DataFrame:
    """Return the chosen squad with in_squad / starting / captain flags.

    `players` must have columns: pred, price, position (GK/DEF/MID/FWD), team.
    """
    p = players.reset_index(drop=True)
    idx = list(p.index)
    pred = p["pred"].to_dict()
    price = p["price"].to_dict()
    pos = p["position"].to_dict()

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("squad", idx, cat="Binary")    # in 15-man squad
    y = pulp.LpVariable.dicts("start", idx, cat="Binary")    # in starting XI
    c = pulp.LpVariable.dicts("capt", idx, cat="Binary")     # captain

    # Objective: XI points, with the captain's points counted a second time.
    prob += pulp.lpSum(pred[i] * y[i] for i in idx) + pulp.lpSum(pred[i] * c[i] for i in idx)

    for i in idx:
        prob += y[i] <= x[i]      # can only start if in the squad
        prob += c[i] <= y[i]      # can only captain if starting
    prob += pulp.lpSum(c[i] for i in idx) == 1
    prob += pulp.lpSum(y[i] for i in idx) == 11
    prob += pulp.lpSum(price[i] * x[i] for i in idx) <= budget

    for position, need in SQUAD.items():
        members = [i for i in idx if pos[i] == position]
        prob += pulp.lpSum(x[i] for i in members) == need
        prob += pulp.lpSum(y[i] for i in members) >= XI_MIN[position]
        prob += pulp.lpSum(y[i] for i in members) <= XI_MAX[position]

    for club in p["team"].unique():
        members = [i for i in idx if p.loc[i, "team"] == club]
        prob += pulp.lpSum(x[i] for i in members) <= MAX_PER_CLUB

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")

    p["in_squad"] = [int(round(x[i].value())) for i in idx]
    p["starting"] = [int(round(y[i].value())) for i in idx]
    p["captain"] = [int(round(c[i].value())) for i in idx]
    return p[p["in_squad"] == 1].copy()


def best_single_transfer(current_ids, players, budget_left=0):
    """Best one-in/one-out swap that maximises predicted points gain.

    `current_ids` is the set of player ids currently owned; `players` is the
    candidate pool (same columns as above plus `id`). Returns (out, in, gain).
    Respects position match and the budget freed by selling.
    """
    owned = players[players["id"].isin(current_ids)]
    pool = players[~players["id"].isin(current_ids)]
    best = (None, None, 0.0)
    for _, o in owned.iterrows():
        cash = budget_left + o["price"]
        cands = pool[(pool["position"] == o["position"]) & (pool["price"] <= cash)]
        if cands.empty:
            continue
        top = cands.loc[cands["pred"].idxmax()]
        gain = top["pred"] - o["pred"]
        if gain > best[2]:
            best = (o, top, gain)
    return best
