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

from src.config import SQUAD, XI_MIN, XI_MAX, MAX_PER_CLUB, DEFAULT_BUDGET, BENCH_WEIGHT


def conflict_dock(prob, y, p, conflict_penalty, min_pred=2.5):
    """LP dock for starting pairs that face each other with opposed point sources.

    Needs a `match` column (an id equal for both players in the same fixture),
    plus `team`, `position` and `pred`. Two starters conflict when they share a
    match but sit on opposite sides (different `team`) with opposed roles: an
    attacker (MID/FWD) vs a defender/keeper (weight 1.0), or two clean-sheet
    reliant defenders (0.6); two attackers don't conflict. Adds the auxiliary
    binary vars/constraints to `prob` and returns the penalty expression to
    subtract from the objective (0 when not applicable). The dock scales with the
    weaker projection, so cheap bench enablers are effectively unaffected.
    """
    if conflict_penalty <= 0 or "match" not in p.columns:
        return 0
    idx = list(p.index)
    pred = p["pred"].to_dict()
    match = p["match"].to_dict()
    team = p["team"].to_dict()
    pos = p["position"].to_dict()
    xgi = p["xgi"].to_dict() if "xgi" in p.columns else {}
    XGI_ATTACK = 0.2                       # per-game xGI above which a mid/fwd is a
    #                                        genuine attacking threat (vs defcon)

    def role(i):
        if pos[i] in ("GK", "DEF"):
            return "def"                   # clean-sheet reliant
        # A mid/fwd only threatens an opposing defence if they actually attack; a
        # defensive midfielder scoring off tackles/CBIT (defcon) does not, so it
        # conflicts with nobody. Default to attacker when xGI is unknown.
        return "att" if xgi.get(i, 1.0) >= XGI_ATTACK else "neutral"

    cand = [i for i in idx if pred[i] >= min_pred and pd.notna(match.get(i))]
    docks = []
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            i, j = cand[a], cand[b]
            if match[i] != match[j] or team[i] == team[j]:
                continue                       # not the same match, or same side
            ri, rj = role(i), role(j)
            if ri == "neutral" or rj == "neutral":
                weight = 0.0                    # a defensive mid conflicts with no-one
            elif ri != rj:
                weight = 1.0                    # attacker vs clean-sheet defence
            elif ri == "def":
                weight = 0.6                    # two opposing clean-sheet defenders
            else:
                weight = 0.0                    # two attackers
            if weight <= 0:
                continue
            z = pulp.LpVariable(f"conflict_{i}_{j}", cat="Binary")
            prob += z <= y[i]
            prob += z <= y[j]
            prob += z >= y[i] + y[j] - 1
            docks.append(conflict_penalty * weight * min(pred[i], pred[j]) * z)
    return pulp.lpSum(docks) if docks else 0


def optimize_squad(players: pd.DataFrame, budget: int = DEFAULT_BUDGET,
                   force_ids=None, conflict_penalty: float = 0.0,
                   bench_boost: bool = False) -> pd.DataFrame:
    """Return the chosen squad with in_squad / starting / captain flags.

    `players` must have columns: pred, price, position (GK/DEF/MID/FWD), team.
    `force_ids`, if given, is a set of values from the `id` column that must be
    included in the 15-man squad (used to pin must-have picks like a premium
    forward at the start of the season).

    `conflict_penalty` (> 0, needs a `match` column giving each player's fixture
    id for the gameweek) makes the optimiser *stack-aware*: it docks the objective
    when two **starting** players face each other with opposed point sources — an
    attacker (MID/FWD) against a defender/keeper, or two clean-sheet reliant
    defenders — so it avoids picking both sides of the same match. See
    ``conflict_dock``.
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

    # Objective: XI points, with the captain's points counted a second time, plus
    # a reward for the bench's projections. Normally small (nailed cheap cover);
    # under Bench Boost all 15 score, so the bench counts in full.
    bench_w = 1.0 if bench_boost else BENCH_WEIGHT
    objective = (pulp.lpSum(pred[i] * y[i] for i in idx)
                 + pulp.lpSum(pred[i] * c[i] for i in idx)
                 + bench_w * pulp.lpSum(pred[i] * (x[i] - y[i]) for i in idx))
    objective = objective - conflict_dock(prob, y, p, conflict_penalty)

    prob += objective

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

    if force_ids and "id" in p.columns:
        forced = set(force_ids)
        for i in idx:
            if p.loc[i, "id"] in forced:
                prob += x[i] == 1     # must be in the squad

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
