"""Personalised FPL advice for a real manager, by team_id.

Pulls the manager's actual squad, free transfers and chip status from the FPL
API, scores every player for the target gameweek with the saved model, then:
  - recommends the best XI, captain and an ordered bench from their 15;
  - plans transfers with a -4 penalty per transfer beyond the free ones, so the
    optimal number (0/1/2.. taking hits) falls out of the optimisation;
  - detects which chips are still available in the current half of the season
    and, for each, finds the best gameweek to play it before it expires
    (deadline-aware "use it or lose it"), via a multi-gameweek projection.

Run:  python -m src.advisor.personal 3136812 15
      python -m src.advisor.personal 3136812 15 --ft 2    # override the FT estimate
"""
import sys
import logging
import os
from functools import lru_cache

import joblib
import pandas as pd
import pulp
import requests

from src.model.features import build_feature_frame
from src.data.history import load_history
from src.model.model import attach_opponent_strength
from src.optim.optimizer import optimize_squad, conflict_dock
from src.data.teams import teams_meta
from src.config import (
    ACTIVE_SEASON as SEASON, FPL_API as FPL, MODEL_PATH, CONFLICT_PENALTY,
    DATA_SOURCE, BENCH_WEIGHT, FIXTURE_WEIGHTS, DEFAULT_BUDGET,
    SQUAD, XI_MIN, XI_MAX, MAX_PER_CLUB, HIT, MAX_FT, BANK_THRESHOLD,
    POSITION_ORDER as ORDER, CHIP_LABEL, HALF_DEADLINE, HALF_LABEL,
)


LOGGER = logging.getLogger(__name__)


# --------------------------- predictions ----------------------------------
@lru_cache(maxsize=4)
def predict_all(season: str) -> pd.DataFrame:
    """Predicted points for every player in every gameweek of the season.

    Cached per season: the feature build + model inference is the heaviest step
    (tens of seconds over several seasons of history), so it runs once per
    session rather than on every request. Call ``predict_all.cache_clear()``
    after re-ingesting fresh data.
    """
    return _predict_from_history(season, load_history())


def _predict_from_history(season: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Score one history snapshot; live callers must not reuse a season-only cache."""
    bundle = joblib.load(MODEL_PATH)
    clf, reg, cols = bundle["clf"], bundle["reg"], bundle["cols"]
    feats = build_feature_frame(attach_opponent_strength(raw))
    feats = feats[feats["season"] == season].copy()
    if feats.empty:                       # early season: no current-season rows yet
        return pd.DataFrame(columns=["gw", "id", "name", "position", "team",
                                     "price", "pred", "actual"])
    rs = raw[raw["season"] == season] if "season" in raw.columns else raw
    from src.model.projections import score_features
    observed_gws = set(rs.loc[rs["minutes"].notna(), "gw"].unique())
    scored = []
    for target, group in feats.groupby("gw"):
        group = group.copy()
        played_before = sum(g < target for g in observed_gws)
        group["pred"], group["pred_model"], group["p_play"], weight = score_features(
            group, bundle, played_before)
        group["form_weight"] = weight
        scored.append(group)
    feats = pd.concat(scored)
    feats["opponent_rating_source"] = feats.get("opponent_rating_source", "historical")
    return feats.groupby(["gw", "element"]).agg(
        name=("name", "first"), position=("position", "first"),
        team=("team", "first"), price=("value", "first"),
        pred=("pred", "sum"), pred_model=("pred_model", "sum"),
        xgi=("roll_expected_goal_involvements_5", "first"),
        p_play=("p_play", "mean"), recent_minutes=("roll_minutes_5", "first"),
        recent_starts=("starts_rate_5", "first"), recent_form=("roll_total_points_3", "first"),
        form_weight=("form_weight", "first"), fixture_count=("fixture", "nunique"),
        opponent_rating_source=("opponent_rating_source", "first"),
        actual=("total_points", "sum"),
    ).reset_index().rename(columns={"element": "id"})


# --------------------------- FPL API --------------------------------------
def _get(path):
    r = requests.get(f"{FPL}/{path}", timeout=30,
                     headers={"User-Agent": "fpl-ai-advisor/0.1"})
    r.raise_for_status()
    return r.json()


def get_picks(team_id, gw):
    return _get(f"entry/{team_id}/event/{gw}/picks/")


def get_transfers(team_id):
    return _get(f"entry/{team_id}/transfers/")


def selling_prices(picks, pool, transfers, history, bootstrap, target_gw):
    """Public acquisition prices where known; explicitly flag incomplete budgets."""
    market = pool.set_index("id")["price"].to_dict()
    freehit_gws = {c["event"] for c in history.get("chips", []) if c["name"] == "freehit"}
    bought = {}
    for move in sorted(transfers or [], key=lambda m: m.get("time", "")):
        if move.get("event", 0) >= target_gw or move.get("event") in freehit_gws:
            continue
        bought[move["element_in"]] = move.get("element_in_cost")
    initial = {e["id"]: e["now_cost"]-e.get("cost_change_start", 0)
               for e in bootstrap.get("elements", []) if "cost_change_start" in e}
    started = min((h["event"] for h in history.get("current", [])), default=0)
    result, unknown = {}, []
    for pick in picks.get("picks", []):
        pid = pick["element"]
        if pid not in market:
            continue
        if pick.get("selling_price") is not None:
            result[pid] = int(pick["selling_price"])
            continue
        purchase = pick.get("purchase_price", bought.get(pid))
        if purchase is None and pid not in bought and transfers is not None and started == 1:
            purchase = initial.get(pid)
        now = int(market[pid])
        if purchase is None:
            result[pid] = now
            unknown.append(pid)
        else:
            result[pid] = now if now <= purchase else int(purchase + (now-purchase)//2)
    return result, unknown


def get_history(team_id):
    return _get(f"entry/{team_id}/history/")


# --------------------------- context from history -------------------------
def estimate_free_transfers(history: dict, target_gw: int, season: str = SEASON) -> int:
    """Reconstruct available free transfers entering target_gw (an estimate).

    GW1 is unlimited (a fresh squad), so nothing rolls over from it — entering
    GW2 you have exactly 1. Rollovers only start accruing from GW2 onward.
    """
    made = {c["event"]: c.get("event_transfers", 0) for c in history.get("current", [])}
    chip = {c["event"]: c["name"] for c in history.get("chips", [])}
    first_gw = min(made, default=1)
    if target_gw <= first_gw + 1:
        return 1
    ft = 1
    for g in range(first_gw + 1, target_gw):
        # Transfer chips preserve the banked count; they do not earn another FT.
        if chip.get(g) not in ("wildcard", "freehit"):
            ft = min(MAX_FT, max(0, ft - made.get(g, 0)) + 1)
        if season == "2025-26" and g + 1 == 16:
            ft = MAX_FT  # One-off AFCON top-up, not a permanent FPL rule.
    return ft


def chips_available(history: dict, target_gw: int):
    """Which chips remain in the current half, and when they expire."""
    half = 1 if target_gw <= 19 else 2
    lo, hi = (1, 19) if half == 1 else (20, 38)
    # only chips already used up to (and including) the current GW count as gone
    used = {c["name"] for c in history.get("chips", [])
            if lo <= c["event"] <= min(hi, target_gw)}
    available = [c for c in CHIP_LABEL if c not in used]
    if target_gw == 1:
        available = [c for c in available if c not in ("wildcard", "freehit")]
    if any(c.get("name") == "freehit" and c.get("event") == target_gw - 1
           for c in history.get("chips", [])):
        available = [c for c in available if c != "freehit"]
    return available, HALF_DEADLINE[half], half


def chips_used(history: dict, target_gw: int) -> dict:
    """{chip_name: gameweek} for chips already played in the current half."""
    half = 1 if target_gw <= 19 else 2
    lo, hi = (1, 19) if half == 1 else (20, 38)
    hi = min(hi, target_gw)
    return {c["name"]: int(c["event"]) for c in history.get("chips", [])
            if lo <= c["event"] <= hi and c["name"] in CHIP_LABEL}


# Free Hit / Wildcard are worth burning only for a big swing — a blank or double
# gameweek, or an injury crisis — not to tidy a slightly sub-optimal team early on
# (normal transfers do that). Require a meaningful value before recommending them.
# (Tunable: raise to hold chips longer, lower to deploy them more readily.)
CHIP_MIN_VALUE = {"freehit": 8.0, "wildcard": 18.0, "bboost": 16.0, "3xc": 8.0}


def recommend_chip(timing: dict, gw: int, deadline_gw: int = None):
    """The chip to play THIS gameweek, or None.

    `timing` maps each chip to a value-sorted ``[(gameweek, value), ...]`` over
    the remaining window. A chip is recommended when this gameweek is **near** its
    best week (within a margin, so a huge, roughly-flat gain — e.g. a Wildcard for
    a broken team — isn't held five weeks for a marginally better peak) *and* its
    value clears ``CHIP_MIN_VALUE`` (a real swing, not a small early-season gap or
    a weak bench). If several qualify, the highest-value one wins; else None (hold).
    This matches the per-chip "worth it now" badge shown in the table.
    """
    candidates = {}
    for chip, ranked in timing.items():
        if not ranked:
            continue
        best_gw, best_val = ranked[0]
        this_val = dict(ranked).get(gw)
        if this_val is None:
            continue
        margin = max(1.0, 0.1 * best_val)
        if this_val < best_val - margin:
            continue                       # a clearly better week is still ahead
        minimum = 0.0 if deadline_gw == gw else CHIP_MIN_VALUE.get(chip, 0.0)
        if this_val <= 0 or this_val < minimum:
            continue                       # not a big enough swing to burn the chip
        candidates[chip] = this_val
    return max(candidates, key=candidates.get) if candidates else None


# --------------------------- optimisation ---------------------------------
def optimize_xi(squad: pd.DataFrame, conflict_penalty: float = 0.0):
    s = squad.reset_index(drop=True)
    idx = list(s.index)
    pred, pos = s["pred"].to_dict(), s["position"].to_dict()
    prob = pulp.LpProblem("xi", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("start", idx, cat="Binary")
    c = pulp.LpVariable.dicts("capt", idx, cat="Binary")
    objective = (pulp.lpSum(pred[i] * y[i] for i in idx)
                 + pulp.lpSum(pred[i] * c[i] for i in idx))
    objective = objective - conflict_dock(prob, y, s, conflict_penalty)
    prob += objective
    for i in idx:
        prob += c[i] <= y[i]
    prob += pulp.lpSum(y.values()) == 11
    prob += pulp.lpSum(c.values()) == 1
    for p in XI_MIN:
        mem = [i for i in idx if pos[i] == p]
        prob += pulp.lpSum(y[i] for i in mem) >= XI_MIN[p]
        prob += pulp.lpSum(y[i] for i in mem) <= XI_MAX[p]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")
    s["starting"] = [int(round(y[i].value())) for i in idx]
    s["captain"] = [int(round(c[i].value())) for i in idx]
    return pulp.value(prob.objective), s


def selection_explanations(squad):
    """Compare each benched player with the best legal one-player lineup swap."""
    starters = squad[squad["starting"] == 1]
    counts = starters["position"].value_counts().to_dict()
    comparisons = []
    for _, bench in squad[squad["starting"] == 0].iterrows():
        legal = []
        for _, starter in starters.iterrows():
            proposed = counts.copy()
            proposed[starter["position"]] -= 1
            proposed[bench["position"]] = proposed.get(bench["position"], 0) + 1
            if all(XI_MIN[p] <= proposed.get(p, 0) <= XI_MAX[p] for p in XI_MIN):
                legal.append(starter)
        if legal:
            out = min(legal, key=lambda r: r["pred"])
            comparisons.append({"bench": bench["name"], "starter": out["name"],
                                "gap": round(float(out["pred"] - bench["pred"]), 2),
                                "bench_pred": round(float(bench["pred"]), 2),
                                "starter_pred": round(float(out["pred"]), 2)})
    return sorted(comparisons, key=lambda r: r["gap"])


def _prune_pool(pool: pd.DataFrame, keep_pred=50, keep_cheap=20) -> pd.DataFrame:
    """Trim the candidate pool to players that can plausibly make the optimal squad:
    the top scorers plus the cheapest budget-enablers in each position. Optimising
    over this subset yields the same squad as the full pool (verified across every
    gameweek), but the MILP solves several times faster."""
    keep = []
    for ps in SQUAD:
        sub = pool[pool["position"] == ps].dropna(subset=["pred", "price"])
        keep.append(sub.nlargest(keep_pred, "pred"))
        keep.append(sub.nsmallest(keep_cheap, "price"))
    return pd.concat(keep).drop_duplicates("id")


def best_squad_value(pool: pd.DataFrame, budget=1000) -> float:
    """XI value of the optimal squad built from scratch (for FH/WC valuation)."""
    sq = optimize_squad(_prune_pool(pool), budget=budget)
    return sq[sq["starting"] == 1]["pred"].sum() + sq[sq["captain"] == 1]["pred"].sum()


def _blend_pred(preds_all: pd.DataFrame, gw: int) -> dict:
    """{element_id: weighted-average projection over the next FIXTURE_WEIGHTS
    gameweeks from ``gw``}. Blanks contribute zero, doubles sum their fixtures.
    Used so transfer decisions weigh a player's whole upcoming run."""
    frames = []
    for k, g in enumerate(range(gw, gw + len(FIXTURE_WEIGHTS))):
        f = preds_all[preds_all["gw"] == g][["id", "pred"]]
        if len(f):
            frames.append(f.assign(w=FIXTURE_WEIGHTS[k]))
    if not frames:
        return {}
    allf = pd.concat(frames)
    allf["wp"] = allf["pred"] * allf["w"]
    agg = allf.groupby("id").agg(wp=("wp", "sum"), w=("w", "sum"))
    horizon = sum(FIXTURE_WEIGHTS[k] for k in range(len(FIXTURE_WEIGHTS))
                  if gw + k <= 38)
    return (agg["wp"] / horizon).to_dict()


def plan_transfers(current_ids, pool, bank, ft):
    p = pool.dropna(subset=["pred", "price"]).reset_index(drop=True)
    idx = list(p.index)
    cur = set(current_ids)
    owned = {i: (p.loc[i, "id"] in cur) for i in idx}
    pred, price, pos = p["pred"].to_dict(), p.get("budget_price", p["price"]).to_dict(), p["position"].to_dict()
    cur_val = sum(price[i] for i in idx if owned[i])
    prob = pulp.LpProblem("plan", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("sq", idx, cat="Binary")
    y = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    c = pulp.LpVariable.dicts("cap", idx, cat="Binary")
    hits = pulp.LpVariable("hits", lowBound=0)
    objective = (pulp.lpSum(pred[i] * y[i] for i in idx)
                 + pulp.lpSum(pred[i] * c[i] for i in idx) - HIT * hits
                 + BENCH_WEIGHT * pulp.lpSum(pred[i] * (x[i] - y[i]) for i in idx))
    objective = objective - conflict_dock(prob, y, p, CONFLICT_PENALTY)
    prob += objective
    for i in idx:
        prob += y[i] <= x[i]
        prob += c[i] <= y[i]
    prob += pulp.lpSum(c.values()) == 1
    prob += pulp.lpSum(y.values()) == 11
    prob += pulp.lpSum(price[i] * x[i] for i in idx) <= cur_val + bank
    for ps, need in SQUAD.items():
        mem = [i for i in idx if pos[i] == ps]
        prob += pulp.lpSum(x[i] for i in mem) == need
        prob += pulp.lpSum(y[i] for i in mem) >= XI_MIN[ps]
        prob += pulp.lpSum(y[i] for i in mem) <= XI_MAX[ps]
    for club in p["team"].unique():
        mem = [i for i in idx if p.loc[i, "team"] == club]
        prob += pulp.lpSum(x[i] for i in mem) <= MAX_PER_CLUB
    prob += hits >= pulp.lpSum(x[i] for i in idx if not owned[i]) - ft
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")
    p["starting"] = [int(round(y[i].value())) for i in idx]
    p["captain"] = [int(round(c[i].value())) for i in idx]
    new = p[[int(round(x[i].value())) == 1 for i in idx]].copy()
    new_ids = set(new["id"])
    xi_val = new[new["starting"] == 1]["pred"].sum() + new[new["captain"] == 1]["pred"].sum()
    return xi_val, new, cur - new_ids, new_ids - cur, int(round(hits.value()))


def plan_exact(current_ids, pool, bank, k):
    """Best squad making EXACTLY k transfers (no hit penalty). For marginal analysis."""
    p = pool.dropna(subset=["pred", "price"]).reset_index(drop=True)
    idx = list(p.index)
    cur = set(current_ids)
    owned = {i: (p.loc[i, "id"] in cur) for i in idx}
    pred, price, pos = p["pred"].to_dict(), p.get("budget_price", p["price"]).to_dict(), p["position"].to_dict()
    cur_val = sum(price[i] for i in idx if owned[i])
    prob = pulp.LpProblem("plan_exact", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("sq", idx, cat="Binary")
    y = pulp.LpVariable.dicts("xi", idx, cat="Binary")
    c = pulp.LpVariable.dicts("cap", idx, cat="Binary")
    objective = (pulp.lpSum(pred[i] * y[i] for i in idx)
                 + pulp.lpSum(pred[i] * c[i] for i in idx)
                 + BENCH_WEIGHT * pulp.lpSum(pred[i] * (x[i] - y[i]) for i in idx))
    objective = objective - conflict_dock(prob, y, p, CONFLICT_PENALTY)
    prob += objective
    for i in idx:
        prob += y[i] <= x[i]
        prob += c[i] <= y[i]
    prob += pulp.lpSum(c.values()) == 1
    prob += pulp.lpSum(y.values()) == 11
    prob += pulp.lpSum(price[i] * x[i] for i in idx) <= cur_val + bank
    for ps, need in SQUAD.items():
        mem = [i for i in idx if pos[i] == ps]
        prob += pulp.lpSum(x[i] for i in mem) == need
        prob += pulp.lpSum(y[i] for i in mem) >= XI_MIN[ps]
        prob += pulp.lpSum(y[i] for i in mem) <= XI_MAX[ps]
    for club in p["team"].unique():
        mem = [i for i in idx if p.loc[i, "team"] == club]
        prob += pulp.lpSum(x[i] for i in mem) <= MAX_PER_CLUB
    prob += pulp.lpSum(x[i] for i in idx if not owned[i]) == k
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    p["starting"] = [int(round(y[i].value())) for i in idx]
    p["captain"] = [int(round(c[i].value())) for i in idx]
    sel = p[[int(round(x[i].value())) == 1 for i in idx]].copy()
    new_ids = set(sel["id"])
    xi_val = sel[sel["starting"] == 1]["pred"].sum() + sel[sel["captain"] == 1]["pred"].sum()
    return xi_val, sel, cur - new_ids, new_ids - cur


# --------------------------- chip timing ----------------------------------
def chip_timing(squad_ids, preds_all, current_gw, available, deadline_gw, budget=DEFAULT_BUDGET):
    """For each available chip, its value in each gameweek of the remaining window.

    Free Hit  = one-week gain (your XI vs the best achievable that single week).
    Wildcard  = a permanent change, so we pick ONE squad optimised over the whole
                remaining half, then value it week-by-week against your current
                squad — i.e. how the changes play out over the long run.
    """
    future = preds_all[preds_all["gw"].between(current_gw, deadline_gw)].copy()
    if future.empty:
        return {}, []
    # Fill known blank weeks with zero rather than dropping owned players or weeks.
    template = future.sort_values("gw").drop_duplicates("id").set_index("id")
    gws = list(range(current_gw, min(deadline_gw, int(future["gw"].max())) + 1))
    completed = []
    for g in gws:
        week = future[future["gw"] == g].set_index("id").reindex(template.index)
        for col in [c for c in ("name", "position", "team", "price", "budget_price") if c in template]:
            week[col] = week[col].fillna(template[col])
        week["pred"] = week["pred"].fillna(0.0)
        week["gw"] = g
        completed.append(week.reset_index())
    preds_all = pd.concat(completed, ignore_index=True)
    cap_pred, bench_sum, your_xi, opt_xi = {}, {}, {}, {}
    for g in gws:
        sq = preds_all[(preds_all["gw"] == g) & (preds_all["id"].isin(squad_ids))]
        if len(sq) < 11:
            continue
        _, xi = optimize_xi(sq)
        start, bench = xi[xi["starting"] == 1], xi[xi["starting"] == 0]
        cap_pred[g] = start["pred"].max()
        bench_sum[g] = bench["pred"].sum()
        your_xi[g] = start["pred"].sum() + start[start["captain"] == 1]["pred"].sum()
        opt_xi[g] = best_squad_value(preds_all[preds_all["gw"] == g], budget=budget)
    valid = sorted(your_xi)
    # Your team's typical weekly shortfall vs the best achievable XI. Free Hit is
    # only worth it when a SPECIFIC week is unusually bad for you (a blank/double
    # gameweek), i.e. the gap exceeds this baseline — not when you're generally
    # sub-optimal, which is a job for transfers or a Wildcard.
    _fh_gaps = sorted(max(0.0, opt_xi[g] - your_xi[g]) for g in valid)
    baseline_gap = _fh_gaps[len(_fh_gaps) // 2] if len(_fh_gaps) > 1 else 0.0   # median

    # Wildcard: one fixed squad optimised over the whole remaining window,
    # then valued week-by-week (re-picking the XI each week from those 15).
    wc_weekly = {}
    if "wildcard" in available and valid:
        horizon = preds_all[preds_all["gw"].isin(valid)].groupby("id").agg(
            pred=("pred", "sum"), name=("name", "first"), position=("position", "first"),
            team=("team", "first"), price=("price", "last")).reset_index()
        if "budget_price" in template:
            horizon["budget_price"] = horizon["id"].map(template["budget_price"])
        wc_ids = set(optimize_squad(horizon, budget=budget)["id"])
        for h in valid:
            sq = preds_all[(preds_all["gw"] == h) & (preds_all["id"].isin(wc_ids))]
            wc_weekly[h] = (optimize_xi(sq)[0] if len(sq) >= 11
                            else sq.sort_values("pred", ascending=False).head(11)["pred"].sum())

    out = {}
    for chip in available:
        per_gw = []
        for g in valid:
            if chip == "3xc":
                val = cap_pred[g]
            elif chip == "bboost":
                val = bench_sum[g]
            elif chip == "freehit":
                val = max(0.0, (opt_xi[g] - your_xi[g]) - baseline_gap)
            else:  # wildcard: sustained weekly advantage of the fixed WC squad from g on
                rest = [h for h in valid if h >= g]
                val = (sum(max(0.0, wc_weekly[h] - your_xi[h]) for h in rest) / len(rest)
                       if rest else 0.0)
            per_gw.append((g, round(val, 1)))
        if per_gw:
            out[chip] = sorted(per_gw, key=lambda t: -t[1])
    return out, valid


# --------------------------- presentation ---------------------------------
def _xi_rows(df, fxmap=None):
    """Serialise a squad df (with starting/captain flags) into ordered UI rows."""
    start = df[df["starting"] == 1].sort_values("position", key=lambda s: s.map(ORDER))
    bench = df[df["starting"] == 0]
    gk = bench[bench["position"] == "GK"]
    outf = bench[bench["position"] != "GK"].sort_values("pred", ascending=False)
    rows = []

    def add(r, role):
        row = {"position": r["position"], "name": r["name"],
               "team": r.get("team"),
               "price": (None if pd.isna(r.get("price")) else int(r["price"])),
               "pred": round(float(r["pred"]), 2),
               "actual": (None if pd.isna(r.get("actual")) else int(round(r["actual"]))),
               "captain": bool(r["captain"]), "vice": False, "role": role}
        if "id" in r and pd.notna(r["id"]):
            row["id"] = int(r["id"])
        row["signal_basis"] = r.get("signal_basis", "recent")
        for key in ("recent_minutes", "recent_starts", "recent_form", "xgi", "p_play", "fixture_count", "form_weight"):
            val = r.get(key)
            if val is not None and pd.notna(val):
                row[key] = round(float(val), 3)
        if "pred_blend" in df.columns and pd.notna(r.get("pred_blend")):
            row["pred_run"] = round(float(r["pred_blend"]), 2)   # next-few-GW average
        if fxmap is not None:
            row["fix"] = fxmap.get(r.get("team"), [])
        rows.append(row)

    for _, r in start.iterrows():
        add(r, "start")
    for _, r in gk.iterrows():
        add(r, "benchgk")
    for _, r in outf.iterrows():
        add(r, "bench")
    # vice-captain = highest-predicted starter who isn't the captain
    non_cap = [r for r in rows if r["role"] == "start" and not r["captain"]]
    if non_cap:
        max(non_cap, key=lambda r: r["pred"])["vice"] = True
    return rows


def _captain_pool(sq):
    """Captain the highest expected scorer, regardless of position."""
    return sq[sq["starting"] == 1], "pred"


def _captain_id(sq):
    """Id of the recommended captain — the top starting attacker."""
    pk, metric = _captain_pool(sq)
    return int(pk.loc[pk[metric].idxmax(), "id"]) if len(pk) else int(sq.iloc[0]["id"])


def _cap_options(sq):
    """Top-3 captain candidates with projected points and each one's share —
    computed the same way the recommended captain is, so they never disagree."""
    pk, metric = _captain_pool(sq)
    top3 = pk.nlargest(3, metric)
    tot = float(top3[metric].sum()) or 1.0
    return [{"name": r["name"], "team": r.get("team"),
             "pred": round(float(r[metric]), 2),
             "share": round(float(r[metric]) / tot * 100)} for _, r in top3.iterrows()]


def _print_rows(rows, w=24):
    print(f"{'pos':<5}{'player':<{w}}{'pred':>7}{'actual':>8}")
    for r in rows:
        if r["role"] != "start":
            continue
        mark = "  (C)" if r["captain"] else ("  (V)" if r.get("vice") else "")
        act = "" if r["actual"] is None else f"{r['actual']:.0f}"
        print(f"{r['position']:<5}{(r['name']+mark):<{w}}{r['pred']:>7.2f}{act:>8}")
    gk = [r for r in rows if r["role"] == "benchgk"]
    outf = [r for r in rows if r["role"] == "bench"]
    gk_str = f"GK {gk[0]['name']}" if gk else ""
    sub_str = " > ".join(r["name"] for r in outf)
    sep = "  | " if gk_str and sub_str else ""
    print(f"bench: {gk_str}{sep}{sub_str}")


def team_fixtures(season, from_gw, n=4):
    """{team_name: [{opp, home, diff}, ...]} for the next n gameweeks after from_gw."""
    hist = load_history()
    if "gw" not in hist.columns and "GW" in hist.columns:
        hist = hist.rename(columns={"GW": "gw"})
    if "season" in hist.columns:
        hist = hist[hist["season"] == season]
    need = {"team", "gw", "opponent_team", "was_home"}
    if not need.issubset(hist.columns):
        return {}
    meta = teams_meta(season)
    fx = hist.dropna(subset=["opponent_team"]).drop_duplicates(["team", "gw"])
    have_diff = "difficulty" in fx.columns
    out = {}
    for team, grp in fx.groupby("team"):
        ups = grp[grp["gw"] >= from_gw].sort_values("gw").head(n)
        lst = []
        for _, r in ups.iterrows():
            code, strength = meta.get(int(r["opponent_team"]), ("?", 3))
            # Prefer FPL's own fixture difficulty (FDR) — it's set even before the
            # season's 1-5 team strengths are — and fall back to team strength.
            fdr = r.get("difficulty") if have_diff else None
            diff = int(fdr) if pd.notna(fdr) else int(strength)
            lst.append({"opp": code, "home": bool(r["was_home"]), "diff": diff})
        out[team] = lst
    return out


def build_advice(team_id, gw=None, ft_override=None, picks_gw=None):
    """Compute the full weekly recommendation and return it as structured data.

    `gw` may be omitted in live mode — advice then targets the upcoming gameweek
    (the one you can still change), while the current team is read from the
    gameweek in progress. Pass `gw` (as the demo does) to advise on a specific,
    already-played gameweek, reading picks from that same gameweek.
    """
    if gw is None or DATA_SOURCE == "live":
        from src.ingestion.live import upcoming_gameweek, current_gameweek
        target = upcoming_gameweek()
        if gw is not None and gw != target:
            return {"ok": False, "error": "Live advice is only available for the next open gameweek."}
        gw = target                                   # one target for every code path
        if picks_gw is None:
            picks_gw = current_gameweek()              # current team: GW in progress
        if gw is None:
            return {"ok": False, "team_id": team_id, "gw": None, "season": SEASON,
                    "error": "No upcoming gameweek with an open deadline is available."}
    if picks_gw is None:
        picks_gw = gw                                  # demo: picks come from the same GW
    picks = get_picks(team_id, picks_gw)
    history = get_history(team_id)
    if DATA_SOURCE == "live" and picks.get("active_chip") == "freehit" and picks_gw > 1:
        picks = get_picks(team_id, picks_gw-1)  # The temporary FH squad reverts next GW.
    ids = [p["element"] for p in picks["picks"]]
    bank = picks.get("entry_history", {}).get("bank", 0)

    ft = ft_override if ft_override is not None else estimate_free_transfers(history, gw)
    available, deadline_gw, half = chips_available(history, gw)
    used_chips = chips_used(history, gw)

    live_raw = None
    live_bootstrap = {}
    form_gw = None
    projection_source = "season_model"
    projection_reason = None
    try:
        if DATA_SOURCE == "live":
            live_raw = load_history()
            form_gw = live_raw.attrs.get("form_gw", 0)
            preds_all = _predict_from_history(SEASON, live_raw)
        else:
            preds_all = predict_all(SEASON)
        pool = preds_all[preds_all["gw"] == gw].copy()
    except Exception:  # noqa: BLE001 — fallback is disclosed in the UI
        LOGGER.exception("Main projection failed for target GW%s; using a disclosed seed", gw)
        projection_reason = "The main forecast could not be completed. Seeded estimates are shown."
        preds_all = pd.DataFrame(columns=["gw", "id", "pred"])
        pool = pd.DataFrame()
    fxmap = {} if DATA_SOURCE == "live" else team_fixtures(SEASON, gw, 4)
    if DATA_SOURCE == "live":
        # Use the authoritative schedule, never opponents inferred from player
        # history (which may include matches played for a previous club).
        try:
            from src.ingestion.fpl_client import FPLClient
            from src.model.preseason import fixture_ticker
            _c = FPLClient()
            live_bootstrap = _c.get_bootstrap_static()
            live_teams = live_bootstrap["teams"]
            fxmap = fixture_ticker(_c.get_fixtures(), live_teams, gw, 4)
            # Normal live predictions use club codes; seeded predictions use IDs.
            fxmap.update({t["short_name"]: fxmap.get(t["id"], []) for t in live_teams})
        except Exception:  # noqa: BLE001
            fxmap = {}

    if pool.empty or (DATA_SOURCE == "live" and not set(ids).issubset(set(pool["id"]))):
        # The upcoming gameweek isn't in the played history yet, so forward-predict
        # it: prefer THIS season's form (from games played so far), falling back to
        # last season's seed only until a gameweek has actually been played.
        from src.model.preseason import preseason_predictions, current_season_seed
        cs = current_season_seed(history=live_raw)
        projection_source = "current_form_seed" if len(cs) else "previous_season_seed"
        if projection_reason is None:
            projection_reason = "A full model forecast is not available for this gameweek."
        form_gw = cs.attrs.get("form_gw", form_gw) if len(cs) else 0
        seed = preseason_predictions(apply_availability=False,
                                     seed_override=cs if len(cs) else None, gw=gw)
        seeded_horizon = seed.attrs.get("weekly_predictions")
        if seeded_horizon is not None:
            preds_all = pd.DataFrame(seeded_horizon)
        if not len(seed):
            return {"ok": False, "team_id": team_id, "gw": gw, "season": SEASON,
                    "error": "Not enough data yet to advise — try again once the "
                             "gameweek's results are in."}
        pool = seed.rename(columns={"pred": "pred_blend", "pred_gw1": "pred"}).copy()
        pool["gw"] = gw                    # match already present from the seed
    else:
        # Retain the target fixture ID for optional risk preferences. The default
        # expected-points objective has no opposition penalty.
        _hist = live_raw if live_raw is not None else load_history()
        if "gw" not in _hist.columns and "GW" in _hist.columns:
            _hist = _hist.rename(columns={"GW": "gw"})
        if "season" in _hist.columns:
            _hist = _hist[_hist["season"] == SEASON]
        if "fixture" in _hist.columns:
            _fx = _hist[_hist["gw"] == gw]
            pool["match"] = pool["id"].map(dict(zip(_fx["element"], _fx["fixture"])))
        # Blended projection over the next few gameweeks — transfer decisions weigh
        # a player's whole upcoming run, not just this gameweek. (The XI you field
        # is still scored on this gameweek only, via base_xi below.)
        pool["pred_blend"] = pool["id"].map(_blend_pred(preds_all, gw)).fillna(pool["pred"])

    # Live availability: scale BOTH this-gameweek and blended projections; drop
    # unavailable players you don't own (your own stay so they can be replaced).
    if DATA_SOURCE == "live":
        from src.ingestion.live import availability
        live_availability = availability()
        mult = pool["id"].map(live_availability).fillna(1.0)
        if not preds_all.empty:
            preds_all = preds_all.copy()
            factors = preds_all["id"].map(live_availability).fillna(1.0)
            preds_all["pred"] = preds_all["pred"] * factors
            if "p_play" in preds_all:
                preds_all["p_play"] = preds_all["p_play"] * factors
            preds_all = preds_all[(factors > 0) | preds_all["id"].isin(ids)]
        pool["pred"] = pool["pred"] * mult
        if "p_play" in pool:
            pool["p_play"] = pool["p_play"] * mult
        pool["pred_blend"] = pool["pred_blend"] * mult
        pool = pool[(mult > 0) | pool["id"].isin(ids)]

    if DATA_SOURCE == "live" and live_raw is not None and not live_raw.empty:
        missing_ids = set(ids) - set(pool["id"])
        missing = live_raw[live_raw["element"].isin(missing_ids)].drop_duplicates("element")
        if len(missing):
            missing = missing.rename(columns={"element": "id", "value": "price"})
            missing = missing[[c for c in ("id", "name", "position", "team", "price") if c in missing]]
            missing = missing.assign(pred=0.0, pred_blend=0.0, gw=gw, fixture_count=0)
            pool = pd.concat([pool, missing], ignore_index=True)

    squad = pool[pool["id"].isin(ids)].copy()
    if len(squad) != 15 or squad["position"].value_counts().to_dict() != SQUAD:
        return {"ok": False, "team_id": team_id, "gw": gw, "season": SEASON,
                "error": "A complete, valid 15-player squad is required. Some player data is unavailable; please retry."}

    # This gameweek's best XI + captain — what to actually field and captain now.
    _, base_xi = optimize_xi(squad, conflict_penalty=CONFLICT_PENALTY)
    base_xi["captain"] = (base_xi["id"] == _captain_id(base_xi)).astype(int)
    cap = base_xi[base_xi["captain"] == 1].iloc[0]
    nm = pool.set_index("id")["name"].to_dict()
    pos = pool.set_index("id")["position"].to_dict()
    prc = pool.set_index("id")["price"].to_dict()
    sell = prc.copy()
    unknown_prices = []
    if DATA_SOURCE == "live":
        try:
            transfer_history = get_transfers(team_id)
        except Exception:
            transfer_history = None
        boot = live_raw.attrs.get("bootstrap", {}) if live_raw is not None else {}
        sell, unknown_prices = selling_prices(picks, pool, transfer_history, history, boot, gw)
    pool["budget_price"] = pool.apply(lambda r: sell.get(r["id"], r["price"]), axis=1)
    total_money = sum(sell[i] for i in ids) + bank
    if not preds_all.empty:
        preds_all = preds_all.copy()
        preds_all["budget_price"] = preds_all.apply(lambda r: sell.get(r["id"], r["price"]), axis=1)


    # Injury/availability flags for the manager's own players (live only), so the
    # UI can surface them even when the best move this week is elsewhere.
    squad_flags = []
    if DATA_SOURCE == "live":
        from src.ingestion.live import player_flags
        for pid, info in player_flags(set(ids)).items():
            squad_flags.append({"name": nm.get(pid, str(pid)), **info})

    # Transfer decisions weigh the next few gameweeks: evaluate the hold baseline
    # and every transfer option on the blended projection, so the comparison is
    # consistent (does this move improve the run enough to beat the -4?).
    run_weight = sum(w for k, w in enumerate(FIXTURE_WEIGHTS) if gw + k <= 38)
    pool_b = pool.copy()
    # A hit is a one-off cost: compare it with a discounted multiweek points total.
    pool_b["pred"] = pool_b["pred_blend"] * run_weight
    squad_b = pool_b[pool_b["id"].isin(ids)].copy()
    base_val, _ = optimize_xi(squad_b, conflict_penalty=CONFLICT_PENALTY)

    # Post-transfer / chip squads are chosen on the blended run (pool_b), so their
    # `pred` is the run average. For display, re-attach this-gameweek + next-4
    # projections, field the XI on this gameweek, and captain the best attacker —
    # exactly like the current XI, so the two figures read the same way.
    gw1_map = pool.set_index("id")["pred"].to_dict()          # this gameweek
    run_map = pool.set_index("id")["pred_blend"].to_dict()    # next-4 average

    def _field(sq):
        sq = sq.copy()
        sq["pred"] = sq["id"].map(gw1_map).fillna(sq["pred"])
        sq["pred_blend"] = sq["id"].map(run_map).fillna(sq["pred"])
        _, sq = optimize_xi(sq, conflict_penalty=CONFLICT_PENALTY)   # field on this GW
        sq["captain"] = (sq["id"] == _captain_id(sq)).astype(int)
        return sq

    # transfer options (0..ft+2 transfers), net of -4 hits
    options = {0: (base_val, base_xi, set(), set())}
    for k in range(1, ft + 3):
        res = plan_exact(ids, pool_b, bank, k)
        if res:
            options[k] = res

    def net(k):
        return options[k][0] - base_val - HIT * max(0, k - ft)

    best_k = max(options, key=net)
    best_free_k = max((k for k in options if k <= ft), key=lambda k: options[k][0])
    free_gain = options[best_free_k][0] - base_val

    # chip decision (one per week)
    chip_error = None
    try:
        timing, _ = chip_timing(ids, preds_all, gw, available, deadline_gw, total_money) if available else ({}, [])
    except (RuntimeError, KeyError, ValueError):
        LOGGER.exception("Chip comparison unavailable for target GW%s", gw)
        timing = {}
        chip_error = "Chip comparison is unavailable for this forecast; lineup and transfer advice remain available."
    this_vals = {c: dict(v).get(gw) for c, v in timing.items()}
    near_deadline = (deadline_gw - gw) <= 1
    rec_chip = recommend_chip(timing, gw, deadline_gw)
    chip_replaces = rec_chip in ("wildcard", "freehit")
    # Free Hit is one week only, so optimise it on THIS gameweek's projection;
    # Wildcard is permanent, so optimise on the blended next-few-GW run (pool_b).
    rec_squad = None
    if chip_replaces:
        chip_pool = pool if rec_chip == "freehit" else pool_b
        rec_squad = optimize_squad(chip_pool, budget=int(round(total_money)))

    # transfer block
    new_sq = None
    final_df = base_xi                      # the squad the captain/picks come from
    if chip_replaces:
        cur_ids, new_ids = set(ids), set(rec_squad["id"])
        outs, ins = cur_ids - new_ids, new_ids - cur_ids
        moves = []
        for p in ("GK", "DEF", "MID", "FWD"):
            op = [i for i in outs if pos.get(i) == p]
            ip = [i for i in ins if pos.get(i) == p]
            for o, i in zip(op, ip):
                moves.append({"out": nm.get(o, o), "in": nm.get(i, i), "out_id": int(o), "in_id": int(i),
                              "out_price": int(sell.get(o, prc.get(o, 0))), "in_price": int(prc.get(i, 0))})
        transfer = {"verdict": "CHIP", "moves": moves,
                    "text": f"Playing {CHIP_LABEL[rec_chip]} this week — normal transfers don't apply."}
    elif best_k == 0 or net(best_k) <= 1e-6:
        transfer = {"verdict": "HOLD",
                    "text": "No transfer improves the discounted next-4 outlook; keep your free transfer."}
    elif best_k <= ft and free_gain < BANK_THRESHOLD and ft < MAX_FT:
        note = " You have a Wildcard, so no need to hoard for a rebuild." if "wildcard" in available else ""
        transfer = {"verdict": "BANK",
                    "text": f"The best move gains only +{free_gain:.2f} pts (below the "
                            f"~{BANK_THRESHOLD:.0f}-pt bar), so bank the transfer for flexibility.{note}"}
    else:
        _, new_sq, outs, ins = options[best_k]
        moves = [{"out": nm.get(o, o), "in": nm.get(i, i), "out_id": int(o), "in_id": int(i),
                  "out_price": int(sell.get(o, prc.get(o, 0))), "in_price": int(prc.get(i, 0))} for o, i in
                 zip(sorted(outs, key=lambda i: ORDER.get(pos.get(i), 9)),
                     sorted(ins, key=lambda i: ORDER.get(pos.get(i), 9)))]
        final_df = _field(new_sq)
        transfer = {"verdict": "TRANSFER", "moves": moves, "k": best_k,
                    "hits": max(0, best_k - ft), "net": round(net(best_k), 2),
                    "post_xi": _xi_rows(final_df, fxmap),
                    "bank_after": bank + sum(sell[i] for i in outs) - sum(prc[i] for i in ins),
                    "ft_left": max(0, ft - best_k),
                    "text": f"Worth +{net(best_k):.2f} discounted pts over the next {min(4, 39-gw)} GWs (after hits)."}
    if not chip_replaces and (ft + 1) in options:
        extra = options[ft + 1][0] - options[best_free_k][0]
        transfer["hit_worth"] = bool(best_k > ft)
        transfer["hit_note"] = (f"A -4 hit {'IS' if best_k > ft else 'is NOT'} worth it: "
                                f"best extra move adds +{extra:.2f} discounted pts over the next {min(4,39-gw)} GWs.")

    # chip table + detail — list every chip; mark the ones already played
    chip_table = []
    for chip in CHIP_LABEL:
        if chip in used_chips:
            chip_table.append({"chip": chip, "label": CHIP_LABEL[chip],
                               "used": True, "used_gw": used_chips[chip]})
            continue
        ranked = timing.get(chip)
        if not ranked:
            chip_table.append({"chip": chip, "label": CHIP_LABEL[chip], "no_data": True})
            continue
        bg, bval = ranked[0]
        tv = this_vals.get(chip)
        margin = max(1.0, 0.1 * bval)
        near_best = (tv is not None and bval > 0 and tv >= bval - margin
                     and tv >= (0.0 if gw == deadline_gw else CHIP_MIN_VALUE.get(chip, 0.0)))
        wait_gain = None if tv is None else round(max(0.0, bval - tv), 1)
        row = {"chip": chip, "label": CHIP_LABEL[chip], "best_gw": int(bg), "best_val": float(bval),
               "this_val": (None if tv is None else float(tv)),
               "near_best": bool(near_best), "wait_gain": wait_gain}
        if len(ranked) > 1:
            row["second_gw"], row["second_val"] = int(ranked[1][0]), float(ranked[1][1])
        chip_table.append(row)
    rec_detail = None
    if rec_chip in ("wildcard", "freehit"):
        final_df = _field(rec_squad)
        rec_detail = {"type": rec_chip, "squad": _xi_rows(final_df, fxmap),
                      "bank_after": bank + sum(sell[i] for i in (set(ids)-set(rec_squad["id"]))) - sum(prc[i] for i in (set(rec_squad["id"])-set(ids))),
                      "ft_left": ft}
    elif rec_chip == "bboost":
        rec_detail = {"type": "bboost",
                      "bench_pts": round(float(base_xi[base_xi["starting"] == 0]["pred"].sum()), 1)}
    elif rec_chip == "3xc":
        fcap = final_df[final_df["captain"] == 1].iloc[0]
        rec_detail = {"type": "3xc", "player": fcap["name"], "pred": round(float(fcap["pred"]), 2)}

    # summary
    if chip_replaces:
        tr = f"Play {CHIP_LABEL[rec_chip]}"
    elif transfer["verdict"] == "HOLD":
        tr = "Hold (no transfer)"
    elif transfer["verdict"] == "BANK":
        tr = "Bank your free transfer"
    else:
        tr = f"{best_k} transfer(s): " + ", ".join(f"{m['out']}->{m['in']}" for m in transfer["moves"])
    summary_cap = final_df[final_df["captain"] == 1].iloc[0]["name"]

    # top-3 captain candidates — computed the same way as the captain, from the
    # same squad, so the picks, the (C) badge and the summary never disagree.
    captain_options = _cap_options(final_df)

    result = {
        "ok": True, "team_id": int(team_id), "gw": int(gw), "season": SEASON,
        "form_gw": form_gw,
        "projection": {"source": projection_source, "reason": projection_reason,
                       "opponent_ratings": sorted(pool["opponent_rating_source"].dropna().unique().tolist())
                       if "opponent_rating_source" in pool else []},
        "budget_estimated": bool(unknown_prices),
        "selection": selection_explanations(base_xi),
        "ft": int(ft), "ft_estimated": ft_override is None, "bank": int(bank),
        "half": int(half), "half_label": HALF_LABEL[half], "deadline_gw": int(deadline_gw),
        "current_xi": _xi_rows(base_xi, fxmap), "current_captain": cap["name"],
        "captain_options": captain_options, "squad_flags": squad_flags,
        "transfer": transfer,
        "chips": {"available": list(available), "table": chip_table,
                  "recommended": rec_chip, "near_deadline": near_deadline,
                  "detail": rec_detail,
                  "forecast_end": int(preds_all.loc[preds_all["gw"] >= gw, "gw"].max()) if not preds_all.empty else int(gw),
                  "error": chip_error},
        "summary": {"transfers": tr, "captain": summary_cap,
                    "chip": CHIP_LABEL[rec_chip] if rec_chip else "none — hold",
                    "triple": rec_chip == "3xc"},
    }
    if DATA_SOURCE == "live":
        from src.model.monitor import create_snapshot, save_snapshot
        try:
            deadline = next(e["deadline_time"] for e in live_bootstrap.get("events", []) if e["id"] == gw)
            capture = pd.concat([preds_all[preds_all["gw"] != gw], pool], ignore_index=True)
            snapshot = create_snapshot(result, capture, deadline, model_path=MODEL_PATH)
            result["tracking_snapshot"] = snapshot
            if os.getenv("FPL_FORECAST_LOG_DIR"):
                save_snapshot(snapshot, os.environ["FPL_FORECAST_LOG_DIR"])
        except (StopIteration, OSError, ValueError, KeyError):
            LOGGER.exception("Forecast snapshot export unavailable for GW%s", gw)
            result["tracking_error"] = "A pre-deadline forecast snapshot could not be saved for this run."
    return result



def advise(team_id, gw, ft_override=None):
    a = build_advice(team_id, gw, ft_override)
    if not a["ok"]:
        print(a["error"])
        return
    print(f"\n=== Advice for team {a['team_id']}, GW{a['gw']} ({a['season']}) ===")
    print(f"Free transfers: {a['ft']}{' (estimated)' if a['ft_estimated'] else ' (override)'}"
          f"   |   bank £{a['bank']/10:.1f}m   |   {a['half_label']} (chips expire GW{a['deadline_gw']})")

    print("\n-- Best XI from your current squad --")
    _print_rows(a["current_xi"])
    print(f"Captain: {a['current_captain']}")

    t = a["transfer"]
    print("\n-- Transfer plan --")
    if t["verdict"] == "CHIP":
        print(f"  {t['text']} (new squad shown under Chips).")
    elif t["verdict"] in ("HOLD", "BANK"):
        print(f"  Verdict: {t['verdict']} — {t['text']}")
    else:
        print(f"  Verdict: TRANSFER — {t['text']}")
        for m in t["moves"]:
            print(f"  OUT {m['out']:<22} ->  IN {m['in']}")
        print(f"  ({t['k']} transfer(s), {t['hits']} hit(s), -{HIT*t['hits']} pts)")
        print("  Your team after the move(s):")
        _print_rows(t["post_xi"], w=22)
    if "hit_note" in t:
        print(f"  ({t['hit_note']})")

    c = a["chips"]
    print(f"\n-- Chips ({a['half_label']}; expire GW{a['deadline_gw']}; one chip per GW) --")
    if not c["table"]:
        print("  No chips this half.")
    else:
        for row in c["table"]:
            if row.get("used"):
                print(f"  {row['label']}: already played in GW{row['used_gw']} this half.")
                continue
            if row.get("no_data"):
                print(f"  {row['label']}: not enough data in the window.")
                continue
            opts = f"best GW{row['best_gw']} (+{row['best_val']:.1f})"
            if "second_gw" in row:
                opts += f", then GW{row['second_gw']} (+{row['second_val']:.1f})"
            tv = row["this_val"]
            print(f"  {row['label']}: {opts}  |  this GW {('+%.1f' % tv) if tv is not None else 'n/a'}")
        rec = c["recommended"]
        if rec:
            tv = next(r["this_val"] for r in c["table"] if r["chip"] == rec)
            why = " (deadline — use or lose)" if c["near_deadline"] else ""
            print(f"  -> Play THIS week: {CHIP_LABEL[rec]} (+{tv:.1f} pts){why}")
            d = c["detail"]
            if d["type"] in ("wildcard", "freehit"):
                label = "Wildcard (keep going forward)" if d["type"] == "wildcard" else "Free Hit (one week only)"
                print(f"     New squad — {label}:")
                _print_rows(d["squad"], w=22)
            elif d["type"] == "bboost":
                print(f"     Bench Boost: your bench adds a projected {d['bench_pts']:.1f} pts this week.")
            elif d["type"] == "3xc":
                print(f"     Triple Captain on {d['player']} (pred {d['pred']:.2f} -> x3).")
        else:
            tail = f" (play them before the GW{a['deadline_gw']} deadline)" if a["half"] == 1 else ""
            print(f"  -> Hold — every remaining chip's best week is still ahead{tail}.")

    s = a["summary"]
    print("\n-- RECOMMENDED ACTIONS THIS WEEK --")
    print(f"  Transfers: {s['transfers']}")
    print(f"  Captain:   {s['captain']}" + ("  (Triple Captain!)" if s["triple"] else ""))
    if a.get("captain_options"):
        print("  Captain picks: " + ", ".join(
            f"{o['name']} ({o['share']}%)" for o in a["captain_options"]))
    print(f"  Chip:      {s['chip']}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    ft_override = None
    if "--ft" in argv:
        i = argv.index("--ft")
        ft_override = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print("Usage: python -m src.advisor.personal <team_id> [gw] [--ft N]")
        sys.exit(1)
    team_id = int(argv[0])
    gw = int(argv[1]) if len(argv) > 1 else 29
    advise(team_id, gw, ft_override)
