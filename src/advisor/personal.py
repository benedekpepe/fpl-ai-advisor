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
    DATA_SOURCE, BENCH_WEIGHT, FIXTURE_WEIGHTS,
    SQUAD, XI_MIN, XI_MAX, MAX_PER_CLUB, HIT, MAX_FT, BANK_THRESHOLD,
    POSITION_ORDER as ORDER, CHIP_LABEL, HALF_DEADLINE, HALF_LABEL,
)


# --------------------------- predictions ----------------------------------
@lru_cache(maxsize=4)
def predict_all(season: str) -> pd.DataFrame:
    """Predicted points for every player in every gameweek of the season.

    Cached per season: the feature build + model inference is the heaviest step
    (tens of seconds over several seasons of history), so it runs once per
    session rather than on every request. Call ``predict_all.cache_clear()``
    after re-ingesting fresh data.
    """
    bundle = joblib.load(MODEL_PATH)
    clf, reg, cols = bundle["clf"], bundle["reg"], bundle["cols"]
    raw = load_history()
    feats = build_feature_frame(attach_opponent_strength(raw))
    feats = feats[feats["season"] == season].copy()
    if feats.empty:                       # early season: no current-season rows yet
        return pd.DataFrame(columns=["gw", "id", "name", "position", "team",
                                     "price", "pred", "actual"])
    # Games actually played this season (a scalar). Upcoming (unplayed) rows carry
    # NaN stats; a plain cumcount counts them and inflates games_played for later
    # future weeks, which would make the early-season adjustments fade across the
    # horizon and wrongly inflate the Free-Hit gap in the nearest week. So the
    # form-related adjustments key off this scalar instead — uniform across weeks.
    try:
        rs = raw[raw["season"] == season] if "season" in raw.columns else raw
        n_played = int(rs.loc[rs["minutes"].notna(), "gw"].nunique())
    except Exception:  # noqa: BLE001
        n_played = int(feats["games_played"].min()) if len(feats) else 0
    n_played = max(n_played, 0)

    X = feats[cols].copy()
    if "games_played" in X.columns:        # don't let unplayed future rows inflate it
        X["games_played"] = X["games_played"].clip(upper=max(n_played, 1))
    # Early-season value dampening. With little form the model leans on price so
    # hard a pricey flop outranks an in-form cheap player. Shrink value toward the
    # per-position median until ~4 games are in, then use the real price (what the
    # model was validated on — full-season data is unaffected).
    if "value" in X.columns:
        shrink = min(n_played / 4.0, 1.0)
        med = feats.groupby("position")["value"].transform("median").fillna(
            feats["value"].median())
        X["value"] = (med + (feats["value"] - med) * shrink).to_numpy()
    feats["pred"] = clf.predict_proba(X)[:, 1] * reg.predict(X)
    feats["pred_model"] = feats["pred"]        # unblended (reliable) — for captaincy
    # Early-season recency blend (fades out by ~GW6). The model regresses a single
    # game hard toward the mean, so a big GW1 barely moves its projection; blend in
    # actual points-per-game so recent output — not just price — ranks players
    # while the sample is tiny.
    if "cum_ppg" in feats.columns and n_played >= 1:
        fw = max(0.0, min(1.0, 0.25 * (1 - (n_played - 1) / 5.0)))
        feats["pred"] = (1 - fw) * feats["pred"] + fw * feats["cum_ppg"].fillna(0)
    return feats.groupby(["gw", "element"]).agg(
        name=("name", "first"), position=("position", "first"),
        team=("team", "first"), price=("value", "first"),
        pred=("pred", "sum"), pred_model=("pred_model", "sum"),
        xgi=("roll_expected_goal_involvements_5", "first"),
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


def get_history(team_id):
    return _get(f"entry/{team_id}/history/")


# --------------------------- context from history -------------------------
def estimate_free_transfers(history: dict, target_gw: int) -> int:
    """Reconstruct available free transfers entering target_gw (an estimate).

    GW1 is unlimited (a fresh squad), so nothing rolls over from it — entering
    GW2 you have exactly 1. Rollovers only start accruing from GW2 onward.
    """
    made = {c["event"]: c.get("event_transfers", 0) for c in history.get("current", [])}
    chip = {c["event"]: c["name"] for c in history.get("chips", [])}
    if target_gw <= 2:
        return 1                                  # GW1 unlimited → GW2 has 1 FT
    ft = 1                                         # entering GW2
    for g in range(2, target_gw):                 # accrue rollovers from GW2 on
        c = chip.get(g)
        if c == "wildcard":
            ft = 1
        elif c == "freehit":
            pass  # free hit ignores transfers and the FT count
        else:
            ft = max(0, ft - made.get(g, 0))
        ft = min(MAX_FT, ft + 1)
        if g + 1 == 16:           # AFCON: everyone gets 5 in GW16
            ft = 5
    return ft


def chips_available(history: dict, target_gw: int):
    """Which chips remain in the current half, and when they expire."""
    half = 1 if target_gw <= 19 else 2
    lo, hi = (1, 19) if half == 1 else (20, 38)
    # only chips already used up to (and including) the current GW count as gone
    used = {c["name"] for c in history.get("chips", [])
            if lo <= c["event"] <= min(hi, target_gw)}
    available = [c for c in CHIP_LABEL if c not in used]
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


def recommend_chip(timing: dict, gw: int):
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
        if this_val < CHIP_MIN_VALUE.get(chip, 0.0):
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
    s["starting"] = [int(round(y[i].value())) for i in idx]
    s["captain"] = [int(round(c[i].value())) for i in idx]
    return pulp.value(prob.objective), s


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
    gameweeks from ``gw``}. Used so transfer decisions weigh a player's whole
    upcoming run, not just the immediate gameweek. Players are averaged only over
    the gameweeks they actually have a fixture for."""
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
    return (agg["wp"] / agg["w"]).to_dict()


def plan_transfers(current_ids, pool, bank, ft):
    p = pool.dropna(subset=["pred", "price"]).reset_index(drop=True)
    idx = list(p.index)
    cur = set(current_ids)
    owned = {i: (p.loc[i, "id"] in cur) for i in idx}
    pred, price, pos = p["pred"].to_dict(), p["price"].to_dict(), p["position"].to_dict()
    cur_val = p[p["id"].isin(cur)]["price"].sum()
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
    pred, price, pos = p["pred"].to_dict(), p["price"].to_dict(), p["position"].to_dict()
    cur_val = p[p["id"].isin(cur)]["price"].sum()
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
def chip_timing(squad_ids, preds_all, current_gw, available, deadline_gw):
    """For each available chip, its value in each gameweek of the remaining window.

    Free Hit  = one-week gain (your XI vs the best achievable that single week).
    Wildcard  = a permanent change, so we pick ONE squad optimised over the whole
                remaining half, then value it week-by-week against your current
                squad — i.e. how the changes play out over the long run.
    """
    gws = sorted(g for g in preds_all["gw"].unique() if current_gw <= g <= deadline_gw)
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
        opt_xi[g] = best_squad_value(preds_all[preds_all["gw"] == g])
    valid = sorted(your_xi)
    # Your team's typical weekly shortfall vs the best achievable XI. Free Hit is
    # only worth it when a SPECIFIC week is unusually bad for you (a blank/double
    # gameweek), i.e. the gap exceeds this baseline — not when you're generally
    # sub-optimal, which is a job for transfers or a Wildcard.
    _fh_gaps = sorted(max(0.0, opt_xi[g] - your_xi[g]) for g in valid)
    baseline_gap = _fh_gaps[len(_fh_gaps) // 2] if _fh_gaps else 0.0   # median

    # Wildcard: one fixed squad optimised over the whole remaining window,
    # then valued week-by-week (re-picking the XI each week from those 15).
    wc_weekly = {}
    if "wildcard" in available and valid:
        horizon = preds_all[preds_all["gw"].isin(valid)].groupby("id").agg(
            pred=("pred", "sum"), name=("name", "first"), position=("position", "first"),
            team=("team", "first"), price=("price", "last")).reset_index()
        wc_ids = set(optimize_squad(horizon)["id"])
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
    if gw is None:
        from src.ingestion.live import upcoming_gameweek, current_gameweek
        gw = upcoming_gameweek()                       # target: next changeable GW
        if picks_gw is None:
            picks_gw = current_gameweek()              # current team: GW in progress
        if gw is None:
            return {"ok": False, "team_id": team_id, "gw": None, "season": SEASON,
                    "error": "Could not detect the current gameweek."}
    if picks_gw is None:
        picks_gw = gw                                  # demo: picks come from the same GW
    picks = get_picks(team_id, picks_gw)
    ids = [p["element"] for p in picks["picks"]]
    bank = picks.get("entry_history", {}).get("bank", 0)
    history = get_history(team_id)

    ft = ft_override if ft_override is not None else estimate_free_transfers(history, gw)
    available, deadline_gw, half = chips_available(history, gw)
    used_chips = chips_used(history, gw)

    try:
        preds_all = predict_all(SEASON)
        pool = preds_all[preds_all["gw"] == gw].copy()
    except Exception:  # noqa: BLE001 — thin/odd early-season data → use the seed
        preds_all = pd.DataFrame(columns=["gw", "id", "pred"])
        pool = pd.DataFrame()
    fxmap = team_fixtures(SEASON, gw, 4)
    if DATA_SOURCE == "live" and not fxmap:
        # The played history has no future rows, so pull upcoming fixtures from the
        # live endpoint (same ticker the builder uses), keyed by team id.
        try:
            from src.ingestion.fpl_client import FPLClient
            from src.model.preseason import fixture_ticker
            _c = FPLClient()
            fxmap = fixture_ticker(_c.get_fixtures(),
                                   _c.get_bootstrap_static()["teams"], gw, 4)
        except Exception:  # noqa: BLE001
            fxmap = {}

    if pool.empty:
        # The upcoming gameweek isn't in the played history yet, so forward-predict
        # it: prefer THIS season's form (from games played so far), falling back to
        # last season's seed only until a gameweek has actually been played.
        from src.model.preseason import preseason_predictions, current_season_seed
        cs = current_season_seed()
        seed = preseason_predictions(apply_availability=False,
                                     seed_override=cs if len(cs) else None)
        if not len(seed):
            return {"ok": False, "team_id": team_id, "gw": gw, "season": SEASON,
                    "error": "Not enough data yet to advise — try again once the "
                             "gameweek's results are in."}
        pool = seed.rename(columns={"pred": "pred_blend", "pred_gw1": "pred"}).copy()
        pool["gw"] = gw                    # match already present from the seed
    else:
        # Attach each player's fixture id for this gameweek so the XI *and* transfer
        # optimisers are stack-aware (won't stack opposing players from one match).
        _hist = load_history()
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
        mult = pool["id"].map(availability()).fillna(1.0)
        pool["pred"] = pool["pred"] * mult
        pool["pred_blend"] = pool["pred_blend"] * mult
        pool = pool[(mult > 0) | pool["id"].isin(ids)]

    squad = pool[pool["id"].isin(ids)].copy()
    if len(squad) < 11:
        return {"ok": False, "team_id": team_id, "gw": gw, "season": SEASON,
                "error": "Not enough player data to advise this gameweek."}

    # This gameweek's best XI + captain — what to actually field and captain now.
    _, base_xi = optimize_xi(squad, conflict_penalty=CONFLICT_PENALTY)
    # Captain the best attacker (MID/FWD) on the reliable (unblended) projection.
    # Captaincy is about ceiling — which keepers and defenders rarely have — and a
    # single big gameweek can inflate anyone's form, so the armband goes to the top
    # projected attacker, not a one-game fluke at the back.
    if "pred_model" in base_xi.columns:
        xi_start = base_xi[base_xi["starting"] == 1]
        att = xi_start[xi_start["position"].isin(["MID", "FWD"])]
        pick = att if len(att) and att["pred_model"].notna().any() else xi_start
        if len(pick) and pick["pred_model"].notna().any():
            cap_id = pick.loc[pick["pred_model"].idxmax(), "id"]
            base_xi["captain"] = (base_xi["id"] == cap_id).astype(int)
    cap = base_xi[base_xi["captain"] == 1].iloc[0]
    nm = pool.set_index("id")["name"].to_dict()
    pos = pool.set_index("id")["position"].to_dict()
    prc = pool.set_index("id")["price"].to_dict()
    total_money = float(squad["price"].sum()) + bank   # squad value + bank, in tenths

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
    pool_b = pool.copy()
    pool_b["pred"] = pool_b["pred_blend"]
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
        if "pred_model" in sq.columns:
            xs = sq[sq["starting"] == 1]
            att = xs[xs["position"].isin(["MID", "FWD"])]
            pk = att if len(att) and att["pred_model"].notna().any() else xs
            if len(pk) and pk["pred_model"].notna().any():
                cid = pk.loc[pk["pred_model"].idxmax(), "id"]
                sq["captain"] = (sq["id"] == cid).astype(int)
        return _xi_rows(sq, fxmap)

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
    timing, _ = chip_timing(ids, preds_all, gw, available, deadline_gw) if available else ({}, [])
    this_vals = {c: dict(v).get(gw) for c, v in timing.items()}
    near_deadline = (deadline_gw - gw) <= 1
    rec_chip = recommend_chip(timing, gw)
    chip_replaces = rec_chip in ("wildcard", "freehit")
    # Free Hit is one week only, so optimise it on THIS gameweek's projection;
    # Wildcard is permanent, so optimise on the blended next-few-GW run (pool_b).
    rec_squad = None
    if chip_replaces:
        chip_pool = pool if rec_chip == "freehit" else pool_b
        rec_squad = optimize_squad(chip_pool, budget=int(round(total_money)))

    # transfer block
    new_sq = None
    if chip_replaces:
        cur_ids, new_ids = set(ids), set(rec_squad["id"])
        outs, ins = cur_ids - new_ids, new_ids - cur_ids
        moves = []
        for p in ("GK", "DEF", "MID", "FWD"):
            op = [i for i in outs if pos.get(i) == p]
            ip = [i for i in ins if pos.get(i) == p]
            for o, i in zip(op, ip):
                moves.append({"out": nm.get(o, o), "in": nm.get(i, i),
                              "out_price": int(prc.get(o, 0)), "in_price": int(prc.get(i, 0))})
        transfer = {"verdict": "CHIP", "moves": moves,
                    "text": f"Playing {CHIP_LABEL[rec_chip]} this week — normal transfers don't apply."}
    elif best_k == 0 or net(best_k) <= 1e-6:
        transfer = {"verdict": "HOLD",
                    "text": "No transfer improves your team this week; keep your free transfer."}
    elif best_k <= ft and free_gain < BANK_THRESHOLD and ft < MAX_FT:
        note = " You have a Wildcard, so no need to hoard for a rebuild." if "wildcard" in available else ""
        transfer = {"verdict": "BANK",
                    "text": f"The best move gains only +{free_gain:.2f} pts (below the "
                            f"~{BANK_THRESHOLD:.0f}-pt bar), so bank the transfer for flexibility.{note}"}
    else:
        _, new_sq, outs, ins = options[best_k]
        moves = [{"out": nm.get(o, o), "in": nm.get(i, i),
                  "out_price": int(prc.get(o, 0)), "in_price": int(prc.get(i, 0))} for o, i in
                 zip(sorted(outs, key=lambda i: ORDER.get(pos.get(i), 9)),
                     sorted(ins, key=lambda i: ORDER.get(pos.get(i), 9)))]
        transfer = {"verdict": "TRANSFER", "moves": moves, "k": best_k,
                    "hits": max(0, best_k - ft), "net": round(net(best_k), 2),
                    "post_xi": _field(new_sq),
                    "bank_after": total_money - float(new_sq["price"].sum()),
                    "ft_left": max(0, ft - best_k),
                    "text": f"Worth +{net(best_k):.2f} pts this week."}
    if not chip_replaces and (ft + 1) in options:
        extra = options[ft + 1][0] - options[best_free_k][0]
        transfer["hit_worth"] = bool(best_k > ft)
        transfer["hit_note"] = (f"A -4 hit {'IS' if best_k > ft else 'is NOT'} worth it: "
                                f"best extra move adds +{extra:.2f}.")

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
                     and tv >= CHIP_MIN_VALUE.get(chip, 0.0))
        wait_gain = None if tv is None else round(max(0.0, bval - tv), 1)
        row = {"chip": chip, "label": CHIP_LABEL[chip], "best_gw": int(bg), "best_val": float(bval),
               "this_val": (None if tv is None else float(tv)),
               "near_best": bool(near_best), "wait_gain": wait_gain}
        if len(ranked) > 1:
            row["second_gw"], row["second_val"] = int(ranked[1][0]), float(ranked[1][1])
        chip_table.append(row)
    rec_detail = None
    if rec_chip in ("wildcard", "freehit"):
        rec_detail = {"type": rec_chip, "squad": _field(rec_squad),
                      "bank_after": total_money - float(rec_squad["price"].sum()),
                      "ft_left": ft}
    elif rec_chip == "bboost":
        rec_detail = {"type": "bboost",
                      "bench_pts": round(float(base_xi[base_xi["starting"] == 0]["pred"].sum()), 1)}
    elif rec_chip == "3xc":
        rec_detail = {"type": "3xc", "player": cap["name"], "pred": round(float(cap["pred"]), 2)}

    # summary
    if chip_replaces:
        tr = f"Play {CHIP_LABEL[rec_chip]}"
        summary_cap = rec_squad[rec_squad["captain"] == 1].iloc[0]["name"]
    elif transfer["verdict"] == "HOLD":
        tr, summary_cap = "Hold (no transfer)", cap["name"]
    elif transfer["verdict"] == "BANK":
        tr, summary_cap = "Bank your free transfer", cap["name"]
    else:
        tr = f"{best_k} transfer(s): " + ", ".join(f"{m['out']}->{m['in']}" for m in transfer["moves"])
        summary_cap = new_sq[new_sq["captain"] == 1].iloc[0]["name"]

    # top-3 captain candidates from the squad the captain is actually chosen from
    final = rec_squad if chip_replaces else (new_sq if new_sq is not None else base_xi)
    fstart = final[final["starting"] == 1]
    top3 = fstart.nlargest(3, "pred")
    tot = float(top3["pred"].sum()) or 1.0
    captain_options = [{"name": rr["name"], "team": rr.get("team"),
                        "pred": round(float(rr["pred"]), 2),
                        "share": round(float(rr["pred"]) / tot * 100)}
                       for _, rr in top3.iterrows()]

    return {
        "ok": True, "team_id": int(team_id), "gw": int(gw), "season": SEASON,
        "ft": int(ft), "ft_estimated": ft_override is None, "bank": int(bank),
        "half": int(half), "half_label": HALF_LABEL[half], "deadline_gw": int(deadline_gw),
        "current_xi": _xi_rows(base_xi, fxmap), "current_captain": cap["name"],
        "captain_options": captain_options, "squad_flags": squad_flags,
        "transfer": transfer,
        "chips": {"available": list(available), "table": chip_table,
                  "recommended": rec_chip, "near_deadline": near_deadline,
                  "detail": rec_detail},
        "summary": {"transfers": tr, "captain": summary_cap,
                    "chip": CHIP_LABEL[rec_chip] if rec_chip else "none — hold",
                    "triple": rec_chip == "3xc"},
    }


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
