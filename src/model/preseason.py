"""Pre-season / early-season cold-start: recommend a squad before any gameweek
has been played.

The in-season model predicts from *this* season's rolling form. Before GW1 that
form doesn't exist yet, so this module seeds each player's form from their
**previous season** (per-game averages), joined to this season's players by the
permanent FPL player ``code`` (ids change between seasons). Players with no
previous-season data (new signings, promoted clubs, youth) get a position + price
based estimate. Everything else — this season's prices, positions, clubs and the
GW1 fixtures — comes from the live FPL API, and opponent strength from the live
bootstrap (not the lagging historical CSVs).

The seeded features feed the *existing* trained model unchanged, so no retrain is
needed. Once real gameweeks are played the normal in-season path takes over.

Run it:  python -m src.model.preseason
"""
import io
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import requests

from src.config import (MERGED_GW_URL, PLAYERS_RAW_URL, SEED_SEASON, MODEL_PATH,
                        CONFLICT_PENALTY, FIXTURE_WEIGHTS)
from src.ingestion.fpl_client import FPLClient
from src.ingestion.live import availability
from src.optim.optimizer import optimize_squad

POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
STATUS_LABEL = {"a": "available", "d": "doubtful", "i": "injured",
                "s": "suspended", "u": "unavailable", "n": "not in squad"}

# Per-game form stats the model's rolling features are built from.
BASES = ["total_points", "minutes", "expected_goals", "expected_assists",
         "expected_goal_involvements", "bps", "ict_index"]
SEED_COLS = BASES + ["starts_rate_5", "cum_ppg"]

# Cold-start projections blend the next few gameweeks so upcoming fixture
# difficulty matters, not just GW1. Weights decay so the immediate gameweek
# dominates but the run still counts (an easy GW1 with a brutal follow-up is
# rated below a steady run). The length also sets how many gameweeks are blended.


def _full_name(el: dict) -> str:
    """Full display name so the pitch tooltip shows more than the short name."""
    full = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
    return full or el.get("web_name") or ""


def _csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(
        requests.get(url, timeout=60).content.decode("utf-8", "replace")))


@lru_cache(maxsize=2)
def seed_from_previous_season(season: str = SEED_SEASON) -> pd.DataFrame:
    """Per-player, per-game form averages from a completed season, keyed by the
    permanent player ``code`` (stable across seasons), plus ``web_name`` as a
    secondary key for the rare case where a code doesn't carry over."""
    pr = _csv(PLAYERS_RAW_URL.format(season=season))
    id_to_code = dict(zip(pr["id"], pr["code"]))
    code_to_web = dict(zip(pr["code"], pr["web_name"]))
    mg = _csv(MERGED_GW_URL.format(season=season))
    mg["code"] = mg["element"].map(id_to_code)
    agg = {b: (b, "mean") for b in BASES}
    agg["starts_rate_5"] = ("starts", "mean")
    agg["cum_ppg"] = ("total_points", "mean")
    seed = mg.groupby("code").agg(**agg).reset_index()
    seed["web_name"] = seed["code"].map(code_to_web)
    return seed


def _apply_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fill seed columns for players with no previous-season data, from a simple
    per-position price->form fit (price is the FPL-set proxy for expected level)."""
    have = df[SEED_COLS[0]].notna()
    for pos in POSITION.values():
        train = df[(df["position"] == pos) & have]
        target = (df["position"] == pos) & (~have)
        if not target.any():
            continue
        if len(train) >= 2:
            for c in SEED_COLS:
                a, b = np.polyfit(train["value"], train[c], 1)
                df.loc[target, c] = np.clip(a * df.loc[target, "value"] + b, 0, None)
        else:  # not enough to fit -> position mean, else 0
            for c in SEED_COLS:
                df.loc[target, c] = max(train[c].mean() if len(train) else 0.0, 0.0)
    return df


def _next_gameweek(events: list) -> int | None:
    for e in events:
        if e.get("is_current"):
            return e["id"]
    for e in events:
        if e.get("is_next"):
            return e["id"]
    return None


def _fixture_opponents(fixtures: list, gw: int) -> dict:
    """team_id -> (opponent_team_id, was_home) for the given gameweek."""
    out = {}
    for f in fixtures:
        if f.get("event") != gw:
            continue
        out[f["team_h"]] = (f["team_a"], True)
        out[f["team_a"]] = (f["team_h"], False)
    return out


def preseason_frame(bootstrap: dict, fixtures: list,
                    seed_season: str = SEED_SEASON, gw: int = None) -> pd.DataFrame:
    """Build the model's 25-feature frame for a gameweek's players (default: the
    next unplayed one)."""
    if gw is None:
        gw = _next_gameweek(bootstrap["events"])
    opp = _fixture_opponents(fixtures, gw)

    strength = {t["id"]: t for t in bootstrap["teams"]}
    rows = []
    for e in bootstrap["elements"]:
        o = opp.get(e["team"])
        if o is None:            # team not playing this GW (blank) -> skip
            continue
        rows.append({
            "id": e["id"], "code": e["code"], "name": _full_name(e),
            "web_name": e.get("web_name"),
            "position": POSITION.get(e["element_type"]), "team": e["team"],
            "value": e["now_cost"], "opponent_team": o[0], "was_home": o[1],
        })
    seed = seed_from_previous_season(seed_season)
    df = pd.DataFrame(rows).merge(seed.drop(columns=["web_name"]),
                                  on="code", how="left")
    # Recover players whose permanent code didn't carry over (rare, but it drops a
    # real player onto the price-only fallback) via a secondary web_name match.
    # Wrapped defensively: on any issue it silently keeps the code-only result.
    try:
        miss = df["cum_ppg"].isna()
        if miss.any():
            by_name = (seed.dropna(subset=["web_name"])
                           .drop_duplicates("web_name", keep="first")
                           .set_index("web_name"))
            for col in SEED_COLS:
                if col in by_name.columns:
                    df.loc[miss, col] = (
                        df.loc[miss, "web_name"].map(by_name[col]).to_numpy())
    except Exception:  # noqa: BLE001 — never let recovery break the build
        pass
    df = _apply_fallback(df)

    for b in BASES:
        df[f"roll_{b}_3"] = df[b]
        df[f"roll_{b}_5"] = df[b]
    df["games_played"] = 0
    df["is_home"] = df["was_home"].astype(float)
    lo = df[["team", "opponent_team"]].min(axis=1).astype(int).astype(str)
    hi = df[["team", "opponent_team"]].max(axis=1).astype(int).astype(str)
    df["match"] = lo + "-" + hi
    for p in POSITION.values():
        df[f"pos_{p}"] = (df["position"] == p).astype(float)

    opp_str = df["opponent_team"].map(strength)
    home = df["was_home"].to_numpy()
    df["opp_defence"] = [s["strength_defence_away"] if h else s["strength_defence_home"]
                         for s, h in zip(opp_str, home)]
    df["opp_attack"] = [s["strength_attack_away"] if h else s["strength_attack_home"]
                        for s, h in zip(opp_str, home)]
    return df


def preseason_predictions(client: FPLClient = None,
                          seed_season: str = SEED_SEASON,
                          apply_availability: bool = True) -> pd.DataFrame:
    """Projected points for every player for the upcoming (unplayed) gameweek.

    With ``apply_availability`` (default), each projection is scaled by the live
    availability multiplier, so injured/suspended players drop out of selection.
    """
    client = client or FPLClient()
    bootstrap = client.get_bootstrap_static()
    fixtures = client.get_fixtures()

    bundle = joblib.load(MODEL_PATH)
    clf, reg, cols = bundle["clf"], bundle["reg"], bundle["cols"]

    def _predict(frame):
        return clf.predict_proba(frame[cols])[:, 1] * reg.predict(frame[cols])

    # Blend the next few gameweeks so upcoming fixture difficulty counts, not just
    # GW1. Each gameweek is scored against its own opponent; a weighted average
    # (GW1 heaviest, see FIXTURE_WEIGHTS) becomes the projection. Players are
    # averaged only over the gameweeks they actually play (blanks are skipped).
    gw0 = _next_gameweek(bootstrap["events"])
    wsum, wtot = {}, {}
    base, gw1_pred = None, {}
    for k, gw in enumerate(range(gw0, gw0 + len(FIXTURE_WEIGHTS))):
        frame = preseason_frame(bootstrap, fixtures, seed_season, gw=gw)
        if not len(frame):
            continue
        w = FIXTURE_WEIGHTS[k]
        p = _predict(frame)
        for pid, pv in zip(frame["id"].to_numpy(), p):
            wsum[pid] = wsum.get(pid, 0.0) + w * pv
            wtot[pid] = wtot.get(pid, 0.0) + w
        if gw == gw0:
            base = frame                       # GW1 frame carries id/price/match
            gw1_pred = dict(zip(frame["id"].to_numpy(), p))   # this-gameweek only
    if base is None:
        return pd.DataFrame(columns=["id", "name", "position", "team", "match",
                                     "price", "pred", "pred_gw1"])
    df = base.copy()
    df["pred"] = [wsum[i] / wtot[i] for i in df["id"].to_numpy()]      # blended run
    df["pred_gw1"] = df["id"].map(gw1_pred).fillna(df["pred"])          # this GW only

    if apply_availability:
        av = availability(client)
        mult = df["id"].map(av).fillna(1.0)
        df = df[mult > 0].copy()          # drop unavailable (injured/suspended/loaned)
        m = df["id"].map(av).fillna(1.0)
        df["pred"] = df["pred"] * m
        df["pred_gw1"] = df["pred_gw1"] * m
    return df[["id", "name", "position", "team", "match", "value",
               "pred", "pred_gw1"]].rename(columns={"value": "price"})


def fixture_ticker(fixtures: list, teams: list, gw: int, n: int = 4) -> dict:
    """{team_id: [{"opp": short_name, "diff": 1-5, "home": bool}, ...]} for the
    next n gameweeks from `gw`, using FPL's own fixture difficulty ratings."""
    short = {t["id"]: t["short_name"] for t in teams}
    rows: dict = {}
    for f in fixtures:
        ev = f.get("event")
        if ev is None or ev < gw:
            continue
        h, a = f["team_h"], f["team_a"]
        rows.setdefault(h, []).append((ev, short.get(a, "?"), f.get("team_h_difficulty") or 3, True))
        rows.setdefault(a, []).append((ev, short.get(h, "?"), f.get("team_a_difficulty") or 3, False))
    out = {}
    for tid, lst in rows.items():
        lst.sort(key=lambda x: x[0])
        out[tid] = [{"opp": opp, "diff": int(diff), "home": home}
                    for _, opp, diff, home in lst[:n]]
    return out


def flagged_players(bootstrap: dict) -> pd.DataFrame:
    """Players the live API marks as not fully available (injured/doubtful/etc)."""
    rows = [{"name": e.get("web_name"), "price": e["now_cost"],
             "position": POSITION.get(e["element_type"]),
             "status": STATUS_LABEL.get(e.get("status"), e.get("status")),
             "chance": e.get("chance_of_playing_next_round"),
             "news": (e.get("news") or "").strip()}
            for e in bootstrap["elements"] if e.get("status", "a") != "a"]
    df = pd.DataFrame(rows)
    return df.sort_values("price", ascending=False) if len(df) else df


def build_squad(preds: pd.DataFrame, force_ids=None,
                bench_boost: bool = False) -> pd.DataFrame:
    """Two-stage cold-start pick: choose the 15 for the *run* (blended `pred`, so
    a good next few gameweeks — fewer forced transfers later), then choose the
    starting XI + captain for the *immediate* gameweek (`pred_gw1`). Falls back to
    a single-stage pick if no this-gameweek column is present.

    With ``bench_boost``, all 15 are optimised for the immediate gameweek and the
    bench counts in full (you plan to play the Bench Boost chip in GW1, so every
    player scores) — no cheap bench fillers."""
    run_map = preds.set_index("id")["pred"].to_dict()          # blended next-X-GW avg
    gw1_map = (preds.set_index("id")["pred_gw1"].to_dict()
               if "pred_gw1" in preds.columns else run_map)     # this gameweek only

    def _annotate(sq):
        sq = sq.copy()
        sq["pred_run"] = sq["id"].map(run_map)                  # squad-selection metric
        sq["pred_gw1"] = sq["id"].map(gw1_map)                  # this-gameweek metric
        return sq

    if bench_boost and "pred_gw1" in preds.columns:
        p = preds.copy()
        p["pred"] = p["pred_gw1"]
        sq = optimize_squad(p, force_ids=force_ids or None,
                            conflict_penalty=CONFLICT_PENALTY, bench_boost=True)
        sq["pred"] = sq["id"].map(gw1_map).fillna(sq["pred"])
        return _annotate(sq)
    squad = optimize_squad(preds, force_ids=force_ids or None,
                           conflict_penalty=CONFLICT_PENALTY)
    if "pred_gw1" not in preds.columns:
        return _annotate(squad)
    from src.advisor.personal import optimize_xi
    sq = squad.copy()
    sq["pred"] = sq["id"].map(gw1_map).fillna(sq["pred"])   # XI chosen on this GW
    _, sq = optimize_xi(sq, conflict_penalty=CONFLICT_PENALTY)
    return _annotate(sq)


def preseason_squad(client: FPLClient = None,
                    seed_season: str = SEED_SEASON) -> pd.DataFrame:
    """The optimal 15-man squad (with XI + captain) for the upcoming gameweek."""
    return build_squad(preseason_predictions(client, seed_season))


def _print_squad(squad: pd.DataFrame, preds: pd.DataFrame) -> None:
    name = dict(zip(preds["id"], preds["name"]))
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi = squad[squad["starting"] == 1].sort_values(
        "position", key=lambda s: s.map(order))
    bench = squad[squad["starting"] == 0].sort_values(
        "position", key=lambda s: s.map(order))
    print("\nStarting XI")
    for _, r in xi.iterrows():
        c = " (C)" if r["captain"] else ""
        print(f"  {r['position']:<4}{name.get(r['id'], r['id']):<20}"
              f"£{r['price']/10:>4.1f}m  proj {r['pred']:>4.1f}{c}")
    print("Bench")
    for _, r in bench.iterrows():
        print(f"  {r['position']:<4}{name.get(r['id'], r['id']):<20}"
              f"£{r['price']/10:>4.1f}m  proj {r['pred']:>4.1f}")
    print(f"\nSquad cost: £{squad['price'].sum()/10:.1f}m / £100.0m")


def _resolve_pins(preds: pd.DataFrame, names: list, avail: dict = None) -> list:
    """Map pinned player names to ids (best/most-played match per name).

    Players the live API marks as out (availability 0 — injured/suspended) are
    skipped with a warning: pinning an unavailable player makes no sense.
    """
    avail = avail or {}
    ids = []
    for n in names:
        hit = preds[preds["name"].str.contains(n, case=False, na=False)]
        if not len(hit):
            print(f"Pin not found (ignored): {n}")
            continue
        row = hit.sort_values("pred", ascending=False).iloc[0]
        pid = int(row["id"])
        if avail.get(pid, 1.0) == 0.0:
            print(f"Pin skipped (unavailable): {row['name']} — injured/suspended")
            continue
        ids.append(pid)
        print(f"Pinned: {row['name']} (£{row['price'] / 10:.1f}m)")
    return ids


def _xi_points(squad: pd.DataFrame) -> float:
    """Objective the optimiser maximises: XI points + captain counted again."""
    xi = squad[squad["starting"] == 1]
    cap = squad[squad["captain"] == 1]
    return xi["pred"].sum() + (cap["pred"].iloc[0] if len(cap) else 0.0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Recommend a Fantasy Premier League squad for the upcoming "
                    "(unplayed) gameweek, seeded from last season's form.")
    parser.add_argument("--pin", default="",
                        help="comma-separated player names to force into the "
                             "squad, e.g. --pin Haaland")
    args = parser.parse_args()

    client = FPLClient()
    preds = preseason_predictions(client)
    force_ids = _resolve_pins(
        preds, [n.strip() for n in args.pin.split(",") if n.strip()],
        availability(client))

    squad = build_squad(preds, force_ids=force_ids or None)
    _print_squad(squad, preds)

    if force_ids:
        cost = _xi_points(squad) - _xi_points(
            optimize_squad(preds, conflict_penalty=CONFLICT_PENALTY))
        print(f"\nForced picks cost: {cost:+.1f} projected pts vs the "
              f"unconstrained squad")

    flagged = flagged_players(client.get_bootstrap_static())
    notable = flagged[(flagged["price"] >= 55) | flagged["chance"].notna()] \
        if len(flagged) else flagged
    if len(notable):
        print("\nAvailability flags (live FPL API) — excluded or downweighted:")
        for _, r in notable.head(20).iterrows():
            chance = f"  {int(r['chance'])}%" if pd.notna(r["chance"]) else ""
            news = f" — {r['news']}" if r["news"] else ""
            print(f"  {r['position']:<4}{r['name']:<20}£{r['price']/10:>4.1f}m  "
                  f"{r['status']}{chance}{news}")


if __name__ == "__main__":
    main()
