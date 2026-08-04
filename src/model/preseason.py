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

from src.config import (MERGED_GW_URL, PLAYERS_RAW_URL, SEED_SEASON, MODEL_PATH)
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


def _csv(url: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(
        requests.get(url, timeout=60).content.decode("utf-8", "replace")))


@lru_cache(maxsize=2)
def seed_from_previous_season(season: str = SEED_SEASON) -> pd.DataFrame:
    """Per-player, per-game form averages from a completed season, keyed by the
    permanent player ``code`` (stable across seasons)."""
    id_to_code = dict(zip(*(lambda p: (p["id"], p["code"]))(
        _csv(PLAYERS_RAW_URL.format(season=season)))))
    mg = _csv(MERGED_GW_URL.format(season=season))
    mg["code"] = mg["element"].map(id_to_code)
    agg = {b: (b, "mean") for b in BASES}
    agg["starts_rate_5"] = ("starts", "mean")
    agg["cum_ppg"] = ("total_points", "mean")
    return mg.groupby("code").agg(**agg).reset_index()


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
                    seed_season: str = SEED_SEASON) -> pd.DataFrame:
    """Build the model's 25-feature frame for the upcoming gameweek's players."""
    gw = _next_gameweek(bootstrap["events"])
    opp = _fixture_opponents(fixtures, gw)

    strength = {t["id"]: t for t in bootstrap["teams"]}
    rows = []
    for e in bootstrap["elements"]:
        o = opp.get(e["team"])
        if o is None:            # team not playing this GW (blank) -> skip
            continue
        rows.append({
            "id": e["id"], "code": e["code"], "name": e.get("web_name"),
            "position": POSITION.get(e["element_type"]), "team": e["team"],
            "value": e["now_cost"], "opponent_team": o[0], "was_home": o[1],
        })
    df = pd.DataFrame(rows).merge(seed_from_previous_season(seed_season),
                                  on="code", how="left")
    df = _apply_fallback(df)

    for b in BASES:
        df[f"roll_{b}_3"] = df[b]
        df[f"roll_{b}_5"] = df[b]
    df["games_played"] = 0
    df["is_home"] = df["was_home"].astype(float)
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
    df = preseason_frame(bootstrap, fixtures, seed_season)

    bundle = joblib.load(MODEL_PATH)
    clf, reg, cols = bundle["clf"], bundle["reg"], bundle["cols"]
    df["pred"] = clf.predict_proba(df[cols])[:, 1] * reg.predict(df[cols])
    if apply_availability:
        av = availability(client)
        df["pred"] = df["pred"] * df["id"].map(av).fillna(1.0)
    return df[["id", "name", "position", "team", "value", "pred"]].rename(
        columns={"value": "price"})


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


def preseason_squad(client: FPLClient = None,
                    seed_season: str = SEED_SEASON) -> pd.DataFrame:
    """The optimal 15-man squad (with XI + captain) for the upcoming gameweek."""
    return optimize_squad(preseason_predictions(client, seed_season))


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


def main() -> None:
    client = FPLClient()
    preds = preseason_predictions(client)
    squad = optimize_squad(preds)
    _print_squad(squad, preds)

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
