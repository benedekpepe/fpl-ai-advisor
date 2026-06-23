"""Live FPL data for in-season use (the 2026-27 "live mode").

The demo runs on a finished season from stored history; this module is the
other side of the ``DATA_SOURCE`` switch (config.py): it builds the same
per-gameweek frame from the *live* FPL API instead, plus two helpers the live
mode needs — the current gameweek and player availability (injuries/doubts).

STATUS: drafted but NOT yet verified against a live season. The FPL API shape
is stable, but ``current_season_frame`` issues one request per player and a few
fields (notably FPL's own expected points) aren't in the per-gameweek history,
so this must be exercised and checked once the 2026-27 season is live. The
high-confidence pieces are ``current_gameweek`` and ``availability``.

Wiring (done once, in August):
  1. set the environment variable  FPL_DATA_SOURCE=live
  2. in app.py, default the gameweek to ``current_gameweek()`` instead of a
     manual slider (the value is already detected here);
  3. in advisor/personal.py ``build_advice``, after building the gameweek's
     player pool, downweight by availability, e.g.:
         av = availability()
         pool["pred"] *= pool["id"].map(av).fillna(1.0)
     so injured/suspended players stop being recommended.
"""
import pandas as pd

from src.ingestion.fpl_client import FPLClient
from src.config import CURRENT_SEASON

# FPL element_type -> our position label.
POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Per-gameweek history fields the model's feature build relies on; mapped 1:1
# from the API's element-summary "history" entries onto our column names.
_HISTORY_FIELDS = [
    "minutes", "total_points", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "bonus", "bps", "saves", "starts",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "ict_index", "influence", "creativity",
    "threat", "value", "selected", "transfers_in", "transfers_out",
]


def _num(v):
    """Parse a possibly-stringified number, tolerating None / empty."""
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def current_gameweek(client: FPLClient | None = None) -> int | None:
    """The gameweek to advise on: the current one, else the next upcoming."""
    client = client or FPLClient()
    events = client.get_bootstrap_static()["events"]
    for e in events:
        if e.get("is_current"):
            return e["id"]
    for e in events:
        if e.get("is_next"):
            return e["id"]
    return None


def availability(client: FPLClient | None = None) -> dict[int, float]:
    """Map element_id -> a play-probability multiplier in [0, 1].

    'a' (available) -> 1.0; a known chance_of_playing -> that / 100; otherwise
    'd' (doubtful) -> 0.5 and 'i'/'s'/'u'/'n' (out) -> 0.0. Multiply a player's
    projected points by this to stop recommending players who won't feature.
    """
    elements = client_or_default(client).get_bootstrap_static()["elements"]
    out: dict[int, float] = {}
    for e in elements:
        status = e.get("status")
        chance = e.get("chance_of_playing_next_round")
        if status == "a":
            out[e["id"]] = 1.0
        elif chance is not None:
            out[e["id"]] = chance / 100.0
        elif status == "d":
            out[e["id"]] = 0.5
        else:  # 'i', 's', 'u', 'n', or unknown -> treat as not playing
            out[e["id"]] = 0.0
    return out


def client_or_default(client: FPLClient | None) -> FPLClient:
    return client or FPLClient()


def current_season_frame(client: FPLClient | None = None) -> pd.DataFrame:
    """Per-gameweek history for the live season, shaped like the stored table.

    Returns the same columns as ``load_history_from_db`` so it can flow straight
    into ``attach_opponent_strength`` + ``build_feature_frame``. Player meta
    (name, position, club) comes from bootstrap-static; the per-gameweek rows
    come from each player's element-summary.

    NOTE: one request per player (~600). Verify volume/rate-limit behaviour and
    the field mapping against a live season before relying on this.
    """
    client = client_or_default(client)
    boot = client.get_bootstrap_static()
    team_name = {t["id"]: t["short_name"] for t in boot["teams"]}

    rows = []
    for el in boot["elements"]:
        meta = {
            "season": CURRENT_SEASON,
            "element": el["id"],
            "name": el.get("web_name"),
            "position": POSITION.get(el.get("element_type")),
            "team": team_name.get(el.get("team")),
        }
        summary = client.get_element_summary(el["id"])
        for h in summary.get("history", []):
            row = dict(meta)
            row["gw"] = h.get("round")
            row["fixture"] = h.get("fixture")
            row["opponent_team"] = h.get("opponent_team")
            row["was_home"] = h.get("was_home")
            row["kickoff_time"] = h.get("kickoff_time")
            row["xp"] = None  # FPL's own xP isn't in per-gameweek history
            for f in _HISTORY_FIELDS:
                row[f] = _num(h.get(f))
            rows.append(row)
    return pd.DataFrame(rows)
