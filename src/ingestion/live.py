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
from src.config import ACTIVE_SEASON

# FPL element_type -> our position label.
POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# How many upcoming (unplayed) gameweeks to project per player — enough for the
# next-gameweek advice, the fixture blend, and near-term chip timing.
UPCOMING_HORIZON = 8

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


def season_started(client: FPLClient | None = None) -> bool:
    """True once at least one gameweek's deadline has passed (FPL marks it
    ``is_current``). Before that it's pre-season — build the opening squad.
    Defensive: any error is treated as not-started (safe: shows the builder)."""
    try:
        events = (client or FPLClient()).get_bootstrap_static()["events"]
        return any(e.get("is_current") for e in events)
    except Exception:  # noqa: BLE001
        return False


def upcoming_gameweek(client: FPLClient | None = None) -> int | None:
    """The next gameweek you can still change — the one whose deadline is in the
    future (FPL ``is_next``). Falls back to ``is_current``, then None. This is the
    gameweek in-season advice should target (you can't change a live one)."""
    try:
        events = (client or FPLClient()).get_bootstrap_static()["events"]
    except Exception:  # noqa: BLE001
        return None
    for e in events:
        if e.get("is_next"):
            return e["id"]
    for e in events:
        if e.get("is_current"):
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


STATUS_LABEL = {"a": "available", "d": "doubtful", "i": "injured",
                "s": "suspended", "u": "unavailable", "n": "not in squad"}


def player_flags(ids=None, client: FPLClient | None = None) -> dict:
    """{id: {"status": label, "chance": int|None, "news": str}} for players the
    live API marks as not fully available (status != 'a'). Optionally limited to
    a set of element ids (e.g. the manager's own squad)."""
    client = client_or_default(client)
    out = {}
    for e in client.get_bootstrap_static()["elements"]:
        if ids is not None and e["id"] not in ids:
            continue
        if e.get("status", "a") != "a":
            out[e["id"]] = {
                "status": STATUS_LABEL.get(e.get("status"), e.get("status")),
                "chance": e.get("chance_of_playing_next_round"),
                "news": (e.get("news") or "").strip(),
            }
    return out


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
            "season": ACTIVE_SEASON,
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
        # Upcoming (unplayed) fixtures — so the model can project the NEXT few
        # gameweeks from this season's form (rolling features come from the games
        # already played), instead of having no row to score. Stats are 0 (not yet
        # played); price is the current price so the value feature is right.
        for fx in summary.get("fixtures", [])[:UPCOMING_HORIZON]:
            if fx.get("event") is None:
                continue
            home = bool(fx.get("is_home"))
            row = dict(meta)
            row["gw"] = fx.get("event")
            row["fixture"] = fx.get("id")
            row["opponent_team"] = fx.get("team_a") if home else fx.get("team_h")
            row["was_home"] = home
            row["kickoff_time"] = fx.get("kickoff_time")
            row["xp"] = None
            for f in _HISTORY_FIELDS:
                row[f] = 0
            row["value"] = el.get("now_cost", 0)   # current price for the value feature
            rows.append(row)
    return pd.DataFrame(rows)
