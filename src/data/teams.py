"""Premier League team metadata: short codes and strength ratings.

A single loader for each season's ``teams.csv`` from the vaastav dataset,
shared by the model (opponent attack/defence strength, used as
fixture-difficulty features) and the advisor (opponent short code plus a 1-5
difficulty for the fixture ticker). Each season is downloaded at most once and
cached in-process, so the previously duplicated network calls collapse into
one.
"""
import io
import math
from functools import lru_cache

import pandas as pd
import requests

from src.config import TEAMS_URL, SEED_SEASON

_STRENGTH_COLS = [
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
]


@lru_cache(maxsize=None)
def _teams_csv(season: str) -> pd.DataFrame:
    """Download and cache one season's raw ``teams.csv``."""
    text = requests.get(TEAMS_URL.format(season=season), timeout=60).content.decode(
        "utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def team_strength(seasons) -> pd.DataFrame:
    """Per-team attack/defence strengths for the given seasons.

    Keyed by ``opponent_team`` so it can be merged onto match rows to attach the
    opponent's strength.
    """
    frames = []
    for s in seasons:
        t = _teams_csv(s)[["id", *_STRENGTH_COLS]].copy()
        t["season"] = s
        frames.append(t)
    cols = ["season", "id", *_STRENGTH_COLS]
    return (pd.concat(frames, ignore_index=True)[cols]
            .rename(columns={"id": "opponent_team"}))


def teams_meta(season: str) -> dict:
    """``{team_id: (short_code, strength 1-5)}`` for one season (for the UI).

    Early in a season FPL hasn't set the 1-5 ``strength`` yet (it's NaN), so the
    difficulty defaults to 3 (medium) while the short name still resolves."""
    try:
        t = _teams_csv(season)
    except Exception:
        return {}
    out = {}
    for _, r in t.iterrows():
        try:
            s = int(r["strength"]) if pd.notna(r.get("strength")) else 3
        except (ValueError, TypeError):
            s = 3
        out[int(r["id"])] = (str(r["short_name"]), s)
    return out


def resolve_live_teams(teams: list, previous_season: str = SEED_SEASON) -> list:
    """Reject uninitialised ratings; match archived clubs by code, never numeric ID.

    Keep valid live fields. Missing/zero fields use the last completed season;
    unmatched/promoted clubs use the valid rating population's median. If no
    defensible prior exists, fail rather than interpret zero as a weak opponent.
    """
    def valid(value):
        try:
            return math.isfinite(float(value)) and float(value) > 0
        except (TypeError, ValueError):
            return False

    needs_prior = any(not valid(t.get(c)) for t in teams for c in _STRENGTH_COLS)
    archived = pd.DataFrame()
    if needs_prior:
        try:
            archived = _teams_csv(previous_season).copy()
        except Exception:
            archived = pd.DataFrame()
    by_code = (archived.drop_duplicates("short_name").set_index("short_name").to_dict("index")
               if "short_name" in archived else {})
    medians = {}
    for col in _STRENGTH_COLS:
        candidates = [v for v in archived.get(col, []) if valid(v)]
        if not candidates:
            candidates = [t.get(col) for t in teams if valid(t.get(col))]
        medians[col] = float(pd.Series(candidates, dtype=float).median()) if candidates else None
    resolved = []
    for team in teams:
        row = dict(team)
        source = "live"
        old = by_code.get(team.get("short_name"), {})
        for col in _STRENGTH_COLS:
            if valid(row.get(col)):
                continue
            if valid(old.get(col)):
                row[col] = float(old[col])
                if source != "neutral":
                    source = "previous_season"
            elif medians[col] is not None:
                row[col] = medians[col]
                source = "neutral"
            else:
                raise ValueError("Opponent strength ratings are unavailable; refusing zero-strength forecasts.")
        row["opponent_rating_source"] = source
        resolved.append(row)
    return resolved
