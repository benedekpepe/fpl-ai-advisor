"""Premier League team metadata: short codes and strength ratings.

A single loader for each season's ``teams.csv`` from the vaastav dataset,
shared by the model (opponent attack/defence strength, used as
fixture-difficulty features) and the advisor (opponent short code plus a 1-5
difficulty for the fixture ticker). Each season is downloaded at most once and
cached in-process, so the previously duplicated network calls collapse into
one.
"""
import io
from functools import lru_cache

import pandas as pd
import requests

from src.config import TEAMS_URL

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
