"""Load per-gameweek player history for the configured data source."""
import io
from functools import lru_cache

import pandas as pd
import requests

from src.db.connection import get_connection


def load_history_from_db() -> pd.DataFrame:
    """Every player's per-gameweek history across the ingested seasons."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM player_gameweek_history", conn)


@lru_cache(maxsize=2)
def load_history_from_csv(seasons: tuple = None) -> pd.DataFrame:
    """Per-gameweek history pulled straight from the vaastav CSVs (no database).

    Used for the hosted demo so it needs no Postgres: the current season is
    enough to advise on it. Columns are renamed to match the stored table so
    everything downstream is identical.
    """
    from src.config import CURRENT_SEASON, MERGED_GW_URL
    seasons = seasons or (CURRENT_SEASON,)
    frames = []
    for s in seasons:
        text = requests.get(MERGED_GW_URL.format(season=s), timeout=60).content.decode(
            "utf-8", "replace")
        d = pd.read_csv(io.StringIO(text))
        d["season"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"GW": "gw", "xP": "xp"})


def load_history() -> pd.DataFrame:
    """Per-gameweek history for the configured data source (config.DATA_SOURCE).

    "db" (default, local) reads the stored history; "csv" pulls the vaastav CSVs
    directly (the hosted demo, no database); "live" builds the current season
    from the live FPL API. Everything downstream (feature build, prediction,
    fixture ticker) is identical regardless of where the rows come from.
    """
    from src.config import DATA_SOURCE
    if DATA_SOURCE == "csv":
        return load_history_from_csv()
    if DATA_SOURCE == "live":
        from src.ingestion.live import current_season_frame
        return current_season_frame()
    return load_history_from_db()
