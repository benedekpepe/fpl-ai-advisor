"""Load per-gameweek player history for the configured data source."""
import pandas as pd

from src.db.connection import get_connection


def load_history_from_db() -> pd.DataFrame:
    """Every player's per-gameweek history across the ingested seasons."""
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM player_gameweek_history", conn)


def load_history() -> pd.DataFrame:
    """Per-gameweek history for the configured data source (config.DATA_SOURCE).

    "db" (default) reads the stored history; "live" builds the current season
    from the live FPL API. Everything downstream (feature build, prediction,
    fixture ticker) is identical regardless of where the rows come from.
    """
    from src.config import DATA_SOURCE
    if DATA_SOURCE == "live":
        from src.ingestion.live import current_season_frame
        return current_season_frame()
    return load_history_from_db()
