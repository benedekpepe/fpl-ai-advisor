"""Load historical per-gameweek player data for backtesting.

Source: vaastav/Fantasy-Premier-League (one merged_gw.csv per season).
That dataset already includes expected-goals/assists style stats, so it is
enough to train and backtest the points model without a separate scrape.

Run:
    python -m src.ingestion.load_history                 # default recent seasons
    python -m src.ingestion.load_history 2023-24 2024-25 # specific seasons
"""
import io
import sys

import pandas as pd
import requests
from psycopg2.extras import execute_values

from src.db.connection import get_connection
from src.config import MERGED_GW_URL

DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]

# CSV columns we keep (older seasons may miss a few — handled gracefully).
KEEP = [
    "GW", "element", "fixture", "name", "position", "team", "opponent_team",
    "was_home", "minutes", "total_points", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "bonus", "bps", "saves", "starts", "xP",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "ict_index", "influence", "creativity",
    "threat", "value", "selected", "transfers_in", "transfers_out",
    "kickoff_time",
]


def _int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool(v):
    if v is None:
        return None
    return str(v).strip().lower() in ("true", "1")


def fetch_season(season: str) -> pd.DataFrame:
    url = MERGED_GW_URL.format(season=season)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # some older seasons are latin-1; decode defensively
    text = resp.content.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def build_rows(df: pd.DataFrame, season: str) -> list:
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols].where(pd.notnull(df[cols]), None)
    rows = []
    for r in df.itertuples(index=False):
        d = dict(zip(cols, r))
        rows.append((
            season, _int(d.get("GW")), _int(d.get("element")), _int(d.get("fixture")),
            d.get("name"), d.get("position"), d.get("team"),
            _int(d.get("opponent_team")), _bool(d.get("was_home")),
            _int(d.get("minutes")), _int(d.get("total_points")),
            _int(d.get("goals_scored")), _int(d.get("assists")),
            _int(d.get("clean_sheets")), _int(d.get("goals_conceded")),
            _int(d.get("bonus")), _int(d.get("bps")), _int(d.get("saves")),
            _int(d.get("starts")), _float(d.get("xP")),
            _float(d.get("expected_goals")), _float(d.get("expected_assists")),
            _float(d.get("expected_goal_involvements")),
            _float(d.get("expected_goals_conceded")), _float(d.get("ict_index")),
            _float(d.get("influence")), _float(d.get("creativity")),
            _float(d.get("threat")), _int(d.get("value")), _int(d.get("selected")),
            _int(d.get("transfers_in")), _int(d.get("transfers_out")),
            d.get("kickoff_time"),
        ))
    return rows


INSERT_SQL = """
    INSERT INTO player_gameweek_history (
        season, gw, element, fixture, name, position, team, opponent_team,
        was_home, minutes, total_points, goals_scored, assists, clean_sheets,
        goals_conceded, bonus, bps, saves, starts, xp,
        expected_goals, expected_assists, expected_goal_involvements,
        expected_goals_conceded, ict_index, influence, creativity, threat,
        value, selected, transfers_in, transfers_out, kickoff_time
    ) VALUES %s
    ON CONFLICT (season, element, fixture) DO NOTHING
"""


def load_rows(rows: list) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, INSERT_SQL, rows, page_size=1000)


def run(seasons: list[str]) -> None:
    total = 0
    for season in seasons:
        print(f"Fetching {season} ...", end=" ", flush=True)
        df = fetch_season(season)
        rows = build_rows(df, season)
        load_rows(rows)
        total += len(rows)
        print(f"loaded {len(rows)} rows")
    print(f"Done. {total} historical rows across {len(seasons)} season(s).")


if __name__ == "__main__":
    seasons = sys.argv[1:] or DEFAULT_SEASONS
    run(seasons)
