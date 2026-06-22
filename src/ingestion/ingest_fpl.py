"""Pull the current FPL state and store it in Postgres.

Run with:  python -m src.ingestion.ingest_fpl

What it does:
  1. bootstrap-static -> teams, positions, gameweeks, players (upsert)
  2. writes a row per player into player_snapshots (time series)
  3. fixtures -> fixtures table (upsert)

Designed to be run repeatedly (e.g. daily via cron / GitHub Actions).
Upserts keep the latest state; snapshots keep the history.
"""
from datetime import datetime, timezone

from psycopg2.extras import execute_values

from src.db.connection import get_connection
from src.ingestion.fpl_client import FPLClient


# --- small parsing helpers (the API returns numbers as strings sometimes) ---
def _f(value):
    """Parse to float, tolerating None / empty string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _current_gameweek(events: list) -> int | None:
    for e in events:
        if e.get("is_current"):
            return e["id"]
    for e in events:
        if e.get("is_next"):
            return e["id"]
    return None


# ----------------------------- upserts ------------------------------------
def upsert_teams(cur, teams: list) -> None:
    rows = [
        (
            t["id"], t["name"], t["short_name"], t.get("strength"),
            t.get("strength_overall_home"), t.get("strength_overall_away"),
            t.get("strength_attack_home"), t.get("strength_attack_away"),
            t.get("strength_defence_home"), t.get("strength_defence_away"),
        )
        for t in teams
    ]
    execute_values(
        cur,
        """
        INSERT INTO teams (
            id, name, short_name, strength,
            strength_overall_home, strength_overall_away,
            strength_attack_home, strength_attack_away,
            strength_defence_home, strength_defence_away
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            short_name = EXCLUDED.short_name,
            strength = EXCLUDED.strength,
            strength_overall_home = EXCLUDED.strength_overall_home,
            strength_overall_away = EXCLUDED.strength_overall_away,
            strength_attack_home = EXCLUDED.strength_attack_home,
            strength_attack_away = EXCLUDED.strength_attack_away,
            strength_defence_home = EXCLUDED.strength_defence_home,
            strength_defence_away = EXCLUDED.strength_defence_away
        """,
        rows,
    )


def upsert_positions(cur, positions: list) -> None:
    rows = [
        (
            p["id"], p["singular_name"], p["singular_name_short"],
            p.get("squad_select"), p.get("squad_min_play"), p.get("squad_max_play"),
        )
        for p in positions
    ]
    execute_values(
        cur,
        """
        INSERT INTO positions (
            id, singular_name, singular_name_short,
            squad_select, squad_min_play, squad_max_play
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            singular_name = EXCLUDED.singular_name,
            singular_name_short = EXCLUDED.singular_name_short,
            squad_select = EXCLUDED.squad_select,
            squad_min_play = EXCLUDED.squad_min_play,
            squad_max_play = EXCLUDED.squad_max_play
        """,
        rows,
    )


def upsert_gameweeks(cur, events: list) -> None:
    rows = [
        (
            e["id"], e["name"], e.get("deadline_time"),
            e.get("is_current"), e.get("is_next"), e.get("finished"),
            e.get("average_entry_score"), e.get("highest_score"),
        )
        for e in events
    ]
    execute_values(
        cur,
        """
        INSERT INTO gameweeks (
            id, name, deadline_time, is_current, is_next, finished,
            average_entry_score, highest_score
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            deadline_time = EXCLUDED.deadline_time,
            is_current = EXCLUDED.is_current,
            is_next = EXCLUDED.is_next,
            finished = EXCLUDED.finished,
            average_entry_score = EXCLUDED.average_entry_score,
            highest_score = EXCLUDED.highest_score
        """,
        rows,
    )


def upsert_players(cur, elements: list) -> None:
    rows = [
        (
            p["id"], p.get("first_name"), p.get("second_name"), p.get("web_name"),
            p["team"], p["element_type"], _i(p.get("now_cost")),
            _i(p.get("total_points")), _f(p.get("form")), _f(p.get("points_per_game")),
            _f(p.get("selected_by_percent")), _i(p.get("minutes")),
            _i(p.get("goals_scored")), _i(p.get("assists")), _i(p.get("clean_sheets")),
            _f(p.get("expected_goals")), _f(p.get("expected_assists")),
            p.get("status"), _i(p.get("chance_of_playing_next_round")), p.get("news"),
        )
        for p in elements
    ]
    execute_values(
        cur,
        """
        INSERT INTO players (
            id, first_name, second_name, web_name, team_id, element_type,
            now_cost, total_points, form, points_per_game, selected_by_percent,
            minutes, goals_scored, assists, clean_sheets,
            expected_goals, expected_assists, status,
            chance_of_playing_next_round, news
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            first_name = EXCLUDED.first_name,
            second_name = EXCLUDED.second_name,
            web_name = EXCLUDED.web_name,
            team_id = EXCLUDED.team_id,
            element_type = EXCLUDED.element_type,
            now_cost = EXCLUDED.now_cost,
            total_points = EXCLUDED.total_points,
            form = EXCLUDED.form,
            points_per_game = EXCLUDED.points_per_game,
            selected_by_percent = EXCLUDED.selected_by_percent,
            minutes = EXCLUDED.minutes,
            goals_scored = EXCLUDED.goals_scored,
            assists = EXCLUDED.assists,
            clean_sheets = EXCLUDED.clean_sheets,
            expected_goals = EXCLUDED.expected_goals,
            expected_assists = EXCLUDED.expected_assists,
            status = EXCLUDED.status,
            chance_of_playing_next_round = EXCLUDED.chance_of_playing_next_round,
            news = EXCLUDED.news,
            updated_at = now()
        """,
        rows,
    )


def insert_snapshots(cur, elements: list, gameweek: int | None) -> None:
    """Append a time-series row per player (no upsert — history grows)."""
    captured_at = datetime.now(timezone.utc)
    rows = [
        (
            p["id"], captured_at, gameweek, _i(p.get("now_cost")), _f(p.get("form")),
            _i(p.get("total_points")), _f(p.get("points_per_game")),
            _f(p.get("selected_by_percent")), _i(p.get("minutes")),
            p.get("status"), _i(p.get("chance_of_playing_next_round")),
        )
        for p in elements
    ]
    execute_values(
        cur,
        """
        INSERT INTO player_snapshots (
            player_id, captured_at, gameweek, now_cost, form, total_points,
            points_per_game, selected_by_percent, minutes, status,
            chance_of_playing_next_round
        ) VALUES %s
        """,
        rows,
    )


def upsert_fixtures(cur, fixtures: list) -> None:
    rows = [
        (
            fx["id"], fx.get("event"), fx.get("kickoff_time"),
            fx.get("team_h"), fx.get("team_a"),
            fx.get("team_h_difficulty"), fx.get("team_a_difficulty"),
            fx.get("team_h_score"), fx.get("team_a_score"), fx.get("finished"),
        )
        for fx in fixtures
    ]
    execute_values(
        cur,
        """
        INSERT INTO fixtures (
            id, event, kickoff_time, team_h, team_a,
            team_h_difficulty, team_a_difficulty,
            team_h_score, team_a_score, finished
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            event = EXCLUDED.event,
            kickoff_time = EXCLUDED.kickoff_time,
            team_h = EXCLUDED.team_h,
            team_a = EXCLUDED.team_a,
            team_h_difficulty = EXCLUDED.team_h_difficulty,
            team_a_difficulty = EXCLUDED.team_a_difficulty,
            team_h_score = EXCLUDED.team_h_score,
            team_a_score = EXCLUDED.team_a_score,
            finished = EXCLUDED.finished,
            updated_at = now()
        """,
        rows,
    )


# ----------------------------- orchestrator -------------------------------
def run() -> None:
    client = FPLClient()

    print("Fetching bootstrap-static ...")
    boot = client.get_bootstrap_static()
    teams = boot["teams"]
    positions = boot["element_types"]
    events = boot["events"]
    elements = boot["elements"]
    gw = _current_gameweek(events)

    print("Fetching fixtures ...")
    fixtures = client.get_fixtures()

    with get_connection() as conn:
        with conn.cursor() as cur:
            upsert_teams(cur, teams)
            upsert_positions(cur, positions)
            upsert_gameweeks(cur, events)
            upsert_players(cur, elements)
            insert_snapshots(cur, elements, gw)
            upsert_fixtures(cur, fixtures)

    print(
        f"Done. teams={len(teams)} positions={len(positions)} "
        f"gameweeks={len(events)} players={len(elements)} "
        f"fixtures={len(fixtures)} (current GW={gw})"
    )


if __name__ == "__main__":
    run()
