"""Thin client for the public Fantasy Premier League API.

No API key required. The endpoints are not officially documented but are
stable and used widely. Be a good citizen: don't hammer them in tight loops.
"""
import requests

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    def __init__(self, base_url: str = FPL_BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "fpl-ai-advisor/0.1"})

    def _get(self, path: str) -> dict | list:
        resp = self.session.get(f"{self.base_url}/{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_bootstrap_static(self) -> dict:
        """The backbone: players (elements), teams, positions, gameweeks."""
        return self._get("bootstrap-static/")

    def get_fixtures(self) -> list:
        """All fixtures with difficulty ratings and (once played) scores."""
        return self._get("fixtures/")

    def get_element_summary(self, player_id: int) -> dict:
        """Per-gameweek history and upcoming fixtures for one player."""
        return self._get(f"element-summary/{player_id}/")

    def get_entry(self, team_id: int) -> dict:
        """A user's overall entry: bank, value, chip history, etc."""
        return self._get(f"entry/{team_id}/")

    def get_entry_picks(self, team_id: int, gameweek: int) -> dict:
        """A user's 15 picks, captain and bench for a given gameweek."""
        return self._get(f"entry/{team_id}/event/{gameweek}/picks/")
