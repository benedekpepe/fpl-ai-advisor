"""Live deadline, data-cutoff and target consistency regressions; no network."""
from datetime import datetime, timezone
from itertools import combinations

import pandas as pd
import pytest

from src.ingestion import live
from src.model import preseason
from src.advisor import personal
from src.config import CONFLICT_PENALTY, XI_MIN, XI_MAX


NOW = datetime(2026, 11, 7, tzinfo=timezone.utc)


def test_deadlines_override_stale_event_flags_and_never_return_locked_week():
    events = [
        dict(id=10, deadline_time="2026-11-06T18:00:00Z", is_next=True),
        dict(id=11, deadline_time="2026-11-14T11:00:00Z", is_current=True),
        dict(id=12, deadline_time="2026-11-21T11:00:00Z"),
    ]
    assert live.next_open_gameweek(list(reversed(events)), NOW) == 11
    assert live.next_open_gameweek(events, datetime(2026, 11, 14, 11, tzinfo=timezone.utc)) == 12
    assert live.next_open_gameweek(events, datetime(2027, 1, 1, tzinfo=timezone.utc)) is None


class FakeClient:
    def __init__(self, events, partial_points=99):
        self.events = events
        self.partial_points = partial_points

    def get_bootstrap_static(self):
        return {
            "events": self.events,
            "teams": [dict(id=1, short_name="ARS"), dict(id=2, short_name="HUL")],
            "elements": [dict(id=1, code=101, web_name="Player", element_type=2,
                              team=1, now_cost=50)],
        }

    def get_fixtures(self):
        return []

    def get_element_summary(self, player_id):
        return {
            "history": [dict(round=g, fixture=g, minutes=90, total_points=p,
                             starts=1, was_home=True) for g, p in [(9, 6), (10, self.partial_points)]],
            "fixtures": [dict(event=g, id=g, team_a=2, team_h=1, is_home=True)
                         for g in [10, 11, 12]],
        }


def events(checked=True):
    return [dict(id=9, finished=True, data_checked=checked, deadline_time="2000-01-01T00:00:00Z"),
            dict(id=10, finished=False, data_checked=False, deadline_time="2000-01-02T00:00:00Z"),
            dict(id=11, finished=False, data_checked=False, deadline_time="2099-01-01T00:00:00Z")]


@pytest.mark.parametrize("checked", [True, False])
def test_partial_and_unchecked_stats_never_enter_form_even_with_no_closed_weeks(checked):
    frame = live.current_season_frame(FakeClient(events(checked)))
    observed = frame.loc[frame.minutes.notna()]
    assert observed.gw.tolist() == ([9] if checked else [])
    assert frame.loc[frame.minutes.isna(), "gw"].tolist() == [11, 12]
    assert frame.attrs["form_gw"] == (9 if checked else 0)


def test_changing_partial_scores_does_not_change_seed_then_closure_refreshes_it(monkeypatch):
    client = FakeClient(events())
    monkeypatch.setattr(live, "client_or_default", lambda _: client)
    first = preseason.current_season_seed(history=live.current_season_frame(client))
    client.partial_points = 1000
    second = preseason.current_season_seed(history=live.current_season_frame(client))
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].cum_ppg == 6
    client.events[1].update(finished=True, data_checked=True)
    third = preseason.current_season_seed(history=live.current_season_frame(client))
    assert third.iloc[0].cum_ppg == 503
    assert third.attrs["form_gw"] == 10


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("freehit", [False, True])
def test_live_normal_and_fallback_paths_pass_same_target_to_xi_and_ticker(monkeypatch, fallback, freehit):
    monkeypatch.setattr(personal, "DATA_SOURCE", "live")
    monkeypatch.setattr(live, "upcoming_gameweek", lambda: 11)
    monkeypatch.setattr(live, "current_gameweek", lambda: 10)
    raw = pd.DataFrame(dict(gw=[11]*15, element=range(15), fixture=[111]*15))
    raw.attrs["form_gw"] = 9
    monkeypatch.setattr(personal, "load_history", lambda: raw)
    pick_weeks = []
    def picks(team, gw):
        pick_weeks.append(gw)
        return {"picks": [{"element": i} for i in range(15)],
                "active_chip": "freehit" if freehit and gw == 10 else None,
                "entry_history": {"bank": 7 if gw == 9 else 0}}
    monkeypatch.setattr(personal, "get_picks", picks)
    monkeypatch.setattr(personal, "get_transfers", lambda _: [])
    monkeypatch.setattr(personal, "get_history", lambda _: {"current": [], "chips": []})
    positions = ["GK"]*2 + ["DEF"]*5 + ["MID"]*5 + ["FWD"]*3
    pool = pd.DataFrame(dict(id=range(15), gw=[11]*15, pred=[4.0]*15,
                             position=positions, team=["ARS"]*15, price=[50]*15,
                             name=[f"Player{i}" for i in range(15)]))
    monkeypatch.setattr(personal, "_predict_from_history", lambda season, history: pool.iloc[:0] if fallback else pool)
    monkeypatch.setattr(personal, "predict_all", lambda *_: pytest.fail("live used season-only cache"))
    monkeypatch.setattr(personal, "team_fixtures", lambda *_: pytest.fail("live ticker used historical fixtures"))
    monkeypatch.setattr(live, "availability", lambda: {i: 1.0 for i in range(15)})
    from src.ingestion import fpl_client
    monkeypatch.setattr(fpl_client, "FPLClient", lambda: FakeClient(events()))
    seen = {}
    def ticker(fixtures, teams, gw, n):
        seen["ticker"] = gw
        return {1: [{"opp": "HUL"}]}
    monkeypatch.setattr(preseason, "fixture_ticker", ticker)
    seed = pd.DataFrame({"code": [101]})
    seed.attrs["form_gw"] = 9
    monkeypatch.setattr(preseason, "current_season_seed", lambda **_: seed)
    def seed_predictions(**kwargs):
        seen["seed"] = kwargs["gw"]
        result = pool.drop(columns="gw").copy()
        result["pred_gw1"] = 5.0
        result["match"] = "1-2"
        result["team"] = 1
        return result
    monkeypatch.setattr(preseason, "preseason_predictions", seed_predictions)
    original_xi = personal.optimize_xi
    calls = []
    def xi(squad, conflict_penalty):
        if not calls:
            assert squad.gw.unique().tolist() == [11]
            assert squad.pred.unique().tolist() == ([5.0] if fallback else [4.0])
        calls.append(squad)
        assert conflict_penalty == 0
        return original_xi(squad, conflict_penalty)
    monkeypatch.setattr(personal, "optimize_xi", xi)
    monkeypatch.setattr(personal, "plan_exact", lambda *_: None)
    monkeypatch.setattr(personal, "chip_timing", lambda *_: ({}, []))
    monkeypatch.setattr(live, "player_flags", lambda *_: {})
    advice = personal.build_advice(123)
    assert advice["ok"]
    assert advice["gw"] == 11
    assert pick_weeks == ([10, 9] if freehit else [10])
    assert advice["bank"] == (7 if freehit else 0)
    assert advice["form_gw"] == 9
    assert sum(row["role"] == "start" for row in advice["current_xi"]) == 11
    assert all(row["fix"] == [{"opp": "HUL"}] for row in advice["current_xi"])
    assert all(row["pred"] == (5.0 if fallback else 4.0) for row in advice["current_xi"])
    expected_run = 4.0 if fallback else round(4.0 / sum(preseason.FIXTURE_WEIGHTS), 2)
    assert all(row["pred_run"] == expected_run for row in advice["current_xi"])
    assert advice["projection"]["source"] == ("current_form_seed" if fallback else "season_model")
    assert len(advice["selection"]) == 4
    assert seen["ticker"] == 11
    assert not fallback or seen["seed"] == 11


def test_screenshot_xi_maximises_points_without_hidden_opposition_dock():
    data = [
        ("Trafford", "GK", 4.5, "LEE", "LEE-TOT"),
        ("Verbruggen", "GK", 4.3, "BHA", "BHA-BRE"),
        ("Calafiori", "DEF", 4.8, "ARS", "ARS-HUL"),
        ("Gvardiol", "DEF", 4.7, "MCI", "MCI-NFO"),
        ("Hall", "DEF", 5.1, "NEW", "NEW-FUL"),
        ("De Cuyper", "DEF", 6.3, "BHA", "BHA-BRE"),
        ("Konsa", "DEF", 3.4, "ARS", "ARS-HUL"),
        ("Belloumi", "MID", 5.7, "HUL", "ARS-HUL"),
        ("Gross", "MID", 5.6, "BHA", "BHA-BRE"),
        ("Tavernier", "MID", 4.8, "BOU", "BOU-IPS"),
        ("Rogers", "MID", 4.9, "CHE", "CHE-SUN"),
        ("Manzambi", "MID", 4.0, "AVL", "AVL-MUN"),
        ("N.Jackson", "FWD", 4.3, "AVL", "AVL-MUN"),
        ("Evanilson", "FWD", 5.1, "BOU", "BOU-IPS"),
        ("Haaland", "FWD", 6.2, "MCI", "MCI-NFO"),
    ]
    squad = pd.DataFrame(data, columns=["name", "position", "pred", "team", "match"])
    score, chosen = personal.optimize_xi(squad, conflict_penalty=CONFLICT_PENALTY)
    starters = chosen[chosen.starting == 1]
    assert {"Calafiori", "Belloumi"} <= set(starters.name)
    best = -1
    for ids in combinations(squad.index, 11):
        xi = squad.loc[list(ids)]
        counts = xi.position.value_counts()
        if all(XI_MIN[p] <= counts.get(p, 0) <= XI_MAX[p] for p in XI_MIN):
            best = max(best, xi.pred.sum() + xi.pred.max())
    assert score == pytest.approx(best)
    _, penalised = personal.optimize_xi(squad, conflict_penalty=.35)
    assert penalised.loc[penalised.name == "Calafiori", "starting"].item() == 0


def test_no_checked_form_produces_empty_seed_not_nan_players(monkeypatch):
    client = FakeClient(events(checked=False))
    monkeypatch.setattr(live, "client_or_default", lambda _: client)
    seed = preseason.current_season_seed(history=live.current_season_frame(client))
    assert seed.empty


def test_seed_projection_uses_explicit_target_and_correct_weighted_average(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    client = FakeClient(events())
    bundle = {
        "clf": SimpleNamespace(predict_proba=lambda x: np.array([[0., 1.]])),
        "reg": SimpleNamespace(predict=lambda x: x["score"].to_numpy()),
        "cols": ["score"],
    }
    monkeypatch.setattr(preseason.joblib, "load", lambda _: bundle)
    seen = []
    def frame(bootstrap, fixtures, seed_season, gw, seed_df):
        seen.append(gw)
        return pd.DataFrame([dict(id=1, name="Player", position="DEF", team=1,
                                  match=f"GW{gw}", value=50, score=float(gw))])
    monkeypatch.setattr(preseason, "preseason_frame", frame)
    result = preseason.preseason_predictions(client=client, gw=11, apply_availability=False)
    assert seen == [11, 12, 13, 14]
    assert result.iloc[0].pred_gw1 == 11
    assert result.iloc[0]["match"] == "GW11"
    expected = sum(g*w for g, w in zip(seen, preseason.FIXTURE_WEIGHTS)) / sum(preseason.FIXTURE_WEIGHTS)
    assert result.iloc[0].pred == pytest.approx(expected)


def test_live_rejects_expired_target_and_handles_end_of_season(monkeypatch):
    monkeypatch.setattr(personal, "DATA_SOURCE", "live")
    monkeypatch.setattr(live, "upcoming_gameweek", lambda: 11)
    monkeypatch.setattr(live, "current_gameweek", lambda: 10)
    monkeypatch.setattr(personal, "get_picks", lambda *_: pytest.fail("fetched picks for invalid target"))
    assert not personal.build_advice(123, gw=10)["ok"]
    monkeypatch.setattr(live, "upcoming_gameweek", lambda: None)
    assert not personal.build_advice(123)["ok"]
