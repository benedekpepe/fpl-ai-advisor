"""Forecast integrity: double/blank weeks, forward form and escaped UI."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import preseason
from src.model.features import build_feature_frame, ROLL_BASES
from src.advisor.personal import _blend_pred, selection_explanations
from src.ui.insights import lineup_insights, forecast_status, validation_panel


def boot():
    teams = [dict(id=i, short_name=f'T{i}', strength_attack_home=1000,
                  strength_attack_away=1000, strength_defence_home=1000,
                  strength_defence_away=1000) for i in range(1, 4)]
    elements = [dict(id=i, code=i, web_name=f'Player{i}', first_name=f'Player{i}',
                     second_name='', element_type=2, team=i, now_cost=50) for i in range(1, 4)]
    return dict(teams=teams, elements=elements, events=[])


def fixtures():
    return [dict(id=1, event=11, team_h=1, team_a=2),
            dict(id=2, event=11, team_h=3, team_a=1),
            dict(id=3, event=13, team_h=2, team_a=3)]


def test_seed_sums_double_matches_keeps_blank_players_and_zero_weeks(monkeypatch):
    seed = pd.DataFrame([dict(code=i, web_name=f'Player{i}', **{key: 1.0 for key in preseason.SEED_COLS})
                         for i in range(1, 4)])
    client = SimpleNamespace(get_bootstrap_static=boot, get_fixtures=fixtures)
    model = dict(cols=['value'], clf=SimpleNamespace(predict_proba=lambda x: np.tile([0., 1.], (len(x), 1))),
                 reg=SimpleNamespace(predict=lambda x: np.full(len(x), 4.0)))
    monkeypatch.setattr(preseason.joblib, 'load', lambda _: model)
    predictions = preseason.preseason_predictions(client, seed_override=seed, gw=11, apply_availability=False).set_index('id')
    assert predictions.index.is_unique
    assert len(predictions) == 3
    assert predictions.loc[1, 'pred_gw1'] == 8.0
    assert predictions.loc[1, 'fixture_count'] == 2
    assert predictions.loc[1, 'pred'] == pytest.approx(8 / sum(preseason.FIXTURE_WEIGHTS))
    assert predictions.loc[2, 'pred'] == pytest.approx((4 + 4*.36) / sum(preseason.FIXTURE_WEIGHTS))
    blank = preseason.preseason_predictions(client, seed_override=seed, gw=12, apply_availability=False)
    assert len(blank) == 3
    assert (blank.pred_gw1 == 0).all()
    assert (blank.fixture_count == 0).all()
    last_week = preseason.preseason_predictions(client, seed_override=seed, gw=38, apply_availability=False)
    assert (last_week.pred == 0).all()


def test_ticker_has_one_cell_per_week_with_double_and_blank():
    ticker = preseason.fixture_ticker(fixtures(), boot()['teams'], 11, 4)
    assert [cell['gw'] for cell in ticker[1]] == [11, 12, 13, 14]
    assert ticker[1][0]['opp'] == 'T2 + T3'
    assert ticker[1][0]['count'] == 2
    assert ticker[1][1]['opp'] == 'BLANK'
    assert ticker[1][1]['count'] == 0
    assert ticker[1][1]['diff'] == 0


def test_weighted_average_counts_missing_blank_week_and_truncates_at_season_end():
    data = pd.DataFrame([dict(id=1, gw=11, pred=8.), dict(id=1, gw=13, pred=4.)])
    assert _blend_pred(data, 11)[1] == pytest.approx((8 + 4*.36)/sum(preseason.FIXTURE_WEIGHTS))
    assert _blend_pred(pd.DataFrame([dict(id=1, gw=38, pred=5.)]), 38)[1] == 5


def history():
    rows=[]
    for i, points in enumerate([2, 4, 8, None, None, None, None], start=1):
        row=dict(season='2024-25', element=1, gw=i, fixture=i,
                 kickoff_time=f'2024-09-{i:02d}T12:00:00Z', position='MID',
                 starts=1 if points is not None else np.nan, was_home=True)
        row.update({b: float(points) if points is not None else np.nan for b in ROLL_BASES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_future_rows_do_not_erode_observed_rolling_form():
    data=history()
    result=build_feature_frame(data)
    future=result[result.gw >= 4]
    assert future.roll_total_points_3.tolist() == pytest.approx([14/3]*4)
    assert future.games_played.tolist() == [3]*4
    assert future.starts_rate_5.tolist() == [1]*4


def test_second_double_fixture_cannot_see_first_fixtures_result():
    data=history().iloc[:3].copy()
    second=data.iloc[-1].copy()
    second['fixture']=99
    second['kickoff_time']='2024-09-04T12:00:00Z'
    data=pd.concat([data,pd.DataFrame([second])],ignore_index=True)
    result=build_feature_frame(data)
    assert result[result.gw == 3].roll_total_points_3.tolist() == [3.,3.]


def test_ui_escapes_names_discloses_fallback_and_unavailable_metrics():
    advice=dict(gw=11, form_gw=9, projection=dict(source='current_form_seed', reason='Seeded estimates.'),
                current_xi=[dict(name='<script>alert(1)</script>', role='start', pred=4., position='DEF')],
                selection=[dict(starter='<img src=x>', bench='A & B', gap=.1)])
    html=forecast_status(advice)+lineup_insights(advice)
    assert '<script>' not in html
    assert '<img src=x>' not in html
    assert '&lt;script&gt;' in html
    assert 'Seeded from current form' in html
    assert '+0.10 pts' in html
    assert '—' in html
    assert validation_panel(Path('missing-validation-report.json')) == ''


def test_validation_rejects_training_on_test_or_future_season():
    from src.model.validate import evaluate
    with pytest.raises(ValueError, match='precede'):
        evaluate(pd.DataFrame(), pd.DataFrame(), '2024-25', '2024-25')
    with pytest.raises(ValueError, match='precede'):
        evaluate(pd.DataFrame(), pd.DataFrame(), '2025-26', '2024-25')
