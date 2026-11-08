"""Zero opponent ratings are missing data, not extremely weak opponents."""
import pandas as pd
import pytest
from src.data import teams
from src.model.model import attach_opponent_strength
from src.model.preseason import preseason_frame, SEED_COLS
from src.ui.insights import forecast_status

FIELDS=teams._STRENGTH_COLS


def archived():
    return pd.DataFrame([dict(id=91+i,short_name=f'T{i}',
                             **{field:1000+100*i+10*j for j,field in enumerate(FIELDS)})
                         for i in [1,2]])


def live_clubs():
    return [dict(id=i,short_name=f'T{i}',**{field:0 for field in FIELDS}) for i in [1,2,3]]


def test_zero_live_ratings_match_previous_club_code_not_changed_numeric_id(monkeypatch):
    monkeypatch.setattr(teams,'_teams_csv',lambda _:archived())
    fixed=teams.resolve_live_teams(live_clubs())
    assert fixed[1]['strength_attack_home'] == 1200
    assert fixed[1]['opponent_rating_source'] == 'previous_season'
    assert fixed[2]['strength_attack_home'] == 1150
    assert fixed[2]['opponent_rating_source'] == 'neutral'
    assert all(team[field]>0 for team in fixed for field in FIELDS)
    assert live_clubs()[0]['strength_attack_home'] == 0  # resolver does not mutate bootstrap


def test_partial_invalid_fields_do_not_overwrite_valid_live_values(monkeypatch):
    monkeypatch.setattr(teams,'_teams_csv',lambda _:archived())
    clubs=live_clubs();clubs[0]['strength_attack_home']=1400
    clubs[0]['strength_defence_home']=float('nan')
    clubs[0]['strength_defence_away']=float('inf')
    fixed=teams.resolve_live_teams(clubs)
    assert fixed[0]['strength_attack_home'] == 1400
    assert fixed[0]['strength_defence_home'] == 1120
    assert fixed[0]['strength_defence_away'] == 1130


def test_missing_ratings_with_no_valid_prior_fail_instead_of_inflating_points(monkeypatch):
    def unavailable(_):
        raise OSError('unavailable')
    monkeypatch.setattr(teams,'_teams_csv',unavailable)
    with pytest.raises(ValueError,match='refusing zero-strength'):
        teams.resolve_live_teams(live_clubs())


def test_valid_live_ratings_require_no_archive_and_are_preserved(monkeypatch):
    monkeypatch.setattr(teams,'_teams_csv',lambda _:pytest.fail('unnecessary archive request'))
    clubs=[dict(id=1,short_name='ARS',**{field:1300 for field in FIELDS})]
    fixed=teams.resolve_live_teams(clubs)
    assert fixed[0]['strength_attack_home']==1300
    assert fixed[0]['opponent_rating_source']=='live'


def test_normal_and_seeded_paths_resolve_the_same_positive_opponent_strength(monkeypatch):
    monkeypatch.setattr(teams,'_teams_csv',lambda _:archived())
    boot=dict(teams=live_clubs(),events=[],elements=[dict(id=1,code=101,web_name='A',
              first_name='A',second_name='',element_type=2,team=1,now_cost=50)])
    raw=pd.DataFrame([dict(season='2026-27',opponent_team=2,was_home=True)])
    raw.attrs['bootstrap']=boot
    normal=attach_opponent_strength(raw)
    seed=pd.DataFrame([dict(code=101,web_name='A',**{c:1. for c in SEED_COLS})])
    seeded=preseason_frame(boot,[dict(id=1,event=11,team_h=1,team_a=2)],gw=11,seed_df=seed)
    assert normal.iloc[0].opp_attack == seeded.iloc[0].opp_attack == 1210
    assert normal.iloc[0].opp_defence == seeded.iloc[0].opp_defence == 1230
    assert seeded.iloc[0].opponent_rating_source == 'previous_season'


def test_forecast_banner_discloses_both_prior_sources():
    html=forecast_status(dict(gw=11,form_gw=9,projection=dict(source='season_model',
                         opponent_ratings=['previous_season','neutral'])))
    assert 'previous-season values' in html
    assert 'Neutral opponent-rating estimates' in html
