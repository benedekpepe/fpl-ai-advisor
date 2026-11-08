"""Defensive contribution is a lagged input, never an extra double-counted award."""
import numpy as np
import pytest

from src.model.features import build_feature_frame,context_columns,feature_columns
from test_projection_integrity import history


def fixture_history(position='DEF'):
    frame=history()
    frame['season']='2025-26'
    frame['position']=position
    frame['minutes']=[90.,90.,90.,np.nan,np.nan,np.nan,np.nan]
    frame['defensive_contribution']=[10.,20.,9.,np.nan,np.nan,np.nan,np.nan]
    frame['clean_sheets']=[1.,0.,1.,np.nan,np.nan,np.nan,np.nan]
    frame['expected_goals_conceded']=[.5,1.5,.5,np.nan,np.nan,np.nan,np.nan]
    return frame


def test_defensive_points_are_capped_and_lagged_without_changing_total_points():
    raw=fixture_history()
    result=build_feature_frame(raw).set_index('gw')
    assert result.loc[3,'roll_defensive_points_3']==2
    assert result.loc[4,'roll_defensive_points_3']==pytest.approx(4/3)
    assert result.loc[3,'total_points']==raw.loc[2,'total_points']
    assert result.loc[4:,'roll_defensive_points_3'].nunique()==1
    assert result.loc[3,'roll_clean_sheets_3']==.5


@pytest.mark.parametrize('position', ['MID','FWD'])
def test_midfielders_and_forwards_use_twelve_action_threshold(position):
    result=build_feature_frame(fixture_history(position)).set_index('gw')
    assert result.loc[3,'roll_defensive_points_3']==1


def test_unknown_and_old_rule_contributions_are_not_fabricated_zeroes():
    old=fixture_history();old['season']='2024-25'
    assert build_feature_frame(old).roll_defensive_points_3.isna().all()
    missing=fixture_history().drop(columns=['defensive_contribution'])
    assert build_feature_frame(missing).roll_defensive_points_3.isna().all()
    assert not set(context_columns()) & set(feature_columns())


def test_live_ingestion_keeps_defensive_stats_and_leaves_future_values_unknown():
    from src.ingestion.live import current_season_frame
    class Client:
        def get_bootstrap_static(self):
            return dict(teams=[dict(id=1,short_name='ARS')],
                        elements=[dict(id=31,web_name='Player',element_type=2,team=1,now_cost=48)],
                        events=[dict(id=1,deadline_time='2020-01-01T12:00:00Z',finished=True,data_checked=True),
                                dict(id=2,deadline_time='2099-01-01T12:00:00Z')])
        def get_element_summary(self,pid):
            return dict(history=[dict(round=1,fixture=1,minutes=90,defensive_contribution='15',
                                      clearances_blocks_interceptions='10',tackles='5',recoveries='5')],
                        fixtures=[dict(event=2,id=2,is_home=True,team_h=1,team_a=2)])
    frame=current_season_frame(Client()).set_index('gw')
    assert frame.loc[1,'defensive_contribution']==15
    assert frame.loc[1,'clearances_blocks_interceptions']==10
    assert np.isnan(frame.loc[2,'defensive_contribution'])
