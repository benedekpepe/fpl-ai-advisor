"""Playing-time probabilities stay coherent and never change point forecasts."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from src.model.playing_time import predict, enrich, softmax_temperature
from src.model.features import build_feature_frame
from src.ui.insights import projection_details
from test_projection_integrity import history


def constant(value):
    return SimpleNamespace(predict=lambda x: np.full(len(x), value))


def bundle():
    return dict(cols=['value'], temperature=1., start_temperature=1.,
                state=SimpleNamespace(predict_proba=lambda x: np.tile([.2,.3,.5],(len(x),1))),
                start=SimpleNamespace(predict_proba=lambda x: np.tile([.25,.75],(len(x),1))),
                models=dict(minutes_1=constant(120), minutes_2=constant(-1),
                            points_1=constant(1),points_2=constant(5)))


def test_hierarchical_probabilities_and_bounded_conditional_minutes():
    scored=predict(pd.DataFrame({'value':[50,60]}),bundle())
    assert (scored.p_start<=scored.minutes_p_play).all()
    assert (scored.p_sixty<=scored.minutes_p_play).all()
    assert scored.p_start.iloc[0]==pytest.approx(.6)
    assert scored.p_cameo.iloc[0]==pytest.approx(.2)
    assert scored.expected_minutes.iloc[0]==pytest.approx(.3*59+.5*60)
    assert scored.state_points.iloc[0]==pytest.approx(2.8)


def test_temperature_calibration_preserves_total_probability():
    p=softmax_temperature(np.array([[.001,.2,.799],[0,0,1]]),1.5)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1),1)


def test_missing_optional_artifact_preserves_forecasts(tmp_path):
    frame=pd.DataFrame({'pred':[4.]})
    assert enrich(frame,tmp_path/'missing.pkl') is frame


def test_broken_optional_artifact_does_not_break_point_advice(tmp_path):
    broken=tmp_path/'broken.pkl';broken.write_bytes(b'not a model')
    frame=pd.DataFrame({'pred':[4.]})
    assert enrich(frame,broken) is frame


def test_minutes_rates_are_lagged_and_future_rows_do_not_consume_form():
    raw=history();raw.loc[:2,'minutes']=[90.,20.,80.]
    raw.loc[:2,'starts']=[1.,0.,1.]
    result=build_feature_frame(raw).set_index('gw')
    assert result.loc[3,'sixty_rate_5']==pytest.approx(.5)
    assert result.loc[3,'cameo_rate_5']==pytest.approx(.5)
    assert result.loc[4:,'sixty_rate_5'].nunique()==1
    changed=raw.copy();changed.loc[2,'minutes']=0
    other=build_feature_frame(changed).set_index('gw')
    assert other.loc[3,'sixty_rate_5']==result.loc[3,'sixty_rate_5']


def test_double_matches_share_pre_round_minutes_rates():
    raw=history();second=raw.iloc[[2]].copy();second['fixture']=99
    second['kickoff_time']='2024-09-03T18:00:00Z'
    result=build_feature_frame(pd.concat([raw,second],ignore_index=True))
    assert result.loc[result.gw==3,'sixty_rate_5'].nunique()==1


def test_probability_details_show_model_and_observed_signals_separately():
    row=dict(name='<Player>',role='start',pred=4.,minutes_basis='playing_time_model',
             p_start=.8,p_sixty=.7,p_cameo=.1,expected_minutes=65.,recent_minutes=60.)
    html=projection_details([row])
    assert 'Start / match' in html and '60+ min / match' in html
    assert 'Estimated minutes / GW' in html and 'point forecast unchanged' in html
    assert '&lt;Player&gt;' in html and '<Player>' not in html
