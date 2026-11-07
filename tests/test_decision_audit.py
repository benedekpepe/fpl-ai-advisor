"""Decision audit regressions: prices, chips, captains and temporal scoring."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.advisor import personal
from src.ingestion import live
from src.model.model import attach_opponent_strength
from src.model import preseason
from src.config import SQUAD


def squad():
    rows=[]
    for pos, count in SQUAD.items():
        for _ in range(count):
            i=len(rows)
            rows.append(dict(id=i,name=f'{pos}{i}',position=pos,pred=3.,price=50,team=f'T{i%12}'))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('chip',['wildcard','freehit'])
def test_transfer_chips_preserve_banked_count_without_extra_ft(chip):
    hist=dict(current=[dict(event=g,event_transfers=0) for g in range(1,6)], chips=[dict(event=5,name=chip)])
    assert personal.estimate_free_transfers(hist,6,season='2026-27') == 4


def test_afcon_topup_only_applies_to_2025_26_and_late_entries_start_with_one():
    hist=dict(current=[dict(event=g,event_transfers=2) for g in range(1,16)],chips=[])
    assert personal.estimate_free_transfers(hist,16,season='2025-26') == 5
    assert personal.estimate_free_transfers(hist,16,season='2026-27') == 1
    assert personal.estimate_free_transfers(dict(current=[dict(event=5,event_transfers=15)]),6) == 1


def test_freehit_cannot_be_used_in_consecutive_halves_or_opening_gw():
    available,_,_=personal.chips_available(dict(chips=[dict(event=19,name='freehit')]),20)
    assert 'freehit' not in available
    first,_,_=personal.chips_available({},1)
    assert 'wildcard' not in first and 'freehit' not in first


def test_highest_projected_defender_can_captain_and_is_in_options():
    data=squad(); data.loc[2,'pred']=10.
    _, xi=personal.optimize_xi(data)
    assert personal._captain_id(xi) == 2
    assert personal._cap_options(xi)[0]['name'] == data.loc[2,'name']


def test_injured_status_overrides_stale_chance_and_probabilities_are_bounded():
    client=SimpleNamespace(get_bootstrap_static=lambda:dict(elements=[
        dict(id=1,status='i',chance_of_playing_next_round=100),
        dict(id=2,status='s',chance_of_playing_next_round=75),
        dict(id=3,status='d',chance_of_playing_next_round=150),
        dict(id=4,status='d',chance_of_playing_next_round=-10),
    ]))
    assert live.availability(client) == {1:0.,2:0.,3:1.,4:0.}


def test_selling_prices_use_half_profit_and_ignore_temporary_freehit_moves():
    pool=pd.DataFrame(dict(id=[1,2,3],price=[54,47,60]))
    picks=dict(picks=[dict(element=i) for i in [1,2,3]])
    moves=[dict(event=2,element_in=1,element_in_cost=50,time='a'),
           dict(event=2,element_in=2,element_in_cost=50,time='a'),
           dict(event=3,element_in=1,element_in_cost=54,time='b')]
    history=dict(chips=[dict(name='freehit',event=3)],current=[dict(event=1)])
    prices,unknown=personal.selling_prices(picks,pool,moves,history,{},4)
    assert prices == {1:52,2:47,3:60}
    assert unknown == [3]
    exact,_=personal.selling_prices(dict(picks=[dict(element=3,selling_price=55)]),pool,None,{}, {},4)
    assert exact[3] == 55


def test_initial_purchase_price_is_reconstructed_only_with_complete_public_history():
    pool=pd.DataFrame(dict(id=[1],price=[54]))
    picks=dict(picks=[dict(element=1)])
    boot=dict(elements=[dict(id=1,now_cost=54,cost_change_start=4)])
    history=dict(current=[dict(event=1)])
    assert personal.selling_prices(picks,pool,[],history,boot,11) == ({1:52},[])
    assert personal.selling_prices(picks,pool,None,history,boot,11) == ({1:54},[1])


def test_transfer_solver_cannot_spend_unrealised_price_profit():
    current=squad()
    current['budget_price']=50
    current.loc[7,'price']=54
    star=pd.DataFrame([dict(id=99,name='Star',position='MID',pred=20.,price=52,budget_price=52,team='NEW')])
    pool=pd.concat([current,star],ignore_index=True)
    assert personal.plan_exact(set(current.id),pool,0,1) is None
    assert personal.plan_exact(set(current.id),pool,2,1) is not None


def test_chip_timing_preserves_blank_owned_player_and_uses_manager_budget(monkeypatch):
    current=squad(); current['gw']=11
    second=current.copy(); second['gw']=12
    current=current[current.id != 0]  # blank for one keeper
    allpred=pd.concat([current,second],ignore_index=True)
    budgets=[]
    monkeypatch.setattr(personal,'best_squad_value',lambda pool,budget: budgets.append(budget) or 50.)
    timing,valid=personal.chip_timing(set(second.id),allpred,11,['bboost','3xc'],19,750)
    assert valid == [11,12]
    assert budgets == [750,750]
    assert dict(timing['bboost'])[11] == 9.  # 3 bench outfielders, blank reserve GK
    assert dict(timing['bboost'])[12] == 12.


def test_live_strengths_use_bootstrap_without_historical_team_download(monkeypatch):
    raw=pd.DataFrame([dict(season='2026-27',opponent_team=2,was_home=True)])
    raw.attrs['bootstrap']=dict(teams=[dict(id=2,strength_attack_home=1000,strength_attack_away=1100,
                                           strength_defence_home=1200,strength_defence_away=1300)])
    from src.data import teams
    monkeypatch.setattr(teams,'_teams_csv',lambda *_:pytest.fail('live strength used archived CSV'))
    out=attach_opponent_strength(raw)
    assert out.iloc[0].opp_attack == 1100
    assert out.iloc[0].opp_defence == 1300


def test_historical_recency_weight_uses_only_rounds_before_target(monkeypatch):
    raw=pd.DataFrame(dict(season=['2024-25']*3,gw=[1,2,3],minutes=[90]*3))
    feats=pd.DataFrame([dict(season='2024-25',gw=g,element=1,name='A',position='MID',team='ARS',
        value=50,games_played=g-1,cum_ppg=5.,roll_expected_goal_involvements_5=.2,
        roll_minutes_5=90.,starts_rate_5=1.,roll_total_points_3=5.,fixture=g,total_points=5.)
        for g in [2,3,4]])
    model=dict(cols=['value'],clf=SimpleNamespace(predict_proba=lambda x:np.tile([0,1],(len(x),1))),
               reg=SimpleNamespace(predict=lambda x:np.full(len(x),5.)))
    monkeypatch.setattr(personal,'attach_opponent_strength',lambda d:d)
    monkeypatch.setattr(personal,'build_feature_frame',lambda d:feats.copy())
    monkeypatch.setattr(personal.joblib,'load',lambda _:model)
    result=personal._predict_from_history('2024-25',raw)
    assert result.form_weight.tolist() == pytest.approx([.15,.3,.45])

def test_paid_transfer_compares_four_week_total_with_one_off_hit(monkeypatch):
    current=squad()
    star=pd.DataFrame([dict(id=99,name='Star',position='MID',pred=4.,price=50,team='STAR')])
    pool=pd.concat([current,star],ignore_index=True)
    horizon=pd.concat([pool.assign(gw=g) for g in range(11,15)],ignore_index=True)
    monkeypatch.setattr(personal,'DATA_SOURCE','csv')
    monkeypatch.setattr(personal,'get_picks',lambda *_:dict(picks=[dict(element=i) for i in current.id]))
    monkeypatch.setattr(personal,'get_history',lambda *_:{})
    monkeypatch.setattr(personal,'chips_available',lambda *_:([],19,1))
    monkeypatch.setattr(personal,'predict_all',lambda *_:horizon)
    monkeypatch.setattr(personal,'team_fixtures',lambda *_:{})
    monkeypatch.setattr(personal,'load_history',lambda:pd.DataFrame(dict(gw=[11]*16,element=pool.id,fixture=range(16))))
    result=personal.build_advice(123,gw=11,ft_override=0)
    assert result['transfer']['verdict'] == 'TRANSFER'
    assert result['transfer']['hits'] == 1
    assert result['transfer']['net'] == pytest.approx(.36)
    assert 'next 4 GWs' in result['transfer']['text']
    assert 'this week' not in result['transfer']['text']


def test_seeded_horizon_exposes_per_week_forecasts_for_chip_evaluation(monkeypatch):
    from test_projection_integrity import boot,fixtures
    seed=pd.DataFrame([dict(code=i,web_name=f'Player{i}',**{key:1. for key in preseason.SEED_COLS})
                       for i in range(1,4)])
    model=dict(cols=['value'],clf=SimpleNamespace(predict_proba=lambda x:np.tile([0.,1.],(len(x),1))),
               reg=SimpleNamespace(predict=lambda x:np.full(len(x),4.)))
    monkeypatch.setattr(preseason.joblib,'load',lambda _:model)
    result=preseason.preseason_predictions(SimpleNamespace(get_bootstrap_static=boot,get_fixtures=fixtures),
                                          seed_override=seed,gw=11,apply_availability=False)
    weeks=pd.DataFrame(result.attrs['weekly_predictions'])
    assert set(weeks.gw) == {11,12,13,14}
    assert not weeks.duplicated(['gw','id']).any()
    assert weeks.query('gw == 11 and id == 1').pred.item() == 8.
    assert (weeks.query('gw == 12').pred == 0).all()


def test_expiring_chip_uses_positive_gain_instead_of_waiting_past_expiry():
    timing={'3xc':[(19,5.)], 'bboost':[(19,3.)]}
    assert personal.recommend_chip(timing,19) is None
    assert personal.recommend_chip(timing,19,deadline_gw=19) == '3xc'
    assert personal.recommend_chip({'bboost':[(19,0.)]},19,deadline_gw=19) is None


def test_single_week_freehit_does_not_subtract_its_entire_gain_as_typical_gap(monkeypatch):
    data=squad().assign(gw=19)
    monkeypatch.setattr(personal,'best_squad_value',lambda pool,budget:50.)
    timing,_=personal.chip_timing(set(data.id),data,19,['freehit'],19,750)
    assert dict(timing['freehit'])[19] == 14.
