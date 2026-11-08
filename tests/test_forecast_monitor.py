"""Prospective scoring must not use late forecasts, partial results or hindsight."""
from datetime import datetime,timezone
import copy
import json
import pandas as pd
import pytest
from src.model.monitor import create_snapshot,parse_snapshot,save_snapshot,evaluate_snapshot,frozen_lineup_points,digest,fetch_outcomes


def lineup():
    positions=['GK']*2+['DEF']*5+['MID']*5+['FWD']*3
    starters={0,2,3,4,7,8,9,10,12,13,14}
    order=[0,2,3,4,7,8,9,10,12,13,14,1,5,6,11]
    return [dict(id=i,name=f'P{i}',position=positions[i],role='start' if i in starters else ('benchgk' if i==1 else 'bench'),captain=i==12,vice=i==7) for i in order]


def snapshot():
    before=lineup();after=copy.deepcopy(before)
    for row in after:
        if row['id']==12:row.update(id=99,name='New')
    advice=dict(gw=11,season='2026-27',team_id=123,form_gw=9,current_xi=before,
                projection=dict(source='season_model'),chips=dict(recommended=None),
                transfer=dict(verdict='TRANSFER',hits=1,moves=[dict(out_id=12,in_id=99)],post_xi=after))
    records=[dict(id=i,name=f'P{i}',position='FWD' if i in [12,99] else 'DEF',team='T',gw=g,
                  pred=3.,p_play=.75,recent_minutes=80.,fixture_count=1) for g in range(11,15) for i in list(range(15))+[99]]
    return create_snapshot(advice,pd.DataFrame(records),'2026-11-14T11:00:00Z',captured_at=datetime(2026,11,8,tzinfo=timezone.utc))


def outcomes(gws=(11,)):
    return [dict(id=i,gw=g,points=8 if i==99 else 2,minutes=90) for g in gws for i in list(range(15))+[99]]


def test_snapshot_checksum_and_deadline_reject_tampering_and_late_capture():
    saved=snapshot();assert parse_snapshot(json.dumps(saved))['target_gw']==11
    saved['predictions'][0]['pred']=99
    with pytest.raises(ValueError,match='checksum'):parse_snapshot(saved)
    saved=snapshot();saved['captured_at']=saved['deadline'];saved['snapshot_id']=digest(saved)
    with pytest.raises(ValueError,match='before'):parse_snapshot(saved)
    with pytest.raises(ValueError):parse_snapshot('[]')


def test_snapshot_storage_is_append_only_and_idempotent(tmp_path):
    saved=snapshot();first=save_snapshot(saved,tmp_path);again=save_snapshot(saved,tmp_path)
    assert first==again and len(list(tmp_path.glob('*.json')))==1
    first.write_text('corrupt',encoding='utf-8')
    with pytest.raises(ValueError,match='overwritten'):save_snapshot(saved,tmp_path)


def test_unchecked_round_cannot_be_scored_even_when_stats_are_present():
    report=evaluate_snapshot(snapshot(),outcomes(),{9})
    assert report['status']=='pending'
    assert report['target_metrics'] is None
    assert report['frozen_lineup'] is None
    assert report['transfer_contribution']['net'] is None


def test_checked_results_use_saved_lineups_hits_and_wait_for_whole_horizon():
    saved=snapshot();report=evaluate_snapshot(saved,outcomes(),{11})
    assert report['status']=='complete'
    assert report['frozen_lineup']==dict(before=24,after=36,hit_cost=4,net=8)
    assert report['transfer_contribution']['complete'] is False
    assert report['transfer_contribution']['net'] is None
    full=evaluate_snapshot(saved,outcomes((11,12,13,14)),{11,12,13,14})
    assert full['transfer_contribution']['complete'] is True
    assert full['transfer_contribution']['net']==pytest.approx(6*2.18-4)


def test_missing_result_is_not_a_zero_but_known_blank_is_zero():
    saved=snapshot();partial=[r for r in outcomes() if r['id']!=0]
    assert evaluate_snapshot(saved,partial,{11})['status']=='partial'
    for row in saved['predictions']:
        if row['id']==0:row['fixture_count']=0;row['pred']=0
    saved['snapshot_id']=digest(saved)
    assert evaluate_snapshot(saved,partial,{11})['status']=='complete'


def test_autosubs_preserve_defender_minimum_and_bench_priority():
    rows=lineup();rows=[r for r in rows if r['role']!='bench']+[r for pid in [11,5,6] for r in rows if r['id']==pid]
    actual={i:dict(points=2,minutes=90) for i in range(15)}
    actual[2]=dict(points=0,minutes=0);actual[11]=dict(points=20,minutes=90);actual[5]=dict(points=8,minutes=90)
    assert frozen_lineup_points(rows,actual)==30  # MID cannot replace the missing third DEF.


def test_captain_falls_back_to_vice_and_goalkeeper_is_replaced():
    rows=lineup();actual={i:dict(points=2,minutes=90) for i in range(15)}
    actual[0]=dict(points=0,minutes=0);actual[12]=dict(points=0,minutes=0)
    assert frozen_lineup_points(rows,actual)==24


def test_double_gameweek_excluded_from_single_match_probability_score():
    saved=snapshot()
    for p in saved['predictions']:p['fixture_count']=2
    saved['snapshot_id']=digest(saved)
    report=evaluate_snapshot(saved,outcomes(),{11})
    assert report['play_brier'] is None
    assert report['recent_minutes_guide_mae'] is None


def test_wrong_live_season_is_rejected_before_fetching_players():
    class Client:
        def get_bootstrap_static(self):return dict(events=[dict(id=1,deadline_time='2027-08-01T12:00:00Z')])
        def get_element_summary(self,_):pytest.fail('wrong season fetched')
    with pytest.raises(ValueError,match='differs'):fetch_outcomes(snapshot(),Client())
