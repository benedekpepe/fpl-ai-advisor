"""Deadline-safe forecast snapshots and retrospective evaluation.

Snapshots are append-only exports, not hindsight backtests. No cloud storage or
scheduler is assumed. Configure FPL_FORECAST_LOG_DIR only on persistent storage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
import os
import re
from pathlib import Path
from statistics import mean


def utc(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00')) if isinstance(value,str) else value
    if result.tzinfo is None:
        raise ValueError('A timezone-aware timestamp is required.')
    return result.astimezone(timezone.utc)


def digest(payload):
    data={k:v for k,v in payload.items() if k!='snapshot_id'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def number(value):
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (TypeError,ValueError):
        return None


def create_snapshot(advice, forecasts, deadline, *, captured_at=None, model_path=None):
    """Freeze exactly the shown decision before its target deadline."""
    now=utc(captured_at or datetime.now(timezone.utc)); deadline=utc(deadline)
    if now>=deadline:
        raise ValueError('Cannot save a prospective forecast after the target deadline.')
    from src.config import FIXTURE_WEIGHTS
    gw=int(advice['gw']); horizon=list(range(gw,min(39,gw+len(FIXTURE_WEIGHTS))))
    records=[]
    for row in forecasts.to_dict('records'):
        if int(row['gw']) not in horizon:
            continue
        record={k:row.get(k) for k in ['name','position','team','opponent_rating_source']}
        record.update(id=int(row['id']),gw=int(row['gw']))
        for key in ['pred','p_play','recent_minutes','fixture_count','minutes_p_play','p_start','p_sixty','p_cameo','expected_minutes']:
            record[key]=number(row.get(key))
        if record['pred'] is not None:
            records.append(record)
    if not records or len({(r['gw'],r['id']) for r in records})!=len(records):
        raise ValueError('Forecast records must be nonempty and unique per player/gameweek.')
    def lineup(rows):
        if len(rows)!=15 or any('id' not in r for r in rows):
            raise ValueError('A snapshot needs all 15 stable player IDs.')
        return [{k:r.get(k) for k in ['id','name','position','role','captain','vice']} for r in rows]
    before=lineup(advice['current_xi'])
    transfer=advice['transfer']
    after=transfer.get('post_xi') if transfer['verdict']=='TRANSFER' else None
    if transfer['verdict']=='CHIP':
        after=(advice['chips'].get('detail') or {}).get('squad')
    snapshot=dict(schema=1,season=advice['season'],target_gw=gw,team_id=int(advice['team_id']),
        captured_at=now.isoformat(),deadline=deadline.isoformat(),form_gw=advice.get('form_gw'),
        projection=advice.get('projection',{}),horizon_weights={str(g):FIXTURE_WEIGHTS[i] for i,g in enumerate(horizon)},
        predictions=sorted(records,key=lambda r:(r['gw'],r['id'])),before=before,
        after=lineup(after) if after else before,
        recommendation=dict(verdict=transfer['verdict'],hits=int(transfer.get('hits',0)),
                            moves=transfer.get('moves',[])),chip=advice['chips'].get('recommended'))
    snapshot['model_sha256']=hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None
    root=Path(__file__).resolve().parents[2]
    pipeline=hashlib.sha256()
    for filename in ['src/advisor/personal.py','src/model/playing_time.py','src/model/features.py','src/model/projections.py','src/data/teams.py','src/config.py']:
        pipeline.update((root/filename).read_bytes())
    snapshot['pipeline_sha256']=pipeline.hexdigest()
    from src.model.playing_time import ARTIFACT
    snapshot['minutes_model_sha256']=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() if ARTIFACT.exists() else None
    snapshot['snapshot_id']=digest(snapshot)
    return snapshot


def parse_snapshot(data):
    if isinstance(data,bytes):
        if len(data)>10_000_000:
            raise ValueError('Snapshot exceeds the 10 MB limit.')
        data=data.decode('utf-8')
    snapshot=json.loads(data) if isinstance(data,str) else data
    if not isinstance(snapshot,dict):raise ValueError('Snapshot must be a JSON object.')
    try:
        if snapshot.get('schema')!=1 or snapshot.get('snapshot_id')!=digest(snapshot):
            raise ValueError('Snapshot format/checksum is invalid.')
        if utc(snapshot['captured_at'])>=utc(snapshot['deadline']):
            raise ValueError('This forecast was not captured before its deadline.')
        if not 1<=int(snapshot['target_gw'])<=38 or not isinstance(snapshot['predictions'],list):
            raise ValueError('Invalid snapshot gameweek or predictions.')
        if not isinstance(snapshot.get('season'),str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}',snapshot['season']):
            raise ValueError('Invalid snapshot season.')
    except (KeyError,AttributeError,TypeError) as exc:
        raise ValueError('Snapshot fields are invalid or missing.') from exc

    return snapshot


def save_snapshot(snapshot, directory):
    parse_snapshot(snapshot)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    filename=f"{snapshot['season']}-gw{snapshot['target_gw']:02d}-{snapshot['snapshot_id'][:16]}.json"
    path=directory/filename
    payload=json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        with path.open('x',encoding='utf-8') as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_text(encoding='utf-8')!=payload:
            raise ValueError('An existing snapshot cannot be overwritten.')
    return path


def metrics(rows):
    residuals=[r['predicted']-r['actual'] for r in rows]
    return dict(n=len(rows),mae=round(mean(abs(r) for r in residuals),4),
                rmse=round(math.sqrt(mean(r*r for r in residuals)),4),bias=round(mean(residuals),4)) if rows else None


def frozen_lineup_points(rows,outcomes,chip=None):
    """Actual points for the saved lineup, legal automatic subs and vice-captain."""
    from src.config import XI_MIN,XI_MAX
    def played(row): return outcomes[int(row['id'])]['minutes']>0
    starters=[r for r in rows if r['role']=='start']
    if len(starters)!=11 or any(int(r['id']) not in outcomes for r in rows):
        raise ValueError('All 15 outcomes and a valid starting XI are required.')
    if chip=='bboost':
        scoring=list(rows)
    else:
        keeper=next(r for r in starters if r['position']=='GK')
        scoring=[r for r in starters if r['position']!='GK']
        if not played(keeper):
            keeper=next((r for r in rows if r['role']=='benchgk' and played(r)),keeper)
        missing=[r for r in scoring if not played(r)]
        bench=[r for r in rows if r['role']=='bench' and played(r)]
        best=None
        for size in range(min(len(bench),len(missing))+1):
            for indices in combinations(range(len(bench)),size):
                subs=[bench[i] for i in indices]
                for removed in combinations(missing,size):
                    selected=[r for r in scoring if r not in removed]+subs+[keeper]
                    counts={p:sum(r['position']==p for r in selected) for p in XI_MIN}
                    if all(XI_MIN[p]<=counts[p]<=XI_MAX[p] for p in XI_MIN):
                        priority=(size,tuple(int(i in indices) for i in range(len(bench))))
                        if best is None or priority>best[0]:best=(priority,selected)
        scoring=best[1] if best else scoring+[keeper]
    total=sum(outcomes[int(r['id'])]['points'] for r in scoring)
    cap=next((r for r in rows if r.get('captain')),None)
    if cap is None or not played(cap):cap=next((r for r in rows if r.get('vice') and played(r)),None)
    if cap is not None and cap in scoring:total+=outcomes[int(cap['id'])]['points']*(2 if chip=='3xc' else 1)
    return total


def evaluate_snapshot(snapshot,outcomes,completed_gws):
    """Never score unchecked/in-progress rounds or treat missing results as zero."""
    snapshot=parse_snapshot(snapshot); completed=set(completed_gws)
    indexed={(int(r['gw']),int(r['id'])):r for r in outcomes}
    predicted={(p['gw'],p['id']):p for p in snapshot['predictions']}
    def outcome(gw,pid):
        if (gw,pid) in indexed:return indexed[(gw,pid)]
        pred=predicted.get((gw,pid))
        return dict(points=0,minutes=0) if pred and pred.get('fixture_count')==0 else None
    rows=[];missing=0
    for p in snapshot['predictions']:
        if p['gw'] not in completed:continue
        actual=outcome(p['gw'],p['id'])
        if actual is None:missing+=1;continue
        rows.append(dict(gw=p['gw'],id=p['id'],name=p['name'],position=p['position'],
                         predicted=p['pred'],actual=actual['points'],minutes=actual['minutes'],
                         p_play=p.get('p_play'),recent_minutes=p.get('recent_minutes'),fixture_count=p.get('fixture_count'),
                         starts=actual.get('starts'),
                         **{k:p.get(k) for k in ['minutes_p_play','p_start','p_sixty','p_cameo','expected_minutes']}))
    target=snapshot['target_gw'];target_rows=[r for r in rows if r['gw']==target]
    positions={p:metrics([r for r in target_rows if r['position']==p]) for p in ['GK','DEF','MID','FWD']}
    single=[r for r in target_rows if r['fixture_count']==1 and r['p_play'] is not None]
    guide=[r for r in target_rows if r['fixture_count']==1 and r['recent_minutes'] is not None]
    diagnostics={}
    for name,observed_event in [('p_start',lambda r:r['starts']>0),('p_sixty',lambda r:r['minutes']>=60),
                        ('p_cameo',lambda r:r['minutes']>0 and r['starts']==0)]:
        valid=[r for r in target_rows if r['fixture_count']==1 and r[name] is not None and
               (name=='p_sixty' or r['starts'] is not None)]
        diagnostics[name]=dict(n=len(valid),brier=round(mean((r[name]-int(observed_event(r)))**2 for r in valid),4)) if valid else None
    minute_rows=[r for r in target_rows if r['fixture_count']==1 and r['expected_minutes'] is not None]
    diagnostics['expected_minutes']=dict(n=len(minute_rows),mae=round(mean(abs(r['expected_minutes']-r['minutes']) for r in minute_rows),4)) if minute_rows else None
    lineup=None
    if target in completed:
        ids={int(r['id']) for r in snapshot['before']+snapshot['after']}
        actual={pid:outcome(target,pid) for pid in ids}
        if all(r is not None for r in actual.values()):
            before=frozen_lineup_points(snapshot['before'],actual)
            after_without_chip=frozen_lineup_points(snapshot['after'],actual)
            after=frozen_lineup_points(snapshot['after'],actual,snapshot['chip'])
            hit=4*snapshot['recommendation']['hits']
            lineup=dict(before=before,after=after,hit_cost=hit,net=after-before-hit,
                        after_without_chip=after_without_chip,
                        team_gain=after_without_chip-before,chip_gain=after-after_without_chip)
    contribution=[]
    moves=snapshot['recommendation']['moves']
    for g,weight in snapshot['horizon_weights'].items():
        gw=int(g)
        if gw not in completed:continue
        if moves and all('in_id' in m and 'out_id' in m and outcome(gw,m['in_id']) is not None and outcome(gw,m['out_id']) is not None for m in moves):
            gain=sum(outcome(gw,m['in_id'])['points']-outcome(gw,m['out_id'])['points'] for m in moves)
            contribution.append(dict(gw=gw,gain=gain,discounted_gain=gain*weight))
    complete=len(contribution)==len(snapshot['horizon_weights']) if moves else False
    return dict(snapshot_id=snapshot['snapshot_id'],target_gw=target,
        status='pending' if target not in completed else ('complete' if len(target_rows)==len([p for p in snapshot['predictions'] if p['gw']==target]) else 'partial'),
        playing_time=diagnostics,target_metrics=metrics(target_rows),by_position=positions,missing_results=missing,
        play_brier=round(mean((r['p_play']-int(r['minutes']>0))**2 for r in single),4) if single else None,
        recent_minutes_guide_mae=round(mean(abs(r['recent_minutes']-r['minutes']) for r in guide),4) if guide else None,
        frozen_lineup=lineup,transfer_contribution=dict(weeks=contribution,complete=complete,
            net=round(sum(r['discounted_gain'] for r in contribution)-4*snapshot['recommendation']['hits'],4) if complete else None),
        limitations='Frozen target-week lineups include automatic substitutes and vice-captain, but not later manual decisions. The baseline has no new chip; team_gain holds chip scoring off and chip_gain applies it to the suggested lineup. Wildcard/Free Hit roster changes are included in team_gain, not valued separately as future chip opportunities. Multiweek transfer contribution compares ownership points, not XI/captain effects. Brier/observed-minute guide metrics exclude doubles. Snapshots are checksummed exports, not digitally authenticated records.')


def fetch_outcomes(snapshot,client):
    snapshot=parse_snapshot(snapshot);boot=client.get_bootstrap_static()
    first=min(boot['events'],key=lambda e:e['id'])
    if str(first.get('deadline_time',''))[:4]!=snapshot['season'][:4]:
        raise ValueError('Live API season differs from the snapshot; use archived outcome data.')
    completed={e['id'] for e in boot['events'] if e.get('finished') and e.get('data_checked')}
    settled=completed&{int(g) for g in snapshot['horizon_weights']}
    if not settled:return [],completed
    ids={p['id'] for p in snapshot['predictions'] if p['gw'] in settled}
    def fetch(pid):
        try:return pid,client.get_element_summary(pid).get('history',[])
        except Exception:return pid,[]
    collected=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pid,history in pool.map(fetch,sorted(ids)):
            for gw in sorted(settled):
                matches=[h for h in history if h.get('round')==gw]
                if matches:collected.append(dict(id=pid,gw=gw,points=sum(h['total_points'] for h in matches),minutes=sum(h['minutes'] for h in matches),
                                                starts=sum(h['starts'] for h in matches) if all('starts' in h for h in matches) else None))
    return collected,completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    capture=sub.add_parser('capture');capture.add_argument('--team-id',type=int,required=True);capture.add_argument('--directory',type=Path,default=Path('data/forecast_logs'))
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('snapshot',type=Path);evaluate.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='capture':
        os.environ['FPL_DATA_SOURCE']='live'
        from src.advisor.personal import build_advice
        advice=build_advice(args.team_id)
        if not advice.get('tracking_snapshot'):raise ValueError(advice.get('error') or advice.get('tracking_error') or 'No prospective snapshot was generated.')
        print(save_snapshot(advice['tracking_snapshot'],args.directory))
    else:
        from src.ingestion.fpl_client import FPLClient
        snapshot=parse_snapshot(args.snapshot.read_bytes())
        outcomes,completed=fetch_outcomes(snapshot,FPLClient())
        report=evaluate_snapshot(snapshot,outcomes,completed)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
