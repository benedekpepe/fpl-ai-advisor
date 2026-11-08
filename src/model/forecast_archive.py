"""Rolling 90-day forecast ledger backed by GitHub Actions artifacts.

No credentials are stored in snapshots. Restoration errors fail the job rather
than silently publishing an empty replacement ledger.
"""
import argparse
from datetime import datetime,timedelta,timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import zipfile

import requests

from src.model.monitor import parse_snapshot,save_snapshot,utc,fetch_outcomes,evaluate_snapshot

ARTIFACT_NAME='forecast-ledger'
MAX_ARCHIVE_BYTES=150_000_000


def import_zip(payload,directory):
    """Read only validated JSON snapshots; never extract arbitrary ZIP paths."""
    if len(payload)>MAX_ARCHIVE_BYTES:raise ValueError('Archive is too large.')
    result=[]
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if sum(item.file_size for item in archive.infolist())>MAX_ARCHIVE_BYTES:
            raise ValueError('Expanded archive is too large.')
        for item in archive.infolist():
            if item.is_dir():continue
            if not re.fullmatch(r'snapshots/[0-9]{4}-[0-9]{2}-gw[0-9]{2}-[0-9a-f]{16}\.json',item.filename):
                continue
            if item.file_size>10_000_000:raise ValueError('Snapshot is too large.')
            snapshot=parse_snapshot(archive.read(item))
            result.append(save_snapshot(snapshot,Path(directory)/'snapshots'))
    return result


def restore(directory,repository,token,session=None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):
        raise ValueError('Invalid GitHub repository.')
    session=session or requests.Session()
    headers={'Accept':'application/vnd.github+json','Authorization':f'Bearer {token}',
             'X-GitHub-Api-Version':'2022-11-28'}
    base=f'https://api.github.com/repos/{repository}/actions/artifacts'
    response=session.get(base,params={'name':ARTIFACT_NAME,'per_page':100},headers=headers,timeout=30)
    response.raise_for_status()
    available=[a for a in response.json()['artifacts'] if not a['expired'] and
               a.get('workflow_run',{}).get('head_branch')==os.getenv('FPL_ARCHIVE_BRANCH','main')]
    if not available:return 0
    artifact=max(available,key=lambda a:(a['created_at'],a['id']))
    download=session.get(f"{base}/{int(artifact['id'])}/zip",headers=headers,timeout=90)
    download.raise_for_status()
    return len(import_zip(download.content,directory))


def load_snapshots(directory,now=None):
    """Prune by original capture time, not by the latest artifact upload date."""
    now=utc(now or datetime.now(timezone.utc));cutoff=now-timedelta(days=90)
    snapshots=[]
    for path in sorted((Path(directory)/'snapshots').glob('*.json')):
        snapshot=parse_snapshot(path.read_bytes())
        captured=utc(snapshot['captured_at'])
        if captured<cutoff:
            path.unlink()  # Only individual files in the explicitly selected archive.
        elif captured>now:
            raise ValueError('Archive contains a future-dated snapshot.')
        else:snapshots.append(snapshot)
    return snapshots


def latest_per_round(snapshots):
    selected={}
    for s in snapshots:
        key=(s['season'],s['team_id'],s['target_gw'])
        if key not in selected or (utc(s['captured_at']),s['snapshot_id'])>(utc(selected[key]['captured_at']),selected[key]['snapshot_id']):
            selected[key]=s
    return [selected[key] for key in sorted(selected)]


def should_capture(snapshots,team_id,season,event,now):
    relevant=[s for s in snapshots if s['team_id']==team_id and s['season']==season and s['target_gw']==event['id']]
    remaining=utc(event['deadline_time'])-utc(now)
    if remaining.total_seconds()<=0:return False
    if not relevant:return True
    return remaining<=timedelta(hours=48) and utc(now)-max(utc(s['captured_at']) for s in relevant)>=timedelta(hours=3)


def run(directory,team_id,client,capture,now=None):
    now=utc(now or datetime.now(timezone.utc));directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    snapshots=load_snapshots(directory,now)
    boot=client.get_bootstrap_static()
    first=min(boot['events'],key=lambda e:e['id'])
    year=int(first['deadline_time'][:4]);season=f'{year}-{str(year+1)[-2:]}'
    upcoming=[e for e in boot['events'] if utc(e['deadline_time'])>now]
    target=min(upcoming,key=lambda e:utc(e['deadline_time'])) if upcoming else None
    captured=False
    if target and should_capture(snapshots,team_id,season,target,now):
        advice=capture(team_id)
        if not advice.get('tracking_snapshot'):raise ValueError(advice.get('tracking_error') or advice.get('error') or 'No snapshot produced.')
        saved=advice['tracking_snapshot']
        if saved['target_gw']!=target['id'] or saved['team_id']!=team_id or saved['season']!=season:
            raise ValueError('Capture target changed; retry with fresh live data.')
        save_snapshot(saved,directory/'snapshots');snapshots.append(saved);captured=True
    selected=latest_per_round(snapshots)
    reports=[]
    # Reuse outcomes across the handful of retained gameweek snapshots.
    cache={}
    class CachedClient:
        def get_bootstrap_static(self):return boot
        def get_element_summary(self,pid):
            if pid not in cache:cache[pid]=client.get_element_summary(pid)
            return cache[pid]
    for saved in selected:
        if saved['season']!=season:continue
        # Older complete reports remain in the bundle; only the latest per GW is scored.
        outcomes,completed=fetch_outcomes(saved,CachedClient())
        report=evaluate_snapshot(saved,outcomes,completed)
        reports.append(dict(team_id=saved['team_id'],season=season,**report))
    result=dict(updated_at=now.isoformat(),retention_days=90,captured=captured,
                snapshots=len(snapshots),selection='Latest pre-deadline snapshot per team/season/gameweek; no hindsight selection.',
                reports=reports,limitations='Public artifacts; rolling 90-day retention, not permanent storage. GitHub schedules can be delayed or disabled after inactivity. Missing rounds/results stay pending or partial.')
    (directory/'review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Forecast archive','',f'Updated: {now.isoformat()}',f'Snapshots retained: {len(snapshots)}','',
           '| GW | Team | Status | Team changes | Chip bonus | Net after hits |','| --- | --- | --- | ---: | ---: | ---: |']
    for report in reports:
        scored=report.get('frozen_lineup')
        vals=[f"{scored[k]:+.0f}" for k in ['team_gain','chip_gain','net']] if scored else ['pending']*3
        lines.append(f"| {report['target_gw']} | {report['team_id']} | {report['status']} | {' | '.join(vals)} |")
    lines+=['','Latest snapshot before each deadline is selected automatically. Pending is not zero.','',result['limitations']]
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('data/forecast_archive'))
    parser.add_argument('--team-id',type=int,default=int(os.getenv('FPL_TRACK_TEAM_ID','2942747')))
    parser.add_argument('--restore-github',action='store_true')
    args=parser.parse_args()
    if args.restore_github:
        token=os.environ['GH_TOKEN'];repo=os.environ['GITHUB_REPOSITORY']
        print('Restored snapshots:',restore(args.directory,repo,token))
    os.environ['FPL_DATA_SOURCE']='live'
    from src.ingestion.fpl_client import FPLClient
    from src.advisor.personal import build_advice
    result=run(args.directory,args.team_id,FPLClient(),build_advice)
    print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a',encoding='utf-8') as handle:
            handle.write((args.directory/'summary.md').read_text(encoding='utf-8'))


if __name__=='__main__':main()
