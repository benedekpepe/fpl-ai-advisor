"""Archive retention, deterministic selection and no late/hindsight captures."""
import copy
from datetime import datetime,timedelta,timezone
from io import BytesIO
import json
import zipfile
import pytest

from src.model.forecast_archive import import_zip,load_snapshots,latest_per_round,should_capture,run,restore
from src.model.monitor import digest,save_snapshot
from test_forecast_monitor import snapshot

NOW=datetime(2026,11,8,tzinfo=timezone.utc)


def revised(saved,**changes):
    saved=copy.deepcopy(saved);saved.update(changes);saved['snapshot_id']=digest(saved)
    return saved


def test_retention_uses_capture_time_and_keeps_only_last_90_days(tmp_path):
    old=revised(snapshot(),captured_at=(NOW-timedelta(days=91)).isoformat())
    old_path=save_snapshot(old,tmp_path/'snapshots')
    current=save_snapshot(snapshot(),tmp_path/'snapshots')
    retained=load_snapshots(tmp_path,NOW)
    assert not old_path.exists() and current.exists()
    assert len(retained)==1


def test_latest_pre_deadline_selection_is_deterministic_without_actual_points():
    early=snapshot();later=revised(early,captured_at=(NOW+timedelta(hours=1)).isoformat())
    assert latest_per_round([later,early])[0]['snapshot_id']==later['snapshot_id']


def test_capture_once_early_then_refresh_in_last_48_hours_but_never_late():
    saved=snapshot();event=dict(id=11,deadline_time=saved['deadline'])
    assert should_capture([],123,'2026-27',event,NOW)
    assert not should_capture([saved],123,'2026-27',event,NOW+timedelta(hours=1))
    near=datetime(2026,11,13,tzinfo=timezone.utc)
    assert should_capture([saved],123,'2026-27',event,near)
    assert not should_capture([saved],123,'2026-27',event,near+timedelta(days=2))


def test_zip_restore_validates_snapshots_without_extracting_other_paths(tmp_path):
    saved=snapshot();buffer=BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        z.writestr(f"snapshots/2026-27-gw11-{saved['snapshot_id'][:16]}.json",json.dumps(saved))
        z.writestr('../escape.txt','bad')
    assert len(import_zip(buffer.getvalue(),tmp_path/'ledger'))==1
    assert not (tmp_path/'escape.txt').exists()


def test_pending_archive_run_does_not_fetch_unfinished_player_results(tmp_path):
    saved=snapshot();save_snapshot(saved,tmp_path/'snapshots')
    class Client:
        def get_bootstrap_static(self):
            return {'events':[dict(id=1,deadline_time='2026-08-01T12:00:00Z',finished=True,data_checked=True),
                              dict(id=11,deadline_time=saved['deadline'],finished=False,data_checked=False)]}
        def get_element_summary(self,pid):pytest.fail('unfinished results fetched')
    result=run(tmp_path,123,Client(),lambda _:pytest.fail('unnecessary recapture'),now=NOW)
    assert result['reports'][0]['status']=='pending'
    assert result['reports'][0]['frozen_lineup'] is None
    assert 'pending' in (tmp_path/'summary.md').read_text()


def test_github_restore_failure_is_not_silently_replaced_with_empty_archive(tmp_path):
    class Response:
        def raise_for_status(self):raise RuntimeError('network unavailable')
    class Session:
        def get(self,*args,**kwargs):return Response()
    with pytest.raises(RuntimeError):restore(tmp_path,'owner/repo','unused',Session())


def test_forged_snapshot_season_cannot_escape_archive_directory(tmp_path):
    forged=revised(snapshot(),season='../outside')
    with pytest.raises(ValueError,match='season'):
        save_snapshot(forged,tmp_path/'snapshots')
    assert not (tmp_path/'outside').exists()
