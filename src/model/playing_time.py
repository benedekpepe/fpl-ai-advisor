"""Chronological playing-time states, independent of production point forecasts.

Train/calibration/test periods never overlap. No live injury news is replayed.
"""
import argparse
from functools import lru_cache
import json
import logging
import pickle
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.config import LGBM_PARAMS, EXTRA_FEATURES, MODELS_DIR
from src.model.features import build_feature_frame, feature_columns

ARTIFACT = MODELS_DIR / 'playing_time_v1.pkl'
FIELDS = ('minutes_p_play', 'p_start', 'p_sixty', 'p_cameo', 'expected_minutes')


def softmax_temperature(probabilities, temperature):
    logits=np.log(np.clip(probabilities,1e-8,1))/temperature
    logits-=logits.max(axis=1,keepdims=True)
    result=np.exp(logits)
    return result/result.sum(axis=1,keepdims=True)


def temperature(probabilities, labels):
    labels=np.asarray(labels,dtype=int)
    def loss(t):
        p=softmax_temperature(probabilities,t)
        return -np.log(np.clip(p[np.arange(len(labels)),labels],1e-8,1)).mean()
    return float(minimize_scalar(loss,bounds=(0.5,3.),method='bounded').x)


def state_labels(frame):
    return np.where(frame.minutes>=60,2,np.where(frame.minutes>0,1,0))


def fit(frame):
    latest=frame.season.max()
    train=frame[(frame.season<latest)|((frame.season==latest)&(frame.gw<31))]
    cal=frame[(frame.season==latest)&(frame.gw>=31)]
    if train.empty or cal.empty or set(state_labels(train))!={0,1,2}:
        raise ValueError('Need all three minute states and a separate final-eight-GW calibration period.')
    cols=feature_columns()+EXTRA_FEATURES+['sixty_rate_5','cameo_rate_5']
    params={**LGBM_PARAMS,'n_estimators':300,'n_jobs':4,'verbosity':-1}
    state=lgb.LGBMClassifier(**params).fit(train[cols],state_labels(train))
    played=train.minutes>0
    start=lgb.LGBMClassifier(**params).fit(train.loc[played,cols],(train.loc[played,'starts']>0).astype(int))
    cal_played=cal.minutes>0
    models={}
    for key,target in [('minutes','minutes'),('points','total_points')]:
        for label in [1,2]:
            mask=state_labels(train)==label
            models[f'{key}_{label}']=lgb.LGBMRegressor(**params).fit(train.loc[mask,cols],train.loc[mask,target])
    return dict(schema=1,cols=cols,state=state,start=start,models=models,
                temperature=temperature(state.predict_proba(cal[cols]),state_labels(cal)),
                start_temperature=temperature(start.predict_proba(cal.loc[cal_played,cols]),(cal.loc[cal_played,'starts']>0).astype(int)),
                train_seasons=sorted(train.season.unique().tolist()),calibration_season=latest,
                train_last_gw_in_calibration_season=30,calibration_gws=[31,38],
                production_points_changed=False)


def predict(frame,bundle):
    x=frame.reindex(columns=bundle['cols'])
    p=softmax_temperature(bundle['state'].predict_proba(x),bundle['temperature'])
    p_play=p[:,1]+p[:,2]
    start=softmax_temperature(bundle['start'].predict_proba(x),bundle['start_temperature'])[:,1]*p_play
    models=bundle['models']
    minutes=p[:,1]*np.clip(models['minutes_1'].predict(x),1,59)+p[:,2]*np.clip(models['minutes_2'].predict(x),60,90)
    points=p[:,1]*models['points_1'].predict(x)+p[:,2]*models['points_2'].predict(x)
    return pd.DataFrame(dict(minutes_p_play=p_play,p_start=start,p_sixty=p[:,2],
                             p_cameo=np.maximum(0,p_play-start),expected_minutes=minutes,
                             state_points=points),index=frame.index)


@lru_cache(maxsize=2)
def _load(path,mtime):
    return joblib.load(path)


def enrich(frame,path=ARTIFACT):
    """Optional diagnostics; missing artifact leaves the main point model usable."""
    path=Path(path)
    if not path.exists():return frame
    try:
        bundle=_load(str(path),path.stat().st_mtime_ns)
    except (OSError,ValueError,EOFError,KeyError,pickle.UnpicklingError):
        logging.getLogger(__name__).exception('Optional playing-time artifact is unavailable.')
        return frame
    if bundle.get('schema')!=1 or 'state' not in bundle:
        logging.getLogger(__name__).warning('Ignoring unsupported playing-time artifact.')
        return frame
    result=frame.copy()
    estimates=predict(result,bundle)
    for field in FIELDS:result[field]=estimates[field]
    result['minutes_basis']='playing_time_model'
    return result


def summary(actual,pred):
    return dict(mae=round(float(np.abs(actual-pred).mean()),4),
                rmse=round(float(np.sqrt(((actual-pred)**2).mean())),4))


def evaluate(train_raw,test_raw):
    from src.model.validate import evaluate as evaluate_points
    from src.model.projections import recency_weight
    train=build_feature_frame(train_raw);test=build_feature_frame(test_raw)
    if train.season.max()>=test.season.min():raise ValueError('Test season must follow all training seasons.')
    bundle=fit(train);est=predict(test,bundle)
    report=dict(train_seasons=bundle['train_seasons'],test_season=test.season.min(),
                calibration_season=bundle['calibration_season'],calibration_gws=bundle['calibration_gws'],
                calibration_temperature=round(bundle['temperature'],4),matches=len(test))
    active=test.roll_minutes_5>=30
    report['minutes']={name:{'all':summary(test.minutes,pred),'active':summary(test.loc[active,'minutes'],pred[active])}
                       for name,pred in [('observed_guide',test.roll_minutes_5),('state_model',est.expected_minutes)]}
    probabilities={}
    for name,target,model,base in [('start',test.starts>0,est.p_start,test.starts_rate_5),
                                  ('sixty',test.minutes>=60,est.p_sixty,test.sixty_rate_5),
                                  ('cameo',(test.minutes>0)&(test.starts==0),est.p_cameo,test.cameo_rate_5)]:
        probabilities[name]={'model_brier':round(float(((model-target.astype(int))**2).mean()),4),
                             'recent_rate_brier':round(float(((base-target.astype(int))**2).mean()),4),
                             'bins':[]}
        for low,high in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.00001)]:
            mask=(model>=low)&(model<high)
            if mask.any():probabilities[name]['bins'].append(dict(n=int(mask.sum()),predicted=round(float(model[mask].mean()),4),observed=round(float(target[mask].mean()),4)))
    report['probabilities']=probabilities
    point_baseline=evaluate_points(train_raw,test_raw,str(train.season.max()),str(test.season.min()))
    weights=test.gw.map(lambda g:recency_weight(max(int(g)-1,0)))
    blended=(1-weights)*est.state_points+weights*test.cum_ppg.fillna(0)
    compared=pd.DataFrame(dict(gw=test.gw,id=test.element,actual=test.total_points,pred=blended,recent=test.roll_minutes_5))
    grouped=compared.groupby(['gw','id']).agg(actual=('actual','sum'),pred=('pred','sum'),recent=('recent','first'))
    active_gw=grouped.recent>=30
    report['point_candidate']={**summary(grouped.actual,grouped.pred),'active_mae':summary(grouped.loc[active_gw,'actual'],grouped.loc[active_gw,'pred'])['mae']}
    report['point_baseline']=point_baseline['metrics']['current_blend']
    candidate_round=grouped.groupby(level='gw').apply(lambda g:float(np.abs(g.actual-g.pred).mean()))
    deltas=np.array([candidate_round.loc[int(g)]-v['current_blend'] for g,v in point_baseline['gameweeks'].items()])
    rng=np.random.default_rng(42)
    sampled=rng.choice(deltas,size=(2000,len(deltas)),replace=True).mean(axis=1)
    report['point_round_mae_delta_bootstrap95']=[round(float(x),5) for x in np.quantile(sampled,[.025,.975])]
    report['points_pass']=all(report['point_candidate'][key]<report['point_baseline'][key] for key in ['mae','rmse','active_mae'])
    report['limitations']=['Separate chronological evaluation, not measured live accuracy.',
        'Historical opponent ratings may reflect later revisions; archived injuries are not replayed.',
        'Probabilities are per fixture; matches in doubles share pre-GW form, their outcomes are dependent.',
        'Minute labels exclude stoppage time; a substitute may also reach 60 minutes.',
        'Calibration uses only the final eight gameweeks of the training season.']
    return report


def main():
    from src.model.validate import read_season
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,default=Path('docs'))
    parser.add_argument('--artifact',type=Path,default=ARTIFACT)
    args=parser.parse_args()
    raw={s:read_season(s,args.cache_dir) for s in ['2023-24','2024-25','2025-26']}
    reports=[evaluate(raw[a],raw[b]) for a,b in [('2023-24','2024-25'),('2024-25','2025-26')]]
    result=dict(holdouts=reports,points_adopted=False,
                decision='Playing-time probability diagnostics only; production expected-points artifact and weights remain unchanged. Expected minutes are supplementary, not a replacement for the observed-minute guide; their MAE does not improve uniformly.')
    args.report_dir.mkdir(parents=True,exist_ok=True)
    (args.report_dir/'playing_time_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Probabilistic diagnostics are gated on Brier, a proper probability score.
    # Expected minutes remain supplementary: RMSE may improve while MAE worsens.
    # Never substitute them silently for the observed guide or promote point scores.
    valid=all(all(r['probabilities'][k]['model_brier']<r['probabilities'][k]['recent_rate_brier'] for k in ['start','sixty','cameo']) for r in reports)
    if not valid:raise ValueError('Diagnostics failed the two-holdout gate; no new artifact exported.')
    bundle=fit(build_feature_frame(pd.concat(raw.values(),ignore_index=True)))
    args.artifact.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,args.artifact,compress=3)
    print(json.dumps(result,indent=2))
    print('Diagnostics artifact:',args.artifact)


if __name__=='__main__':main()
