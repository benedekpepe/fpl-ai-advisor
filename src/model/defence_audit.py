"""Chronological audit of dedicated defensive-context inputs; never promotes models.

FPL_DATA_SOURCE=live python -m src.model.defence_audit --cache-dir work/validation --output docs/defence_context_validation.json --compare Konsa Calafiori
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from src.model.features import build_feature_frame,feature_columns,context_columns
from src.model.validate import read_season
from src.model.projections import score_features
from src.config import LGBM_PARAMS,CLF_N_ESTIMATORS,EXTRA_FEATURES,MODEL_PATH

def fit(train,cols):
 params={**LGBM_PARAMS,'n_jobs':4,'verbosity':-1}
 clf=lgb.LGBMClassifier(**{**params,'n_estimators':CLF_N_ESTIMATORS})
 reg=lgb.LGBMRegressor(**params)
 mask=train.minutes>0
 clf.fit(train[cols],mask.astype(int));reg.fit(train.loc[mask,cols],train.loc[mask,'total_points'])
 return dict(clf=clf,reg=reg,cols=cols)
def score(test,bundle):
 result=[]
 for gw,group in test.groupby('gw'):
  p,_,_,_=score_features(group,bundle,max(int(gw)-1,0))
  result.append(pd.Series(p,index=group.index))
 return pd.concat(result).reindex(test.index)
def metrics(test,pred):
 frame=pd.DataFrame(dict(gw=test.gw,id=test.element,actual=test.total_points,pred=pred,mins=test.roll_minutes_5,pos=test.position))
 f=frame.groupby(['gw','id']).agg(actual=('actual','sum'),pred=('pred','sum'),mins=('mins','first'),pos=('pos','first'))
 def m(g):
  d=g.pred-g.actual
  return dict(n=len(g),mae=float(d.abs().mean()),rmse=float(np.sqrt((d*d).mean())),bias=float(d.mean()))
 return dict(all=m(f),active=m(f[f.mins>=30]),def_active=m(f[(f.mins>=30)&(f.pos=='DEF')]),
             positions={pos:m(g) for pos,g in f.groupby('pos')},
             round_mae={int(gw):m(g)['mae'] for gw,g in f.groupby(level=0)})
def evaluate(cache_dir,live_raw,compare=()):
    raw={s:read_season(s,cache_dir) for s in ['2023-24','2024-25','2025-26']}
    feat={s:build_feature_frame(r) for s,r in raw.items()}
    live=build_feature_frame(live_raw)
    if set(live.season) != {'2026-27'}:
        raise ValueError('This audit expects the later 2026-27 live season; set FPL_DATA_SOURCE=live.')
    cutoff=int(live.loc[live.minutes.notna(),'gw'].max())
    target=int(live.loc[live.minutes.isna(),'gw'].min())
    old_cols=feature_columns()+EXTRA_FEATURES
    new_cols=old_cols+context_columns()
    reports={};last={}
    for case,train,test in [
     ('2025-26_GW25-38',pd.concat([feat['2023-24'],feat['2024-25'],feat['2025-26'][feat['2025-26'].gw<=24]],ignore_index=True),feat['2025-26'][feat['2025-26'].gw>=25]),
     ('2026-27_checked',pd.concat(list(feat.values()),ignore_index=True),live[live.minutes.notna()])]:
     print('Training',case,flush=True);reports[case]={}
     for name,cols in [('same_features_retrained',old_cols),('defence_context',new_cols)]:
      b=fit(train,cols);p=score(test,b);reports[case][name]=metrics(test,p)
      print(name,{k:v for k,v in reports[case][name].items() if k in ['all','active','def_active']},flush=True)
      if case=='2026-27_checked':last[name]=b
     if case=='2026-27_checked':reports[case]['deployed_artifact']=metrics(test,score(test,joblib.load(MODEL_PATH)))
     a=reports[case]['defence_context']['round_mae'];base=reports[case]['same_features_retrained']['round_mae']
     d=np.array([a[g]-base[g] for g in sorted(a)])
     rng=np.random.default_rng(42);interval=np.quantile(rng.choice(d,size=(2000,len(d)),replace=True).mean(axis=1),[.025,.975])
     reports[case]['round_mae_delta_bootstrap95']=interval.tolist()
    for name,b in last.items():
        future=live[(live.gw==target)&live.name.isin(compare)].copy()
        if not future.empty:
            pred,model,play,weight=score_features(future,b,cutoff)
            reports['pair_'+name]=[dict(name=r['name'],pred=float(pred[i]),model=float(model[i]),p_play=float(play[i]))
                                       for i,(_,r) in enumerate(future.iterrows())]
    reports['method']={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'live_form_cutoff':cutoff,'target_gw':int(target),
        'historical_holdout':'Train 2023-24, 2024-25 and 2025-26 through GW24; test GW25-38.',
        'live_holdout':'Train all 2023-24 through 2025-26; test only checked live matches with prior form.',
        'candidate_features':context_columns(),
        'limitations':['Historical ratings may include later revisions. No archived injury flags are replayed.',
                       'The test periods have now been inspected and are not untouched tuning sets.',
                       'Deployed artifact has no training metadata; its reported comparison is only on the later live season.'],
        'decision':'Do not deploy defence-context candidate: historical MAE worsens. No player-specific correction.'}
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',nargs='*',default=[])
    args=parser.parse_args()
    import src.ingestion.live as ingestion
    from src.model.model import attach_opponent_strength
    raw=attach_opponent_strength(ingestion.current_season_frame())
    result=evaluate(args.cache_dir,raw,args.compare)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Audit report:',args.output)


if __name__=='__main__':main()
