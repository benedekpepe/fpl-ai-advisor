"""Reproducible chronological holdout; never overwrites the deployed model.

python -m src.model.validate --cache-dir work/validation --output docs/model_validation.json
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from src.config import MERGED_GW_URL, TEAMS_URL, LGBM_PARAMS, CLF_N_ESTIMATORS, EXTRA_FEATURES
from src.model.features import build_feature_frame, feature_columns
from src.model.projections import score_features


def read_season(season, cache):
    cache.mkdir(parents=True, exist_ok=True)
    def csv(kind, url):
        path = cache / f'{season}-{kind}.csv'
        if not path.exists():
            response = requests.get(url.format(season=season), timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        return pd.read_csv(path)
    raw = csv('history', MERGED_GW_URL).rename(columns={'GW': 'gw', 'xP': 'xp'})
    raw['season'] = season
    strength = csv('teams', TEAMS_URL).rename(columns={'id': 'opponent_team'})
    cols = ['opponent_team', 'strength_attack_home', 'strength_attack_away',
            'strength_defence_home', 'strength_defence_away']
    raw = raw.merge(strength[cols], on='opponent_team', how='left')
    for kind in ['attack', 'defence']:
        raw[f'opp_{kind}'] = raw[f'strength_{kind}_away'].where(raw.was_home, raw[f'strength_{kind}_home'])
    return raw


def errors(actual, pred):
    return {'mae': round(float(np.mean(np.abs(actual-pred))), 4),
            'rmse': round(float(np.sqrt(np.mean((actual-pred)**2))), 4)}


def evaluate(train_raw, test_raw, train_season, test_season):
    if train_season >= test_season:
        raise ValueError('Training seasons must precede the test season.')
    train = build_feature_frame(train_raw)
    test = build_feature_frame(test_raw)
    cols = feature_columns() + EXTRA_FEATURES
    params = {**LGBM_PARAMS, 'n_jobs': 4, 'verbosity': -1}
    clf = lgb.LGBMClassifier(**{**params, 'n_estimators': CLF_N_ESTIMATORS})
    reg = lgb.LGBMRegressor(**params)
    played = train.minutes > 0
    clf.fit(train[cols], played.astype(int))
    reg.fit(train.loc[played, cols], train.loc[played, 'total_points'])
    bundle = dict(clf=clf, reg=reg, cols=cols)
    policies = {'current_blend': 1., 'model_only': 0., 'lighter_blend': .5, 'stronger_blend': 1.5}
    scored = []
    for gw, rows in test.groupby('gw'):
        rows = rows.copy()
        for name, scale in policies.items():
            rows[name], _, rows['p_play'], _ = score_features(rows, bundle, max(int(gw)-1, 0), scale)
        recent = rows.copy()
        recent['cum_ppg'] = recent['roll_total_points_5'].fillna(recent['cum_ppg'])
        rows['recent5_blend'], _, _, _ = score_features(recent, bundle, max(int(gw)-1, 0))
        conf = (rows['games_played']/3).clip(0, 1)
        rows['sample_aware'] = rows['model_only'] + conf*(rows['current_blend']-rows['model_only'])
        rows['form3'] = rows['roll_total_points_3'].fillna(0)
        rows['minutes_guard'] = rows.current_blend * (rows.roll_minutes_5.fillna(0)/60).clip(.25, 1)
        scored.append(rows)
    scored = pd.concat(scored)
    names = list(policies) + ['form3', 'minutes_guard', 'recent5_blend', 'sample_aware']
    grouped = scored.groupby(['gw', 'element']).agg(
        actual=('total_points', 'sum'), recent_minutes=('roll_minutes_5', 'first'),
        position=('position', 'first'), name=('name', 'first'), minutes=('minutes', 'sum'),
        **{name: (name, 'sum') for name in names})
    active = grouped[grouped.recent_minutes >= 30]
    metrics = {name: {**errors(grouped.actual, grouped[name]),
                      'active_mae': errors(active.actual, active[name])['mae']} for name in names}
    gw_metrics = {str(gw): {name: errors(g.actual, g[name])['mae'] for name in names}
                  for gw, g in grouped.groupby(level='gw')}
    def segment_metrics(frame):
        result = errors(frame.actual, frame.current_blend)
        result.update(n=len(frame), bias=round(float((frame.current_blend-frame.actual).mean()),4))
        return result
    by_position = {str(pos): segment_metrics(g) for pos,g in grouped.groupby('position')}
    minute_bands = pd.cut(grouped.recent_minutes, [-1,30,60,float('inf')], right=False,
                          labels=['under_30','30_to_59','60_plus'])
    by_minutes = {str(band): segment_metrics(g) for band,g in grouped.groupby(minute_bands,observed=True)}
    calibration = []
    bins = pd.cut(scored.p_play, [0,.2,.4,.6,.8,1.],include_lowest=True)
    for band,g in scored.groupby(bins,observed=True):
        calibration.append(dict(band=str(band),n=len(g), predicted=round(float(g.p_play.mean()),4),
                                observed=round(float((g.minutes>0).mean()),4)))
    player_bias = grouped.groupby(level='element').agg(name=('name','first'), position=('position','first'),
        n=('actual','size'), actual=('actual','mean'), predicted=('current_blend','mean'),minutes=('minutes','mean'))
    player_bias = player_bias[(player_bias.n>=10)&(player_bias.minutes>=30)].copy()
    player_bias['bias']=player_bias.predicted-player_bias.actual
    bias_examples = {label: rows.reset_index().round(4).to_dict('records')
        for label,rows in [('over',player_bias.nlargest(8,'bias')),('under',player_bias.nsmallest(8,'bias'))]}
    minute_rows = scored.dropna(subset=['roll_minutes_5', 'minutes'])
    return {
        'train_season': train_season, 'test_season': test_season,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'train_rows': len(train), 'test_player_gameweeks': len(grouped),
        'active_player_gameweeks': len(active), 'metrics': metrics, 'gameweeks': gw_metrics,
        'segments': {'positions':by_position, 'recent_minutes':by_minutes},
        'play_calibration': calibration, 'player_bias_examples':bias_examples,
        'minutes': {'guide_mae': errors(minute_rows.minutes, minute_rows.roll_minutes_5)['mae'],
                    'guide_active_mae': errors(minute_rows.loc[minute_rows.roll_minutes_5>=30,'minutes'],minute_rows.loc[minute_rows.roll_minutes_5>=30,'roll_minutes_5'])['mae'],
                    'play_brier': round(float(np.mean((scored.p_play-(scored.minutes>0).astype(int))**2)),4)},
        'method': 'Train on an earlier season; evaluate later season with only prior observed form. All matches in a double share pre-GW features. Metrics aggregate points per player/gameweek. Active cohort: prior average minutes >=30.',
        'limitations': ['Historical opponent strengths are season-level snapshots and may reflect later revisions.', 'Historical Assistant Manager chip rows are excluded; only GK/DEF/MID/FWD players are evaluated.', 'Historical injury news and deadline-time availability snapshots are not replayed.', 'Evaluation model is separate from the deployed artifact; one holdout season is insufficient for selecting new production weights.', 'Metrics evaluate points per player/gameweek, not transfer hits or whole-season manager rank.'],
        'decision': 'Exploratory comparison only. Keep deployed weights unchanged; confirm on another untouched season before tuning.',
        'sources': [MERGED_GW_URL.format(season=s) for s in [train_season, test_season]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-season', default='2023-24')
    parser.add_argument('--test-season', default='2024-25')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.train_season >= args.test_season:
        parser.error('Training season must be earlier than test season.')
    print('Downloading/caching public historical data...', flush=True)
    train = read_season(args.train_season, args.cache_dir)
    test = read_season(args.test_season, args.cache_dir)
    print('Training an isolated evaluation model and comparing policies...', flush=True)
    result = evaluate(train, test, args.train_season, args.test_season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2), flush=True)
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
