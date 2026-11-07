"""Expected points and transparent supporting signals, shared with validation."""
import numpy as np


def recency_weight(played):
    weight = min(0.15 * played, 0.45)
    return max(0.15, 0.45 - (played - 6) * 0.05) if played > 6 else weight


def score_features(feats, bundle, played, blend_scale=1.0):
    X = feats[bundle["cols"]].copy()
    if "value" in X:
        shrink = min(played / 4.0, 1.0)
        med = feats.groupby("position")["value"].transform("median").fillna(feats["value"].median())
        X["value"] = (med + (feats["value"] - med) * shrink).to_numpy()
    p_play = np.clip(bundle["clf"].predict_proba(X)[:, 1], 0, 1)
    model = p_play * bundle["reg"].predict(X)
    weight = min(1.0, recency_weight(played) * blend_scale)
    form = feats["cum_ppg"].fillna(0).to_numpy()
    pred = (1-weight) * model + weight * form
    return pred, model, p_play, weight
