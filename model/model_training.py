import os
import joblib
import numpy as np
import xgboost as xgb
from feature_engineering import (
    ENGINE_DATA, ALL_TARGETS, FEATURE_SETS_XGB,
    build_modeling_frame, training_filter, compute_sample_weights,
)

MODELS_DIR = os.path.join(ENGINE_DATA, 'Projection Models', 'inference')

# Inference models are always XGBoost. Bootstrap models are always Ridge (see model_bootstrap.py).
MODEL_CONFIG = {
    'ev_atoi': {'decay': 0.05, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'pp_atoi': {'decay': 0.05, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'pk_atoi': {'decay': 0.05, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'gp_rate': {'decay': 0.05, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'evg60':   {'decay': 0.10, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'eva60':   {'decay': 0.10, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'ppg60':   {'decay': 0.10, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
    'ppa60':   {'decay': 0.10, 'xgb_params': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}},
}

DEFAULT_XGB_PARAMS = dict(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method='hist', objective='reg:squarederror', n_jobs=4, verbosity=0,
)


def fit_target_model(target, frame, projection_year, config=None, verbose=False):
    if config is None:
        config = MODEL_CONFIG[target]

    feats = FEATURE_SETS[target]
    mask = training_filter(frame, target) & (frame['season'] < projection_year)
    sub = frame.loc[mask].copy()
    if sub.empty:
        raise RuntimeError(f'No training rows for {target}')

    X = sub[feats]
    y = sub[target].astype(float).values
    weights = compute_sample_weights(sub, target, config['decay'], projection_year)
    X_imp, feat_means = impute_features(X)

    if config['family'] == 'ridge':
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_imp.values)
        model = Ridge(alpha=config['alpha'])
        model.fit(Xs, y, sample_weight=weights)
    else:
        scaler = None
        params = {**DEFAULT_XGB_PARAMS, **config.get('xgb_params', {})}
        model = xgb.XGBRegressor(**params)
        model.fit(X_imp.values, y, sample_weight=weights)

    if verbose:
        print(f'  {target:<8} family={config["family"]:<5} n={len(sub):>5}')

    return {
        'target': target,
        'family': config['family'],
        'config': config,
        'features': feats,
        'feature_means': feat_means.to_dict(),
        'scaler': scaler,
        'model': model,
        'n_train': int(len(sub)),
    }


def save_target_model(bundle):
    os.makedirs(MODELS_DIR, exist_ok=True)
    target = bundle['target']
    path = os.path.join(MODELS_DIR, f'{target}_model.pkl')
    joblib.dump(bundle, path)
    return path


def load_target_model(target):
    path = os.path.join(MODELS_DIR, f'{target}_model.pkl')
    return joblib.load(path)


def train_all_models(projection_year, retrain=False, verbose=False):
    bundles = {}
    if not retrain and all(os.path.exists(os.path.join(MODELS_DIR, f'{t}_model.pkl')) for t in ALL_TARGETS):
        if verbose:
            print('Loading cached models from disk.')
        for t in ALL_TARGETS:
            bundles[t] = load_target_model(t)
        return bundles

    if verbose:
        print(f'Building modeling frame for projection_year={projection_year}...')
    frame = build_modeling_frame()

    if verbose:
        print(f'Training {len(ALL_TARGETS)} models...')
    for target in ALL_TARGETS:
        bundle = fit_target_model(target, frame, projection_year, verbose=verbose)
        save_target_model(bundle)
        bundles[target] = bundle

    return bundles
