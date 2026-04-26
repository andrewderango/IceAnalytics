import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from feature_engineering import (
    ENGINE_DATA, ALL_TARGETS, FEATURE_SETS, SH_TO_EV_GOAL_RATIO, SH_TO_EV_ASSIST_RATIO,
    build_modeling_frame, build_inference_frame, training_filter,
    impute_features, compute_sample_weights,
)
from model_training import MODEL_CONFIG, DEFAULT_XGB_PARAMS

BOOTSTRAPS_DIR = os.path.join(ENGINE_DATA, 'Projection Models', 'bootstraps')
BOOTSTRAP_MODELS_DIR = os.path.join(BOOTSTRAPS_DIR, 'models')
N_BOOTSTRAPS = 500
HOLDOUT_FRAC = 0.01  # PRUS OOS residual holdout fraction


def _bundle_path(target, projection_year):
    return os.path.join(BOOTSTRAP_MODELS_DIR, str(projection_year), f'{target}_bootstraps.pkl')

def _bundle_exists(target, projection_year):
    return os.path.exists(_bundle_path(target, projection_year))

def _fit_and_score_bootstrap(target, sub, inf_X_imp, inf_feats, projection_year, config, rng):
    n = len(sub)
    n_hold = max(1, int(n * HOLDOUT_FRAC))
    all_idx = rng.permutation(n)
    hold_idx = all_idx[:n_hold]
    train_pool_idx = all_idx[n_hold:]
    boot_idx = rng.integers(0, len(train_pool_idx), size=len(train_pool_idx))
    train_idx = train_pool_idx[boot_idx]

    train_rows = sub.iloc[train_idx]
    hold_rows = sub.iloc[hold_idx]

    feats = inf_feats
    X_tr = train_rows[feats]
    y_tr = train_rows[target].astype(float).values
    w_tr = compute_sample_weights(train_rows, target, config['decay'], projection_year)
    X_tr_imp, feat_means = impute_features(X_tr)

    X_hold = hold_rows[feats]
    y_hold = hold_rows[target].astype(float).values
    w_hold = compute_sample_weights(hold_rows, target, config['decay'], projection_year)
    means_series = pd.Series(feat_means.to_dict() if hasattr(feat_means, 'to_dict') else feat_means)
    X_hold_imp, _ = impute_features(X_hold, fitted_means=means_series)

    sc = None
    if config['family'] == 'ridge':
        sc = StandardScaler()
        Xs_tr = sc.fit_transform(X_tr_imp.values)
        m = Ridge(alpha=config['alpha'])
        m.fit(Xs_tr, y_tr, sample_weight=w_tr)
        hold_preds = m.predict(sc.transform(X_hold_imp.values))
        inf_preds = m.predict(sc.transform(inf_X_imp))
    else:
        params = {**DEFAULT_XGB_PARAMS, **config.get('xgb_params', {})}
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr_imp.values, y_tr, sample_weight=w_tr)
        hold_preds = m.predict(X_hold_imp.values)
        inf_preds = m.predict(inf_X_imp)

    hold_resid_var = float(np.average((y_hold - hold_preds) ** 2, weights=w_hold))
    return inf_preds, hold_resid_var, m, sc

def _save_bundle(target, projection_year, bundle):
    path = _bundle_path(target, projection_year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)

def _load_bundle_preds(target, projection_year, config, inf_X_imp_arr):
    bundle = joblib.load(_bundle_path(target, projection_year))
    preds_list, resid_vars = [], []
    for entry in bundle:
        model, scaler, rv = entry['model'], entry['scaler'], entry['resid_var']
        if config['family'] == 'ridge':
            preds_list.append(model.predict(scaler.transform(inf_X_imp_arr)))
        else:
            preds_list.append(model.predict(inf_X_imp_arr))
        resid_vars.append(rv)
    return np.array(preds_list, dtype=np.float32).T, np.array(resid_vars)

# Bootstrap one target, returns per-player stdev series
def bootstrap_target(target, projection_year, n_boots=N_BOOTSTRAPS, seed=42, verbose=False, retrain=True):
    config = MODEL_CONFIG[target]
    feats = FEATURE_SETS[target]

    inf_frame = build_inference_frame(projection_year)
    X_inf_raw = inf_frame[feats]
    X_inf_imp, _ = impute_features(X_inf_raw)
    inf_X_imp_arr = X_inf_imp.values

    preds_matrix = np.zeros((len(inf_frame), n_boots), dtype=np.float32)
    resid_vars = np.zeros(n_boots, dtype=np.float64)

    if not retrain and _bundle_exists(target, projection_year):
        if verbose:
            print(f'Loading saved bootstraps for {target}')
        preds_matrix, resid_vars = _load_bundle_preds(target, projection_year, config, inf_X_imp_arr)
    else:
        if not retrain and verbose:
            print(f'Saved bootstrap bundle for {target} not found, training from scratch')
        train_frame = build_modeling_frame()
        mask = training_filter(train_frame, target) & (train_frame['season'] < projection_year)
        sub = train_frame.loc[mask].copy().reset_index(drop=True)
        if sub.empty:
            raise RuntimeError(f'No training rows for bootstrap target {target}')

        bundle = []
        rng = np.random.default_rng(seed)
        iterator = range(n_boots)
        if verbose:
            iterator = tqdm(iterator, desc=f'bootstrap {target}')
        for i in iterator:
            inf_preds, rv, model, scaler = _fit_and_score_bootstrap(
                target, sub, inf_X_imp_arr, feats, projection_year, config, rng)
            preds_matrix[:, i] = inf_preds
            resid_vars[i] = rv
            bundle.append({'model': model, 'scaler': scaler, 'resid_var': rv})
        _save_bundle(target, projection_year, bundle)

    # PRUS
    residual_var = float(np.mean(resid_vars))
    ens_var = preds_matrix.var(axis=1)
    mean_ens_var = float(np.mean(ens_var)) if ens_var.size else 1.0
    if mean_ens_var > 0:
        scaled_var = ens_var * (residual_var / mean_ens_var)
    else:
        scaled_var = ens_var
    stdev = np.sqrt(np.clip(scaled_var, 0, None))

    stdev_df = inf_frame[['playerId']].assign(**{f'{target}_stdev': stdev})
    return stdev_df, residual_var, mean_ens_var


# Reduce variance as season progresses
def apply_season_progress_scaling(stdev_df, projection_year):
    path = os.path.join(ENGINE_DATA, 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv')
    if os.path.exists(path):
        cur = pd.read_csv(path, usecols=['playerId', 'gamesPlayed'])
        cur = cur.rename(columns={'gamesPlayed': 'cur_gp'})
    else:
        cur = pd.DataFrame({'playerId': [], 'cur_gp': []})

    out = stdev_df.merge(cur, on='playerId', how='left')
    out['cur_gp'] = out['cur_gp'].fillna(0).clip(lower=0, upper=82)
    factor = np.sqrt(1.0 - out['cur_gp'] / 82.0)
    for col in out.columns:
        if col.endswith('_stdev'):
            out[col] = out[col] * factor
    return out.drop(columns=['cur_gp'])


# Run bootstraps for all targets, returns merged stdev frame
def run_all_bootstraps(projection_year, n_boots=N_BOOTSTRAPS, verbose=False, retrain=True):
    os.makedirs(BOOTSTRAPS_DIR, exist_ok=True)
    merged = None
    summary = {}
    for i, target in enumerate(ALL_TARGETS):
        df, residual_var, mean_ens_var = bootstrap_target(target, projection_year, n_boots=n_boots, seed=42 + i, verbose=verbose, retrain=retrain)
        summary[target] = {
            'n_boots': n_boots,
            'mean_stdev': float(df[f'{target}_stdev'].mean()),
            'residual_var': residual_var,
            'mean_ens_var': mean_ens_var,
        }
        merged = df if merged is None else merged.merge(df, on='playerId', how='outer')

    # Heuristic PK stdevs proportional to EV stdevs
    if 'evg60_stdev' in merged.columns:
        merged['pkg60_stdev'] = merged['evg60_stdev'] * SH_TO_EV_GOAL_RATIO
    if 'eva60_stdev' in merged.columns:
        merged['pka60_stdev'] = merged['eva60_stdev'] * SH_TO_EV_ASSIST_RATIO

    merged = apply_season_progress_scaling(merged, projection_year)

    with open(os.path.join(BOOTSTRAPS_DIR, 'bootstrap_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    out_dir = os.path.join(ENGINE_DATA, 'Projections', str(projection_year), 'Skaters')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{projection_year}_skater_bootstrap_stdevs.csv')
    merged.to_csv(path, index=False)
    if verbose:
        print(f'Wrote bootstrap stdevs to {path}')

    return merged
