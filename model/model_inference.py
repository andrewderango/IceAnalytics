import os
import numpy as np
import pandas as pd
from feature_engineering import (
    ENGINE_DATA, ALL_TARGETS,
    SH_TO_EV_GOAL_RATIO, SH_TO_EV_ASSIST_RATIO,
    build_inference_frame, impute_features, load_bios,
)

def _predict_one(bundle, X_df):
    feats = bundle['features']
    X = X_df[feats].copy()
    means = pd.Series(bundle['feature_means'])
    X_imp, _ = impute_features(X, fitted_means=means)
    if bundle['family'] == 'ridge':
        Xs = bundle['scaler'].transform(X_imp.values)
        return bundle['model'].predict(Xs)
    return bundle['model'].predict(X_imp.values)

# Run every trained model on the projection-year inference frame and return a dataframe of predictions
def run_inference(projection_year, bundles, verbose=False):

    frame = build_inference_frame(projection_year)
    if verbose:
        print(f'Inference frame: {len(frame)} player rows for {projection_year}.')

    out = frame[['playerId', 'positionCode', 'age']].copy()
    out['season'] = projection_year

    for target in ALL_TARGETS:
        preds = _predict_one(bundles[target], frame)
        out[target] = preds

    # Sanity clipping
    out['ev_atoi'] = out['ev_atoi'].clip(lower=0, upper=25)
    out['pp_atoi'] = out['pp_atoi'].clip(lower=0, upper=8)
    out['pk_atoi'] = out['pk_atoi'].clip(lower=0, upper=6)
    out['gp_rate'] = out['gp_rate'].clip(lower=0, upper=1)
    for c in ['evg60', 'eva60', 'ppg60', 'ppa60']:
        out[c] = out[c].clip(lower=0)

    # PK heuristic
    out['pkg60'] = out['evg60'] * SH_TO_EV_GOAL_RATIO
    out['pka60'] = out['eva60'] * SH_TO_EV_ASSIST_RATIO

    out['gp'] = (out['gp_rate'] * 82).round(2)
    out['ev_g_per_game'] = out['evg60'] * out['ev_atoi'] / 60.0
    out['ev_a_per_game'] = out['eva60'] * out['ev_atoi'] / 60.0
    out['pp_g_per_game'] = out['ppg60'] * out['pp_atoi'] / 60.0
    out['pp_a_per_game'] = out['ppa60'] * out['pp_atoi'] / 60.0
    out['pk_g_per_game'] = out['pkg60'] * out['pk_atoi'] / 60.0
    out['pk_a_per_game'] = out['pka60'] * out['pk_atoi'] / 60.0

    out['proj_goals'] = (out['ev_g_per_game'] + out['pp_g_per_game'] + out['pk_g_per_game']) * out['gp']
    out['proj_assists'] = (out['ev_a_per_game'] + out['pp_a_per_game'] + out['pk_a_per_game']) * out['gp']
    out['proj_points'] = out['proj_goals'] + out['proj_assists']

    bios = load_bios()
    if 'skaterFullName' in bios.columns:
        out = out.merge(bios[['playerId', 'skaterFullName']], on='playerId', how='left')

    return out


def save_inference(projection_year, df, verbose=False):
    out_dir = os.path.join(ENGINE_DATA, 'Projections', str(projection_year), 'Skaters')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{projection_year}_skater_metaprojections.csv')
    df.sort_values('proj_points', ascending=False).to_csv(path, index=False)
    if verbose:
        print(f'Wrote {len(df)} player projections to {path}')
    return path
