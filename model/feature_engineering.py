import os
import glob
import numpy as np
import pandas as pd

ENGINE_DATA = os.path.join(os.path.dirname(__file__), '..', 'engine_data')
SKATER_DATA_DIR = os.path.join(ENGINE_DATA, 'Historical Skater Data')
SKATER_BIOS_PATH = os.path.join(ENGINE_DATA, 'Player Bios', 'Skaters', 'skater_bios.csv')
LAG_YEARS = [1, 2, 3]
SEASON_MAX_GP = {
    2013: 48,   # 2012-13 lockout
    2020: 70,   # 2019-20 COVID
    2021: 56,   # 2020-21 COVID
}
DEFAULT_MAX_GP = 82

SH_TO_EV_GOAL_RATIO = 0.4417666384460248
SH_TO_EV_ASSIST_RATIO = 0.25892504539760747

RATE_TARGETS = ['evg60', 'eva60', 'ppg60', 'ppa60']
ATOI_TARGETS = ['ev_atoi', 'pp_atoi', 'pk_atoi']
GP_TARGET = 'gp_rate'
ALL_TARGETS = ATOI_TARGETS + [GP_TARGET] + RATE_TARGETS

MIN_EV_TOI_TRAIN = 200 * 60  # 200 mins
MIN_PP_TOI_TRAIN = 30 * 60   # 30 mins
MIN_PK_TOI_TRAIN = 30 * 60   # 30 mins


def _season_end_year_from_filename(filename):
    base = os.path.basename(filename)
    return int(base.split('_')[0].split('-')[1])

def season_max_gp(year):
    return SEASON_MAX_GP.get(year, DEFAULT_MAX_GP)

# Aggregates player-season data into big modeling frame with season column and derived features
def load_raw_skater_seasons(min_year=None, max_year=None):
    files = sorted(glob.glob(os.path.join(SKATER_DATA_DIR, '*_skater_data.csv')))
    frames = []
    for path in files:
        year = _season_end_year_from_filename(path)
        if min_year is not None and year < min_year:
            continue
        if max_year is not None and year > max_year:
            continue
        df = pd.read_csv(path)
        df['season'] = year
        frames.append(df)
    if not frames:
        raise RuntimeError(f'No skater data found in {SKATER_DATA_DIR}')
    raw = pd.concat(frames, ignore_index=True, sort=False)
    return _add_derived_columns(raw)

# Compute per 60 rates and ATOI splits used as features
def _add_derived_columns(df):
    df = df.copy()

    for col_in, col_out in [
        ('evTimeOnIcePerGame', 'ev_atoi'),
        ('ppTimeOnIcePerGame', 'pp_atoi'),
        ('shTimeOnIcePerGame', 'pk_atoi'),
        ('timeOnIcePerGame', 'all_atoi'),
    ]:
        if col_in in df.columns:
            df[col_out] = pd.to_numeric(df[col_in], errors='coerce') / 60.0
        else:
            df[col_out] = np.nan

    df['gp'] = pd.to_numeric(df.get('gamesPlayed'), errors='coerce')
    df['gp_rate'] = df['gp'] / df['season'].map(season_max_gp).astype(float)

    # EV per 60
    ev_toi_sec = pd.to_numeric(df.get('evTimeOnIce'), errors='coerce')
    ev_toi_hr = ev_toi_sec / 3600.0
    ev_goals = pd.to_numeric(df.get('evGoals'), errors='coerce')
    ev_points = pd.to_numeric(df.get('evPoints'), errors='coerce')
    ev_assists = ev_points - ev_goals
    df['ev_toi_sec'] = ev_toi_sec
    df['evg60'] = np.where(ev_toi_hr > 0, ev_goals / ev_toi_hr, np.nan)
    df['eva60'] = np.where(ev_toi_hr > 0, ev_assists / ev_toi_hr, np.nan)

    # PP per 60
    pp_toi_sec = pd.to_numeric(df.get('ppTimeOnIce'), errors='coerce')
    df['pp_toi_sec'] = pp_toi_sec
    df['ppg60'] = pd.to_numeric(df.get('ppGoalsPer60'), errors='coerce')
    pp_pts60 = pd.to_numeric(df.get('ppPointsPer60'), errors='coerce')
    df['ppa60'] = pp_pts60 - df['ppg60']

    # PK TOI
    pk_toi_sec = pd.to_numeric(df.get('shTimeOnIce'), errors='coerce')
    df['pk_toi_sec'] = pk_toi_sec

    # Position group
    pos = df.get('positionCode', pd.Series(index=df.index, dtype=object)).astype(str)
    df['is_defense'] = (pos == 'D').astype(int)
    df['is_forward'] = pos.isin(['L', 'C', 'R']).astype(int)

    return df

def load_bios():
    bios = pd.read_csv(SKATER_BIOS_PATH)
    bios['birthDate'] = pd.to_datetime(bios['birthDate'], errors='coerce')
    keep = ['playerId', 'skaterFullName', 'birthDate', 'positionCode',
            'shootsCatches', 'draftRound', 'draftOverall', 'draftYear']
    return bios[[c for c in keep if c in bios.columns]].copy()

# Cols from a player-season record that should lag and be features
LAG_FEATURE_COLS = [
    'gp', 'gp_rate', 'all_atoi', 'ev_atoi', 'pp_atoi', 'pk_atoi',
    'evg60', 'eva60', 'ppg60', 'ppa60',
    'ev_toi_sec', 'pp_toi_sec', 'pk_toi_sec',
]

# Reuturn frame where each row is record for model training, inference, and tuning
# Each record is a player-season (playerId, season=Y0) with features from Y-1/Y-2/Y-3 and targets from Y0.
def build_modeling_frame(target_seasons=None, raw=None, bios=None):
    if raw is None:
        raw = load_raw_skater_seasons()
    if bios is None:
        bios = load_bios()

    seasons_available = sorted(raw['season'].unique())
    if target_seasons is None:
        target_seasons = [y for y in seasons_available if y - max(LAG_YEARS) >= min(seasons_available)]

    rows = []
    for y0 in target_seasons:
        if y0 not in seasons_available:
            continue
        y0_rows = raw[raw['season'] == y0].copy()
        y0_rows = y0_rows[y0_rows['playerId'].notna()]
        target_cols = ['playerId', 'skaterFullName', 'season', 'positionCode'] + ALL_TARGETS + ['gp', 'ev_toi_sec', 'pp_toi_sec', 'pk_toi_sec', 'is_defense', 'is_forward']
        y0_keep = y0_rows[[c for c in target_cols if c in y0_rows.columns]].copy()

        # Attach lag features for each lag year
        for lag in LAG_YEARS:
            year = y0 - lag
            if year not in seasons_available:
                # Fill with NaN for this lag entirely
                for col in LAG_FEATURE_COLS:
                    y0_keep[f'lag{lag}_{col}'] = np.nan
                continue
            lag_df = raw[raw['season'] == year][['playerId'] + LAG_FEATURE_COLS].copy()
            lag_df = lag_df.rename(columns={c: f'lag{lag}_{c}' for c in LAG_FEATURE_COLS})
            y0_keep = y0_keep.merge(lag_df, on='playerId', how='left')

        rows.append(y0_keep)

    frame = pd.concat(rows, ignore_index=True, sort=False)

    # Attach bios, compute age (oct. 1 of season start), draft features
    frame = frame.merge(bios, on='playerId', how='left', suffixes=('', '_bio'))
    season_start = pd.to_datetime(frame['season'].astype(int).astype(str) + '-10-01') - pd.DateOffset(years=1)
    age_days = (season_start - frame['birthDate']).dt.days
    frame['age'] = age_days / 365.25
    frame['age2'] = frame['age'] ** 2
    frame['age3'] = frame['age'] ** 3
    frame['draft_round'] = pd.to_numeric(frame.get('draftRound'), errors='coerce').fillna(8)
    frame['draft_overall'] = pd.to_numeric(frame.get('draftOverall'), errors='coerce').fillna(225)
    frame['is_rookie'] = frame['lag1_gp'].isna().astype(int)
    drop = ['birthDate', 'shootsCatches', 'draftRound', 'draftOverall', 'draftYear']
    frame = frame.drop(columns=[c for c in drop if c in frame.columns])

    return frame

# Feature subsets per target. Kept compact to avoid overfitting at ridge.
def _common_player_features():
    return ['age', 'age2', 'age3', 'is_defense', 'is_rookie', 'draft_round', 'draft_overall']

FEATURE_SETS = {
    'ev_atoi': _common_player_features() + [
        'lag1_ev_atoi', 'lag2_ev_atoi', 'lag3_ev_atoi',
        'lag1_all_atoi', 'lag1_gp_rate', 'lag1_pp_atoi',
    ],
    'pp_atoi': _common_player_features() + [
        'lag1_pp_atoi', 'lag2_pp_atoi', 'lag3_pp_atoi',
        'lag1_ev_atoi', 'lag1_ppg60', 'lag1_ppa60', 'lag1_gp_rate',
    ],
    'pk_atoi': _common_player_features() + [
        'lag1_pk_atoi', 'lag2_pk_atoi', 'lag3_pk_atoi',
        'lag1_ev_atoi', 'lag1_gp_rate',
    ],
    'gp_rate': _common_player_features() + [
        'lag1_gp_rate', 'lag2_gp_rate', 'lag3_gp_rate',
        'lag1_all_atoi', 'lag1_ev_atoi',
    ],
    'evg60': _common_player_features() + [
        'lag1_evg60', 'lag2_evg60', 'lag3_evg60',
        'lag1_ev_atoi', 'lag1_eva60', 'lag1_ev_toi_sec',
    ],
    'eva60': _common_player_features() + [
        'lag1_eva60', 'lag2_eva60', 'lag3_eva60',
        'lag1_ev_atoi', 'lag1_evg60', 'lag1_ev_toi_sec',
    ],
    'ppg60': _common_player_features() + [
        'lag1_ppg60', 'lag2_ppg60', 'lag3_ppg60',
        'lag1_pp_atoi', 'lag1_ppa60', 'lag1_pp_toi_sec',
    ],
    'ppa60': _common_player_features() + [
        'lag1_ppa60', 'lag2_ppa60', 'lag3_ppa60',
        'lag1_pp_atoi', 'lag1_ppg60', 'lag1_pp_toi_sec',
    ],
}


# Per-target row filters for training; returns boolean mask of valid rows
def training_filter(frame, target):
    mask = frame[target].notna()
    if target == 'evg60' or target == 'eva60':
        mask &= frame['ev_toi_sec'].fillna(0) >= MIN_EV_TOI_TRAIN
    elif target == 'ppg60' or target == 'ppa60':
        mask &= frame['pp_toi_sec'].fillna(0) >= MIN_PP_TOI_TRAIN
    elif target == 'pk_atoi':
        mask &= frame['gp'].fillna(0) >= 10
    elif target == 'pp_atoi':
        mask &= frame['gp'].fillna(0) >= 10
    elif target == 'ev_atoi':
        mask &= frame['gp'].fillna(0) >= 10
    elif target == 'gp_rate':
        mask &= frame['gp'].fillna(0) >= 1
    return mask

# Fill NaN feature values with column means
def impute_features(X, fitted_means=None):
    if fitted_means is None:
        fitted_means = X.mean(numeric_only=True)
    X_imp = X.fillna(fitted_means)
    X_imp = X_imp.fillna(0.0)
    return X_imp, fitted_means

# Exponential time decay * sample size factor
def compute_sample_weights(frame_subset, target, decay, current_year):
    seasons = frame_subset['season'].astype(float)
    time_w = np.exp(-decay * (current_year - seasons))

    if target in ('evg60', 'eva60'):
        size = frame_subset['ev_toi_sec'].fillna(0).clip(lower=0)
    elif target in ('ppg60', 'ppa60'):
        size = frame_subset['pp_toi_sec'].fillna(0).clip(lower=0)
    elif target in ('ev_atoi', 'pp_atoi', 'pk_atoi', 'gp_rate'):
        size = frame_subset['gp'].fillna(0).clip(lower=0)
    else:
        size = pd.Series(1.0, index=frame_subset.index)

    size = size / max(size.median(), 1.0)
    return (time_w * (0.25 + size)).astype(float).values


# Build frame for inference for every player eligible for projection
def build_inference_frame(projection_year, raw=None, bios=None):
    if raw is None:
        raw = load_raw_skater_seasons()
    if bios is None:
        bios = load_bios()

    lag_min_year = projection_year - max(LAG_YEARS)
    prior_ids = raw[(raw['season'] >= lag_min_year) & (raw['season'] < projection_year)]['playerId'].dropna().unique()
    y0_ids = raw[raw['season'] == projection_year]['playerId'].dropna().unique()
    candidate_ids = np.union1d(prior_ids, y0_ids)

    # Carry the most recent positionCode for each candidate.
    pos_lookup = (
        raw[raw['playerId'].isin(candidate_ids)]
        .sort_values('season')
        .groupby('playerId')['positionCode']
        .last()
    )

    synth = pd.DataFrame({'playerId': candidate_ids})
    synth['season'] = projection_year
    for col in ALL_TARGETS + ['gp', 'ev_toi_sec', 'pp_toi_sec', 'pk_toi_sec']:
        synth[col] = np.nan
    synth['positionCode'] = synth['playerId'].map(pos_lookup)
    synth['is_defense'] = (synth['positionCode'] == 'D').astype(int)
    synth['is_forward'] = synth['positionCode'].isin(['L', 'C', 'R']).astype(int)

    raw_no_y0 = raw[raw['season'] != projection_year]
    raw_plus = pd.concat([raw_no_y0, synth], ignore_index=True, sort=False)
    frame = build_modeling_frame(target_seasons=[projection_year], raw=raw_plus, bios=bios)
    return frame
