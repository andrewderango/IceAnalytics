import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_training import *
from scraper_functions import *
from sklearn.utils import resample
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

def atoi_model_inference(projection_year, player_stat_df, atoi_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI']]
            df['ATOI'] = df['TOI']/df['GP']
            df = df.drop(columns=['TOI'])
            df = df.rename(columns={'ATOI': f'Y-{projection_year-year} ATOI', 'GP': f'Y-{projection_year-year} GP'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP', 'Y-0 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)

    # Mofify model coefficients to phase in the current season (Y-0) based on its progression into the season
    atoi_model_data = list(atoi_model.coef_)
    atoi_model_data.append(atoi_model.intercept_)
    max_gp = combined_df['Y-0 GP'].max()
    atoi_model_data = np.insert(atoi_model_data, 3, atoi_model_data[2])
    atoi_model_data[0] = (0 - atoi_model_data[0])*max_gp/82 + atoi_model_data[0]
    atoi_model_data[1] = (atoi_model_data[0] - atoi_model_data[1])*max_gp/82 + atoi_model_data[1]
    atoi_model_data[2] = (atoi_model_data[1] - atoi_model_data[2])*max_gp/82 + atoi_model_data[2]

    # Calculate projected ATOI using weighted averages and bias
    combined_df['Y-3 Score'] = combined_df['Y-3 ATOI']*combined_df['Y-3 GP']*atoi_model_data[0]
    combined_df['Y-2 Score'] = combined_df['Y-2 ATOI']*combined_df['Y-2 GP']*atoi_model_data[1]
    combined_df['Y-1 Score'] = combined_df['Y-1 ATOI']*combined_df['Y-1 GP']*atoi_model_data[2]
    combined_df['Y-0 Score'] = combined_df['Y-0 ATOI']*combined_df['Y-0 GP']*atoi_model_data[3]
    combined_df['Age Score'] = combined_df['Y-0 Age']*atoi_model_data[4] + combined_df['Y-0 Age']*combined_df['Y-0 Age']*atoi_model_data[5] + combined_df['Y-0 Age']*combined_df['Y-0 Age']*combined_df['Y-0 Age']*atoi_model_data[6] + atoi_model_data[7]
    combined_df['Weight'] = combined_df['Y-3 GP']*atoi_model_data[0] + combined_df['Y-2 GP']*atoi_model_data[1] + combined_df['Y-1 GP']*atoi_model_data[2] + combined_df['Y-0 GP']*atoi_model_data[3]
    combined_df['Score'] = combined_df['Y-3 Score'] + combined_df['Y-2 Score'] + combined_df['Y-1 Score'] + combined_df['Y-0 Score'] + combined_df['Age Score']
    combined_df['Proj. ATOI'] = combined_df['Score']/combined_df['Weight']
    combined_df['Proj. ATOI'] = combined_df['Proj. ATOI'].apply(lambda x: max(min(x, 30), 2))

    combined_df = combined_df.drop(columns=['Y-3 Score', 'Y-2 Score', 'Y-1 Score', 'Y-0 Score', 'Age Score', 'Weight', 'Score'])
    combined_df.sort_values(by='Proj. ATOI', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'Position', 'Team', 'Y-0 Age', 'Proj. ATOI']]
    combined_df = combined_df.rename(columns={'Y-0 Age': 'Age', 'Proj. ATOI': 'ATOI'})
    combined_df['Age'] = (combined_df['Age'] - 1).astype(int)
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def gp_model_inference(projection_year, player_stat_df, gp_model_data, download_file, verbose):
    player_stat_df = player_stat_df[player_stat_df['PlayerID'] != 0.0]
    player_stat_df = p24_gp_model_inference(projection_year, player_stat_df, gp_model_data[0], verbose)
    player_stat_df = u24_gp_model_inference(projection_year, player_stat_df, gp_model_data[1], verbose)
    player_stat_df['GP_Score'] = player_stat_df['P24_GP_Score'].combine_first(player_stat_df['U24_GP_Score'])
    player_stat_df = player_stat_df.drop(columns=['P24_GP_Score', 'U24_GP_Score'])
    player_stat_df = player_stat_df.sort_values(by=['GP_Score', 'ATOI'], ascending=[False, False])
    player_stat_df = player_stat_df.reset_index(drop=True)

    # Define file paths
    y1_prev_skater_data_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-2}-{projection_year-1}_skater_data.csv')
    y2_prev_skater_data_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-3}-{projection_year-2}_skater_data.csv')
    y3_prev_skater_data_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-4}-{projection_year-3}_skater_data.csv')

    # Read the CSV files
    y1_gp_scaling = pd.read_csv(y1_prev_skater_data_path)['GP'].to_list()
    y2_gp_scaling = pd.read_csv(y2_prev_skater_data_path)['GP'].to_list()
    y3_gp_scaling = pd.read_csv(y3_prev_skater_data_path)['GP'].to_list()

    # Remove nulls
    y1_gp_scaling = [x for x in y1_gp_scaling if str(x) != 'nan']
    y2_gp_scaling = [x for x in y2_gp_scaling if str(x) != 'nan']
    y3_gp_scaling = [x for x in y3_gp_scaling if str(x) != 'nan']

    # Fill the missing rows with random samples
    if len(y1_gp_scaling) >= len(player_stat_df):
        y1_gp_scaling = y1_gp_scaling[:len(player_stat_df)]
    else:
        y1_gp_scaling.extend(random.choices(y1_gp_scaling, k=len(player_stat_df)-len(y1_gp_scaling)))
    if len(y2_gp_scaling) >= len(player_stat_df):
        y2_gp_scaling = y2_gp_scaling[:len(player_stat_df)]
    else:
        y2_gp_scaling.extend(random.choices(y2_gp_scaling, k=len(player_stat_df)-len(y2_gp_scaling)))
    if len(y3_gp_scaling) >= len(player_stat_df):
        y3_gp_scaling = y3_gp_scaling[:len(player_stat_df)]
    else:
        y3_gp_scaling.extend(random.choices(y3_gp_scaling, k=len(player_stat_df)-len(y3_gp_scaling)))

    # Sort lists descending
    y1_gp_scaling.sort(reverse=True)
    y2_gp_scaling.sort(reverse=True)
    y3_gp_scaling.sort(reverse=True)

    # Combine them by averaging
    player_stat_df['GPprb'] = [(y1_gp_scaling[i] + y2_gp_scaling[i] + y3_gp_scaling[i]) / 3 for i in range(len(player_stat_df))]
    player_stat_df['GPprb'] = player_stat_df['GPprb'] / 82
    player_stat_df['GPprb'] = player_stat_df['GPprb'].apply(lambda x: min(x, 0.958))
    player_stat_df = player_stat_df.drop(columns=['GP_Score'])

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def p24_gp_model_inference(projection_year, player_stat_df, model, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP']]
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df[combined_df['Y-0 Age'] >= 24]
    combined_df = combined_df.fillna(0)

    # Inference GP using model
    features = ['Y-3 GP', 'Y-2 GP', 'Y-1 GP']
    combined_df['P24_GP_Score'] = model.predict(MinMaxScaler().fit_transform(combined_df[features]))
    combined_df['P24_GP_Score'] = MinMaxScaler().fit_transform(combined_df[['P24_GP_Score']])
    combined_df.sort_values(by='P24_GP_Score', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    # Phase in current season
    max_gp = min(82, combined_df['Y-0 GP'].max())
    combined_df['P24_GP_Score_Current'] = combined_df['Y-0 GP'].rank(pct=True)
    combined_df['P24_GP_Score'] = combined_df['P24_GP_Score_Current']*max_gp/82 + combined_df['P24_GP_Score']*(1-max_gp/82)
    combined_df = combined_df.drop(columns=['P24_GP_Score_Current'])

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'P24_GP_Score']]
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    return player_stat_df

def u24_gp_model_inference(projection_year, player_stat_df, model, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'First Assists', 'Second Assists']]
            df['ATOI'] = df['TOI']/df['GP']
            df['Points'] = df['Goals'] + df['First Assists'] + df['Second Assists']
            df['P/GP'] = (df['Goals']+df['First Assists']+df['Second Assists'])/df['GP']
            df = df.drop(columns=['TOI', 'Goals', 'First Assists', 'Second Assists'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'Points': f'Y-{projection_year-year} Points',
                'P/GP': f'Y-{projection_year-year} P/GP',
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} Points'] = 0
            df[f'Y-{projection_year-year} P/GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df[combined_df['Y-0 Age'] < 24]
    combined_df = combined_df.fillna(0)

    # Inference GP using model
    features = ['Y-1 GP', 'Y-1 ATOI', 'Y-1 Points']
    boolean_feature = 'PositionBool'
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    scaled_features = MinMaxScaler().fit_transform(combined_df[features])
    combined_features = np.hstack((scaled_features, combined_df[[boolean_feature]].values))
    combined_df['U24_GP_Score'] = model.predict(combined_features)
    combined_df['U24_GP_Score'] = MinMaxScaler().fit_transform(combined_df[['U24_GP_Score']])
    combined_df.sort_values(by='U24_GP_Score', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    # Phase in current season
    max_gp = min(82, combined_df['Y-0 GP'].max())
    combined_df['U24_GP_Score_Current'] = combined_df['Y-0 GP'].rank(pct=True)
    combined_df['U24_GP_Score'] = combined_df['U24_GP_Score_Current']*max_gp/82 + combined_df['U24_GP_Score']*(1-max_gp/82)
    combined_df = combined_df.drop(columns=['U24_GP_Score_Current'])

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'U24_GP_Score']]
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    return player_stat_df

def goal_model_inference(projection_year, player_stat_df, goal_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'ixG', 'Shots', 'iCF', 'Rush Attempts']]
            df['ATOI'] = df['TOI']/df['GP']
            df['Gper1kChunk'] = df['Goals']/df['TOI']/2 * 1000
            df['xGper1kChunk'] = df['ixG']/df['TOI']/2 * 1000
            df['SHper1kChunk'] = df['Shots']/df['TOI']/2 * 1000
            df['iCFper1kChunk'] = df['iCF']/df['TOI']/2 * 1000
            df['RAper1kChunk'] = df['Rush Attempts']/df['TOI']/2 * 1000
            df = df.drop(columns=['TOI', 'Goals', 'ixG', 'Shots', 'iCF', 'Rush Attempts'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'Gper1kChunk': f'Y-{projection_year-year} Gper1kChunk',
                'xGper1kChunk': f'Y-{projection_year-year} xGper1kChunk',
                'SHper1kChunk': f'Y-{projection_year-year} SHper1kChunk',
                'iCFper1kChunk': f'Y-{projection_year-year} iCFper1kChunk',
                'RAper1kChunk': f'Y-{projection_year-year} RAper1kChunk'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} Gper1kChunk'] = 0
            df[f'Y-{projection_year-year} xGper1kChunk'] = 0
            df[f'Y-{projection_year-year} SHper1kChunk'] = 0
            df[f'Y-{projection_year-year} iCFper1kChunk'] = 0
            df[f'Y-{projection_year-year} RAper1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    combined_df = combined_df.dropna(subset=['Player'])
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    features = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    # y3_cols_to_nan = [col for col in combined_df.columns if col.startswith('Y-3') and col != 'Y-3 GP']
    # y2_cols_to_nan = [col for col in combined_df.columns if col.startswith('Y-2') and col != 'Y-2 GP']
    # combined_df.loc[combined_df['Y-3 GP'] < 30, y3_cols_to_nan] = float('nan')
    # combined_df.loc[combined_df['Y-2 GP'] < 30, y2_cols_to_nan] = float('nan')
    y1_cols_to_impute = [col for col in combined_df.columns if col.startswith('Y-1') and col != 'Y-1 GP']
    imputation_qual_df = combined_df[combined_df['Y-1 GP'] >= 70]
    combined_df['SampleReplaceGP'] = combined_df['Y-1 GP'].apply(lambda x: max(50 - x, 0))
    for feature in y1_cols_to_impute:
        replacement_value = imputation_qual_df[feature].mean()-imputation_qual_df[feature].std()
        combined_df[feature] = combined_df.apply(lambda row: (row[feature]*row['Y-1 GP'] + replacement_value*row['SampleReplaceGP']) / (row['Y-1 GP']+row['SampleReplaceGP']), axis=1)

    # create predictions
    try:
        dmatrix = xgb.DMatrix(combined_df[features].values)
        predictions = goal_model.predict(dmatrix)
    except TypeError:
        data = combined_df[features].values
        predictions = goal_model.predict(data)
    combined_df['Proj. Gper1kChunk'] = combined_df['Y-0 GP']/82*combined_df['Y-0 Gper1kChunk'] + (82-combined_df['Y-0 GP'])/82*predictions
    combined_df = combined_df[['PlayerID', 'Player', 'Proj. Gper1kChunk', 'Position', 'Y-0 Age']]
    combined_df.sort_values(by='Proj. Gper1kChunk', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. Gper1kChunk']]
    combined_df = combined_df.rename(columns={'Proj. Gper1kChunk': 'Gper1kChunk'})
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')
    # player_stat_df = player_stat_df[player_stat_df['Player'] != 0]
    # player_stat_df['PlayerID'] = player_stat_df['PlayerID'].astype(int)
    player_stat_df = player_stat_df.reset_index(drop=True)
    player_stat_df = fix_teams(player_stat_df)

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def a1_model_inference(projection_year, player_stat_df, a1_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'First Assists', 'Second Assists', 'Rush Attempts', 'Rebounds Created', 'Takeaways']]
            df['ATOI'] = df['TOI']/df['GP']
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df['RAper1kChunk'] = df['Rush Attempts']/df['TOI']/2 * 1000
            df['RCper1kChunk'] = df['Rebounds Created']/df['TOI']/2 * 1000
            df['TAper1kChunk'] = df['Takeaways']/df['TOI']/2 * 1000
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists', 'Rush Attempts', 'Rebounds Created', 'Takeaways'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk',
                'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk',
                'RAper1kChunk': f'Y-{projection_year-year} RAper1kChunk',
                'RCper1kChunk': f'Y-{projection_year-year} RCper1kChunk',
                'TAper1kChunk': f'Y-{projection_year-year} TAper1kChunk'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0
            df[f'Y-{projection_year-year} RAper1kChunk'] = 0
            df[f'Y-{projection_year-year} RCper1kChunk'] = 0
            df[f'Y-{projection_year-year} TAper1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    features = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    # y3_cols_to_nan = [col for col in combined_df.columns if col.startswith('Y-3') and col != 'Y-3 GP']
    # y2_cols_to_nan = [col for col in combined_df.columns if col.startswith('Y-2') and col != 'Y-2 GP']
    # combined_df.loc[combined_df['Y-3 GP'] < 30, y3_cols_to_nan] = float('nan')
    # combined_df.loc[combined_df['Y-2 GP'] < 30, y2_cols_to_nan] = float('nan')
    y1_cols_to_impute = [col for col in combined_df.columns if col.startswith('Y-1') and col != 'Y-1 GP']
    imputation_qual_df = combined_df[combined_df['Y-1 GP'] >= 70]
    combined_df['SampleReplaceGP'] = combined_df['Y-1 GP'].apply(lambda x: max(50 - x, 0))
    for feature in y1_cols_to_impute:
        replacement_value = imputation_qual_df[feature].mean()-imputation_qual_df[feature].std()
        combined_df[feature] = combined_df.apply(lambda row: (row[feature]*row['Y-1 GP'] + replacement_value*row['SampleReplaceGP']) / (row['Y-1 GP']+row['SampleReplaceGP']), axis=1)

    # create predictions
    predictions = a1_model.predict(combined_df[features], verbose=verbose)
    predictions = predictions.reshape(-1)
    combined_df['Proj. A1per1kChunk'] = combined_df['Y-0 GP']/82*combined_df['Y-0 A1per1kChunk'] + (82-combined_df['Y-0 GP'])/82*predictions

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. A1per1kChunk', 'Position', 'Y-0 Age']]
    combined_df.sort_values(by='Proj. A1per1kChunk', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. A1per1kChunk']]
    combined_df = combined_df.rename(columns={'Proj. A1per1kChunk': 'A1per1kChunk'})
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')
    combined_df['A1per1kChunk'] = combined_df['A1per1kChunk'].apply(lambda x: 0 if x < 0 else x)
    player_stat_df = player_stat_df[player_stat_df['Player'] != 0]
    player_stat_df['PlayerID'] = player_stat_df['PlayerID'].astype(int)
    player_stat_df = player_stat_df.reset_index(drop=True)

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def a2_model_inference(projection_year, player_stat_df, a2_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'First Assists', 'Second Assists', 'Rush Attempts', 'Rebounds Created', 'Takeaways']]
            df['ATOI'] = df['TOI']/df['GP']
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df['RAper1kChunk'] = df['Rush Attempts']/df['TOI']/2 * 1000
            df['RCper1kChunk'] = df['Rebounds Created']/df['TOI']/2 * 1000
            df['TAper1kChunk'] = df['Takeaways']/df['TOI']/2 * 1000
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists', 'Rush Attempts', 'Rebounds Created', 'Takeaways'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk',
                'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk',
                'RAper1kChunk': f'Y-{projection_year-year} RAper1kChunk',
                'RCper1kChunk': f'Y-{projection_year-year} RCper1kChunk',
                'TAper1kChunk': f'Y-{projection_year-year} TAper1kChunk'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0
            df[f'Y-{projection_year-year} RAper1kChunk'] = 0
            df[f'Y-{projection_year-year} RCper1kChunk'] = 0
            df[f'Y-{projection_year-year} TAper1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    features = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    # y3_cols_to_nan = [col for col in combined_df.columns if col.startswith('Y-3') and col != 'Y-3 GP']
    # y2_cols_to_nan = [col for col in combined_df.columns if col.startswith('Y-2') and col != 'Y-2 GP']
    # combined_df.loc[combined_df['Y-3 GP'] < 30, y3_cols_to_nan] = float('nan')
    # combined_df.loc[combined_df['Y-2 GP'] < 30, y2_cols_to_nan] = float('nan')
    y1_cols_to_impute = [col for col in combined_df.columns if col.startswith('Y-1') and col != 'Y-1 GP']
    imputation_qual_df = combined_df[combined_df['Y-1 GP'] >= 70]
    combined_df['SampleReplaceGP'] = combined_df['Y-1 GP'].apply(lambda x: max(50 - x, 0))
    for feature in y1_cols_to_impute:
        replacement_value = imputation_qual_df[feature].mean()-imputation_qual_df[feature].std()
        combined_df[feature] = combined_df.apply(lambda row: (row[feature]*row['Y-1 GP'] + replacement_value*row['SampleReplaceGP']) / (row['Y-1 GP']+row['SampleReplaceGP']), axis=1)

    # create predictions
    predictions = a2_model.predict(combined_df[features], verbose=verbose)
    predictions = predictions.reshape(-1)
    combined_df['Proj. A2per1kChunk'] = combined_df['Y-0 GP']/82*combined_df['Y-0 A2per1kChunk'] + (82-combined_df['Y-0 GP'])/82*predictions

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. A2per1kChunk', 'Position', 'Y-0 Age']]
    combined_df.sort_values(by='Proj. A2per1kChunk', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. A2per1kChunk']]
    combined_df = combined_df.rename(columns={'Proj. A2per1kChunk': 'A2per1kChunk'})
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')
    combined_df['A2per1kChunk'] = combined_df['A2per1kChunk'].apply(lambda x: 0 if x < 0 else x)
    player_stat_df = player_stat_df[player_stat_df['Player'] != 0]
    player_stat_df['PlayerID'] = player_stat_df['PlayerID'].astype(int)
    player_stat_df = player_stat_df.reset_index(drop=True)

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def skater_xga_model_inference(projection_year, player_stat_df, skater_xga_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_onice_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical On-Ice Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'CA/60', 'FA/60', 'SA/60', 'xGA/60', 'GA/60', 'On-Ice SV%']]
            df['ATOI'] = df['TOI']/df['GP']
            df = df.drop(columns=['TOI'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'CA/60': f'Y-{projection_year-year} CA/60',
                'FA/60': f'Y-{projection_year-year} FA/60',
                'SA/60': f'Y-{projection_year-year} SA/60',
                'xGA/60': f'Y-{projection_year-year} xGA/60',
                'GA/60': f'Y-{projection_year-year} GA/60',
                'On-Ice SV%': f'Y-{projection_year-year} oiSV%'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical On-Ice Skater Data', f'{year-2}-{year-1}_skater_onice_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} CA/60'] = 0
            df[f'Y-{projection_year-year} FA/60'] = 0
            df[f'Y-{projection_year-year} SA/60'] = 0
            df[f'Y-{projection_year-year} xGA/60'] = 0
            df[f'Y-{projection_year-year} GA/60'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    try: # model was trained in this session
        predictions = skater_xga_model.predict(combined_df[['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 CA/60', 'Y-2 CA/60', 'Y-1 CA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'PositionBool']])
    except TypeError: # model was loaded in, pre-trained
        data_dmatrix = xgb.DMatrix(combined_df[['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 CA/60', 'Y-2 CA/60', 'Y-1 CA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'PositionBool']])
        predictions = skater_xga_model.predict(data_dmatrix)
    predictions = predictions.reshape(-1)
    combined_df['Proj. xGA/60'] = combined_df['Y-0 GP']/82*combined_df['Y-0 xGA/60'] + (82-combined_df['Y-0 GP'])/82*predictions

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. xGA/60', 'Position', 'Y-0 Age']]
    combined_df.sort_values(by='Proj. xGA/60', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. xGA/60']]
    combined_df = combined_df.rename(columns={'Proj. xGA/60': 'xGA/60'})
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')
    combined_df['xGA/60'] = combined_df['xGA/60'].apply(lambda x: 0 if x < 0 else x)
    player_stat_df = player_stat_df[player_stat_df['Player'] != 0]
    player_stat_df['PlayerID'] = player_stat_df['PlayerID'].astype(int)
    player_stat_df = player_stat_df.reset_index(drop=True)

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID'], how='left')
        player_stat_df = player_stat_df.drop(columns=['Player_y'])
        player_stat_df = player_stat_df.rename(columns={'Player_x': 'Player'})

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def skater_ga_model_inference(projection_year, player_stat_df, skater_ga_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_onice_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical On-Ice Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'CA/60', 'FA/60', 'SA/60', 'xGA/60', 'GA/60', 'On-Ice SV%']]
            df['ATOI'] = df['TOI']/df['GP']
            df = df.drop(columns=['TOI'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'CA/60': f'Y-{projection_year-year} CA/60',
                'FA/60': f'Y-{projection_year-year} FA/60',
                'SA/60': f'Y-{projection_year-year} SA/60',
                'xGA/60': f'Y-{projection_year-year} xGA/60',
                'GA/60': f'Y-{projection_year-year} GA/60',
                'On-Ice SV%': f'Y-{projection_year-year} oiSV%'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical On-Ice Skater Data', f'{year-2}-{year-1}_skater_onice_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} CA/60'] = 0
            df[f'Y-{projection_year-year} FA/60'] = 0
            df[f'Y-{projection_year-year} SA/60'] = 0
            df[f'Y-{projection_year-year} xGA/60'] = 0
            df[f'Y-{projection_year-year} GA/60'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    try: # model was trained in this session
        predictions = skater_ga_model.predict(combined_df[['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'PositionBool']])
    except TypeError: # model was loaded in, pre-trained
        data_dmatrix = xgb.DMatrix(combined_df[['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'PositionBool']])
        predictions = skater_ga_model.predict(data_dmatrix)
    predictions = predictions.reshape(-1)
    combined_df['Proj. GA/60'] = combined_df['Y-0 GP']/82*combined_df['Y-0 GA/60'] + (82-combined_df['Y-0 GP'])/82*predictions

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. GA/60', 'Position', 'Y-0 Age']]
    combined_df.sort_values(by='Proj. GA/60', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['PlayerID', 'Player', 'Proj. GA/60']]
    combined_df = combined_df.rename(columns={'Proj. GA/60': 'GA/60'})
    player_stat_df = player_stat_df.drop_duplicates(subset='PlayerID', keep='last')
    combined_df['GA/60'] = combined_df['GA/60'].apply(lambda x: 0 if x < 0 else x)
    player_stat_df = player_stat_df[player_stat_df['Player'] != 0]
    player_stat_df['PlayerID'] = player_stat_df['PlayerID'].astype(int)
    player_stat_df = player_stat_df.reset_index(drop=True)

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID'], how='left')
        player_stat_df = player_stat_df.drop(columns=['Player_y'])
        player_stat_df = player_stat_df.rename(columns={'Player_x': 'Player'})

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def team_ga_model_inference(projection_year, team_stat_df, player_stat_df, team_ga_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_team_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Team Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return

        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['Team', 'GP', 'TOI', 'Point %', 'CA', 'FA', 'SA', 'GA', 'xGA', 'SCA', 'HDCA', 'HDGA', 'HDSV%', 'SV%']]
            df['CA/GP'] = df['CA']/df['GP']
            df['FA/GP'] = df['FA']/df['GP']
            df['SA/GP'] = df['SA']/df['GP']
            df['GA/GP'] = df['GA']/df['GP']
            df['xGA/GP'] = df['xGA']/df['GP']
            df['SCA/GP'] = df['SCA']/df['GP']
            df['HDCA/GP'] = df['HDCA']/df['GP']
            df['HDGA/GP'] = df['HDGA']/df['GP']
            df = df.drop(columns=['TOI', 'CA', 'FA', 'SA', 'GA', 'xGA', 'SCA', 'HDCA', 'HDGA'])
            df = df.rename(columns={
                'GP': f'Y-{projection_year-year} GP',
                'Point %': f'Y-{projection_year-year} P%',
                'CA/GP': f'Y-{projection_year-year} CA/GP',
                'FA/GP': f'Y-{projection_year-year} FA/GP',
                'SA/GP': f'Y-{projection_year-year} SHA/GP',
                'GA/GP': f'Y-{projection_year-year} GA/GP',
                'xGA/GP': f'Y-{projection_year-year} xGA/GP',
                'SCA/GP': f'Y-{projection_year-year} SCA/GP',
                'HDCA/GP': f'Y-{projection_year-year} HDCA/GP',
                'HDGA/GP': f'Y-{projection_year-year} HDGA/GP',
                'HDSV%': f'Y-{projection_year-year} HDSV%',
                'SV%': f'Y-{projection_year-year} SV%'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Team Data', f'{year-2}-{year-1}_team_data.csv')) # copy last season df
            df = df[['Team']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} P%'] = 0
            df[f'Y-{projection_year-year} CA/GP'] = 0
            df[f'Y-{projection_year-year} FA/GP'] = 0
            df[f'Y-{projection_year-year} SHA/GP'] = 0
            df[f'Y-{projection_year-year} GA/GP'] = 0
            df[f'Y-{projection_year-year} xGA/GP'] = 0
            df[f'Y-{projection_year-year} SCA/GP'] = 0
            df[f'Y-{projection_year-year} HDCA/GP'] = 0
            df[f'Y-{projection_year-year} HDGA/GP'] = 0
            df[f'Y-{projection_year-year} HDSV%'] = 0
            df[f'Y-{projection_year-year} SV%'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Team', how='outer')

    combined_df = combined_df.dropna(subset=['Y-1 GP']).fillna(0)
    try: # model was trained in this session
        predictions = team_ga_model.predict(combined_df[['Y-2 FA/GP', 'Y-1 FA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-2 SV%', 'Y-1 SV%', 'Y-2 P%', 'Y-1 P%']])
    except TypeError: # model was loaded in, pre-trained
        data_dmatrix = xgb.DMatrix(combined_df[['Y-2 FA/GP', 'Y-1 FA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-2 SV%', 'Y-1 SV%', 'Y-2 P%', 'Y-1 P%']])
        predictions = team_ga_model.predict(data_dmatrix)
    predictions = predictions.reshape(-1)
    combined_df['Proj. GA/GP'] = combined_df['Y-0 GP']/82*combined_df['Y-0 GA/GP'] + (82-combined_df['Y-0 GP'])/82*predictions
    nhlapi_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'), index_col=0)
    nhlapi_data = nhlapi_data[['Team Name', 'Abbreviation']].rename(columns={'Team Name': 'Team'})

    combined_df = pd.merge(combined_df[['Team', 'Proj. GA/GP']], nhlapi_data[['Team', 'Abbreviation']], on='Team', how='left')
    combined_df.loc[combined_df['Team'] == 'Montreal Canadiens', 'Abbreviation'] = 'MTL'
    combined_df.loc[combined_df['Team'] == 'St Louis Blues', 'Abbreviation'] = 'STL'
    combined_df.loc[combined_df['Team'] == 'Arizona Coyotes', 'Abbreviation'] = 'UTA'

    combined_df = combined_df[['Team', 'Abbreviation', 'Proj. GA/GP']]
    combined_df.sort_values(by='Proj. GA/GP', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df.rename(columns={'Proj. GA/GP': 'GA/GP'})

    if team_stat_df is None or team_stat_df.empty:
        team_stat_df = combined_df
    else:
        team_stat_df = pd.merge(team_stat_df, combined_df, on='Team', how='left')

    SAMPLED_MU, SAMPLED_SIGMA = 3.110104167, 0.4139170901 ###!
    player_stat_df.rename(columns={'Team': 'Abbreviation'}, inplace=True)
    player_stat_df['pGA/60'] = player_stat_df['GA/60'] * player_stat_df['ATOI']
    player_stat_df['pxGA/60'] = player_stat_df['xGA/60'] * player_stat_df['ATOI']
    team_weighted_ga = player_stat_df.groupby('Abbreviation').apply(lambda x: x['pGA/60'].sum() / x['ATOI'].sum()).reset_index(name='pGA/GP')
    team_weighted_xga = player_stat_df.groupby('Abbreviation').apply(lambda x: x['pxGA/60'].sum() / x['ATOI'].sum()).reset_index(name='pxGA/GP')
    team_stat_df = team_stat_df.merge(team_weighted_ga, on='Abbreviation', how='left')
    team_stat_df = team_stat_df.merge(team_weighted_xga, on='Abbreviation', how='left')
    team_stat_df['Agg GA/GP'] = team_stat_df['GA/GP']*0.81 + team_stat_df['pGA/GP']*0.11 + team_stat_df['pxGA/GP']*0.08 ###!
    team_stat_df['z_score'] = (team_stat_df['Agg GA/GP'] - team_stat_df['Agg GA/GP'].mean()) / team_stat_df['Agg GA/GP'].std()
    team_stat_df['Normalized GA/GP'] = (team_stat_df['z_score'] * SAMPLED_SIGMA) + SAMPLED_MU
    team_stat_df.drop(columns=['z_score'], inplace=True)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_team_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_projections.csv has been downloaded to the following directory: {export_path}')

    return team_stat_df

def display_inferences(projection_year, player_stat_df, bootstrap_df, inference_state, download_file, verbose):

    # Merge standard deviations into player_stat_df
    player_stat_df = player_stat_df.copy()
    bootstrap_df = bootstrap_df.copy()
    bootstrap_df['GPprb'] = bootstrap_df['GP']/82
    bootstrap_df = bootstrap_df.drop(columns=['Player', 'Team', 'Position', 'Age', 'GP'])
    player_stat_df = player_stat_df.merge(bootstrap_df, on=['PlayerID'], how='left', suffixes=('', '_StDev'))
    stable_player_stat_df = player_stat_df.copy()

    # Load the existing stats
    existing_stats = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'First Assists', 'Second Assists', 'Total Assists']]
    gp_remain = 82 - existing_stats.sort_values(by='GP', ascending=False).iloc[29]['GP']
    if verbose:
        print(f'GP remaining: {gp_remain}')

    # Modify the inferences based on the inference state
    if inference_state.upper() in ['TOTAL', 'TOTAL_82']:
        # Show the inferences for the full season (GP-weighted average of current and projected stats)
        existing_stats['ATOI_cur'] = existing_stats['TOI']/existing_stats['GP']
        existing_stats['Gper1kChunk_cur'] = existing_stats['Goals']/existing_stats['TOI']*500
        existing_stats['A1per1kChunk_cur'] = existing_stats['First Assists']/existing_stats['TOI']*500
        existing_stats['A2per1kChunk_cur'] = existing_stats['Second Assists']/existing_stats['TOI']*500
        existing_stats[['ATOI_cur', 'Gper1kChunk_cur', 'A1per1kChunk_cur', 'A2per1kChunk_cur']] = existing_stats[['ATOI_cur', 'Gper1kChunk_cur', 'A1per1kChunk_cur', 'A2per1kChunk_cur']].fillna(0)
        existing_stats = existing_stats.drop(columns=['Player'])
        player_stat_df = player_stat_df.merge(existing_stats, on=['PlayerID'], how='outer')
        player_stat_df['ATOI'] = (player_stat_df['ATOI_cur']*player_stat_df['GP'] + player_stat_df['ATOI']*gp_remain) / (player_stat_df['GP'] + gp_remain)
        player_stat_df['Gper1kChunk'] = (player_stat_df['Gper1kChunk_cur']*player_stat_df['GP'] + player_stat_df['Gper1kChunk']*gp_remain) / (player_stat_df['GP'] + gp_remain)
        player_stat_df['A1per1kChunk'] = (player_stat_df['A1per1kChunk_cur']*player_stat_df['GP'] + player_stat_df['A1per1kChunk']*gp_remain) / (player_stat_df['GP'] + gp_remain)
        player_stat_df['A2per1kChunk'] = (player_stat_df['A2per1kChunk_cur']*player_stat_df['GP'] + player_stat_df['A2per1kChunk']*gp_remain) / (player_stat_df['GP'] + gp_remain)
        player_stat_df['iG/60'] = player_stat_df['Gper1kChunk']/500 * 60
        player_stat_df['iA1/60'] = player_stat_df['A1per1kChunk']/500 * 60
        player_stat_df['iA2/60'] = player_stat_df['A2per1kChunk']/500 * 60
        player_stat_df['iP/60'] = player_stat_df['iG/60'] + player_stat_df['iA1/60'] + player_stat_df['iA2/60']
        player_stat_df[['ATOI', 'iG/60', 'iA1/60', 'iA2/60']] = player_stat_df[['ATOI', 'iG/60', 'iA1/60', 'iA2/60']].fillna(0)
        if inference_state.upper() == 'TOTAL':
            player_stat_df['iGoals'] = player_stat_df['Gper1kChunk']/500 * player_stat_df['ATOI'] * (gp_remain+player_stat_df['GP'])
            player_stat_df['iPoints'] = (player_stat_df['Gper1kChunk']+player_stat_df['A1per1kChunk']+player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI'] * (gp_remain+player_stat_df['GP'])
        elif inference_state.upper() == 'TOTAL_82':
            player_stat_df['iGoals'] = player_stat_df['Gper1kChunk']/500 * player_stat_df['ATOI'] * 82
            player_stat_df['iPoints'] = (player_stat_df['Gper1kChunk']+player_stat_df['A1per1kChunk']+player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI'] * 82
    elif inference_state.upper() == 'REMAIN':
        # Show the inferences for the remaining games (GP estimation)
        player_stat_df['iG/60'] = player_stat_df['Gper1kChunk']/500 * 60
        player_stat_df['iA1/60'] = player_stat_df['A1per1kChunk']/500 * 60
        player_stat_df['iA2/60'] = player_stat_df['A2per1kChunk']/500 * 60
        player_stat_df['iP/60'] = player_stat_df['iG/60'] + player_stat_df['iA1/60'] + player_stat_df['iA2/60']
        player_stat_df['iGoals'] = player_stat_df['Gper1kChunk']/500 * player_stat_df['ATOI'] * gp_remain
        player_stat_df['iPoints'] = (player_stat_df['Gper1kChunk']+player_stat_df['A1per1kChunk']+player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI'] * gp_remain
    elif inference_state.upper() == 'REMAIN_82':
        # Show the inferences for the remaining games, prorated to a full 82 game season
        player_stat_df['iG/60'] = player_stat_df['Gper1kChunk']/500 * 60
        player_stat_df['iA1/60'] = player_stat_df['A1per1kChunk']/500 * 60
        player_stat_df['iA2/60'] = player_stat_df['A2per1kChunk']/500 * 60
        player_stat_df['iP/60'] = player_stat_df['iG/60'] + player_stat_df['iA1/60'] + player_stat_df['iA2/60']
        player_stat_df['iGoals'] = player_stat_df['Gper1kChunk']/500 * player_stat_df['ATOI'] * 82
        player_stat_df['iPoints'] = (player_stat_df['Gper1kChunk']+player_stat_df['A1per1kChunk']+player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI'] * 82
    elif inference_state.upper() == 'EXISTING':
        # Show the existing stats
        player_stat_df = player_stat_df[['PlayerID', 'Position', 'Team', 'Age']]
        player_stat_df = player_stat_df.merge(existing_stats, on=['PlayerID'], how='right')
        player_stat_df['ATOI'] = player_stat_df['TOI']/player_stat_df['GP']
        player_stat_df['Gper1kChunk'] = player_stat_df['Goals']/player_stat_df['TOI']*500
        player_stat_df['A1per1kChunk'] = player_stat_df['First Assists']/player_stat_df['TOI']*500
        player_stat_df['A2per1kChunk'] = player_stat_df['Second Assists']/player_stat_df['TOI']*500
        player_stat_df['iG/60'] = player_stat_df['Gper1kChunk']/500 * 60
        player_stat_df['iA1/60'] = player_stat_df['A1per1kChunk']/500 * 60
        player_stat_df['iA2/60'] = player_stat_df['A2per1kChunk']/500 * 60
        player_stat_df['iP/60'] = player_stat_df['iG/60'] + player_stat_df['iA1/60'] + player_stat_df['iA2/60']
        player_stat_df['iGoals'] = player_stat_df['Goals']
        player_stat_df['iPoints'] = player_stat_df['Total Assists'] + player_stat_df['Goals']
        player_stat_df[['ATOI', 'iG/60', 'iA1/60', 'iA2/60']] = player_stat_df[['ATOI', 'iG/60', 'iA1/60', 'iA2/60']].fillna(0)
    else:
        raise ValueError(f"Invalid inference state '{inference_state}'. Please select an inference state in ('TOTAL', 'REMAING', 'REMAINING_82', 'EXISTING').")
    
    # player_stat_df = player_stat_df[['PlayerID', 'Player', 'Position', 'Team', 'Age', 'ATOI', 'Gper1kChunk', 'A1per1kChunk', 'A2per1kChunk']]
    player_stat_df = player_stat_df[['PlayerID', 'Player', 'Position', 'Team', 'Age', 'ATOI', 'iG/60', 'iA1/60', 'iA2/60', 'iGoals', 'iPoints']]
    player_stat_df = player_stat_df.sort_values(by='iPoints', ascending=False)
    player_stat_df = player_stat_df.reset_index(drop=True)
    # print(player_stat_df.to_string())
    print(player_stat_df.head(50))
    # print(player_stat_df.info())

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        stable_player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

def savitzky_golvay_calibration(projection_year, player_stat_df):
    player_stat_df = savgol_goal_calibration(projection_year, player_stat_df)
    player_stat_df = savgol_a1_calibration(projection_year, player_stat_df)
    player_stat_df = savgol_a2_calibration(projection_year, player_stat_df)
    player_stat_df = gp_inference_calibration(projection_year, player_stat_df)

    return player_stat_df

def savgol_goal_calibration(projection_year, player_stat_df):
    # Train calibration models
    fwd_scaling, fwd_model = train_goal_calibration_model(projection_year=projection_year, retrain_model=False, position='F')
    dfc_scaling, dfc_model = train_goal_calibration_model(projection_year=projection_year, retrain_model=False, position='D')

    # Generate goal inferences and update player_stat_df
    player_stat_df = generate_savgol_goal_inferences(projection_year, player_stat_df, fwd_model, dfc_model)

    # Apply calibration models with sampling
    player_stat_df = player_stat_df.sort_values(by='Gper1kChunk', ascending=False).reset_index(drop=True)
    fwd_scalers_remaining = player_stat_df[player_stat_df['Position'] != 'D'].shape[0] - len(fwd_scaling)
    dfc_scalers_remaining = player_stat_df[player_stat_df['Position'] == 'D'].shape[0] - len(dfc_scaling)
    fwd_scaling_array = np.array(fwd_scaling)
    dfc_scaling_array = np.array(dfc_scaling)
    fwd_threshold = np.mean(fwd_scaling_array) + np.std(fwd_scaling_array)/2
    dfc_threshold = np.mean(dfc_scaling_array) + np.std(dfc_scaling_array)/2
    threshold_fwds = fwd_scaling_array[fwd_scaling < fwd_threshold]
    threshold_dfcs = dfc_scaling_array[dfc_scaling < dfc_threshold]
    sampled_fwds = np.random.choice(threshold_fwds, size=fwd_scalers_remaining, replace=True)
    sampled_dfcs = np.random.choice(threshold_dfcs, size=dfc_scalers_remaining, replace=True)
    fwd_scaling.extend(sampled_fwds)
    dfc_scaling.extend(sampled_dfcs)
    fwd_scaling = sorted(fwd_scaling, reverse=True)
    dfc_scaling = sorted(dfc_scaling, reverse=True)
    fwds_df = player_stat_df[player_stat_df['Position'] != 'D'].copy()
    fwds_df['sGper1kChunk'] = fwd_scaling
    dfcs_df = player_stat_df[player_stat_df['Position'] == 'D'].copy()
    dfcs_df['sGper1kChunk'] = dfc_scaling
    player_stat_df = pd.concat([fwds_df, dfcs_df], axis=0)
    player_stat_df = player_stat_df.sort_values(by='Gper1kChunk', ascending=False).reset_index(drop=True)

    # Apply Savitzky-Golay filter calibration
    player_stat_df['sDiff'] = player_stat_df['sGper1kChunk'] - player_stat_df['Gper1kChunk']
    player_stat_df['RowNum'] = player_stat_df.index + 1
    player_stat_df['Savgol Window'] = player_stat_df['RowNum'].apply(lambda x: 25 - (20 / (1 + np.exp(0.1 * (x - 25)))))
    player_stat_df['sAdj'] = player_stat_df.apply(lambda row: savgol_filter(player_stat_df['sDiff'], int(row['Savgol Window']), 2)[row.name], axis=1)
    sorted_savgol_adj = player_stat_df['sAdj'].sort_values(ascending=False).values
    player_stat_df['sAdj'] = sorted_savgol_adj
    player_stat_df['SavgolGper1kChunk'] = player_stat_df.apply(lambda row: row['Gper1kChunk'] if (row['Gper1kChunk'] + row['sAdj']) < 1 else (row['Gper1kChunk'] + row['sAdj']), axis=1)
    player_stat_df = player_stat_df.drop(columns=['Gper1kChunk', 'sGper1kChunk', 'RowNum', 'Savgol Window', 'sDiff', 'sAdj'])
    player_stat_df = player_stat_df.rename(columns={'SavgolGper1kChunk': 'Gper1kChunk'})

    return player_stat_df

def savgol_a1_calibration(projection_year, player_stat_df):
    # Train calibration models
    fwd_scaling, fwd_model = train_a1_calibration_model(projection_year=projection_year, retrain_model=False, position='F')
    dfc_scaling, dfc_model = train_a1_calibration_model(projection_year=projection_year, retrain_model=False, position='D')

    # Generate goal inferences and update player_stat_df
    player_stat_df = generate_savgol_a1_inferences(projection_year, player_stat_df, fwd_model, dfc_model)

    # Apply calibration models with sampling
    player_stat_df = player_stat_df.sort_values(by='A1per1kChunk', ascending=False).reset_index(drop=True)
    fwd_scalers_remaining = player_stat_df[player_stat_df['Position'] != 'D'].shape[0] - len(fwd_scaling)
    dfc_scalers_remaining = player_stat_df[player_stat_df['Position'] == 'D'].shape[0] - len(dfc_scaling)
    fwd_scaling_array = np.array(fwd_scaling)
    dfc_scaling_array = np.array(dfc_scaling)
    fwd_threshold = np.mean(fwd_scaling_array) + np.std(fwd_scaling_array)/2
    dfc_threshold = np.mean(dfc_scaling_array) + np.std(dfc_scaling_array)/2
    threshold_fwds = fwd_scaling_array[fwd_scaling < fwd_threshold]
    threshold_dfcs = dfc_scaling_array[dfc_scaling < dfc_threshold]
    sampled_fwds = np.random.choice(threshold_fwds, size=fwd_scalers_remaining, replace=True)
    sampled_dfcs = np.random.choice(threshold_dfcs, size=dfc_scalers_remaining, replace=True)
    fwd_scaling.extend(sampled_fwds)
    dfc_scaling.extend(sampled_dfcs)
    fwd_scaling = sorted(fwd_scaling, reverse=True)
    dfc_scaling = sorted(dfc_scaling, reverse=True)
    fwds_df = player_stat_df[player_stat_df['Position'] != 'D'].copy()
    fwds_df['sA1per1kChunk'] = fwd_scaling
    dfcs_df = player_stat_df[player_stat_df['Position'] == 'D'].copy()
    dfcs_df['sA1per1kChunk'] = dfc_scaling
    player_stat_df = pd.concat([fwds_df, dfcs_df], axis=0)
    player_stat_df = player_stat_df.sort_values(by='A1per1kChunk', ascending=False).reset_index(drop=True)

    # Apply Savitzky-Golay filter calibration
    player_stat_df['sDiff'] = player_stat_df['sA1per1kChunk'] - player_stat_df['A1per1kChunk']
    player_stat_df['RowNum'] = player_stat_df.index + 1
    player_stat_df['Savgol Window'] = player_stat_df['RowNum'].apply(lambda x: 25 - (15 / (1 + np.exp(0.1 * (x - 25)))))
    player_stat_df['sAdj'] = player_stat_df.apply(lambda row: savgol_filter(player_stat_df['sDiff'], int(row['Savgol Window']), 2)[row.name], axis=1)
    sorted_savgol_adj = player_stat_df['sAdj'].sort_values(ascending=False).values
    player_stat_df['sAdj'] = sorted_savgol_adj
    player_stat_df['SavgolA1per1kChunk'] = player_stat_df.apply(lambda row: row['A1per1kChunk'] if (row['A1per1kChunk'] + row['sAdj']) < 1 else (row['A1per1kChunk'] + row['sAdj']), axis=1)
    player_stat_df = player_stat_df.drop(columns=['A1per1kChunk', 'sA1per1kChunk', 'RowNum', 'Savgol Window', 'sDiff', 'sAdj'])
    player_stat_df = player_stat_df.rename(columns={'SavgolA1per1kChunk': 'A1per1kChunk'})

    return player_stat_df

def savgol_a2_calibration(projection_year, player_stat_df):
    # Train calibration models
    fwd_scaling, fwd_model = train_a2_calibration_model(projection_year=projection_year, retrain_model=False, position='F')
    dfc_scaling, dfc_model = train_a2_calibration_model(projection_year=projection_year, retrain_model=False, position='D')

    # Generate goal inferences and update player_stat_df
    player_stat_df = generate_savgol_a2_inferences(projection_year, player_stat_df, fwd_model, dfc_model)

    # Apply calibration models with sampling
    player_stat_df = player_stat_df.sort_values(by='A2per1kChunk', ascending=False).reset_index(drop=True)
    fwd_scalers_remaining = player_stat_df[player_stat_df['Position'] != 'D'].shape[0] - len(fwd_scaling)
    dfc_scalers_remaining = player_stat_df[player_stat_df['Position'] == 'D'].shape[0] - len(dfc_scaling)
    fwd_scaling_array = np.array(fwd_scaling)
    dfc_scaling_array = np.array(dfc_scaling)
    fwd_threshold = np.mean(fwd_scaling_array) + np.std(fwd_scaling_array)/2
    dfc_threshold = np.mean(dfc_scaling_array) + np.std(dfc_scaling_array)/2
    threshold_fwds = fwd_scaling_array[fwd_scaling < fwd_threshold]
    threshold_dfcs = dfc_scaling_array[dfc_scaling < dfc_threshold]
    sampled_fwds = np.random.choice(threshold_fwds, size=fwd_scalers_remaining, replace=True)
    sampled_dfcs = np.random.choice(threshold_dfcs, size=dfc_scalers_remaining, replace=True)
    fwd_scaling.extend(sampled_fwds)
    dfc_scaling.extend(sampled_dfcs)
    fwd_scaling = sorted(fwd_scaling, reverse=True)
    dfc_scaling = sorted(dfc_scaling, reverse=True)

    fwds_df = player_stat_df[player_stat_df['Position'] != 'D'].copy()
    fwds_df['sA2per1kChunk'] = fwd_scaling
    dfcs_df = player_stat_df[player_stat_df['Position'] == 'D'].copy()
    dfcs_df['sA2per1kChunk'] = dfc_scaling
    player_stat_df = pd.concat([fwds_df, dfcs_df], axis=0)
    player_stat_df = player_stat_df.sort_values(by='A2per1kChunk', ascending=False).reset_index(drop=True)

    # Apply Savitzky-Golay filter calibration
    player_stat_df['sDiff'] = player_stat_df['sA2per1kChunk'] - player_stat_df['A2per1kChunk']
    player_stat_df['RowNum'] = player_stat_df.index + 1
    player_stat_df['Savgol Window'] = player_stat_df['RowNum'].apply(lambda x: 25 - (15 / (1 + np.exp(0.1 * (x - 25)))))
    player_stat_df['sAdj'] = player_stat_df.apply(lambda row: savgol_filter(player_stat_df['sDiff'], int(row['Savgol Window']), 2)[row.name], axis=1)
    sorted_savgol_adj = player_stat_df['sAdj'].sort_values(ascending=False).values
    player_stat_df['sAdj'] = sorted_savgol_adj
    player_stat_df['SavgolA2per1kChunk'] = player_stat_df.apply(lambda row: row['A2per1kChunk'] if (row['A2per1kChunk'] + row['sAdj']) < 1 else (row['A2per1kChunk'] + row['sAdj']), axis=1)
    player_stat_df = player_stat_df.drop(columns=['A2per1kChunk', 'sA2per1kChunk', 'RowNum', 'Savgol Window', 'sDiff', 'sAdj'])
    player_stat_df = player_stat_df.rename(columns={'SavgolA2per1kChunk': 'A2per1kChunk'})

    return player_stat_df

def gp_inference_calibration(projection_year, player_stat_df):
    # Apply savgol-Pbased GP recalibration
    player_stat_df['PosFD'] = player_stat_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    player_stat_df['Pts'] = (player_stat_df['Gper1kChunk'] + player_stat_df['A1per1kChunk'] + player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI']
    player_stat_df['pPts'] = player_stat_df.groupby('PosFD')['Pts'].rank(pct=True)
    player_stat_df['GPprb'] = player_stat_df['GPprb']*0.227618 + player_stat_df['pPts']*0.772382
    player_stat_df = player_stat_df.drop(columns=['PosFD', 'Pts', 'pPts'])

    # Phase in current season
    y0_skater_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))[['PlayerID', 'Player', 'GP']]
    y0_skater_data = y0_skater_data.dropna()
    y0_gp_max = y0_skater_data['GP'].max()
    player_stat_df = pd.merge(player_stat_df, y0_skater_data, on=['PlayerID', 'Player'], how='outer')
    player_stat_df['GPprbPrime'] = player_stat_df['GP']*y0_gp_max/82 + player_stat_df['GPprb']*(82-player_stat_df['GP'])/82
    player_stat_df['GPprbPrime'] = player_stat_df['GPprbPrime'].fillna(player_stat_df['GPprb'])
    player_stat_df['GPprbPrime'] = player_stat_df['GPprb'].fillna(player_stat_df['GP'] / 82)
    player_stat_df = player_stat_df.dropna(subset=['GP'])
    player_stat_df = player_stat_df.drop(columns=['GP', 'GPprb'])
    player_stat_df = player_stat_df.rename(columns={'GPprbPrime': 'GPprb'})

    return player_stat_df

def generate_savgol_goal_inferences(projection_year, player_stat_df, fwd_model, dfc_model):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'ixG']]
            df['ATOI'] = df['TOI']/df['GP']
            df['Gper1kChunk'] = df['Goals']/df['TOI']/2 * 1000
            df['xGper1kChunk'] = df['ixG']/df['TOI']/2 * 1000
            df = df.drop(columns=['TOI', 'Goals', 'ixG'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'Gper1kChunk': f'Y-{projection_year-year} Gper1kChunk',
                'xGper1kChunk': f'Y-{projection_year-year} xGper1kChunk'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} Gper1kChunk'] = 0
            df[f'Y-{projection_year-year} xGper1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    combined_df = combined_df.dropna(subset=['Player'])
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    young_combined_df = combined_df[combined_df['Y-0 Age'] <= 22].copy()
    young_combined_df['PositionFD'] = young_combined_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    features = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    y1_cols_to_impute = [col for col in features if col.startswith('Y-1') and col != 'Y-1 GP']
    imputation_qual_df = combined_df[combined_df['Y-1 GP'] >= 70]
    young_combined_df['SampleReplaceGP'] = young_combined_df['Y-1 GP'].apply(lambda x: max(50 - x, 0))
    for feature in y1_cols_to_impute:
        replacement_value = imputation_qual_df[feature].mean()-imputation_qual_df[feature].std()
        young_combined_df[feature] = young_combined_df.apply(lambda row: (row[feature]*row['Y-1 GP'] + replacement_value*row['SampleReplaceGP']) / (row['Y-1 GP']+row['SampleReplaceGP']), axis=1)

    # create savgol-based inferences
    fwd_young_combined_df = young_combined_df[young_combined_df['PositionFD'] == 'F'].copy()
    dfc_young_combined_df = young_combined_df[young_combined_df['PositionFD'] == 'D'].copy()
    fwd_young_combined_df['Pre-Adj Gper1kChunk'] = (
        young_combined_df['Y-3 xGper1kChunk']*young_combined_df['Y-3 GP']*fwd_model.coef_[0]*0.5 + 
        young_combined_df['Y-3 Gper1kChunk']*young_combined_df['Y-3 GP']*fwd_model.coef_[0]*0.5 + 
        young_combined_df['Y-2 xGper1kChunk']*young_combined_df['Y-2 GP']*fwd_model.coef_[1]*0.5 + 
        young_combined_df['Y-2 Gper1kChunk']*young_combined_df['Y-2 GP']*fwd_model.coef_[1]*0.5 +
        young_combined_df['Y-1 xGper1kChunk']*young_combined_df['Y-1 GP']*fwd_model.coef_[2]*0.5 +
        young_combined_df['Y-1 Gper1kChunk']*young_combined_df['Y-1 GP']*fwd_model.coef_[2]*0.5
    ) / (young_combined_df['Y-3 GP']*fwd_model.coef_[0] + young_combined_df['Y-2 GP']*fwd_model.coef_[1] + young_combined_df['Y-1 GP']*fwd_model.coef_[2])
    dfc_young_combined_df['Pre-Adj Gper1kChunk'] = (
        young_combined_df['Y-3 xGper1kChunk']*young_combined_df['Y-3 GP']*dfc_model.coef_[0]*0.5 + 
        young_combined_df['Y-3 Gper1kChunk']*young_combined_df['Y-3 GP']*dfc_model.coef_[0]*0.5 + 
        young_combined_df['Y-2 xGper1kChunk']*young_combined_df['Y-2 GP']*dfc_model.coef_[1]*0.5 + 
        young_combined_df['Y-2 Gper1kChunk']*young_combined_df['Y-2 GP']*dfc_model.coef_[1]*0.5 +
        young_combined_df['Y-1 xGper1kChunk']*young_combined_df['Y-1 GP']*dfc_model.coef_[2]*0.5 +
        young_combined_df['Y-1 Gper1kChunk']*young_combined_df['Y-1 GP']*dfc_model.coef_[2]*0.5
    ) / (young_combined_df['Y-3 GP']*dfc_model.coef_[0] + young_combined_df['Y-2 GP']*dfc_model.coef_[1] + young_combined_df['Y-1 GP']*dfc_model.coef_[2])
    young_combined_df = pd.concat([fwd_young_combined_df, dfc_young_combined_df])

    # create adjustments
    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data[train_data['Y-0 Age'] <= 21]
    train_data = train_data[train_data['Y-1 GP'] >= 50]
    train_data['PositionFD'] = train_data['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    train_data['Y-1 BlendGoals'] = train_data.apply(lambda row: row['Y-1 Gper1kChunk']*0.5 + row['Y-1 xGper1kChunk']*0.5 if row['PositionFD'] == 'F' else row['Y-1 Gper1kChunk']*0.2 + row['Y-1 xGper1kChunk']*0.8, axis=1)
    train_data['Y-0 BlendGoals'] = train_data.apply(lambda row: row['Y-0 Gper1kChunk']*0.5 + row['Y-0 xGper1kChunk']*0.5 if row['PositionFD'] == 'F' else row['Y-0 Gper1kChunk']*0.2 + row['Y-0 xGper1kChunk']*0.8, axis=1)
    train_data['BlendDiff'] = train_data['Y-0 BlendGoals'] - train_data['Y-1 BlendGoals']
    train_data['BlendDiff'] = train_data['BlendDiff'].apply(lambda x: x if x >= 0 else 0)
    train_data = train_data[['Player', 'PositionFD', 'Y-0', 'Y-0 Age', 'Y-1 BlendGoals', 'Y-0 BlendGoals', 'BlendDiff']]
    young_adj_df = train_data[['PositionFD', 'Y-0 Age', 'BlendDiff']].groupby(['PositionFD', 'Y-0 Age']).mean().reset_index()
    young_adj_df['AdjBlend'] = young_adj_df.apply(lambda row: young_adj_df.loc[(young_adj_df['PositionFD'] == row['PositionFD']) & (young_adj_df['Y-0 Age'] != row['Y-0 Age']), 'BlendDiff'].values[0]*0.25 + row['BlendDiff']*0.75 if row['Y-0 Age'] == 20 else row['BlendDiff'], axis=1)
    young_adj_df = young_adj_df[['PositionFD', 'Y-0 Age', 'AdjBlend']].rename(columns={'AdjBlend': 'Adjustment'})
    df_age22 = young_adj_df[young_adj_df['Y-0 Age'] == 21].copy()
    df_age22['Y-0 Age'] = 22
    young_adj_df = pd.concat([young_adj_df, df_age22])

    # join in adjustments
    young_combined_df['PositionFD'] = young_combined_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    young_combined_df = young_combined_df.merge(young_adj_df, on=['PositionFD', 'Y-0 Age'], how='left')
    young_combined_df['Adj Gper1kChunk'] = young_combined_df['Pre-Adj Gper1kChunk'] + young_combined_df['Adjustment']
    young_combined_df['Proj. Gper1kChunk'] = young_combined_df['Y-0 GP']/82*young_combined_df['Y-0 Gper1kChunk'] + (82-young_combined_df['Y-0 GP'])/82*young_combined_df['Adj Gper1kChunk']
    young_combined_df = young_combined_df[['PlayerID', 'Player', 'Proj. Gper1kChunk']]
    young_combined_df = young_combined_df.rename(columns={'Proj. Gper1kChunk': 'Gper1kChunk'})

    # merge inferences into player_stat_df
    merged_df = pd.merge(player_stat_df, young_combined_df[['PlayerID', 'Gper1kChunk']], on='PlayerID', how='left', suffixes=('', '_updated'))
    player_stat_df['Gper1kChunk'] = merged_df['Gper1kChunk_updated'].combine_first(merged_df['Gper1kChunk'])

    return player_stat_df

def generate_savgol_a1_inferences(projection_year, player_stat_df, fwd_model, dfc_model):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'First Assists', 'Second Assists']]
            df['ATOI'] = df['TOI']/df['GP']
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk',
                'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    combined_df = combined_df.dropna(subset=['Player'])
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    young_combined_df = combined_df[combined_df['Y-0 Age'] <= 22].copy()
    young_combined_df['PositionFD'] = young_combined_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    features = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    y1_cols_to_impute = [col for col in features if col.startswith('Y-1') and col != 'Y-1 GP']
    imputation_qual_df = combined_df[combined_df['Y-1 GP'] >= 70]
    young_combined_df['SampleReplaceGP'] = young_combined_df['Y-1 GP'].apply(lambda x: max(50 - x, 0))
    for feature in y1_cols_to_impute:
        replacement_value = imputation_qual_df[feature].mean()-imputation_qual_df[feature].std()
        young_combined_df[feature] = young_combined_df.apply(lambda row: (row[feature]*row['Y-1 GP'] + replacement_value*row['SampleReplaceGP']) / (row['Y-1 GP']+row['SampleReplaceGP']), axis=1)

    # create savgol-based inferences
    fwd_young_combined_df = young_combined_df[young_combined_df['PositionFD'] == 'F'].copy()
    dfc_young_combined_df = young_combined_df[young_combined_df['PositionFD'] == 'D'].copy()
    fwd_young_combined_df['Pre-Adj A1per1kChunk'] = (
        young_combined_df['Y-3 A1per1kChunk']*young_combined_df['Y-3 GP']*fwd_model.coef_[0] + 
        young_combined_df['Y-2 A1per1kChunk']*young_combined_df['Y-2 GP']*fwd_model.coef_[1] + 
        young_combined_df['Y-1 A1per1kChunk']*young_combined_df['Y-1 GP']*fwd_model.coef_[2]
    ) / (young_combined_df['Y-3 GP']*fwd_model.coef_[0] + young_combined_df['Y-2 GP']*fwd_model.coef_[1] + young_combined_df['Y-1 GP']*fwd_model.coef_[2])
    dfc_young_combined_df['Pre-Adj A2per1kChunk'] = (
        young_combined_df['Y-3 A1per1kChunk']*young_combined_df['Y-3 GP']*dfc_model.coef_[0] + 
        young_combined_df['Y-2 A1per1kChunk']*young_combined_df['Y-2 GP']*dfc_model.coef_[1] +
        young_combined_df['Y-1 A1per1kChunk']*young_combined_df['Y-1 GP']*dfc_model.coef_[2]
    ) / (young_combined_df['Y-3 GP']*dfc_model.coef_[0] + young_combined_df['Y-2 GP']*dfc_model.coef_[1] + young_combined_df['Y-1 GP']*dfc_model.coef_[2])
    young_combined_df = pd.concat([fwd_young_combined_df, dfc_young_combined_df])

    # create adjustments
    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data[train_data['Y-0 Age'] <= 21]
    train_data = train_data[train_data['Y-1 GP'] >= 50]
    train_data['PositionFD'] = train_data['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    train_data['A1per1kChunkDiff'] = train_data['Y-0 A1per1kChunk'] - train_data['Y-1 A1per1kChunk']
    train_data['A1per1kChunkDiff'] = train_data['A1per1kChunkDiff'].apply(lambda x: x if x >= 0 else 0)
    train_data = train_data[['Player', 'PositionFD', 'Y-0', 'Y-0 Age', 'Y-1 A1per1kChunk', 'Y-0 A1per1kChunk', 'A1per1kChunkDiff']]
    young_adj_df = train_data[['PositionFD', 'Y-0 Age', 'A1per1kChunkDiff']].groupby(['PositionFD', 'Y-0 Age']).mean().reset_index()
    young_adj_df['Adjustment'] = young_adj_df.apply(lambda row: young_adj_df.loc[(young_adj_df['PositionFD'] == row['PositionFD']) & (young_adj_df['Y-0 Age'] != row['Y-0 Age']), 'A1per1kChunkDiff'].values[0]*0.25 + row['A1per1kChunkDiff']*0.75 if row['Y-0 Age'] == 20 else row['A1per1kChunkDiff'], axis=1)
    young_adj_df = young_adj_df[['PositionFD', 'Y-0 Age', 'Adjustment']]
    df_age22 = young_adj_df[young_adj_df['Y-0 Age'] == 21].copy()
    df_age22['Y-0 Age'] = 22
    young_adj_df = pd.concat([young_adj_df, df_age22])

    # join in adjustments
    young_combined_df['PositionFD'] = young_combined_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    young_combined_df = young_combined_df.merge(young_adj_df, on=['PositionFD', 'Y-0 Age'], how='left')
    young_combined_df['Adj A1per1kChunk'] = young_combined_df['Pre-Adj A1per1kChunk'] + young_combined_df['Adjustment']
    young_combined_df['Proj. A1per1kChunk'] = young_combined_df['Y-0 GP']/82*young_combined_df['Y-0 A1per1kChunk'] + (82-young_combined_df['Y-0 GP'])/82*young_combined_df['Adj A1per1kChunk']
    young_combined_df = young_combined_df[['PlayerID', 'Player', 'Proj. A1per1kChunk']]
    young_combined_df = young_combined_df.rename(columns={'Proj. A1per1kChunk': 'A1per1kChunk'})

    # merge inferences into player_stat_df
    merged_df = pd.merge(player_stat_df, young_combined_df[['PlayerID', 'A1per1kChunk']], on='PlayerID', how='left', suffixes=('', '_updated'))
    player_stat_df['A1per1kChunk'] = merged_df['A1per1kChunk_updated'].combine_first(merged_df['A1per1kChunk'])

    return player_stat_df

def generate_savgol_a2_inferences(projection_year, player_stat_df, fwd_model, dfc_model):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'First Assists', 'Second Assists']]
            df['ATOI'] = df['TOI']/df['GP']
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists'])
            df = df.rename(columns={
                'ATOI': f'Y-{projection_year-year} ATOI', 
                'GP': f'Y-{projection_year-year} GP', 
                'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk',
                'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk'
            })
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    combined_df = combined_df.dropna(subset=['Player'])
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    young_combined_df = combined_df[combined_df['Y-0 Age'] <= 22].copy()
    young_combined_df['PositionFD'] = young_combined_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    features = ['Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    y1_cols_to_impute = [col for col in features if col.startswith('Y-1') and col != 'Y-1 GP']
    imputation_qual_df = combined_df[combined_df['Y-1 GP'] >= 70]
    young_combined_df['SampleReplaceGP'] = young_combined_df['Y-1 GP'].apply(lambda x: max(50 - x, 0))
    for feature in y1_cols_to_impute:
        replacement_value = imputation_qual_df[feature].mean()-imputation_qual_df[feature].std()
        young_combined_df[feature] = young_combined_df.apply(lambda row: (row[feature]*row['Y-1 GP'] + replacement_value*row['SampleReplaceGP']) / (row['Y-1 GP']+row['SampleReplaceGP']), axis=1)

    # create savgol-based inferences
    fwd_young_combined_df = young_combined_df[young_combined_df['PositionFD'] == 'F'].copy()
    dfc_young_combined_df = young_combined_df[young_combined_df['PositionFD'] == 'D'].copy()
    fwd_young_combined_df['Pre-Adj A2per1kChunk'] = (
        young_combined_df['Y-3 A2per1kChunk']*young_combined_df['Y-3 GP']*fwd_model.coef_[0] + 
        young_combined_df['Y-2 A2per1kChunk']*young_combined_df['Y-2 GP']*fwd_model.coef_[1] + 
        young_combined_df['Y-1 A2per1kChunk']*young_combined_df['Y-1 GP']*fwd_model.coef_[2]
    ) / (young_combined_df['Y-3 GP']*fwd_model.coef_[0] + young_combined_df['Y-2 GP']*fwd_model.coef_[1] + young_combined_df['Y-1 GP']*fwd_model.coef_[2])
    dfc_young_combined_df['Pre-Adj A2per1kChunk'] = (
        young_combined_df['Y-3 A2per1kChunk']*young_combined_df['Y-3 GP']*dfc_model.coef_[0] + 
        young_combined_df['Y-2 A2per1kChunk']*young_combined_df['Y-2 GP']*dfc_model.coef_[1] +
        young_combined_df['Y-1 A2per1kChunk']*young_combined_df['Y-1 GP']*dfc_model.coef_[2]
    ) / (young_combined_df['Y-3 GP']*dfc_model.coef_[0] + young_combined_df['Y-2 GP']*dfc_model.coef_[1] + young_combined_df['Y-1 GP']*dfc_model.coef_[2])
    young_combined_df = pd.concat([fwd_young_combined_df, dfc_young_combined_df])

    # create adjustments
    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data[train_data['Y-0 Age'] <= 21]
    train_data = train_data[train_data['Y-1 GP'] >= 50]
    train_data['PositionFD'] = train_data['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    train_data['A2per1kChunkDiff'] = train_data['Y-0 A2per1kChunk'] - train_data['Y-1 A2per1kChunk']
    train_data['A2per1kChunkDiff'] = train_data['A2per1kChunkDiff'].apply(lambda x: x if x >= 0 else 0)
    train_data = train_data[['Player', 'PositionFD', 'Y-0', 'Y-0 Age', 'Y-1 A2per1kChunk', 'Y-0 A2per1kChunk', 'A2per1kChunkDiff']]
    young_adj_df = train_data[['PositionFD', 'Y-0 Age', 'A2per1kChunkDiff']].groupby(['PositionFD', 'Y-0 Age']).mean().reset_index()
    young_adj_df['Adjustment'] = young_adj_df.apply(lambda row: young_adj_df.loc[(young_adj_df['PositionFD'] == row['PositionFD']) & (young_adj_df['Y-0 Age'] != row['Y-0 Age']), 'A2per1kChunkDiff'].values[0]*0.25 + row['A2per1kChunkDiff']*0.75 if row['Y-0 Age'] == 20 else row['A2per1kChunkDiff'], axis=1)
    young_adj_df = young_adj_df[['PositionFD', 'Y-0 Age', 'Adjustment']]
    df_age22 = young_adj_df[young_adj_df['Y-0 Age'] == 21].copy()
    df_age22['Y-0 Age'] = 22
    young_adj_df = pd.concat([young_adj_df, df_age22])

    # join in adjustments
    young_combined_df['PositionFD'] = young_combined_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    young_combined_df = young_combined_df.merge(young_adj_df, on=['PositionFD', 'Y-0 Age'], how='left')
    young_combined_df['Adj A2per1kChunk'] = young_combined_df['Pre-Adj A2per1kChunk'] + young_combined_df['Adjustment']
    young_combined_df['Proj. A2per1kChunk'] = young_combined_df['Y-0 GP']/82*young_combined_df['Y-0 A2per1kChunk'] + (82-young_combined_df['Y-0 GP'])/82*young_combined_df['Adj A2per1kChunk']
    young_combined_df = young_combined_df[['PlayerID', 'Player', 'Proj. A2per1kChunk']]
    young_combined_df = young_combined_df.rename(columns={'Proj. A2per1kChunk': 'A2per1kChunk'})

    # merge inferences into player_stat_df
    merged_df = pd.merge(player_stat_df, young_combined_df[['PlayerID', 'A2per1kChunk']], on='PlayerID', how='left', suffixes=('', '_updated'))
    player_stat_df['A2per1kChunk'] = merged_df['A2per1kChunk_updated'].combine_first(merged_df['A2per1kChunk'])

    return player_stat_df

def bootstrap_atoi_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'atoi_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'residual_variance.json')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        train_data['Y-3 Pper1kChunk'] = train_data['Y-3 Gper1kChunk'] + train_data['Y-3 A1per1kChunk'] + train_data['Y-3 A2per1kChunk']
        train_data['Y-2 Pper1kChunk'] = train_data['Y-2 Gper1kChunk'] + train_data['Y-2 A1per1kChunk'] + train_data['Y-2 A2per1kChunk']
        train_data['Y-1 Pper1kChunk'] = train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk']

        features = ['Y-3 ATOI', 'Y-3 GP', 'Y-3 Pper1kChunk', 'Y-2 ATOI', 'Y-2 GP', 'Y-2 Pper1kChunk', 'Y-1 ATOI', 'Y-1 GP', 'Y-1 Pper1kChunk', 'Y-0 Age', 'PositionBool']
        target_var = 'Y-0 ATOI'

        # Define X and y
        X = train_data[features]
        y = train_data[target_var]

        # Hyperparameters for XGBoost
        params = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'objective': 'reg:squarederror'
        }

        # Loop through the bootstrap samples, training new samples and storing in models list
        models, discrepancies = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping ATOI"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            errors = y_test - y_test_pred
            discrepancies.extend(errors)
            models.append(model)
        residual_variance = np.var(discrepancies)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify discrepancies json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['ATOI'] = residual_variance
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['ATOI']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'First Assists', 'Second Assists']]
            df['ATOI'] = df['TOI']/df['GP']
            df['Pper1kChunk'] = (df['Goals'] + df['First Assists'] + df['Second Assists'])/df['TOI']/2 * 1000
            df[['ATOI', 'GP', 'Pper1kChunk']] = df[['ATOI', 'GP', 'Pper1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'Goals', 'First Assists', 'Second Assists'])
            df = df.rename(columns={'ATOI': f'Y-{projection_year-year} ATOI', 'GP': f'Y-{projection_year-year} GP', 'Pper1kChunk': f'Y-{projection_year-year} Pper1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Pper1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP', 'Y-0 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    # Generate predictions
    features = ['Y-3 ATOI', 'Y-3 GP', 'Y-3 Pper1kChunk', 'Y-2 ATOI', 'Y-2 GP', 'Y-2 Pper1kChunk', 'Y-1 ATOI', 'Y-1 GP', 'Y-1 Pper1kChunk', 'Y-0 Age', 'PositionBool']
    X_pred = combined_df[features]
    predictions = np.zeros((len(combined_df), bootstrap_samples))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_pred)
    bootstrap_variances = np.var(predictions, axis=1)
    bootstrap_variances *= residual_variance/np.mean(bootstrap_variances)
    bootstrap_stdevs = np.sqrt(bootstrap_variances)
    combined_df['ATOI'] = bootstrap_stdevs

    # Adjust for current season progress
    combined_df['ATOI'] = combined_df['ATOI'] * np.sqrt(1 - combined_df['Y-0 GP']/82)

    # Merge adjusted variance inferences into bootstrap_df
    if bootstrap_df is None or bootstrap_df.empty:
        combined_df.rename(columns={'Y-0 Age': 'Age'}, inplace=True)
        bootstrap_df = combined_df[['PlayerID', 'Player', 'Team', 'Position', 'Age']].copy()
        bootstrap_df['Age'] = bootstrap_df['Age'] - 1
        bootstrap_df['ATOI'] = bootstrap_variances
    else:
        bootstrap_df = pd.merge(bootstrap_df, combined_df[['PlayerID', 'ATOI']], on='PlayerID', how='left')

    if verbose:
        print(f'Bootstrapped ATOI inferences for {projection_year} have been generated')
        print(bootstrap_df)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_gp_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'gp_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'residual_variance.json')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)

        features = ['Y-3 GP', 'Y-3 Points', 'Y-2 GP', 'Y-2 Points', 'Y-1 GP', 'Y-1 Points', 'Y-0 Age', 'PositionBool']
        target_var = 'Y-0 GP'

        # Define X and y
        X = train_data[features]
        y = train_data[target_var]

        # Hyperparameters for XGBoost
        params = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'objective': 'reg:squarederror'
        }

        # Loop through the bootstrap samples, training new samples and storing in models list
        models, discrepancies = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping GP"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            errors = y_test - y_test_pred
            discrepancies.extend(errors)
            models.append(model)
        residual_variance = np.var(discrepancies)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify discrepancies json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['GP'] = residual_variance
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['GP']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'Total Points']]
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'Total Points': f'Y-{projection_year-year} Points'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Points'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP', 'Y-0 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    # Generate predictions
    features = ['Y-3 GP', 'Y-3 Points', 'Y-2 GP', 'Y-2 Points', 'Y-1 GP', 'Y-1 Points', 'Y-0 Age', 'PositionBool']
    X_pred = combined_df[features]
    predictions = np.zeros((len(combined_df), bootstrap_samples))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_pred)
    bootstrap_variances = np.var(predictions, axis=1)
    bootstrap_variances *= residual_variance/np.mean(bootstrap_variances)
    bootstrap_stdevs = np.sqrt(bootstrap_variances)
    combined_df['GP'] = bootstrap_stdevs

    # Adjust for current season progress
    combined_df['GP'] = combined_df['GP'] * np.sqrt(1 - combined_df['Y-0 GP']/82)

    # Merge adjusted variance inferences into bootstrap_df
    if bootstrap_df is None or bootstrap_df.empty:
        combined_df.rename(columns={'Y-0 Age': 'Age'}, inplace=True)
        bootstrap_df = combined_df[['PlayerID', 'Player', 'Team', 'Position', 'Age']].copy()
        bootstrap_df['Age'] = bootstrap_df['Age'] - 1
        bootstrap_df['GP'] = bootstrap_variances
    else:
        bootstrap_df = pd.merge(bootstrap_df, combined_df[['PlayerID', 'GP']], on='PlayerID', how='left')

    if verbose:
        print(f'Bootstrapped GP inferences for {projection_year} have been generated')
        print(bootstrap_df)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_goal_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'goal_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'residual_variance.json')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)

        features = ['Y-3 GP', 'Y-3 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 GP', 'Y-2 Gper1kChunk', 'Y-2 xGper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 GP', 'Y-1 Gper1kChunk', 'Y-1 xGper1kChunk', 'Y-1 SHper1kChunk', 'Y-0 Age', 'PositionBool']
        target_var = 'Y-0 Gper1kChunk'

        # Define X and y
        X = train_data[features]
        y = train_data[target_var]

        # Hyperparameters for XGBoost
        params = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 150,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'objective': 'reg:squarederror'
        }

        # Loop through the bootstrap samples, training new samples and storing in models list
        models, discrepancies = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping Goals"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            errors = y_test - y_test_pred
            discrepancies.extend(errors)
            models.append(model)
        residual_variance = np.var(discrepancies)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify discrepancies json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['Gper1kChunk'] = residual_variance
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['Gper1kChunk']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'ixG', 'Shots']]
            df['Gper1kChunk'] = df['Goals']/df['TOI']/2 * 1000
            df['xGper1kChunk'] = df['ixG']/df['TOI']/2 * 1000
            df['SHper1kChunk'] = df['Shots']/df['TOI']/2 * 1000
            df[['Gper1kChunk', 'xGper1kChunk', 'SHper1kChunk']] = df[['Gper1kChunk', 'xGper1kChunk', 'SHper1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'Goals', 'ixG', 'Shots'])
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'Gper1kChunk': f'Y-{projection_year-year} Gper1kChunk', 'xGper1kChunk': f'Y-{projection_year-year} xGper1kChunk', 'SHper1kChunk': f'Y-{projection_year-year} SHper1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Gper1kChunk'] = 0
            df[f'Y-{projection_year-year} xGper1kChunk'] = 0
            df[f'Y-{projection_year-year} SHper1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP', 'Y-0 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    # Generate predictions
    features = ['Y-3 GP', 'Y-3 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 GP', 'Y-2 Gper1kChunk', 'Y-2 xGper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 GP', 'Y-1 Gper1kChunk', 'Y-1 xGper1kChunk', 'Y-1 SHper1kChunk', 'Y-0 Age', 'PositionBool']
    X_pred = combined_df[features]
    predictions = np.zeros((len(combined_df), bootstrap_samples))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_pred)
    bootstrap_variances = np.var(predictions, axis=1)
    bootstrap_variances *= residual_variance/np.mean(bootstrap_variances)
    bootstrap_stdevs = np.sqrt(bootstrap_variances)
    combined_df['Gper1kChunk'] = bootstrap_stdevs

    # Adjust for current season progress
    combined_df['Gper1kChunk'] = combined_df['Gper1kChunk'] * np.sqrt(1 - combined_df['Y-0 GP']/82)

    # Merge adjusted variance inferences into bootstrap_df
    if bootstrap_df is None or bootstrap_df.empty:
        combined_df.rename(columns={'Y-0 Age': 'Age'}, inplace=True)
        bootstrap_df = combined_df[['PlayerID', 'Player', 'Team', 'Position', 'Age']].copy()
        bootstrap_df['Age'] = bootstrap_df['Age'] - 1
        bootstrap_df['Gper1kChunk'] = bootstrap_variances
    else:
        bootstrap_df = pd.merge(bootstrap_df, combined_df[['PlayerID', 'Gper1kChunk']], on='PlayerID', how='left')

    if verbose:
        print(f'Bootstrapped Gper1kChunk inferences for {projection_year} have been generated')
        print(bootstrap_df)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_a1_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'a1_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'residual_variance.json')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)

        features = ['Y-3 GP', 'Y-3 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 GP', 'Y-2 A1per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 GP', 'Y-1 A1per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'PositionBool']
        target_var = 'Y-0 A1per1kChunk'

        # Define X and y
        X = train_data[features]
        y = train_data[target_var]

        # Hyperparameters for XGBoost
        params = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 125,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'objective': 'reg:squarederror'
        }

        # Loop through the bootstrap samples, training new samples and storing in models list
        models, discrepancies = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping Primary Assists"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            errors = y_test - y_test_pred
            discrepancies.extend(errors)
            models.append(model)
        residual_variance = np.var(discrepancies)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify discrepancies json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['A1per1kChunk'] = residual_variance
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['A1per1kChunk']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'First Assists', 'Second Assists']]
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df[['A1per1kChunk', 'A2per1kChunk']] = df[['A1per1kChunk', 'A2per1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists'])
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk', 'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP', 'Y-0 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    # Generate predictions
    features = ['Y-3 GP', 'Y-3 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 GP', 'Y-2 A1per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 GP', 'Y-1 A1per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'PositionBool']
    X_pred = combined_df[features]
    predictions = np.zeros((len(combined_df), bootstrap_samples))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_pred)
    bootstrap_variances = np.var(predictions, axis=1)
    bootstrap_variances *= residual_variance/np.mean(bootstrap_variances)
    bootstrap_stdevs = np.sqrt(bootstrap_variances)
    combined_df['A1per1kChunk'] = bootstrap_stdevs

    # Adjust for current season progress
    combined_df['A1per1kChunk'] = combined_df['A1per1kChunk'] * np.sqrt(1 - combined_df['Y-0 GP']/82)

    # Merge adjusted variance inferences into bootstrap_df
    if bootstrap_df is None or bootstrap_df.empty:
        combined_df.rename(columns={'Y-0 Age': 'Age'}, inplace=True)
        bootstrap_df = combined_df[['PlayerID', 'Player', 'Team', 'Position', 'Age']].copy()
        bootstrap_df['Age'] = bootstrap_df['Age'] - 1
        bootstrap_df['A1per1kChunk'] = bootstrap_variances
    else:
        bootstrap_df = pd.merge(bootstrap_df, combined_df[['PlayerID', 'A1per1kChunk']], on='PlayerID', how='left')

    if verbose:
        print(f'Bootstrapped A1per1kChunk inferences for {projection_year} have been generated')
        print(bootstrap_df)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_a2_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'a2_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'bootstraps', 'residual_variance.json')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)

        features = ['Y-3 GP', 'Y-3 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 GP', 'Y-2 A1per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 GP', 'Y-1 A1per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'PositionBool']
        target_var = 'Y-0 A2per1kChunk'

        # Define X and y
        X = train_data[features]
        y = train_data[target_var]

        # Hyperparameters for XGBoost
        params = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 125,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'objective': 'reg:squarederror'
        }

        # Loop through the bootstrap samples, training new samples and storing in models list
        models, discrepancies = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping Secondary Assists"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            errors = y_test - y_test_pred
            discrepancies.extend(errors)
            models.append(model)
        residual_variance = np.var(discrepancies)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify discrepancies json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['A2per1kChunk'] = residual_variance
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['A2per1kChunk']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'Player', 'GP', 'TOI', 'First Assists', 'Second Assists']]
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df[['A1per1kChunk', 'A2per1kChunk']] = df[['A1per1kChunk', 'A2per1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists'])
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk', 'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP', 'Y-0 GP'], how='all')
    combined_df = combined_df.reset_index(drop=True)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    # Generate predictions
    features = ['Y-3 GP', 'Y-3 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 GP', 'Y-2 A1per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 GP', 'Y-1 A1per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'PositionBool']
    X_pred = combined_df[features]
    predictions = np.zeros((len(combined_df), bootstrap_samples))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_pred)
    bootstrap_variances = np.var(predictions, axis=1)
    bootstrap_variances *= residual_variance/np.mean(bootstrap_variances)
    bootstrap_stdevs = np.sqrt(bootstrap_variances)
    combined_df['A2per1kChunk'] = bootstrap_stdevs

    # Adjust for current season progress
    combined_df['A2per1kChunk'] = combined_df['A2per1kChunk'] * np.sqrt(1 - combined_df['Y-0 GP']/82)

    # Merge adjusted variance inferences into bootstrap_df
    if bootstrap_df is None or bootstrap_df.empty:
        combined_df.rename(columns={'Y-0 Age': 'Age'}, inplace=True)
        bootstrap_df = combined_df[['PlayerID', 'Player', 'Team', 'Position', 'Age']].copy()
        bootstrap_df['Age'] = bootstrap_df['Age'] - 1
        bootstrap_df['A2per1kChunk'] = bootstrap_variances
    else:
        bootstrap_df = pd.merge(bootstrap_df, combined_df[['PlayerID', 'A2per1kChunk']], on='PlayerID', how='left')

    if verbose:
        print(f'Bootstrapped A2per1kChunk inferences for {projection_year} have been generated')
        print(bootstrap_df)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df