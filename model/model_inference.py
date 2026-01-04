import os
import random
import numpy as np
import pandas as pd
from model_training import *
from scraper_functions import *

def atoi_model_inference(projection_year, player_stat_df, atoi_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI']]
            df['ATOI'] = df['TOI']/df['GP']
            df = df.drop(columns=['TOI'])
            df = df.rename(columns={'ATOI': f'Y-{projection_year-year} ATOI', 'GP': f'Y-{projection_year-year} GP'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
    y1_prev_skater_data_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{projection_year-2}-{projection_year-1}_skater_data.csv')
    y2_prev_skater_data_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{projection_year-3}-{projection_year-2}_skater_data.csv')
    y3_prev_skater_data_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{projection_year-4}-{projection_year-3}_skater_data.csv')

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP']]
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI', 'Goals', 'First Assists', 'Second Assists']]
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} Points'] = 0
            df[f'Y-{projection_year-year} P/GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI', 'Goals', 'ixG', 'Shots', 'iCF', 'Rush Attempts']]
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
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
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    combined_df = combined_df.dropna(subset=['PlayerID'])
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI', 'First Assists', 'Second Assists', 'Rush Attempts', 'Rebounds Created', 'Takeaways']]
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
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
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI', 'First Assists', 'Second Assists', 'Rush Attempts', 'Rebounds Created', 'Takeaways']]
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
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
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical On-Ice Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI', 'CA/60', 'FA/60', 'SA/60', 'xGA/60', 'GA/60', 'On-Ice SV%']]
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical On-Ice Skater Data', f'{year-2}-{year-1}_skater_onice_data.csv')) # copy last season df
            df = df[['PlayerID']]
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
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical On-Ice Skater Data', filename)
        if not os.path.exists(file_path):
            if year == projection_year:
                season_started = False
            else:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        if season_started == True:
            df = pd.read_csv(file_path)
            df = df[['PlayerID', 'GP', 'TOI', 'CA/60', 'FA/60', 'SA/60', 'xGA/60', 'GA/60', 'On-Ice SV%']]
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical On-Ice Skater Data', f'{year-2}-{year-1}_skater_onice_data.csv')) # copy last season df
            df = df[['PlayerID']]
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
            combined_df = pd.merge(combined_df, df, on=['PlayerID'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID'], how='left')
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Team Data', filename)
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Team Data', f'{year-2}-{year-1}_team_data.csv')) # copy last season df
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
    nhlapi_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'nhlapi_team_data.csv'), index_col=0)
    nhlapi_data = nhlapi_data[['Team Name', 'Abbreviation']].rename(columns={'Team Name': 'Team'})

    combined_df = pd.merge(combined_df[['Team', 'Proj. GA/GP']], nhlapi_data[['Team', 'Abbreviation']], on='Team', how='left')
    combined_df.loc[combined_df['Team'] == 'Montreal Canadiens', 'Abbreviation'] = 'MTL'
    combined_df.loc[combined_df['Team'] == 'St Louis Blues', 'Abbreviation'] = 'STL'
    combined_df.loc[combined_df['Team'] == 'Arizona Coyotes', 'Abbreviation'] = 'UTA'
    combined_df.loc[combined_df['Team'] == 'Utah Hockey Club', 'Team'] = 'Utah Mammoth' ### temp !!!
    combined_df.loc[combined_df['Team'] == 'Utah Mammoth', 'Abbreviation'] = 'UTA'

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

    SAMPLED_MU, SAMPLED_SIGMA = 3.087422, 0.309527 ###!
    player_stat_df.rename(columns={'Team': 'Abbreviation'}, inplace=True)
    player_stat_df['pGA/60'] = player_stat_df['GA/60'] * player_stat_df['ATOI']
    player_stat_df['pxGA/60'] = player_stat_df['xGA/60'] * player_stat_df['ATOI']
    team_weighted_ga = player_stat_df.groupby('Abbreviation').apply(lambda x: x['pGA/60'].sum() / x['ATOI'].sum()).reset_index(name='pGA/GP')
    team_weighted_xga = player_stat_df.groupby('Abbreviation').apply(lambda x: x['pxGA/60'].sum() / x['ATOI'].sum()).reset_index(name='pxGA/GP')
    team_stat_df = team_stat_df.merge(team_weighted_ga, on='Abbreviation', how='left')
    team_stat_df = team_stat_df.merge(team_weighted_xga, on='Abbreviation', how='left')
    team_stat_df['Agg GA/GP'] = team_stat_df['GA/GP']*0.61 + team_stat_df['pGA/GP']*0.28 + team_stat_df['pxGA/GP']*0.11 ###!
    team_stat_df['z_score'] = (team_stat_df['Agg GA/GP'] - team_stat_df['Agg GA/GP'].mean()) / team_stat_df['Agg GA/GP'].std()
    team_stat_df['Normalized GA/GP'] = (team_stat_df['z_score'] * SAMPLED_SIGMA) + SAMPLED_MU
    team_stat_df.drop(columns=['z_score'], inplace=True)

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Teams')
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
    existing_stats = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))[['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'First Assists', 'Second Assists', 'Total Assists']]
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
    if verbose:
        # print(player_stat_df.to_string())
        print(player_stat_df.head(50))
        # print(player_stat_df.info())

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        stable_player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

def gp_inference_calibration(projection_year, player_stat_df):
    # Apply points-based GP recalibration
    player_stat_df['PosFD'] = player_stat_df['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
    player_stat_df['Pts'] = (player_stat_df['Gper1kChunk'] + player_stat_df['A1per1kChunk'] + player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI']
    player_stat_df['pPts'] = player_stat_df.groupby('PosFD')['Pts'].rank(pct=True)
    player_stat_df['GPprb'] = 1/(1 + np.exp(-((player_stat_df['GPprb']*0.939644 + player_stat_df['pPts']*0.060356) - 1/2)/0.108235))
    player_stat_df = player_stat_df.drop(columns=['PosFD', 'Pts', 'pPts'])

    # Phase in current season
    y0_skater_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))[['PlayerID', 'Player', 'GP']]
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
