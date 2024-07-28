import os
import numpy as np
import pandas as pd
from model_training import *
from scraper_functions import *
from scipy.signal import savgol_filter
        
def atoi_model_inference(projection_year, player_stat_df, atoi_model_data, download_file, verbose):

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
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)

    # Edit atoi_model_data to phase in the current season (Y-0) based on its progression into the season
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

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
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    features = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    combined_df['SampleGP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df['SampleReplaceGP'] = combined_df['SampleGP'].apply(lambda x: max(82 - x, 0))
    qual_df = combined_df[combined_df['SampleGP'] >= 82]
    for feature in features:
        if feature not in ['Y-0 Age', 'PositionBool']:
            replacement_value = qual_df[feature].mean()-qual_df[feature].std()
            combined_df[feature] = combined_df.apply(lambda row: (row[feature]*row['SampleGP'] + replacement_value*row['SampleReplaceGP']) / (row['SampleGP']+row['SampleReplaceGP']), axis=1)

    # create predictions
    dmatrix = xgb.DMatrix(combined_df[features])
    predictions = goal_model.predict(dmatrix)
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

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
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
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    features = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    combined_df['SampleGP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df['SampleReplaceGP'] = combined_df['SampleGP'].apply(lambda x: max(82 - x, 0))
    qual_df = combined_df[combined_df['SampleGP'] >= 82]
    for feature in features:
        if feature not in ['Y-0 Age', 'PositionBool']:
            replacement_value = qual_df[feature].mean()-qual_df[feature].std()
            combined_df[feature] = combined_df.apply(lambda row: (row[feature]*row['SampleGP'] + replacement_value*row['SampleReplaceGP']) / (row['SampleGP']+row['SampleReplaceGP']), axis=1)

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
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
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)
    features = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'PositionBool']

    # sample size control
    combined_df['SampleGP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df['SampleReplaceGP'] = combined_df['SampleGP'].apply(lambda x: max(82 - x, 0))
    qual_df = combined_df[combined_df['SampleGP'] >= 82]
    for feature in features:
        if feature not in ['Y-0 Age', 'PositionBool']:
            replacement_value = qual_df[feature].mean()-qual_df[feature].std()
            combined_df[feature] = combined_df.apply(lambda row: (row[feature]*row['SampleGP'] + replacement_value*row['SampleReplaceGP']) / (row['SampleGP']+row['SampleReplaceGP']), axis=1)

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
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
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
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
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
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
            df = df[['PlayerID', 'Player']]
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
        predictions = team_ga_model.predict(combined_df[['Y-3 FA/GP', 'Y-2 FA/GP', 'Y-1 FA/GP', 'Y-3 GA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-3 xGA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-3 SV%', 'Y-2 SV%', 'Y-1 SV%']])
    except TypeError: # model was loaded in, pre-trained
        data_dmatrix = xgb.DMatrix(combined_df[['Y-3 FA/GP', 'Y-2 FA/GP', 'Y-1 FA/GP', 'Y-3 GA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-3 xGA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-3 SV%', 'Y-2 SV%', 'Y-1 SV%']])
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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_team_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_projections.csv has been downloaded to the following directory: {export_path}')

    return team_stat_df

def savitzky_golvay_calibration(projection_year, player_stat_df):
    player_stat_df = savgol_goal_calibration(projection_year, player_stat_df)
    player_stat_df = savgol_a1_calibration(projection_year, player_stat_df)
    player_stat_df = savgol_a2_calibration(projection_year, player_stat_df)

    return player_stat_df

def savgol_goal_calibration(projection_year, player_stat_df):
    # Train calibration models
    fwd_scaling = train_goal_calibration_model(projection_year=projection_year, retrain_model=False, position='F')
    dfc_scaling = train_goal_calibration_model(projection_year=projection_year, retrain_model=False, position='D')

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
    fwd_scaling = train_a1_calibration_model(projection_year=projection_year, retrain_model=False, position='F')
    dfc_scaling = train_a1_calibration_model(projection_year=projection_year, retrain_model=False, position='D')

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
    fwd_scaling = train_a2_calibration_model(projection_year=projection_year, retrain_model=False, position='F')
    dfc_scaling = train_a2_calibration_model(projection_year=projection_year, retrain_model=False, position='D')

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