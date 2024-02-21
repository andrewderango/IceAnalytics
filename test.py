import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from model_training import *
from scraper_functions import *
        
def atoi_model_inference(projection_year, player_stat_df, atoi_model_data, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} GP'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
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
        export_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
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
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    predictions = goal_model.predict(combined_df[['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'PositionBool']], verbose=verbose)
    predictions = predictions.reshape(-1)
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

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
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
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    predictions = a1_model.predict(combined_df[['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-3 GP', 'Y-2 GP', 'Y-1 GP', 'Y-0 Age', 'PositionBool']], verbose=verbose)
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

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters')
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
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
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
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on=['PlayerID', 'Player'], how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)
    combined_df['PositionBool'] = combined_df['Position'].apply(lambda x: 0 if x == 'D' else 1)

    predictions = a2_model.predict(combined_df[['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-3 GP', 'Y-2 GP', 'Y-1 GP', 'Y-0 Age', 'PositionBool']], verbose=verbose)
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

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_metaprojections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_metaprojections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df

def ga_model_inference(projection_year, team_stat_df, ga_model, download_file, verbose):

    combined_df = pd.DataFrame()
    season_started = True

    for year in range(projection_year-3, projection_year+1):
        filename = f'{year-1}-{year}_team_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Team Data', filename)
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
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Team Data', f'{year-2}-{year-1}_team_data.csv')) # copy last season df
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

    try: # model was trained in this session
        predictions = ga_model.predict(combined_df[['Y-3 P%', 'Y-2 P%', 'Y-1 P%', 'Y-3 CA/GP', 'Y-2 CA/GP', 'Y-1 CA/GP', 'Y-3 FA/GP', 'Y-2 FA/GP', 'Y-1 FA/GP', 'Y-3 SHA/GP', 'Y-2 SHA/GP', 'Y-1 SHA/GP', 'Y-3 GA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-3 xGA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-3 SCA/GP', 'Y-2 SCA/GP', 'Y-1 SCA/GP', 'Y-3 HDCA/GP', 'Y-2 HDCA/GP', 'Y-1 HDCA/GP', 'Y-3 HDGA/GP', 'Y-2 HDGA/GP', 'Y-1 HDGA/GP', 'Y-3 HDSV%', 'Y-2 HDSV%', 'Y-1 HDSV%', 'Y-3 SV%', 'Y-2 SV%', 'Y-1 SV%']])
    except TypeError: # model was loaded in, pre-trained
        data_dmatrix = xgb.DMatrix(combined_df[['Y-3 P%', 'Y-2 P%', 'Y-1 P%', 'Y-3 CA/GP', 'Y-2 CA/GP', 'Y-1 CA/GP', 'Y-3 FA/GP', 'Y-2 FA/GP', 'Y-1 FA/GP', 'Y-3 SHA/GP', 'Y-2 SHA/GP', 'Y-1 SHA/GP', 'Y-3 GA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-3 xGA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-3 SCA/GP', 'Y-2 SCA/GP', 'Y-1 SCA/GP', 'Y-3 HDCA/GP', 'Y-2 HDCA/GP', 'Y-1 HDCA/GP', 'Y-3 HDGA/GP', 'Y-2 HDGA/GP', 'Y-1 HDGA/GP', 'Y-3 HDSV%', 'Y-2 HDSV%', 'Y-1 HDSV%', 'Y-3 SV%', 'Y-2 SV%', 'Y-1 SV%']])
        predictions = ga_model.predict(data_dmatrix) 
    predictions = predictions.reshape(-1)
    combined_df['Proj. GA/GP'] = combined_df['Y-0 GP']/82*combined_df['Y-0 GA/GP'] + (82-combined_df['Y-0 GP'])/82*predictions

    combined_df = combined_df[['Team', 'Proj. GA/GP']]
    combined_df.sort_values(by='Proj. GA/GP', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df.rename(columns={'Proj. GA/GP': 'GA/GP'})

    if team_stat_df is None or team_stat_df.empty:
        team_stat_df = combined_df
    else:
        team_stat_df = pd.merge(team_stat_df, combined_df, on=['PlayerID', 'Player'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_team_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_projections.csv has been downloaded to the following directory: {export_path}')

    return team_stat_df

def simulate_season(projetion_year, verbose):
    # load dfs
    schedule_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Team Data', f'game_schedule.csv'), index_col=0)
    metaprojection_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters', f'{projection_year}_skater_metaprojections.csv'), index_col=0)
    metaprojection_df['Aper1kChunk'] = metaprojection_df['A1per1kChunk'] + metaprojection_df['A2per1kChunk']
    metaprojection_df['Pper1kChunk'] = metaprojection_df['Gper1kChunk'] + metaprojection_df['Aper1kChunk']
    teams_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'), index_col=0)

    # configure skater monte carlo projection df
    monte_carlo_skater_proj_df = metaprojection_df[['PlayerID', 'Player', 'Position', 'Team', 'Age']].copy()
    monte_carlo_skater_proj_df = monte_carlo_skater_proj_df.assign(Games_Played=0, Goals=0, Primary_Assists=0, Secondary_Assists=0)
    monte_carlo_skater_proj_df.rename(columns={'Games_Played': 'Games Played', 'Primary_Assists': 'Primary Assists', 'Secondary_Assists': 'Secondary Assists'}, inplace=True)
    
    # configure team monte carlo projection df
    monte_carlo_team_proj_df = teams_df[['Team Name', 'Abbreviation']].copy()
    active_teams = pd.concat([schedule_df['Home Team'], schedule_df['Visiting Team']])
    monte_carlo_team_proj_df = monte_carlo_team_proj_df[monte_carlo_team_proj_df['Team Name'].isin(active_teams)]
    monte_carlo_team_proj_df = monte_carlo_team_proj_df.assign(Wins=0, Losses=0, OTL=0, Goals_For=0, Goals_Against=0)
    monte_carlo_team_proj_df.rename(columns={'Goals_For': 'Goals For', 'Goals_Against': 'Goals Against'}, inplace=True)

    # add team abbreviations to schedule
    schedule_df = schedule_df.merge(teams_df, left_on='Home Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Home Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID'])
    schedule_df = schedule_df.merge(teams_df, left_on='Visiting Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Visiting Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID'])

    # create game scoring dictionary
    game_scoring_dict = {} # {player_id: [games, goals, primary_assists, secondary_assists]}
    for player_id in monte_carlo_skater_proj_df['PlayerID']:
        game_scoring_dict[player_id] = [0, 0, 0, 0]

    # simulate each game
    for index, row in tqdm(schedule_df.iterrows(), total=schedule_df.shape[0]):
        scoring_frequency = 0.05086743957 # average number of goals scored by both teams combined in 30 seconds of game time
        a1_probability = 0.9438426454 # probability of a goal having a primary assistor
        a2_probability = 0.7916037451 # probability of a goal with a primary assistor also having a secondary assistor
        simulate_game(row['Home Abbreviation'], row['Visiting Abbreviation'], metaprojection_df, game_scoring_dict, monte_carlo_team_proj_df, scoring_frequency, a1_probability, a2_probability, verbose)

    # add game scoring dict stats to monte_carlo_skater_proj_df
    scoring_df = pd.DataFrame.from_dict(game_scoring_dict, orient='index', columns=['Games Played', 'Goals', 'Primary Assists', 'Secondary Assists'])
    monte_carlo_skater_proj_df.set_index('PlayerID', inplace=True)
    monte_carlo_skater_proj_df[['Goals', 'Primary Assists', 'Secondary Assists']] += scoring_df
    monte_carlo_skater_proj_df.reset_index(inplace=True)

    monte_carlo_skater_proj_df['Assists'] =monte_carlo_skater_proj_df['Primary Assists'] + monte_carlo_skater_proj_df['Secondary Assists']
    monte_carlo_skater_proj_df['Points'] = monte_carlo_skater_proj_df['Goals'] + monte_carlo_skater_proj_df['Assists']
    monte_carlo_skater_proj_df = monte_carlo_skater_proj_df.sort_values(by=['Points', 'Goals', 'Assists', 'Primary Assists', 'Secondary Assists'], ascending=False)

    monte_carlo_team_proj_df['Points'] = monte_carlo_team_proj_df['Wins']*2 + monte_carlo_team_proj_df['OTL']
    monte_carlo_team_proj_df = monte_carlo_team_proj_df.sort_values(by=['Points', 'Wins', 'Goals For', 'Goals Against'], ascending=False)

    print(monte_carlo_skater_proj_df.head(10))
    print(monte_carlo_team_proj_df.head(10))

# @profile
def simulate_game(home_team, visiting_team, metaprojection_df, game_scoring_dict, monte_carlo_team_proj_df, scoring_frequency, a1_probability, a2_probability, verbose):
    # set initial score to 0-0
    home_score = 0
    visitor_score = 0

    # Get home and visiting team rosters
    home_team_roster = metaprojection_df[metaprojection_df['Team'] == home_team]
    visiting_team_roster = metaprojection_df[metaprojection_df['Team'] == visiting_team]

    # Precompute team scores
    home_team_point_score = home_team_roster['Pper1kChunk'].sum()
    visiting_team_point_score = visiting_team_roster['Pper1kChunk'].sum()
    home_team_scoring_prob = home_team_point_score / (home_team_point_score + visiting_team_point_score)

    # Precompute team abbreviations
    home_team_abbr = monte_carlo_team_proj_df.loc[monte_carlo_team_proj_df['Abbreviation'] == home_team]
    visiting_team_abbr = monte_carlo_team_proj_df.loc[monte_carlo_team_proj_df['Abbreviation'] == visiting_team]

    # Precompute team rosters
    home_team_roster_players = home_team_roster['PlayerID'].tolist()
    visiting_team_roster_players = visiting_team_roster['PlayerID'].tolist()

    for chunk in range(120):
        if np.random.uniform(0, 1) < scoring_frequency:
            if np.random.uniform(0, 1) < home_team_scoring_prob:
                home_score += 1
                scoring_roster = home_team_roster_players
                scoring_team_abbr = home_team_abbr
            else:
                visitor_score += 1
                scoring_roster = visiting_team_roster_players
                scoring_team_abbr = visiting_team_abbr

            scorer_id = np.random.choice(scoring_roster)
            game_scoring_dict[scorer_id][1] += 1

            # Assign assists
            if np.random.uniform(0, 1) < a1_probability:
                possible_assists = [p for p in scoring_roster if p != scorer_id]
                sampled_assists = np.random.choice(possible_assists, size=min(2, len(possible_assists)), replace=False)

                a1_id = sampled_assists[0]
                game_scoring_dict[a1_id][2] += 1

                if len(sampled_assists) == 2 and np.random.uniform(0, 1) < a2_probability:
                    a2_id = sampled_assists[1]
                    game_scoring_dict[a2_id][3] += 1

    # Update monte_carlo_team_proj_df
    home_abbr_index = home_team_abbr.index[0]
    visiting_abbr_index = visiting_team_abbr.index[0]
    monte_carlo_team_proj_df.loc[home_abbr_index, 'Wins'] += home_score > visitor_score
    monte_carlo_team_proj_df.loc[home_abbr_index, 'Losses'] += home_score < visitor_score
    monte_carlo_team_proj_df.loc[visiting_abbr_index, 'Wins'] += home_score < visitor_score
    monte_carlo_team_proj_df.loc[visiting_abbr_index, 'Losses'] += home_score > visitor_score

    monte_carlo_team_proj_df.loc[home_abbr_index, 'Goals For'] += home_score
    monte_carlo_team_proj_df.loc[home_abbr_index, 'Goals Against'] += visitor_score
    monte_carlo_team_proj_df.loc[visiting_abbr_index, 'Goals For'] += visitor_score
    monte_carlo_team_proj_df.loc[visiting_abbr_index, 'Goals Against'] += home_score

    return game_scoring_dict

global start_time
start_time = time.time()
projection_year = 2024

# Scrape or fetch player data
scrape_historical_player_data(2008, 2024, True, False, True, False)
scrape_historical_player_data(2008, 2024, False, False, True, False)
scrape_historical_player_data(2008, 2024, True, True, True, False)
scrape_historical_player_data(2008, 2024, False, True, True, False)
scrape_nhlapi_data(2008, 2024, False, True, False)
scrape_nhlapi_data(2008, 2024, True, True, False)
aggregate_player_bios(True, True, False)
aggregate_player_bios(False, True, False)

# Scrape or fetch team data
scrape_historical_team_data(2008, 2024, True, False)
scrape_teams(True, False)
scrape_games(projection_year, True, False)

# Train models
atoi_model_data = train_atoi_model(projection_year, False, False)
goal_model = train_goal_model(projection_year, False, False)
a1_model = train_a1_model(projection_year, False, False)
a2_model = train_a2_model(projection_year, False, False)
ga_model = train_ga_model(projection_year, False, False)

# Make player inferences
player_stat_df = pd.DataFrame()
player_stat_df = atoi_model_inference(projection_year, player_stat_df, atoi_model_data, True, False)
player_stat_df = goal_model_inference(projection_year, player_stat_df, goal_model, True, False)
player_stat_df = a1_model_inference(projection_year, player_stat_df, a1_model, True, False)
player_stat_df = a2_model_inference(projection_year, player_stat_df, a2_model, True, False)
# player_stat_df = player_stat_df.sort_values(by='Gper1kChunk', ascending=False)
# print(player_stat_df)

# Make team inferences
team_stat_df = pd.DataFrame()
team_stat_df = ga_model_inference(projection_year, team_stat_df, ga_model, True, False)
# team_stat_df = team_stat_df.sort_values(by='GA/GP', ascending=False)
# print(team_stat_df)

# Simulate season
simulate_season(projection_year, True)

print(f"Runtime: {time.time()-start_time:.3f} seconds")