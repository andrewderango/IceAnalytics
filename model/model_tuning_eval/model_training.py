import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def aggregate_skater_offence_training_data(projection_year):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data')
    files = sorted(os.listdir(file_path))
    for file in files:
        if file[-15:] != 'skater_data.csv':
            files.remove(file) # Remove files like .DS_Store or other unexpected files

    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combinations = [files[i:i+4] for i in range(len(files)-3)]
    combined_data = pd.DataFrame()

    for file_list in combinations:
        combined_df = None
        for index, file in enumerate(file_list):
            df = pd.read_csv(os.path.join(file_path, file), usecols=['PlayerID', 'Player', 'GP', 'TOI', 'Goals', 'ixG', 'Shots', 'iCF', 'Rush Attempts', 'First Assists', 'Second Assists', 'Rebounds Created', 'Takeaways'])
            df['ATOI'] = df['TOI']/df['GP']
            df['Gper1kChunk'] = df['Goals']/df['TOI']/2 * 1000
            df['xGper1kChunk'] = df['ixG']/df['TOI']/2 * 1000
            df['SHper1kChunk'] = df['Shots']/df['TOI']/2 * 1000
            df['iCFper1kChunk'] = df['iCF']/df['TOI']/2 * 1000
            df['RAper1kChunk'] = df['Rush Attempts']/df['TOI']/2 * 1000
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df['RCper1kChunk'] = df['Rebounds Created']/df['TOI']/2 * 1000
            df['TAper1kChunk'] = df['Takeaways']/df['TOI']/2 * 1000
            df = df.drop(columns=['Player', 'TOI', 'Goals', 'ixG', 'Shots', 'iCF', 'Rush Attempts', 'First Assists', 'Second Assists', 'Rebounds Created', 'Takeaways'])
            df = df.rename(columns={
                'ATOI': f'Y-{3-index} ATOI', 
                'GP': f'Y-{3-index} GP', 
                'Gper1kChunk': f'Y-{3-index} Gper1kChunk',
                'xGper1kChunk': f'Y-{3-index} xGper1kChunk',
                'SHper1kChunk': f'Y-{3-index} SHper1kChunk',
                'iCFper1kChunk': f'Y-{3-index} iCFper1kChunk',
                'RAper1kChunk': f'Y-{3-index} RAper1kChunk',
                'A1per1kChunk': f'Y-{3-index} A1per1kChunk',
                'A2per1kChunk': f'Y-{3-index} A2per1kChunk',
                'RCper1kChunk': f'Y-{3-index} RCper1kChunk',
                'TAper1kChunk': f'Y-{3-index} TAper1kChunk'
            })
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='PlayerID', how='outer')

        last_file = file_list[-1]
        combined_df = combined_df.merge(bios_df, on='PlayerID', how='left')
        combined_df = combined_df.dropna(subset=['Y-0 GP', 'Y-1 GP'])

        # Calculate Y-0 age and season
        year = int(last_file.split('_')[0].split('-')[1])
        combined_df['Y-0'] = year
        combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
        combined_df['Y-0 Age'] = combined_df['Y-0'] - combined_df['Date of Birth'].dt.year
        combined_df = combined_df.drop(columns=['Date of Birth'])

        combined_data = pd.concat([combined_data, combined_df], ignore_index=True)

    # Data cleaning
    combined_data = combined_data.loc[(combined_data['Y-3 GP'] >= 30) & (combined_data['Y-2 GP'] >= 30) & (combined_data['Y-1 GP'] >= 30) & (combined_data['Y-0 GP'] >= 30)]
    combined_data = combined_data[combined_data['Y-0'] != projection_year]
    combined_data.sort_values(by=['Player', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)

    return combined_data

def train_goal_model_xgboost(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'PositionBool']
        train_data = train_data.dropna(subset=feature_cols)

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 Gper1kChunk']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the XGBoost model with common parameters
        n_estimators = 50
        learning_rate = 0.10
        max_depth = 3
        min_child_weight = 1
        subsample = 1.0
        colsample_bytree = 1.0
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            verbosity=0
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_test_pred = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Evaluate the model on the train set
        y_train_pred = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        if verbose:
            print(f"Train MSE: {mse_train:.2f}")
            print(f"Train MAE: {mae_train:.2f}")
            print(f"Train R²: {r2_train:.2f}")
            print(f"Test MSE: {mse_test:.2f}")
            print(f"Test MAE: {mae_test:.2f}")
            print(f"Test R²: {r2_test:.2f}")

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'goal_model.json'))

        return model, train_data
    
def generate_predictions(model, train_data):
    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data['Y-0 Gper1kChunk']

    # Generate predictions
    train_data['Proj. G/1kChunk'] = model.predict(X)

    return train_data

def goal_model_inference_xgboost(projection_year, player_stat_df, goal_model, download_file, verbose):

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
    predictions = goal_model.predict(combined_df[features])
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

# goal_model, train_data = train_goal_model_xgboost(projection_year=2025, retrain_model=True, verbose=True)
# train_data = generate_predictions(goal_model, train_data)
# train_data = train_data[train_data['Y-0'] == 2024]
# train_data = train_data.sort_values(by='Proj. G/1kChunk', ascending=False)
# print(train_data[['Player', 'Proj. G/1kChunk']])

player_stat_df = pd.DataFrame()
goal_model, train_data = train_goal_model_xgboost(projection_year=2025, retrain_model=True, verbose=True)
player_stat_df = goal_model_inference_xgboost(projection_year=2025, player_stat_df=player_stat_df, goal_model=goal_model, download_file=False, verbose=True)
player_stat_df['G/60'] = player_stat_df['Gper1kChunk']/500 * 60
print(player_stat_df.head(20))