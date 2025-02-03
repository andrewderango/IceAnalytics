import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

def aggregate_skater_offence_training_data(projection_year):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data')
    files = sorted(os.listdir(file_path))
    # files.remove('2023-2024_skater_data.csv')
    files.remove('2024-2025_skater_data.csv')
    for file in files:
        if file[-15:] != 'skater_data.csv':
            files.remove(file) # Remove files like .DS_Store or other unexpected files

    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combinations = [files[i:i+4] for i in range(len(files)-3)]
    combined_data = pd.DataFrame()

    for file_list in combinations:
        combined_df = None
        for index, file in enumerate(file_list):
            # Read in and compute rate stats
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

            # # Set columns to NaN if GP is less than 30
            # columns_to_nan = ['ATOI', 'Gper1kChunk', 'xGper1kChunk', 'SHper1kChunk', 'iCFper1kChunk', 'RAper1kChunk', 'A1per1kChunk', 'A2per1kChunk', 'RCper1kChunk', 'TAper1kChunk']
            # df.loc[df['GP'] < 30, columns_to_nan] = float('nan')

            # Add year index to column names
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
    # combined_data = combined_data.loc[(combined_data['Y-3 GP'] >= 30) & (combined_data['Y-2 GP'] >= 30) & (combined_data['Y-1 GP'] >= 30) & (combined_data['Y-0 GP'] >= 30)]
    combined_data = combined_data.loc[(combined_data['Y-1 GP'] >= 30) & (combined_data['Y-0 GP'] >= 30)]
    combined_data = combined_data[combined_data['Y-0'] != projection_year]
    # combined_data.sort_values(by='Y-0 ATOI', ascending=False, inplace=True)
    combined_data.sort_values(by=['Player', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    # print(combined_data.to_string())
    # print(combined_data)

    return combined_data

def tune_goal_model(projection_year, verbose):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
    
    if verbose:
        print(train_data)

    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    feature_cols = ['Y-0', 'Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data[['Y-0', 'Y-0 Gper1kChunk']]

    # Split the data into training and testing sets
    X_train = X.loc[X['Y-0'] != 2024]
    y_train = y.loc[y['Y-0'] != 2024]
    X_test = X.loc[X['Y-0'] == 2024]
    y_test = y.loc[y['Y-0'] == 2024]

    # remove y-0
    X_train = X_train.drop(columns=['Y-0'])
    X_test = X_test.drop(columns=['Y-0'])
    y_train = y_train.drop(columns=['Y-0'])
    y_test = y_test.drop(columns=['Y-0'])

    # Define the model
    n_estimators = 50
    learning_rate = 0.1
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

    # # Perform 10-fold cross-validation
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # mse_scores = []
    # mae_scores = []

    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)

    #     mse_scores.append(mean_squared_error(y_test, y_pred))
    #     mae_scores.append(mean_absolute_error(y_test, y_pred))


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse_scores = mean_squared_error(y_test, y_pred)
    mae_scores = mean_absolute_error(y_test, y_pred)

    if verbose:
        # print(f'MSE Scores: {mse_scores}')
        # print(f'MAE Scores: {mae_scores}')
        # print(f'Mean MSE: {sum(mse_scores) / len(mse_scores)}')
        # print(f'Mean MAE: {sum(mae_scores) / len(mae_scores) / 1000*2*60}')
        print(f'MSE Score: {mse_scores}')
        print(f'MAE Score: {mae_scores / 1000*2*60}')

    # create a df for 2024 predictions
    y_test['Y-0 Gper1kChunk Prediction'] = y_pred
    y_test['PlayerID'] = train_data.loc[train_data['Y-0'] == 2024, 'PlayerID'].values
    y_test['Player'] = train_data.loc[train_data['Y-0'] == 2024, 'Player'].values
    y_test['Position'] = train_data.loc[train_data['Y-0'] == 2024, 'Position'].values
    y_test['Y-0 Age'] = train_data.loc[train_data['Y-0'] == 2024, 'Y-0 Age'].values
    y_test['G/60 Pred'] = y_test['Y-0 Gper1kChunk Prediction'] / 1000 * 2 * 60
    y_test['G/60 Actual'] = y_test['Y-0 Gper1kChunk'] / 1000 * 2 * 60
    y_test = y_test.drop(columns=['Y-0 Gper1kChunk', 'Y-0 Gper1kChunk Prediction'])
    y_test = y_test.sort_values(by='G/60 Pred', ascending=False)
    print(y_test)
    quit()

    # Train the model on the full dataset
    model.fit(X, y)
    
    return model

PROJECTION_YEAR = 2025
goal_model = tune_goal_model(projection_year=PROJECTION_YEAR, verbose=True)