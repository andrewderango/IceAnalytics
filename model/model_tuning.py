import os
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
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
    # combined_data = combined_data[combined_data['Y-0'] != projection_year]
    # combined_data.sort_values(by='Y-0 ATOI', ascending=False, inplace=True)
    combined_data.sort_values(by=['Player', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    # print(combined_data.to_string())
    # print(combined_data)

    return combined_data

def tune_goal_model(projection_year, model, verbose):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]

    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']

    if model == 'xgboost':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
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
    
    elif model == 'ridge':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Ridge(alpha=0.001)

    elif model == 'lasso':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Lasso(alpha=0.001)

    elif model == 'neural_net':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        model = MLPRegressor(
            hidden_layer_sizes=(50, 30, 10),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.005,
            alpha=0.00001,
            solver='adam',
            activation='relu', 
            batch_size=64,
            random_state=42
        )

    elif model == 'neural_net2':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-0 Age', 'Position']
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.035,
            learning_rate='adaptive',
            learning_rate_init=0.005,
            alpha=0.01,
            solver='adam',
            activation='relu', 
            batch_size=64,
            random_state=42
        )

    elif model == 'support_vector':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        model = SVR(
            kernel='rbf', 
            C=1.0, 
            epsilon=0.80,
            gamma='scale'
        )

    # Drop rows with missing values of features
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data['Y-0 Gper1kChunk']

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    if verbose:
        print(f'Mean G/60 MSE: {sum(mse_scores) / len(mse_scores) * 9/625}')
        print(f'Mean G/60 MAE: {sum(mae_scores) / len(mae_scores) * 3/25}')

    # Train the model on the full dataset
    model.fit(X, y)
    
    return model

def tune_a1_model(projection_year, model, verbose):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]

    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-0 Age', 'Position']

    if model == 'xgboost':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
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
    
    elif model == 'ridge':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Ridge(alpha=0.001)

    elif model == 'lasso':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Lasso(alpha=0.001)

    elif model == 'elastic_net':
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-0 Age', 'Position']
        model = ElasticNet(alpha=0.001, l1_ratio=0.5)

    elif model == 'neural_net':
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = MLPRegressor(
            hidden_layer_sizes=(60, 30, 15),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.005,
            alpha=0.00001,
            solver='adam',
            activation='relu', 
            batch_size=64,
            random_state=42
        )

    elif model == 'support_vector':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = SVR(
            kernel='rbf', 
            C=1.0, 
            epsilon=1.0,
            gamma='scale'
        )

    # Drop rows with missing values of features
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data['Y-0 A1per1kChunk']

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    if verbose:
        print(f'Mean A1/60 MSE: {sum(mse_scores) / len(mse_scores) * 9/625}')
        print(f'Mean A1/60 MAE: {sum(mae_scores) / len(mae_scores) * 3/25}')

    # Train the model on the full dataset
    model.fit(X, y)
    
    return model

def tune_a2_model(projection_year, model, verbose):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]

    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-0 Age', 'Position']

    if model == 'xgboost':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
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
    
    elif model == 'ridge':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Ridge(alpha=0.001)

    elif model == 'lasso':
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Lasso(alpha=0.001)

    elif model == 'elastic_net':
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-0 Age', 'Position']
        model = ElasticNet(alpha=0.01, l1_ratio=0.35)

    elif model == 'neural_net':
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = MLPRegressor(
            hidden_layer_sizes=(60, 30, 15),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.005,
            alpha=0.00001,
            solver='adam',
            activation='relu', 
            batch_size=64,
            random_state=42
        )

    elif model == 'support_vector':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = SVR(
            kernel='rbf', 
            C=1.0, 
            epsilon=1.20,
            gamma='scale'
        )

    # Drop rows with missing values of features
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data['Y-0 A2per1kChunk']

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    if verbose:
        print(f'Mean A2/60 MSE: {sum(mse_scores) / len(mse_scores) * 9/625}')
        print(f'Mean A2/60 MAE: {sum(mae_scores) / len(mae_scores) * 3/25}')

    # Train the model on the full dataset
    model.fit(X, y)
    
    return model

def tune_atoi_model(projection_year, model, verbose):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]

    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    train_data['Y-3 Pper1kChunk'] = train_data['Y-3 Gper1kChunk'] + train_data['Y-3 A1per1kChunk'] + train_data['Y-3 A2per1kChunk']
    train_data['Y-2 Pper1kChunk'] = train_data['Y-2 Gper1kChunk'] + train_data['Y-2 A1per1kChunk'] + train_data['Y-2 A2per1kChunk']
    train_data['Y-1 Pper1kChunk'] = train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk']    
    train_data['Y-3 PperGame'] = train_data['Y-3 Pper1kChunk']*train_data['Y-3 ATOI']/1000*2
    train_data['Y-2 PperGame'] = train_data['Y-2 Pper1kChunk']*train_data['Y-2 ATOI']/1000*2
    train_data['Y-1 PperGame'] = train_data['Y-1 Pper1kChunk']*train_data['Y-1 ATOI']/1000*2
    feature_cols = ['Y-3 PperGame', 'Y-2 PperGame', 'Y-1 PperGame', 'Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-3 Pper1kChunk', 'Y-2 Pper1kChunk', 'Y-1 Pper1kChunk', 'Y-0 Age', 'Position']

    if model == 'xgboost':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
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
    
    elif model == 'ridge':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        model = Ridge(alpha=0.001)

    elif model == 'lasso':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Lasso(alpha=0.001)

    elif model == 'elastic_net':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-0 Age', 'Position']
        model = ElasticNet(alpha=0.0008, l1_ratio=0.50)

    elif model == 'neural_net':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = MLPRegressor(
            hidden_layer_sizes=(60, 30, 15),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.005,
            alpha=0.00001,
            solver='adam',
            activation='relu', 
            batch_size=64,
            random_state=42
        )

    elif model == 'support_vector':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = SVR(
            kernel='rbf', 
            C=1.0, 
            epsilon=1.20,
            gamma='scale'
        )

    # Drop rows with missing values of features
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data['Y-0 ATOI']

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    if verbose:
        print(f'Mean ATOI MSE: {sum(mse_scores) / len(mse_scores)}')
        print(f'Mean ATOI MAE: {sum(mae_scores) / len(mae_scores)}')

    # Train the model on the full dataset
    model.fit(X, y)
    
    return model

def tune_gp_model(projection_year, model, verbose):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]

    # Define the feature columns
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    train_data['Y-3 Pper1kChunk'] = train_data['Y-3 Gper1kChunk'] + train_data['Y-3 A1per1kChunk'] + train_data['Y-3 A2per1kChunk']
    train_data['Y-2 Pper1kChunk'] = train_data['Y-2 Gper1kChunk'] + train_data['Y-2 A1per1kChunk'] + train_data['Y-2 A2per1kChunk']
    train_data['Y-1 Pper1kChunk'] = train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk']    
    train_data['Y-3 PperGame'] = train_data['Y-3 Pper1kChunk']*train_data['Y-3 ATOI']/1000*2
    train_data['Y-2 PperGame'] = train_data['Y-2 Pper1kChunk']*train_data['Y-2 ATOI']/1000*2
    train_data['Y-1 PperGame'] = train_data['Y-1 Pper1kChunk']*train_data['Y-1 ATOI']/1000*2
    feature_cols = ['Y-3 PperGame', 'Y-2 PperGame', 'Y-1 PperGame', 'Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-3 Pper1kChunk', 'Y-2 Pper1kChunk', 'Y-1 Pper1kChunk', 'Y-3 GP', 'Y-2 GP', 'Y-1 GP', 'Y-0 Age', 'Position']

    if model == 'xgboost':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
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
    
    elif model == 'ridge':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-0 Age', 'Position']
        model = Ridge(alpha=0.001)

    elif model == 'lasso':
        # feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        model = Lasso(alpha=0.001)

    elif model == 'elastic_net':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-0 Age', 'Position']
        model = ElasticNet(alpha=0.005, l1_ratio=0.50)

    elif model == 'neural_net':
        feature_cols = ['Y-3 PperGame', 'Y-2 PperGame', 'Y-1 PperGame', 'Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-3 Pper1kChunk', 'Y-2 Pper1kChunk', 'Y-1 Pper1kChunk', 'Y-3 GP', 'Y-2 GP', 'Y-1 GP', 'Y-0 Age', 'Position']
        model = MLPRegressor(
            hidden_layer_sizes=(60, 30, 15),
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.005,
            alpha=0.00001,
            solver='adam',
            activation='relu', 
            batch_size=64,
            random_state=42
        )

    elif model == 'support_vector':
        # feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-0 Age', 'Position']
        model = SVR(
            kernel='rbf', 
            C=1.0, 
            epsilon=1.20,
            gamma='scale'
        )

    # Drop rows with missing values of features
    train_data = train_data.dropna(subset=feature_cols)

    # Separate the features and the target
    X = train_data[feature_cols]
    y = train_data['Y-0 GP']

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    if verbose:
        print(f'Mean GP MSE: {sum(mse_scores) / len(mse_scores)}')
        print(f'Mean GP MAE: {sum(mae_scores) / len(mae_scores)}')

    # Train the model on the full dataset
    model.fit(X, y)
    
    return model

# compute MSE and MAE for P/60 using the models
def compute_p60_metrics(projection_year, goal_model, a1_model, a2_model, atoi_model, gp_model):

    train_data = aggregate_skater_offence_training_data(projection_year)
    train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]

    # make predictions for each model
    train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
    train_data['Y-3 Pper1kChunk'] = train_data['Y-3 Gper1kChunk'] + train_data['Y-3 A1per1kChunk'] + train_data['Y-3 A2per1kChunk']
    train_data['Y-2 Pper1kChunk'] = train_data['Y-2 Gper1kChunk'] + train_data['Y-2 A1per1kChunk'] + train_data['Y-2 A2per1kChunk']
    train_data['Y-1 Pper1kChunk'] = train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk']
    train_data['Y-3 PperGame'] = train_data['Y-3 Pper1kChunk']*train_data['Y-3 ATOI']/1000*2
    train_data['Y-2 PperGame'] = train_data['Y-2 Pper1kChunk']*train_data['Y-2 ATOI']/1000*2
    train_data['Y-1 PperGame'] = train_data['Y-1 Pper1kChunk']*train_data['Y-1 ATOI']/1000*2
    feature_cols = ['Y-3 PperGame', 'Y-2 PperGame', 'Y-1 PperGame', 'Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-3 Pper1kChunk', 'Y-2 Pper1kChunk', 'Y-1 Pper1kChunk', 'Y-0 Age', 'Position']

    train_data = train_data.dropna(subset=feature_cols)
    X = train_data[feature_cols]
    y = train_data['Y-0 Pper1kChunk']

PROJECTION_YEAR = 2025
goal_model = tune_goal_model(projection_year=PROJECTION_YEAR, model='support_vector', verbose=True)
a1_model = tune_a1_model(projection_year=PROJECTION_YEAR, model='elastic_net', verbose=True)
a2_model = tune_a2_model(projection_year=PROJECTION_YEAR, model='elastic_net', verbose=True)
atoi_model = tune_atoi_model(projection_year=PROJECTION_YEAR, model='elastic_net', verbose=True)
gp_model = tune_gp_model(projection_year=PROJECTION_YEAR, model='neural_net', verbose=True)
compute_p60_metrics(projection_year=PROJECTION_YEAR, goal_model=goal_model, a1_model=a1_model, a2_model=a2_model, atoi_model=atoi_model, gp_model=gp_model)