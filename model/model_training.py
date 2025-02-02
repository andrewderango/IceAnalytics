import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

def aggregate_skater_offence_training_data(projection_year):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data')
    files = sorted(os.listdir(file_path))
    files.remove('2023-2024_skater_data.csv')
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

def aggregate_skater_defence_training_data(projection_year):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical On-Ice Skater Data')
    files = sorted(os.listdir(file_path))
    for file in files:
        if file[-15:] != 'skater_data.csv':
            files.remove(file) # Remove files like .DS_Store or other unexpected files

    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position'])
    combinations = [files[i:i+4] for i in range(len(files)-3)]
    combined_data = pd.DataFrame()

    for file_list in combinations:
        combined_df = None
        for index, file in enumerate(file_list):
            df = pd.read_csv(os.path.join(file_path, file), usecols=['PlayerID', 'Player', 'GP', 'TOI', 'CA/60', 'FA/60', 'SA/60', 'GA/60', 'xGA/60', 'On-Ice SV%'])
            df['ATOI'] = df['TOI']/df['GP']
            df = df.rename(columns={
                'ATOI': f'Y-{3-index} ATOI', 
                'GP': f'Y-{3-index} GP', 
                'CA/60': f'Y-{3-index} CA/60',
                'FA/60': f'Y-{3-index} FA/60',
                'SA/60': f'Y-{3-index} SA/60',
                'GA/60': f'Y-{3-index} GA/60',
                'xGA/60': f'Y-{3-index} xGA/60',
                'On-Ice SV%': f'Y-{3-index} oiSV%'
            })
            df = df.drop(columns=['Player', 'TOI'])
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
    combined_data = combined_data.loc[(combined_data['Y-3 GP'] >= 35) & (combined_data['Y-2 GP'] >= 35) & (combined_data['Y-1 GP'] >= 35) & (combined_data['Y-0 GP'] >= 35)]
    combined_data = combined_data[combined_data['Y-0'] != projection_year]
    # combined_data.sort_values(by='Y-0 ATOI', ascending=False, inplace=True)
    combined_data.sort_values(by=['Player', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    # print(combined_data.to_string())
    # print(combined_data)

    return combined_data

def aggregate_team_training_data(projection_year):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Team Data')
    files = sorted(os.listdir(file_path))
    for file in files:
        if file[-13:] != 'team_data.csv':
            files.remove(file) # Remove files like .DS_Store or other unexpected files

    combinations = [files[i:i+4] for i in range(len(files)-3)]
    combined_data = pd.DataFrame()

    for file_list in combinations:
        combined_df = None
        for index, file in enumerate(file_list):
            df = pd.read_csv(os.path.join(file_path, file), usecols=['Team', 'GP', 'TOI', 'Point %', 'CA', 'FA', 'SA', 'GA', 'xGA', 'SCA', 'HDCA', 'HDGA', 'HDSV%', 'SV%'])
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
                'GP': f'Y-{3-index} GP',
                'Point %': f'Y-{3-index} P%',
                'CA/GP': f'Y-{3-index} CA/GP',
                'FA/GP': f'Y-{3-index} FA/GP',
                'SA/GP': f'Y-{3-index} SHA/GP',
                'GA/GP': f'Y-{3-index} GA/GP',
                'xGA/GP': f'Y-{3-index} xGA/GP',
                'SCA/GP': f'Y-{3-index} SCA/GP',
                'HDCA/GP': f'Y-{3-index} HDCA/GP',
                'HDGA/GP': f'Y-{3-index} HDGA/GP',
                'HDSV%': f'Y-{3-index} HDSV%',
                'SV%': f'Y-{3-index} SV%'
            })
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='Team', how='outer')

        last_file = file_list[-1]
        year = int(last_file.split('_')[0].split('-')[1])
        combined_df['Y-0'] = year

        combined_data = pd.concat([combined_data, combined_df], ignore_index=True)

    # Data cleaning
    combined_data = combined_data.loc[(combined_data['Y-3 GP'] >= 30) & (combined_data['Y-2 GP'] >= 30) & (combined_data['Y-1 GP'] >= 30) & (combined_data['Y-0 GP'] >= 30)]
    combined_data = combined_data[combined_data['Y-0'] != projection_year]
    # combined_data.sort_values(by='Y-0 ATOI', ascending=False, inplace=True)
    combined_data.sort_values(by=['Team', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    # print(combined_data.to_string())
    # print(combined_data)

    return combined_data

def train_atoi_model(projection_year, retrain_model, verbose):

    # Define the model path for saving and loading
    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'atoi_model.pkl')

    if retrain_model:
        # Aggregate and filter the training data
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        train_data = train_data.dropna(subset=['Y-0 Age'])

        if verbose:
            print(train_data)

        # Create and define polynomial features and target variable
        poly = PolynomialFeatures(3, include_bias=False)
        train_data[['Y-0 Age', 'Y-0 Age^2', 'Y-0 Age^3']] = poly.fit_transform(train_data[['Y-0 Age']])
        features = ['Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-0 Age', 'Y-0 Age^2', 'Y-0 Age^3']
        target_var = 'Y-0 ATOI'

        # Define X and y
        X = train_data[features]
        y = train_data[target_var]

        # Create and train the Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        if verbose:
            print("ATOI model linear coefficients:")
            for i in range(len(features)):
                print(features[i], '\t', model.coef_[i])

            print("Linear Regression Intercept:\t", model.intercept_)

        # Save the model
        joblib.dump(model, model_path)
        if verbose:
            print(f'atoi_model.pkl has been saved to: {model_path}')
    
    else:
        # Load the model
        model = joblib.load(model_path)

    return model
        
def train_gp_model(projection_year, retrain_model, verbose):
    p24_gp_model = train_p24_gp_model(projection_year=projection_year, retrain_model=retrain_model, verbose=verbose)
    u24_gp_model = train_u24_gp_model(projection_year=projection_year, retrain_model=retrain_model, verbose=verbose)
    return [p24_gp_model, u24_gp_model]

def train_p24_gp_model(projection_year, retrain_model, verbose):

    # Define the model path for saving and loading
    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'p24_gp_model.pkl')

    if retrain_model:
        # Train model
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 20) & (train_data['Y-2 GP'] >= 20) & (train_data['Y-1 GP'] >= 20) & (train_data['Y-0 GP'] >= 20)]
        feature_cols = ['Y-3 GP', 'Y-2 GP', 'Y-1 GP']
        X = MinMaxScaler().fit_transform(train_data[feature_cols])
        y = MinMaxScaler().fit_transform(train_data['Y-0 GP'].values.reshape(-1, 1)).flatten()
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        model.coef_ = model.coef_ / model.coef_.sum()

        # Save the model
        joblib.dump(model, model_path)
    
    else:
        # Load model
        model = joblib.load(model_path)

    return model

def train_u24_gp_model(projection_year, retrain_model, verbose):

    # Define the model path for saving and loading
    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'u24_gp_model.joblib')

    if retrain_model:
        # Train model
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-1 GP'] >= 10) & (train_data['Y-0 GP'] >= 10)]
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        train_data['Y-1 P/GP'] = (train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk']) / 500 * train_data['Y-1 ATOI']
        train_data['Y-1 Points'] = (train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk']) / 500 * train_data['Y-1 ATOI'] * train_data['Y-1 GP']
        feature_cols = ['Y-1 GP', 'Y-1 ATOI', 'Y-1 Points']
        boolean_feature = 'PositionBool'
        train_data = train_data.dropna(subset=feature_cols + [boolean_feature])
        train_data = train_data[train_data['Y-0 Age'] < 24]
        scaled_features = MinMaxScaler().fit_transform(train_data[feature_cols])
        X = np.hstack((scaled_features, train_data[[boolean_feature]].values))
        y = MinMaxScaler().fit_transform(train_data['Y-0 GP'].values.reshape(-1, 1)).flatten()
        model = SVR(kernel='linear', gamma='scale', C=1.0, epsilon=0.1)
        model.fit(X, y)

        # Save the model
        joblib.dump(model, model_path)
    
    else:
        # Load model
        model = joblib.load(model_path)

    return model

def train_goal_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
        train_data = train_data.dropna(subset=feature_cols)

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 Gper1kChunk']

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

        # Train the model
        model.fit(X, y)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'goal_model.json'))
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'goal_model.xgb'))
    
    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'goal_model.xgb'))
    
    return model
    
def train_a1_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'Position']
        train_data = train_data.dropna(subset=feature_cols)

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 A1per1kChunk']

        # Define the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(len(feature_cols), input_dim=len(feature_cols), kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(10, kernel_initializer='normal'))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X, y, epochs=100, batch_size=5, verbose=verbose)

        # Save the model
        model.save(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'primary_assist_model.keras'))

        return model
    
    else:
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'primary_assist_model.keras'), compile=False)
        return model
    
def train_a2_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'Position']
        train_data = train_data.dropna(subset=feature_cols)

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 A2per1kChunk']

        # Define the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(len(feature_cols), input_dim=len(feature_cols), kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(10, kernel_initializer='normal'))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X, y, epochs=100, batch_size=5, verbose=verbose)

        # Save the model
        model.save(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'secondary_assist_model.keras'))

        return model
    
    else:
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'secondary_assist_model.keras'), compile=False)
        return model
    
def train_ga_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_team_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        feature_cols = ['Y-2 FA/GP', 'Y-1 FA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-2 SV%', 'Y-1 SV%', 'Y-2 P%', 'Y-1 P%']

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 GA/GP']

        # Define the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1,
                        max_depth = 5, n_estimators = 100)

        # Train the model
        model.fit(X, y, verbose=verbose)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'goals_against_model.xgb'))

        return model

    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'goals_against_model.xgb'))
        return model
    
def train_skater_xga_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_defence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 CA/60', 'Y-2 CA/60', 'Y-1 CA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'PositionBool']

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 xGA/60']

        # Define the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.10, max_depth = 3, n_estimators = 60)

        # Train the model
        model.fit(X, y, verbose=verbose)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'skater_xga_model.xgb'))
        return model

    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'skater_xga_model.xgb'))
        return model
    
def train_skater_ga_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_defence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'PositionBool']

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 GA/60']

        # Define the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.075, max_depth = 2, n_estimators = 55)

        # Train the model
        model.fit(X, y, verbose=verbose)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'skater_ga_model.xgb'))
        return model

    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'skater_ga_model.xgb'))
        return model
    
def train_goal_calibration_model(projection_year, retrain_model, position):

    if retrain_model:
        # Train model
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        if position == 'F':
            train_data = train_data[train_data['Position'] != 'D']
        elif position == 'D':
            train_data = train_data[train_data['Position'] == 'D']
        else:
            raise ValueError(f"Invalid position argument: {position}. Must be either 'F' or 'D'.")
        feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk']
        train_data = train_data.dropna(subset=feature_cols)
        X = train_data[feature_cols]
        y = train_data['Y-0 Gper1kChunk']
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        model.coef_ = model.coef_ / model.coef_.sum()

        # Save the model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'fwd_goal_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'dfc_goal_calibration_model.pkl')
        joblib.dump(model, model_path)
    
    else:
        # Load model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'fwd_goal_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'dfc_goal_calibration_model.pkl')
        model = joblib.load(model_path)

    # Get data from past 3 years
    combined_df = pd.DataFrame()
    for year in range(projection_year-3, projection_year):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year != projection_year:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        df = pd.read_csv(file_path).iloc[:, 1:]
        df = df.dropna(subset=['PlayerID', 'GP', 'Goals', 'TOI'])
        df = df[df['Position'] != 'D'] if position == 'F' else df[df['Position'] == 'D']
        df[f'Y-{projection_year-year} GP'] = df['GP']
        df[f'Y-{projection_year-year} Gper1kChunk'] = df['Goals']/df['TOI']/2 * 1000
        df = df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} Gper1kChunk']]

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} Gper1kChunk']], on=['PlayerID', 'Player'], how='outer')

    # Generate scaling
    combined_df = combined_df.fillna(0)
    combined_df['GP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df = combined_df[combined_df['GP'] >= 82]
    combined_df['Scaled Gper1kChunk'] = (model.coef_[0]*combined_df['Y-3 Gper1kChunk']*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 Gper1kChunk']*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 Gper1kChunk']*combined_df['Y-1 GP']) / (model.coef_[0]*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 GP'])
    combined_df = combined_df.sort_values(by='Scaled Gper1kChunk', ascending=False)
    combined_df = combined_df.reset_index(drop=True)
    scaling = combined_df['Scaled Gper1kChunk'].to_list()
    
    return scaling, model

def train_a1_calibration_model(projection_year, retrain_model, position):

    if retrain_model:
        # Train model
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        if position == 'F':
            train_data = train_data[train_data['Position'] != 'D']
        elif position == 'D':
            train_data = train_data[train_data['Position'] == 'D']
        else:
            raise ValueError(f"Invalid position argument: {position}. Must be either 'F' or 'D'.")
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk']
        train_data = train_data.dropna(subset=feature_cols)
        X = train_data[feature_cols]
        y = train_data['Y-0 A1per1kChunk']
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        model.coef_ = model.coef_ / model.coef_.sum()

        # Save the model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'fwd_a1_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'dfc_a1_calibration_model.pkl')
        joblib.dump(model, model_path)
    
    else:
        # Load model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'fwd_a1_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'dfc_a1_calibration_model.pkl')
        model = joblib.load(model_path)

    # Get data from past 3 years
    combined_df = pd.DataFrame()
    for year in range(projection_year-3, projection_year):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year != projection_year:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        df = pd.read_csv(file_path).iloc[:, 1:]
        df = df.dropna(subset=['PlayerID', 'GP', 'First Assists', 'TOI'])
        df = df[df['Position'] != 'D'] if position == 'F' else df[df['Position'] == 'D']
        df[f'Y-{projection_year-year} GP'] = df['GP']
        df[f'Y-{projection_year-year} A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
        df = df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} A1per1kChunk']]
        # print(df)

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} A1per1kChunk']], on=['PlayerID', 'Player'], how='outer')

    # Generate scaling
    combined_df = combined_df.fillna(0)
    combined_df['GP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df = combined_df[combined_df['GP'] >= 82]
    combined_df['Scaled A1per1kChunk'] = (model.coef_[0]*combined_df['Y-3 A1per1kChunk']*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 A1per1kChunk']*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 A1per1kChunk']*combined_df['Y-1 GP']) / (model.coef_[0]*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 GP'])
    combined_df = combined_df.sort_values(by='Scaled A1per1kChunk', ascending=False)
    combined_df = combined_df.reset_index(drop=True)
    scaling = combined_df['Scaled A1per1kChunk'].to_list()
    
    return scaling, model

def train_a2_calibration_model(projection_year, retrain_model, position):

    if retrain_model:
        # Train model
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.loc[(train_data['Y-3 GP'] >= 30) & (train_data['Y-2 GP'] >= 30) & (train_data['Y-1 GP'] >= 30) & (train_data['Y-0 GP'] >= 30)]
        if position == 'F':
            train_data = train_data[train_data['Position'] != 'D']
        elif position == 'D':
            train_data = train_data[train_data['Position'] == 'D']
        else:
            raise ValueError(f"Invalid position argument: {position}. Must be either 'F' or 'D'.")
        feature_cols = ['Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk']
        train_data = train_data.dropna(subset=feature_cols)
        X = train_data[feature_cols]
        y = train_data['Y-0 A2per1kChunk']
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        model.coef_ = model.coef_ / model.coef_.sum()

        # Save the model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'fwd_a2_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'dfc_a2_calibration_model.pkl')
        joblib.dump(model, model_path)
    
    else:
        # Load model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'fwd_a2_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'inference', 'dfc_a2_calibration_model.pkl')
        model = joblib.load(model_path)

    # Get data from past 3 years
    combined_df = pd.DataFrame()
    for year in range(projection_year-3, projection_year):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            if year != projection_year:
                print(f'{filename} does not exist in the following directory: {file_path}')
                return
    
        df = pd.read_csv(file_path).iloc[:, 1:]
        df = df.dropna(subset=['PlayerID', 'GP', 'Second Assists', 'TOI'])
        df = df[df['Position'] != 'D'] if position == 'F' else df[df['Position'] == 'D']
        df[f'Y-{projection_year-year} GP'] = df['GP']
        df[f'Y-{projection_year-year} A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
        df = df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} A2per1kChunk']]
        # print(df)

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} A2per1kChunk']], on=['PlayerID', 'Player'], how='outer')

    # Generate scaling
    combined_df = combined_df.fillna(0)
    combined_df['GP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df = combined_df[combined_df['GP'] >= 82]
    combined_df['Scaled A2per1kChunk'] = (model.coef_[0]*combined_df['Y-3 A2per1kChunk']*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 A2per1kChunk']*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 A2per1kChunk']*combined_df['Y-1 GP']) / (model.coef_[0]*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 GP'])
    combined_df = combined_df.sort_values(by='Scaled A2per1kChunk', ascending=False)
    combined_df = combined_df.reset_index(drop=True)
    scaling = combined_df['Scaled A2per1kChunk'].to_list()

    return scaling, model