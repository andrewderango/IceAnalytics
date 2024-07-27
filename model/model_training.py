import os
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
    # combined_data.sort_values(by='Y-0 ATOI', ascending=False, inplace=True)
    combined_data.sort_values(by=['Player', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    # print(combined_data.to_string())
    # print(combined_data)

    return combined_data

def aggregate_skater_defence_training_data(projection_year):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical On-Ice Skater Data')
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
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Team Data')
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

    filename = 'atoi_model.csv'
    file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', filename)

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(train_data, test_size=0.5, random_state=42)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        test_data = test_data.dropna(subset=['Y-0 Age'])

        # Define the input variables and target variable
        input_vars = ['Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-0 Age']
        target_var = 'Y-0 ATOI'

        # Create polynomial features for 'Y-0 Age'
        poly = PolynomialFeatures(3, include_bias=False)
        train_data[['Y-0 Age', 'Y-0 Age^2', 'Y-0 Age^3']] = poly.fit_transform(train_data[['Y-0 Age']])
        test_data[['Y-0 Age', 'Y-0 Age^2', 'Y-0 Age^3']] = poly.transform(test_data[['Y-0 Age']])

        # Update the input variables to include the polynomial features
        input_vars = ['Y-3 ATOI', 'Y-2 ATOI', 'Y-1 ATOI', 'Y-0 Age', 'Y-0 Age^2', 'Y-0 Age^3']

        # Create the Linear Regression model
        model = LinearRegression()

        # Train the Linear Regression model
        model.fit(train_data[input_vars], train_data[target_var])

        if verbose:
            # Make predictions on the test data
            predictions = model.predict(test_data[input_vars])

            mse = mean_squared_error(test_data[target_var], predictions)
            print("MSE for ATOI model:", mse)

            print("ATOI model linear coefficients:")
            for i in range(len(input_vars)):
                print(input_vars[i], '\t', model.coef_[i])

            print("Linear Regression Intercept:\t", model.intercept_)

        # Save the model coefficients and intercept
        coef_df = pd.DataFrame(model.coef_, index=input_vars, columns=['Coefficient'])
        coef_df.index.name = 'Label'
        coef_df.loc['Intercept'] = model.intercept_

        export_path = os.path.dirname(file_path)
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        coef_df.to_csv(os.path.join(export_path, filename), index=True)
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

        return np.append(model.coef_, model.intercept_)
    
    else:
        if os.path.exists(file_path):
            coef_df = pd.read_csv(file_path, index_col=0)
            if verbose:
                print(coef_df)
            return coef_df['Coefficient'].values
        else:
            print(f'{filename} does not exist in the following directory: {file_path}')
            return None

def train_goal_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        
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
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'goal_model.json'))
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'goal_model.xgb'))
    
    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'goal_model.xgb'))
    
    return model
    
def train_a1_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'Position']
        train_data = train_data.dropna(subset=feature_cols)

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 A1per1kChunk']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(len(feature_cols), input_dim=len(feature_cols), kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(10, kernel_initializer='normal'))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=verbose)

        # Evaluate the model
        mse = model.evaluate(X_test, y_test, verbose=0)
        if verbose:
            print("MSE: %.2f" % mse)

        # Save the model
        model.save(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'primary_assist_model.keras'))

        return model
    
    else:
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'primary_assist_model.keras'), compile=False)
        return model
    
def train_a2_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_offence_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 A1per1kChunk', 'Y-2 A1per1kChunk', 'Y-1 A1per1kChunk', 'Y-3 A2per1kChunk', 'Y-2 A2per1kChunk', 'Y-1 A2per1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-3 RCper1kChunk', 'Y-2 RCper1kChunk', 'Y-1 RCper1kChunk', 'Y-3 TAper1kChunk', 'Y-2 TAper1kChunk', 'Y-1 TAper1kChunk', 'Y-0 Age', 'Position']
        train_data = train_data.dropna(subset=feature_cols)

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 A2per1kChunk']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(len(feature_cols), input_dim=len(feature_cols), kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(10, kernel_initializer='normal'))
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=verbose)

        # Evaluate the model
        mse = model.evaluate(X_test, y_test, verbose=0)
        if verbose:
            print("MSE: %.2f" % mse)

        # Save the model
        model.save(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'secondary_assist_model.keras'))

        return model
    
    else:
        model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'secondary_assist_model.keras'), compile=False)
        return model
    
def train_ga_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_team_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        feature_cols = ['Y-3 FA/GP', 'Y-2 FA/GP', 'Y-1 FA/GP', 'Y-3 GA/GP', 'Y-2 GA/GP', 'Y-1 GA/GP', 'Y-3 xGA/GP', 'Y-2 xGA/GP', 'Y-1 xGA/GP', 'Y-3 SV%', 'Y-2 SV%', 'Y-1 SV%']

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 GA/GP']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1,
                        max_depth = 5, n_estimators = 100)

        # Train the model
        model.fit(X_train, y_train, verbose=verbose)

        # Evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        if verbose:
            print("MSE: %.2f" % mse)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'goals_against_model.xgb'))

        return model

    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'goals_against_model.xgb'))
        return model
    
def train_skater_xga_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_defence_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 CA/60', 'Y-2 CA/60', 'Y-1 CA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'Position']

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 xGA/60']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.10, max_depth = 3, n_estimators = 60)

        # Train the model
        model.fit(X_train, y_train, verbose=verbose)

        # Evaluate the model on the test dataset
        predictions_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, predictions_test)
        if verbose:
            print("Test MSE: %.3f" % mse_test)

        # Evaluate the model on the train dataset
        predictions_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, predictions_train)
        if verbose:
            print("Train MSE: %.3f" % mse_train)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'skater_xga_model.xgb'))
        return model

    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'skater_xga_model.xgb'))
        return model
    
def train_skater_ga_model(projection_year, retrain_model, verbose):

    if retrain_model == True:

        train_data = aggregate_skater_defence_training_data(projection_year)
        
        if verbose:
            print(train_data)

        # Define the feature columns
        train_data['Position'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        feature_cols = ['Y-3 GA/60', 'Y-2 GA/60', 'Y-1 GA/60', 'Y-3 xGA/60', 'Y-2 xGA/60', 'Y-1 xGA/60', 'Y-3 SA/60', 'Y-2 SA/60', 'Y-1 SA/60', 'Y-0 Age', 'Position']

        # Separate the features and the target
        X = train_data[feature_cols]
        y = train_data['Y-0 GA/60']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the model
        model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.075, max_depth = 2, n_estimators = 55)

        # Train the model
        model.fit(X_train, y_train, verbose=verbose)

        # Evaluate the model on the test dataset
        predictions_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, predictions_test)
        if verbose:
            print("Test MSE: %.3f" % mse_test)

        # Evaluate the model on the train dataset
        predictions_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, predictions_train)
        if verbose:
            print("Train MSE: %.3f" % mse_train)

        # Save the model
        model.save_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'skater_ga_model.xgb'))
        return model

    else:
        model = xgb.Booster()
        model.load_model(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'skater_ga_model.xgb'))
        return model