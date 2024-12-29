import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_training import *
from scraper_functions import *
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def bootstrap_atoi_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'atoi_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'bootstrap_variance.json')

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
        models, cumulative_residuals = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping ATOI"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            bootstrap_residuals = y_test - y_test_pred
            cumulative_residuals.extend(bootstrap_residuals)
            models.append(model)
        residual_variance = np.var(cumulative_residuals)

        # Get total variance of actual ATOI
        actual_variance = np.var(y)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify bootstrap variance json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['ATOI'] = {'Residual': residual_variance, 'Actual': actual_variance}
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['ATOI']['Residual']
        actual_variance = json_data['ATOI']['Actual']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
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
            df['Pper1kChunk'] = (df['Goals'] + df['First Assists'] + df['Second Assists'])/df['TOI']/2 * 1000
            df[['ATOI', 'GP', 'Pper1kChunk']] = df[['ATOI', 'GP', 'Pper1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'Goals', 'First Assists', 'Second Assists'])
            df = df.rename(columns={'ATOI': f'Y-{projection_year-year} ATOI', 'GP': f'Y-{projection_year-year} GP', 'Pper1kChunk': f'Y-{projection_year-year} Pper1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} ATOI'] = 0
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Pper1kChunk'] = 0

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
    print(residual_variance)
    print(actual_variance)
    r_squared = 1 - residual_variance/actual_variance # proportion of variance explained by model
    evp = 1 - r_squared # error variance proportion (evp) = 1 - r2
    combined_df['ATOI'] = combined_df['ATOI'] * np.sqrt(1 - np.minimum(combined_df['Y-0 GP'], 82)/82) * r_squared + evp

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_gp_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'gp_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'bootstrap_variance.json')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        train_data['Y-3 Points'] = train_data['Y-3 GP']*train_data['Y-3 ATOI']*train_data['Y-3 Gper1kChunk']/1000*2
        train_data['Y-2 Points'] = train_data['Y-2 GP']*train_data['Y-2 ATOI']*train_data['Y-2 Gper1kChunk']/1000*2
        train_data['Y-1 Points'] = train_data['Y-1 GP']*train_data['Y-1 ATOI']*train_data['Y-1 Gper1kChunk']/1000*2

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
        models, cumulative_residuals = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping GP"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            bootstrap_residuals = y_test - y_test_pred
            cumulative_residuals.extend(bootstrap_residuals)
            models.append(model)
        residual_variance = np.var(cumulative_residuals)

        # Get total variance of actual GP
        actual_variance = np.var(y)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify bootstrap variance json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['GP'] = {'Residual': residual_variance, 'Actual': actual_variance}
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['GP']['Residual']
        actual_variance = json_data['GP']['Actual']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
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
            df = df[['PlayerID', 'GP', 'Total Points']]
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'Total Points': f'Y-{projection_year-year} Points'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Points'] = 0

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
    print(residual_variance)
    print(actual_variance)
    r_squared = 1 - residual_variance/actual_variance # proportion of variance explained by model
    evp = 1 - r_squared # error variance proportion (evp) = 1 - r2
    combined_df['GP'] = combined_df['GP'] * np.sqrt(1 - np.minimum(combined_df['Y-0 GP'], 82)/82) * r_squared + evp

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_goal_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'goal_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'bootstrap_variance.json')

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
        models, cumulative_residuals = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping Goals"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            bootstrap_residuals = y_test - y_test_pred
            cumulative_residuals.extend(bootstrap_residuals)
            models.append(model)
        residual_variance = np.var(cumulative_residuals)

        # Get total variance of actual Gper1kChunk
        actual_variance = np.var(y)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify bootstrap variance json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['Gper1kChunk'] = {'Residual': residual_variance, 'Actual': actual_variance}
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['Gper1kChunk']['Residual']
        actual_variance = json_data['Gper1kChunk']['Actual']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
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
            df = df[['PlayerID', 'GP', 'TOI', 'Goals', 'ixG', 'Shots']]
            df['Gper1kChunk'] = df['Goals']/df['TOI']/2 * 1000
            df['xGper1kChunk'] = df['ixG']/df['TOI']/2 * 1000
            df['SHper1kChunk'] = df['Shots']/df['TOI']/2 * 1000
            df[['Gper1kChunk', 'xGper1kChunk', 'SHper1kChunk']] = df[['Gper1kChunk', 'xGper1kChunk', 'SHper1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'Goals', 'ixG', 'Shots'])
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'Gper1kChunk': f'Y-{projection_year-year} Gper1kChunk', 'xGper1kChunk': f'Y-{projection_year-year} xGper1kChunk', 'SHper1kChunk': f'Y-{projection_year-year} SHper1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Gper1kChunk'] = 0
            df[f'Y-{projection_year-year} xGper1kChunk'] = 0
            df[f'Y-{projection_year-year} SHper1kChunk'] = 0

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
    print(residual_variance)
    print(actual_variance)
    r_squared = 1 - residual_variance/actual_variance # proportion of variance explained by model
    evp = 1 - r_squared # error variance proportion (evp) = 1 - r2
    combined_df['Gper1kChunk'] = combined_df['Gper1kChunk'] * np.sqrt(1 - np.minimum(combined_df['Y-0 GP'], 82)/82) * r_squared + evp

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_a1_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'a1_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'bootstrap_variance.json')

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
        models, cumulative_residuals = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping Primary Assists"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            bootstrap_residuals = y_test - y_test_pred
            cumulative_residuals.extend(bootstrap_residuals)
            models.append(model)
        residual_variance = np.var(cumulative_residuals)

        # Get total variance of actual A1per1kChunk
        actual_variance = np.var(y)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify bootstrap variance json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['A1per1kChunk'] = {'Residual': residual_variance, 'Actual': actual_variance}
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['A1per1kChunk']['Residual']
        actual_variance = json_data['A1per1kChunk']['Actual']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
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
            df = df[['PlayerID', 'GP', 'TOI', 'First Assists', 'Second Assists']]
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df[['A1per1kChunk', 'A2per1kChunk']] = df[['A1per1kChunk', 'A2per1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists'])
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk', 'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0

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
    print(residual_variance)
    print(actual_variance)
    r_squared = 1 - residual_variance/actual_variance # proportion of variance explained by model
    evp = 1 - r_squared # error variance proportion (evp) = 1 - r2
    combined_df['A1per1kChunk'] = combined_df['A1per1kChunk'] * np.sqrt(1 - np.minimum(combined_df['Y-0 GP'], 82)/82) * r_squared + evp

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df

def bootstrap_a2_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'a2_bootstrapped_models.pkl')
    json_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projection Models', 'bootstraps', 'bootstrap_variance.json')

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
        models, cumulative_residuals = [], []
        bootstrap_samples = 500
        for i in tqdm(range(bootstrap_samples), desc="Bootstrapping Secondary Assists"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=i)
            X_sample, y_sample = resample(X_train, y_train, random_state=i)
            model = xgb.XGBRegressor(**params)
            model.fit(X_sample, y_sample)
            y_test_pred = model.predict(X_test)
            bootstrap_residuals = y_test - y_test_pred
            cumulative_residuals.extend(bootstrap_residuals)
            models.append(model)
        residual_variance = np.var(cumulative_residuals)

        # Get total variance of actual A2per1kChunk
        actual_variance = np.var(y)

        # Download models
        models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        joblib.dump(models_dict, model_path)

        # Modify bootstrap variance json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        json_data['A2per1kChunk'] = {'Residual': residual_variance, 'Actual': actual_variance}
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

    else:
        # Load residual variance
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        residual_variance = json_data['A2per1kChunk']['Residual']
        actual_variance = json_data['A2per1kChunk']['Actual']

        # Load models
        models_dict = joblib.load(model_path)
        models = [models_dict[model] for model in models_dict]
        bootstrap_samples = len(models)

    # Generate bootstrap inferences
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
            df = df[['PlayerID', 'GP', 'TOI', 'First Assists', 'Second Assists']]
            df['A1per1kChunk'] = df['First Assists']/df['TOI']/2 * 1000
            df['A2per1kChunk'] = df['Second Assists']/df['TOI']/2 * 1000
            df[['A1per1kChunk', 'A2per1kChunk']] = df[['A1per1kChunk', 'A2per1kChunk']].fillna(0)
            df = df.drop(columns=['TOI', 'First Assists', 'Second Assists'])
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'A1per1kChunk': f'Y-{projection_year-year} A1per1kChunk', 'A2per1kChunk': f'Y-{projection_year-year} A2per1kChunk'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} A1per1kChunk'] = 0
            df[f'Y-{projection_year-year} A2per1kChunk'] = 0

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
    print(residual_variance)
    print(actual_variance)
    r_squared = 1 - residual_variance/actual_variance # proportion of variance explained by model
    evp = 1 - r_squared # error variance proportion (evp) = 1 - r2
    combined_df['A2per1kChunk'] = combined_df['A2per1kChunk'] * np.sqrt(1 - np.minimum(combined_df['Y-0 GP'], 82)/82) * r_squared + evp

    # Drop columns without Player ID
    bootstrap_df = bootstrap_df.dropna(subset=['PlayerID'])

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
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        bootstrap_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_bootstraps.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_bootstraps.csv has been downloaded to the following directory: {export_path}')

    return bootstrap_df