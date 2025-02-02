import os
import numpy as np
import pandas as pd
from model_training import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

def bootstrap_gp_inferences(projection_year, bootstrap_df, retrain_model, download_file, verbose):

    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'engine_data', 'Projection Models', 'bootstraps', 'gp_bootstrapped_models.pkl')

    # Retrain model if specified
    if retrain_model:
        train_data = aggregate_skater_offence_training_data(projection_year)
        train_data = train_data.dropna(subset=['Y-0 Age'])
        train_data['PositionBool'] = train_data['Position'].apply(lambda x: 0 if x == 'D' else 1)
        train_data['Y-3 Points'] = (train_data['Y-3 Gper1kChunk'] + train_data['Y-3 A1per1kChunk'] + train_data['Y-3 A2per1kChunk'])/1000*2 * train_data['Y-3 GP'] * train_data['Y-3 ATOI']
        train_data['Y-2 Points'] = (train_data['Y-2 Gper1kChunk'] + train_data['Y-2 A1per1kChunk'] + train_data['Y-2 A2per1kChunk'])/1000*2 * train_data['Y-3 GP'] * train_data['Y-3 ATOI']
        train_data['Y-1 Points'] = (train_data['Y-1 Gper1kChunk'] + train_data['Y-1 A1per1kChunk'] + train_data['Y-1 A2per1kChunk'])/1000*2 * train_data['Y-3 GP'] * train_data['Y-3 ATOI']

        # features = ['Y-3 GP', 'Y-3 Points', 'Y-2 GP', 'Y-2 Points', 'Y-1 GP', 'Y-1 Points', 'Y-0 Age', 'PositionBool']
        features = ['Y-3 GP', 'Y-3 Points', 'Y-3 ATOI', 'Y-2 GP', 'Y-2 Points', 'Y-2 ATOI', 'Y-1 GP', 'Y-1 Points', 'Y-1 ATOI', 'Y-0 Age', 'PositionBool']
        target_var = 'Y-0 GP'

        # cols = ['Player', 'Y-0'] + features
        # print(train_data[cols])
        # quit()
        
        # Define X and y
        X = train_data[features]
        y = train_data[target_var]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameters for XGBoost
        params = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 3,
            'n_estimators': 100,
            'reg_alpha': 0.00,
            'reg_lambda': 0.00,
            'subsample': 0.8,
            'objective': 'reg:squarederror'
        }
        
        # Initialize and train the model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict on the training and testing sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate the Mean Squared Error for training and testing sets
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        
        print(f'Training MSE: {train_mse}, Training MAE: {train_mae}')
        print(f'Testing MSE: {test_mse}, Testing MAE: {test_mae}')

        quit()

        # # Download models
        # models_dict = {f'model_{i}': model for i, model in enumerate(models)}
        # joblib.dump(models_dict, model_path)

    else:
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
            df = df[['PlayerID', 'Player', 'GP', 'Total Points']]
            df = df.rename(columns={'GP': f'Y-{projection_year-year} GP', 'Total Points': f'Y-{projection_year-year} Points'})
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{year-2}-{year-1}_skater_data.csv')) # copy last season df
            df = df[['PlayerID', 'Player']]
            df[f'Y-{projection_year-year} GP'] = 0
            df[f'Y-{projection_year-year} Points'] = 0

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on=['PlayerID', 'Player'], how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['PlayerID', 'Player', 'Date of Birth', 'Position', 'Team'])
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
    std_devs = np.std(predictions, axis=1)
    combined_df['GP'] = std_devs

    # Merge inferences into bootstrap_df
    if bootstrap_df is None or bootstrap_df.empty:
        combined_df.rename(columns={'Y-0 Age': 'Age'}, inplace=True)
        bootstrap_df = combined_df[['PlayerID', 'Player', 'Team', 'Position', 'Age']].copy()
        bootstrap_df['Age'] = bootstrap_df['Age'] - 1
        bootstrap_df['GP'] = std_devs
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

# Bootstrap player inferences
bootstrap_df = pd.DataFrame()
bootstrap_df = bootstrap_gp_inferences(projection_year=2025, bootstrap_df=bootstrap_df, retrain_model=True, download_file=True, verbose=False)
print(bootstrap_df)
quit()