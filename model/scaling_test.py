import os
import pandas as pd
from model_training import *
from sklearn.linear_model import LinearRegression
import joblib

def train_goal_calibration_model(projection_year, retrain_model, position):

    if retrain_model:
        # Train model
        train_data = aggregate_skater_offence_training_data(projection_year)
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
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'fwd_goal_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'dfc_goal_calibration_model.pkl')
        joblib.dump(model, model_path)
    
    else:
        # Load model
        if position == 'F':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'fwd_goal_calibration_model.pkl')
        else:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projection Models', 'dfc_goal_calibration_model.pkl')
        model = joblib.load(model_path)

    combined_df = pd.DataFrame()
    for year in range(projection_year-3, projection_year):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
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
        print(df)

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df[['PlayerID', 'Player', f'Y-{projection_year-year} GP', f'Y-{projection_year-year} Gper1kChunk']], on=['PlayerID', 'Player'], how='outer')

    combined_df = combined_df.fillna(0)
    combined_df['GP'] = combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP']
    combined_df = combined_df[combined_df['GP'] >= 82]
    combined_df['Scaled Gper1kChunk'] = (model.coef_[0]*combined_df['Y-3 Gper1kChunk']*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 Gper1kChunk']*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 Gper1kChunk']*combined_df['Y-1 GP']) / (model.coef_[0]*combined_df['Y-3 GP'] + model.coef_[1]*combined_df['Y-2 GP'] + model.coef_[2]*combined_df['Y-1 GP'])
    combined_df = combined_df.sort_values(by='Scaled Gper1kChunk', ascending=False)
    combined_df = combined_df.reset_index(drop=True)
    scaling_list = combined_df['Scaled Gper1kChunk'].to_list()
    
    return scaling_list