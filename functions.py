import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def scrape_historical_data(start_year, end_year, skaters, bios, verbose):
    for year in range(start_year, end_year+1):
        if skaters == True and bios == False:
            filename = f'{year-1}-{year}_skater_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
        elif skaters == False and bios == False:
            filename = f'{year-1}-{year}_goalie_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Goaltending Data', filename)
        elif skaters == True and bios == True:
            filename = f'{year-1}-{year}_skater_bios.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'Historical Skater Bios', filename)
        elif skaters == False and bios == True:
            filename = f'{year-1}-{year}_goalie_bios.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Goaltenders', 'Historical Goaltender Bios', filename)

        if os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        if skaters == True and bios == False:
            url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        elif skaters == False and bios == False:
            url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=g&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        elif skaters == True and bios == True:
            url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        elif skaters == False and bios == True:
            url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=G&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        
        df = pd.read_html(url)[0]
        df = df.iloc[:, 1:]
        if verbose == True:
            print(df)

        export_path = os.path.dirname(file_path)
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        df.to_csv(os.path.join(export_path, filename))
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

    return None

def aggregate_player_bios(skaters, check_preexistence, verbose):
    if skaters == True:
        filename = f'skater_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', filename)
    else:
        filename = f'goalie_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Goaltenders', filename)
    
    if check_preexistence == True:
        if os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            return

    if skaters == True:
        files = sorted(os.listdir(os.path.join(os.path.dirname(file_path), 'Historical Skater Bios')))
    else:
        files = sorted(os.listdir(os.path.join(os.path.dirname(file_path), 'Historical Goaltender Bios')))
    dataframes = []
    for file in files:
        if skaters == True:
            df = pd.read_csv(os.path.join(os.path.dirname(file_path), 'Historical Skater Bios', file), index_col=0)
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(file_path), 'Historical Goaltender Bios', file), index_col=0)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=False)
    combined_df.drop_duplicates(subset='Player', keep='first', inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    export_path = os.path.dirname(file_path)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    combined_df.to_csv(os.path.join(export_path, filename), index=True)
    if verbose:
        print(f'{filename} has been downloaded to the following directory: {export_path}')

    return combined_df

def aggregate_training_data():
    file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data')
    files = sorted(os.listdir(file_path))
    for file in files:
        if file[-15:] != 'skater_data.csv':
            files.remove(file)
    combinations = [files[i:i+4] for i in range(len(files)-3)]

    combined_data = pd.DataFrame()

    for file_list in combinations:
        combined_df = None
        for index, file in enumerate(file_list):
            df = pd.read_csv(os.path.join(file_path, file), usecols=['Player', 'GP', 'TOI'])
            df['ATOI'] = df['TOI']/df['GP']
            df = df.drop(columns=['TOI'])
            df = df.rename(columns={'ATOI': f'Y-{3-index} ATOI', 'GP': f'Y-{3-index} GP'})
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='Player', how='outer')

        last_file = file_list[-1]
        bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['Player', 'Date of Birth'])
        combined_df = combined_df.merge(bios_df, on='Player', how='left')
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
    # combined_data.sort_values(by='Y-0 ATOI', ascending=False, inplace=True)
    combined_data.sort_values(by=['Player', 'Y-0'], ascending=[True, False], inplace=True)
    combined_data = combined_data.reset_index(drop=True)
    # print(combined_data.to_string())
    # print(combined_data)

    return combined_data

def train_atoi_model(retrain_model, verbose):

    filename = 'atoi_model.csv'
    file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projection Models', filename)

    if retrain_model == True:

        atoi_train_data = aggregate_training_data()
        print(atoi_train_data)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(atoi_train_data, test_size=0.5, random_state=42)

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
        coef_df.to_csv('atoi_model.csv')

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
        
### Needs to be edited to include current season
def project_atoi(projection_year, player_stat_df, atoi_model_data, download_file, verbose):

    combined_df = pd.DataFrame()

    for year in range(projection_year-3, projection_year):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
        if not os.path.exists(file_path):
            print(f'{filename} does not exist in the following directory: {file_path}')
            return
    
        # print the stat df
        df = pd.read_csv(file_path)
        df = df[['Player', 'GP', 'TOI']]
        df['ATOI'] = df['TOI']/df['GP']
        df = df.drop(columns=['TOI'])
        df = df.rename(columns={'ATOI': f'Y-{projection_year-year} ATOI', 'GP': f'Y-{projection_year-year} GP'})

        if combined_df is None or combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='Player', how='outer')

    # Calculate projection age
    bios_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), usecols=['Player', 'Date of Birth', 'Position'])
    combined_df = combined_df.merge(bios_df, on='Player', how='left')
    combined_df['Date of Birth'] = pd.to_datetime(combined_df['Date of Birth'])
    combined_df['Y-0 Age'] = projection_year - combined_df['Date of Birth'].dt.year
    combined_df = combined_df.drop(columns=['Date of Birth'])
    combined_df = combined_df.dropna(subset=['Y-1 GP'])
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.fillna(0)

    # combined_df['Proj. ATOI'] = atoi_model_data[0]*combined_df['Y-3 ATOI'] + atoi_model_data[1]*combined_df['Y-2 ATOI'] + atoi_model_data[2]*combined_df['Y-1 ATOI'] + atoi_model_data[3]*combined_df['Y-0 Age'] + atoi_model_data[4]*combined_df['Y-0 Age']*combined_df['Y-0 Age'] + atoi_model_data[5]*combined_df['Y-0 Age']*combined_df['Y-0 Age']*combined_df['Y-0 Age'] + atoi_model_data[6]
    # combined_df['Weighted'] = (combined_df['Y-3 ATOI']*combined_df['Y-3 GP'] + combined_df['Y-2 ATOI']*combined_df['Y-2 GP'] + combined_df['Y-1 ATOI']*combined_df['Y-1 GP'])/(combined_df['Y-3 GP'] + combined_df['Y-2 GP'] + combined_df['Y-1 GP'])

    combined_df['Y-3 Score'] = combined_df['Y-3 ATOI']*combined_df['Y-3 GP']*atoi_model_data[0]
    combined_df['Y-2 Score'] = combined_df['Y-2 ATOI']*combined_df['Y-2 GP']*atoi_model_data[1]
    combined_df['Y-1 Score'] = combined_df['Y-1 ATOI']*combined_df['Y-1 GP']*atoi_model_data[2]
    combined_df['Age Score'] = combined_df['Y-0 Age']*atoi_model_data[3] + combined_df['Y-0 Age']*combined_df['Y-0 Age']*atoi_model_data[4] + combined_df['Y-0 Age']*combined_df['Y-0 Age']*combined_df['Y-0 Age']*atoi_model_data[5] + atoi_model_data[6]
    combined_df['Weight'] = combined_df['Y-3 GP']*atoi_model_data[0] + combined_df['Y-2 GP']*atoi_model_data[1] + combined_df['Y-1 GP']*atoi_model_data[2]
    combined_df['Score'] = combined_df['Y-3 Score'] + combined_df['Y-2 Score'] + combined_df['Y-1 Score'] + combined_df['Age Score']
    combined_df['Proj. ATOI'] = combined_df['Score']/combined_df['Weight']
    combined_df = combined_df.drop(columns=['Y-3 Score', 'Y-2 Score', 'Y-1 Score', 'Age Score', 'Weight', 'Score'])
    combined_df.sort_values(by='Proj. ATOI', ascending=False, inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    if verbose:
        print()
        print(combined_df)

    combined_df = combined_df[['Player', 'Position', 'Y-0 Age', 'Proj. ATOI']]
    combined_df = combined_df.rename(columns={'Y-0 Age': 'Age', 'Proj. ATOI': 'ATOI'})
    combined_df['Age'] = (combined_df['Age'] - 1).astype(int)

    if player_stat_df is None or player_stat_df.empty:
        player_stat_df = combined_df
    else:
        player_stat_df = pd.merge(player_stat_df, combined_df, on=['Player', 'Position', 'Age'], how='left')

    if download_file:
        export_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        player_stat_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_projections.csv has been downloaded to the following directory: {export_path}')

    return player_stat_df


start_time = time.time()

projection_year = 2024
scrape_historical_data(2008, 2023, True, False, False)
scrape_historical_data(2008, 2023, False, False, False)
scrape_historical_data(2008, 2023, True, True, False)
scrape_historical_data(2008, 2023, False, True, False)
aggregate_player_bios(True, True, False)
aggregate_player_bios(False, True, False)

atoi_model_data = train_atoi_model(False, False)
print(atoi_model_data) 
'''
Need to accomodate for same season data. 
Phase each year of data to the previous year's weight based on gameNumber/82, which determines the progress of the season.
For ex: For ATOI model gives [ 9.90060537e-02  1.05083392e-01  7.43612821e-01  -1.32400542e-01  2.21274207e-04  -1.24648414e-06  4.51331597e+00]
Start of season (0th game): [9.90060537e-02, 1.05083392e-01, 7.43612821e-01, 7.43612821e-01, ...]
40th game of season: [(0 - 9.90060537e-02)*40/82 + 9.90060537e-02 || (9.90060537e-02 - 1.05083392e-01)*40/82 + 1.05083392e-01 || (1.05083392e-01 - 7.43612821e-01)*0/82 + 7.43612821e-01 || 7.43612821e-01, ...]
Use max GP for all players to determine season progress.
Make sure that this adjustment of weights is accounted for when calculating the projected ATOI (weighted average).

Also, if proj_year = 2024 and there is no df for that season, then copy the 2023 df, keep all names, set stats to 0.
This is like if the season hasn't started yet.
'''

player_stat_df = pd.DataFrame()
player_stat_df = project_atoi(projection_year, player_stat_df, atoi_model_data, True, False)
print(player_stat_df)

print(f"Runtime: {time.time()-start_time:.3f} seconds")

## Should also make a new file for the scraper functions (historical stats and bio aggregator; first 2 functions)