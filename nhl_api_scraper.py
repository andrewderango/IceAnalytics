import os
import requests
import pandas as pd

def scrape_teams(check_preexistence, verbose):

    filename = f'nhlapi_team_data.csv'
    file_path = os.path.dirname(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Team Data', filename))

    if check_preexistence == True:
        if os.path.isfile(file_path):
            return
    else:
        response = requests.get('https://api.nhle.com/stats/rest/en/team')
        data = response.json()

        df = pd.DataFrame(data['data'])[['id', 'fullName', 'triCode']]
        df.columns = ['TeamID', 'Team Name', 'Abbreviation']

        if verbose:
            print(df)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, filename))

def scrape_games(projection_year, check_preexistence, verbose):

    filename = f'game_schedule.csv'
    file_path = os.path.dirname(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Team Data', filename))

    if check_preexistence == True:
        if os.path.isfile(file_path):
            return
    else:
        response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
        data = response.json()

        df = pd.DataFrame(data['data'])
        # join with team data to get team names
        team_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'))
        df = df.merge(team_data[['TeamID', 'Team Name']], left_on='homeTeamId', right_on='TeamID', how='left')
        df = df.merge(team_data[['TeamID', 'Team Name']], left_on='visitingTeamId', right_on='TeamID', how='left')

        df = df[df['gameType'] == 2][['id', 'easternStartTime', 'gameNumber', 'gameStateId', 'period', 'homeScore', 'Team Name_x', 'visitingScore', 'Team Name_y']]        
        df.columns = ['GameID', 'Time (EST)', 'Game Number', 'Game State', 'Period', 'Home Score', 'Home Team', 'Visiting Score', 'Visiting Team']
        df['Period'] = df['Period'].fillna(0).astype(int)
        df = df.sort_values(by='GameID')
        df = df.reset_index(drop=True)

        if verbose:
            print(df)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, filename))

projection_year = 2024
scrape_teams(True, False)
scrape_games(projection_year, True, False)