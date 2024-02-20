import os
import unidecode
import requests
import pandas as pd

# Function to scrape raw historical data from Natural Stat Trick
def scrape_historical_player_data(start_year, end_year, skaters, bios, check_preexistence, verbose):
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

        if check_preexistence == True:
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

# Function to scrape raw historical data from Natural Stat Trick
def scrape_historical_team_data(start_year, end_year, check_preexistence, verbose):
    for year in range(start_year, end_year+1):
        filename = f'{year-1}-{year}_team_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Team Data', filename)

        if check_preexistence == True:
            if os.path.exists(file_path):
                if verbose:
                    print(f'{filename} already exists in the following directory: {file_path}')
                continue

        url = f'https://www.naturalstattrick.com/teamtable.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&rate=n&team=all&loc=B&gpf=410&fd=&td='
        
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

# Function to aggregate historical player bios for all players in the Sim Engine database
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
    for file in files:
        if file[-4:] != '.csv':
            files.remove(file)

    dataframes = []
    for file in files:
        if skaters == True:
            df = pd.read_csv(os.path.join(os.path.dirname(file_path), 'Historical Skater Bios', file), index_col=0)
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(file_path), 'Historical Goaltender Bios', file), index_col=0)
        df['Last Season'] = file[:9]
        dataframes.append(df)
        
    combined_df = pd.concat(dataframes, ignore_index=False)
    combined_df.sort_values(by=['Date of Birth', 'Last Season'], na_position='first', inplace=True)
    combined_df.drop_duplicates(subset='playerStripped', keep='last', inplace=True)
    combined_df.drop(columns=['Last Season'], inplace=True)
    combined_df = combined_df[combined_df['Date of Birth'] != '-']
    combined_df = combined_df.reset_index(drop=True)

    export_path = os.path.dirname(file_path)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    combined_df.to_csv(os.path.join(export_path, filename), index=True)
    if verbose:
        print(f'{filename} has been downloaded to the following directory: {export_path}')

    return combined_df

# Scrape team data from NHL API
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

# Scrape schedule data from NHL API
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

# Add player ids from NHL API to historical skater data
def scrape_nhlapi_data(start_year, end_year, bios, check_preexistence, verbose):

    for year in range(start_year, end_year+1):
        if bios == True:
            filename = f'{year-1}-{year}_skater_bios.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'Historical Skater Bios', filename)
        elif bios == False:
            filename = f'{year-1}-{year}_skater_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
        
        # check if the column 'Headshot' is already in the dataframe
        if check_preexistence == True:
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path, index_col=0)
                if 'Headshot' in df.columns:
                    continue

        response = requests.get(f'https://api-web.nhle.com/v1/skater-stats-leaders/{year-1}{year}/2?categories=toi&limit=9999')
        data = response.json()
        df = pd.DataFrame(columns=['PlayerID', 'Player', 'Team', 'Position', 'Headshot', 'Team Logo'])
        
        for player in data['toi']:
            new_row = pd.DataFrame({
                'PlayerID': [player['id']], 
                'Player': [player['firstName']['default'] + ' ' + player['lastName']['default']], 
                'Team': [player['teamAbbrev']], 
                'Position': [player['position']],
                'Headshot': [player['headshot']],
                'Team Logo': [player['teamLogo']]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        # get sim engine data historical skater data
        if bios == True:
            historical_skater_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'Historical Skater Bios', f'{year-1}-{year}_skater_bios.csv'), index_col=0)
        elif bios == False:
            historical_skater_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', f'{year-1}-{year}_skater_data.csv'), index_col=0)

        # merge 
        try:
            historical_skater_data = historical_skater_data.drop(columns=['PlayerID', 'Headshot', 'Team Logo'])
        except KeyError:
            pass
        historical_skater_data['playerStripped'] = historical_skater_data['Player'].apply(lambda x: unidecode.unidecode(x)).str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names)
        df['playerStripped'] = df['Player'].apply(lambda x: unidecode.unidecode(x)).str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names)
        historical_skater_data['playerStripped'] = historical_skater_data.apply(handle_duplicate_names, axis=1)
        df['playerStripped'] = df.apply(handle_duplicate_names, axis=1)
        historical_skater_data.drop(columns=['Player', 'Position', 'Team'], inplace=True)
        combined_df = pd.merge(df, historical_skater_data, on='playerStripped', how='outer')

        export_path = os.path.dirname(file_path)
        combined_df.to_csv(os.path.join(export_path, filename))
        if verbose:
            print(f'{filename} has been modified in the following directory: {export_path}')

def replace_names(s):
    name_replacement_dict = {
        'mike': 'michael',
        'matthew': 'matt',
        'christopher': 'chris',
        'evgenii': 'evgeny',
        'alexander': 'alex',
        'patrick': 'pat',
        'nicholas': 'nick',
        'maxime': 'max',
        'cristovalnieves': 'boonieves',
        'daniel': 'danny',
        'tony': 'anthony',
        'jacob': 'jake',
        'william': 'will',
        'timothy': 'tim',
        'gerald': 'gerry',
        'alexchmelevski': 'sashachmelevski',
        'jjmoser': 'janismoser',
        'cameron': 'cam',
        'zacharyhayes': 'zackhayes',
        'sammy': 'samuel',
        'benoitoliviergroulx': 'bogroulx',
        'tommy': 'thomas',
        'alexey': 'alexei',
    }

    for original, replacement in name_replacement_dict.items():
        s = s.replace(original, replacement)
    return s

def handle_duplicate_names(row):
    if row['Player'] == 'Sebastian Aho':
        if 'C' in row['Position'] or 'R' in row['Position']:
            return 'sebastianahoC'
        elif row['Position'] == 'D':
            return 'sebastianahoD'
    elif row['Player'] == 'Ryan Johnson':
        if 'C' in row['Position']:
            return 'ryanjohnsonC'
        elif row['Position'] == 'D':
            return 'ryanjohnsonD'
    elif row['Player'] == 'Colin White':
        if 'C' in row['Position']:
            return 'colinwhiteC'
        elif row['Position'] == 'D':
            return 'colinwhiteD'
    return row['playerStripped']