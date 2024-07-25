import os
import unidecode
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Function to scrape raw historical data from Natural Stat Trick
def scrape_historical_player_data(start_year, end_year, skaters, bios, on_ice, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year+1):
        if skaters == True and bios == False and on_ice == False:
            filename = f'{year-1}-{year}_skater_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        elif skaters == True and bios == False and on_ice == True:
            filename = f'{year-1}-{year}_skater_onice_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical On-Ice Skater Data', filename)
        elif skaters == False and bios == False:
            filename = f'{year-1}-{year}_goalie_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Goaltending Data', filename)
        elif skaters == True and bios == True:
            filename = f'{year-1}-{year}_skater_bios.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'Historical Skater Bios', filename)
        elif skaters == False and bios == True:
            filename = f'{year-1}-{year}_goalie_bios.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Goaltenders', 'Historical Goaltender Bios', filename)

        if check_preexistence == True:
            if os.path.exists(file_path):
                if verbose:
                    print(f'{filename} already exists in the following directory: {file_path}')
                continue

        if projection_year != year or season_state != 'PRESEASON':
            if skaters == True and bios == False and on_ice == False:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == True and bios == False and on_ice == True:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=oi&rate=y&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == False and bios == False:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=g&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == True and bios == True:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == False and bios == True:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=G&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            df = pd.read_html(url)[0]
            df = df.iloc[:, 1:]
        else:
            if skaters == True and bios == False:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-2}{year-1}&thruseason={year-2}{year-1}&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == False and bios == False:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-2}{year-1}&thruseason={year-2}{year-1}&stype=2&sit=all&score=all&stdoi=g&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == True and bios == True:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-2}{year-1}&thruseason={year-2}{year-1}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            elif skaters == False and bios == True:
                url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-2}{year-1}&thruseason={year-2}{year-1}&stype=2&sit=all&score=all&stdoi=bio&rate=n&team=ALL&pos=G&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
            df = pd.read_html(url)[0]
            df = df.iloc[:, 1:]

            if bios == False:
                # Fill stats data with 0
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                specific_columns = ['IPP', 'SH%', 'Faceoffs %']
                for column in df.columns:
                    if column in numeric_columns or column in specific_columns:
                        df[column] = 0

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
def scrape_historical_team_data(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year+1):
        filename = f'{year-1}-{year}_team_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Team Data', filename)

        if check_preexistence == True:
            if os.path.exists(file_path):
                if verbose:
                    print(f'{filename} already exists in the following directory: {file_path}')
                continue

        if projection_year != year or season_state != 'PRESEASON':
            url = f'https://www.naturalstattrick.com/teamtable.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&rate=n&team=all&loc=B&gpf=410&fd=&td='
            df = pd.read_html(url)[0]
            df = df.iloc[:, 1:]
        else:
            response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
            data = response.json()
            df = pd.DataFrame(data['data'])
            team_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'))
            df = df.merge(team_data[['TeamID', 'Team Name']], left_on='homeTeamId', right_on='TeamID', how='left')
            df = df.merge(team_data[['TeamID', 'Team Name']], left_on='visitingTeamId', right_on='TeamID', how='left')
            df = df[df['gameType'] == 2][['Team Name_x']]
            df.columns = ['Team']
            df = df.drop_duplicates()
            df = df.sort_values(by='Team')
            df = df.reset_index(drop=True)
            df['GP'], df['TOI'], df['W'], df['L'], df['OTL'], df['ROW'], df['Points'], df['Point %'], df['CA'], df['FA'], df['SA'], df['GF'], df['GA'], df['xGA'], df['SCA'], df['HDCA'], df['HDGA'], df['HDSV%'], df['SV%'] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
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
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', filename)
    else:
        filename = f'goalie_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Goaltenders', filename)
    
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
    if 'playerStripped' not in combined_df.columns:
        combined_df['playerStripped'] = combined_df['Player'].apply(lambda x: unidecode.unidecode(x)).str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names)
    combined_df.sort_values(by=['Date of Birth', 'Last Season'], na_position='first', inplace=True)
    combined_df.drop_duplicates(subset='playerStripped', keep='last', inplace=True)
    # combined_df = combined_df[combined_df['Date of Birth'] != '-']
    combined_df['Age'] = combined_df['Age'].replace('-', 28)
    combined_df['Date of Birth'] = combined_df['Date of Birth'].replace('-', '1993-01-01')
    combined_df = combined_df.reset_index(drop=True)

    export_path = os.path.dirname(file_path)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    combined_df.to_csv(os.path.join(export_path, filename), index=True)
    if verbose:
        print(f'{filename} has been downloaded to the following directory: {export_path}')

    return combined_df

# Scrape team data from NHL API
def scrape_teams(projection_year, check_preexistence, verbose):

    filename = f'nhlapi_team_data.csv'
    file_path = os.path.dirname(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', filename))

    if check_preexistence == True:
        if os.path.isfile(file_path):
            return
    else:
        teams_response = requests.get('https://api.nhle.com/stats/rest/en/team')
        teams_data = teams_response.json()
        schedule_response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
        schedule_data = schedule_response.json()

        # Get unique team ids for projection season
        home_team_ids = [game['homeTeamId'] for game in schedule_data['data']]
        visiting_team_ids = [game['visitingTeamId'] for game in schedule_data['data']]
        distinct_team_ids = list(set(home_team_ids + visiting_team_ids))

        # Construct teams df
        df = pd.DataFrame(teams_data['data'])[['id', 'fullName', 'triCode']]
        df.columns = ['TeamID', 'Team Name', 'Abbreviation']
        df['Active'] = df['TeamID'].apply(lambda x: x in distinct_team_ids)

        if verbose:
            print(df)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, filename))

# Scrape schedule data from NHL API
def scrape_games(projection_year, check_preexistence, verbose):

    filename = f'{projection_year-1}-{projection_year}_game_schedule.csv'
    file_path = os.path.dirname(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', filename))

    if check_preexistence == True:
        if os.path.isfile(file_path):
            return
    else:
        response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
        data = response.json()

        df = pd.DataFrame(data['data'])
        # join with team data to get team names
        team_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'))
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
def scrape_nhlapi_data(start_year, end_year, bios, projection_year, season_state, check_preexistence, verbose):

    for year in range(start_year, end_year+1):
        if bios == True:
            filename = f'{year-1}-{year}_skater_bios.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'Historical Skater Bios', filename)
        elif bios == False:
            filename = f'{year-1}-{year}_skater_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', filename)
        
        # check if the column 'Headshot' is already in the dataframe
        if check_preexistence == True:
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path, index_col=0)
                if 'Headshot' in df.columns:
                    continue

        if projection_year != year or season_state != 'PRESEASON':
            response = requests.get(f'https://api-web.nhle.com/v1/skater-stats-leaders/{year-1}{year}/2?categories=toi&limit=9999')
        else:
            response = requests.get(f'https://api-web.nhle.com/v1/skater-stats-leaders/{year-2}{year-1}/2?categories=toi&limit=9999')
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
            historical_skater_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Player Bios', 'Skaters', 'Historical Skater Bios', f'{year-1}-{year}_skater_bios.csv'), index_col=0)
        elif bios == False:
            historical_skater_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{year-1}-{year}_skater_data.csv'), index_col=0)

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

def get_season_state(projection_year):
    response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
    data = response.json()

    if data['total'] == 0:
        return 'NO SCHEDULE'
    elif data['data'][0]['gameStateId'] == 1:
        return 'PRESEASON'
    else:
        return 'REGULAR SEASON'
    
def fix_teams(player_stat_df):
    url = "https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=999999&q=%2A&active=true"
    response = requests.get(url)
    data = response.json()
    active_players_df = pd.DataFrame(data)
    active_players_df['playerId'] = active_players_df['playerId'].astype('int64')
    active_players_df.rename(columns={'playerId': 'PlayerID', 'teamAbbrev': 'Team'}, inplace=True)
    player_to_team_map = active_players_df.set_index('PlayerID')['Team'].to_dict()
    player_stat_df['Team'] = player_stat_df['PlayerID'].map(player_to_team_map)
    return player_stat_df
    
def push_to_supabase(table_name, verbose=False):
    load_dotenv()
    SUPABASE_URL = os.getenv('REACT_APP_SUPABASE_PROJ_URL')
    SUPABASE_KEY = os.getenv('REACT_APP_SUPABASE_ANON_KEY')
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    try:
        session = supabase.auth.sign_in_with_password({"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})
    except:
        supabase.auth.sign_up(credentials={"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})
        session = supabase.auth.sign_in_with_password({"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})

    if table_name == 'team-projections':
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Teams', '2025_team_aggregated_projections.csv')
        df = pd.read_csv(file_path)
        df = df.drop(df.columns[0], axis=1)
        rename_dict = {
            'Abbreviation': 'abbrev',
            'Team': 'team',
            'Points': 'points',
            'Wins': 'wins',
            'Losses': 'losses',
            'OTL': 'otl',
            'Goals For': 'goals_for',
            'Goals Against': 'goals_against',
        }
        df.rename(columns=rename_dict, inplace=True)
        df['logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['abbrev'] + '_dark.svg'
        df['playoff_prob'] = 0.50
        df['presidents_trophy_prob'] = 0.03125
        df['stanley_cup_prob'] = 0.03125
    elif table_name == 'player-projections':
        file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters', '2025_skater_aggregated_projections.csv')
        df = pd.read_csv(file_path)
        df = df.drop(df.columns[0], axis=1)
        rename_dict = {
            'PlayerID': 'player_id',
            'Player': 'player',
            'Position': 'position',
            'Team': 'team',
            'Age': 'age',
            'Games Played': 'games',
            'Goals': 'goals',
            'Assists': 'assists',
            'Points': 'points',
        }
        df.rename(columns=rename_dict, inplace=True)
        df['position'] = df['position'].apply(lambda x: 'RW' if x == 'R' else ('LW' if x == 'L' else x))
        df['logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['team'] + '_dark.svg'
    data_to_insert = df.to_dict(orient='records')

    if verbose:
        print(df)
        print(data_to_insert)
    
    delete_response = None
    insert_response = None
    try:
        delete_response = supabase.table(table_name).delete().gt('points', -1).execute()
        insert_response = supabase.table(table_name).insert(data_to_insert).execute()
        print(f"Successfully inserted {len(data_to_insert)} records into '{table_name}' table.")
    except Exception as e:
        print(f"An error occurred: {e}")

    supabase.auth.sign_out()
    return delete_response, insert_response