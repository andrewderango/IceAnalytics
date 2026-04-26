import io
import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

NST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.naturalstattrick.com/",
}

def parse_toi(series):
    if series.dtype == object:
        return series.apply(lambda x: int(str(x).split(':')[0]) + int(str(x).split(':')[1])/60 if isinstance(x, str) and ':' in str(x) else float(x))
    return series


_SKATER_STATS_ENDPOINTS = ['summary', 'scoringpergame', 'powerplay', 'penaltykill', 'timeonice']

def _fetch_nhl_skater_report(report, season_id):
    url = (
        f'https://api.nhle.com/stats/rest/en/skater/{report}'
        f'?isAggregate=true&isGame=false&start=0&limit=-1'
        f'&cayenneExp=gameTypeId=2%20and%20seasonId%3C={season_id}%20and%20seasonId%3E={season_id}'
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json().get('data', []))

def _fetch_skater_team_map(season_id):
    url = (
        f'https://api.nhle.com/stats/rest/en/skater/summary'
        f'?isAggregate=false&isGame=false&start=0&limit=-1'
        f'&cayenneExp=gameTypeId=2%20and%20seasonId%3C={season_id}%20and%20seasonId%3E={season_id}'
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    rows = response.json().get('data', [])
    if not rows:
        return pd.DataFrame(columns=['playerId', 'teamAbbrevs'])
    df = pd.DataFrame(rows)[['playerId', 'teamAbbrevs']]
    return df.groupby('playerId', as_index=False)['teamAbbrevs'].agg(lambda s: ','.join(s.astype(str).unique()))




def _join_skater_reports(season_id):
    base = _fetch_nhl_skater_report('summary', season_id)
    if base.empty:
        return base
    combined = base
    for report in _SKATER_STATS_ENDPOINTS[1:]:
        df = _fetch_nhl_skater_report(report, season_id)
        if df.empty:
            continue
        overlap = [c for c in df.columns if c in combined.columns and c != 'playerId']
        df = df.drop(columns=overlap)
        combined = combined.merge(df, on='playerId', how='outer')
    team_map = _fetch_skater_team_map(season_id)
    if not team_map.empty:
        combined = combined.merge(team_map, on='playerId', how='left')
    return combined

# Scrape per-season skater stats from NHL API
def scrape_skater_data(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year + 1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', filename)

        if check_preexistence and os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        is_preseason_pull = (projection_year == year and season_state == 'PRESEASON')
        fetch_year = year - 1 if is_preseason_pull else year
        season_id = (fetch_year - 1) * 10000 + fetch_year

        df = _join_skater_reports(season_id)

        if is_preseason_pull and not df.empty:
            preserve = {'playerId', 'skaterFullName', 'lastName', 'positionCode', 'shootsCatches', 'teamAbbrevs', 'seasonId'}
            for col in df.columns:
                if col not in preserve and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = 0


        if not df.empty and 'playerId' in df.columns and 'skaterFullName' in df.columns:
            priority = [c for c in ['playerId', 'skaterFullName', 'teamAbbrevs', 'positionCode', 'gamesPlayed', 'goals', 'assists', 'points'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in set(priority)]
            df = df[priority + other_cols]
            sort_keys = [c for c in ['points', 'goals', 'assists', 'playerId'] if c in df.columns]
            sort_asc = [c == 'playerId' for c in sort_keys]
            df = df.sort_values(by=sort_keys, ascending=sort_asc).reset_index(drop=True)

        if verbose:
            print(df)

        export_path = os.path.dirname(file_path)
        os.makedirs(export_path, exist_ok=True)
        df.to_csv(os.path.join(export_path, filename), index=False)
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

    return

# Scrape per-season skater bios from NHL API
def scrape_skater_bios(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year + 1):
        filename = f'{year-1}-{year}_skater_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', 'Historical Skater Bios', filename)

        if check_preexistence and os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        is_preseason_pull = (projection_year == year and season_state == 'PRESEASON')
        fetch_year = year - 1 if is_preseason_pull else year
        season_id = (fetch_year - 1) * 10000 + fetch_year

        df = _fetch_nhl_skater_report('bios', season_id)

        if not df.empty:
            team_map = _fetch_skater_team_map(season_id)
            if not team_map.empty:
                df = df.merge(team_map, on='playerId', how='left')
            if 'playerId' in df.columns and 'skaterFullName' in df.columns:
                priority = [c for c in ['playerId', 'skaterFullName', 'teamAbbrevs', 'positionCode', 'gamesPlayed', 'goals', 'assists', 'points'] if c in df.columns]
                other_cols = [c for c in df.columns if c not in set(priority)]
                df = df[priority + other_cols]
                sort_keys = [c for c in ['points', 'goals', 'assists', 'playerId'] if c in df.columns]
                sort_asc = [c == 'playerId' for c in sort_keys]
                df = df.sort_values(by=sort_keys, ascending=sort_asc).reset_index(drop=True)

        if verbose:
            print(df)

        export_path = os.path.dirname(file_path)
        os.makedirs(export_path, exist_ok=True)
        df.to_csv(os.path.join(export_path, filename), index=False)
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

    return

_GOALIE_STATS_ENDPOINTS = ['summary', 'savesByStrength']

def _fetch_nhl_goalie_report(report, season_id):
    url = (
        f'https://api.nhle.com/stats/rest/en/goalie/{report}'
        f'?isAggregate=true&isGame=false&start=0&limit=-1'
        f'&cayenneExp=gameTypeId=2%20and%20seasonId%3C={season_id}%20and%20seasonId%3E={season_id}'
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json().get('data', []))

def _fetch_goalie_team_map(season_id):
    url = (
        f'https://api.nhle.com/stats/rest/en/goalie/summary'
        f'?isAggregate=false&isGame=false&start=0&limit=-1'
        f'&cayenneExp=gameTypeId=2%20and%20seasonId%3C={season_id}%20and%20seasonId%3E={season_id}'
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    rows = response.json().get('data', [])
    if not rows:
        return pd.DataFrame(columns=['playerId', 'teamAbbrevs'])
    df = pd.DataFrame(rows)[['playerId', 'teamAbbrevs']]
    return df.groupby('playerId', as_index=False)['teamAbbrevs'].agg(lambda s: ','.join(s.astype(str).unique()))

def _join_goalie_reports(season_id):
    base = _fetch_nhl_goalie_report('summary', season_id)
    if base.empty:
        return base
    combined = base
    for report in _GOALIE_STATS_ENDPOINTS[1:]:
        df = _fetch_nhl_goalie_report(report, season_id)
        if df.empty:
            continue
        overlap = [c for c in df.columns if c in combined.columns and c != 'playerId']
        df = df.drop(columns=overlap)
        combined = combined.merge(df, on='playerId', how='outer')
    team_map = _fetch_goalie_team_map(season_id)
    if not team_map.empty:
        combined = combined.merge(team_map, on='playerId', how='left')
    return combined

# Scrape per-season goalie stats from NHL API
def scrape_goalie_data(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year + 1):
        filename = f'{year-1}-{year}_goalie_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Goaltending Data', filename)

        if check_preexistence and os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        is_preseason_pull = (projection_year == year and season_state == 'PRESEASON')
        fetch_year = year - 1 if is_preseason_pull else year
        season_id = (fetch_year - 1) * 10000 + fetch_year

        df = _join_goalie_reports(season_id)

        if is_preseason_pull and not df.empty:
            preserve = {'playerId', 'goalieFullName', 'lastName', 'positionCode', 'catchesGlove', 'teamAbbrevs', 'seasonId'}
            for col in df.columns:
                if col not in preserve and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = 0

        if not df.empty and 'playerId' in df.columns and 'goalieFullName' in df.columns:
            if 'positionCode' not in df.columns:
                df['positionCode'] = 'G'
            priority = [c for c in ['playerId', 'goalieFullName', 'teamAbbrevs', 'positionCode', 'gamesPlayed', 'savePct', 'wins', 'losses', 'otLosses'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in set(priority)]
            df = df[priority + other_cols]
            sort_keys = [c for c in ['wins', 'playerId'] if c in df.columns]
            sort_asc = [c == 'playerId' for c in sort_keys]
            df = df.sort_values(by=sort_keys, ascending=sort_asc).reset_index(drop=True)

        if verbose:
            print(df)

        export_path = os.path.dirname(file_path)
        os.makedirs(export_path, exist_ok=True)
        df.to_csv(os.path.join(export_path, filename), index=False)
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

    return

# Scrape per-season goalie bios from NHL API
def scrape_goalie_bios(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year + 1):
        filename = f'{year-1}-{year}_goalie_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Goaltenders', 'Historical Goaltender Bios', filename)

        if check_preexistence and os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        is_preseason_pull = (projection_year == year and season_state == 'PRESEASON')
        fetch_year = year - 1 if is_preseason_pull else year
        season_id = (fetch_year - 1) * 10000 + fetch_year

        df = _fetch_nhl_goalie_report('bios', season_id)

        if not df.empty:
            team_map = _fetch_goalie_team_map(season_id)
            if not team_map.empty:
                df = df.merge(team_map, on='playerId', how='left')
            if 'playerId' in df.columns and 'goalieFullName' in df.columns:
                if 'positionCode' not in df.columns:
                    df['positionCode'] = 'G'
                priority = [c for c in ['playerId', 'goalieFullName', 'teamAbbrevs', 'positionCode', 'gamesPlayed', 'wins', 'losses', 'otLosses'] if c in df.columns]
                other_cols = [c for c in df.columns if c not in set(priority)]
                df = df[priority + other_cols]
                sort_keys = [c for c in ['wins', 'playerId'] if c in df.columns]
                sort_asc = [c == 'playerId' for c in sort_keys]
                df = df.sort_values(by=sort_keys, ascending=sort_asc).reset_index(drop=True)

        if verbose:
            print(df)

        export_path = os.path.dirname(file_path)
        os.makedirs(export_path, exist_ok=True)
        df.to_csv(os.path.join(export_path, filename), index=False)
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

    return


# Scrape per-season team stats from NHL API
def scrape_team_data(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year + 1):
        filename = f'{year-1}-{year}_team_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Team Data', filename)

        if check_preexistence and os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        is_preseason_pull = (projection_year == year and season_state == 'PRESEASON')
        fetch_year = year - 1 if is_preseason_pull else year
        season_id = (fetch_year - 1) * 10000 + fetch_year

        url = (
            f'https://api.nhle.com/stats/rest/en/team/summary'
            f'?isAggregate=false&isGame=false&start=0&limit=-1'
            f'&cayenneExp=gameTypeId=2%20and%20seasonId%3C={season_id}%20and%20seasonId%3E={season_id}'
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.DataFrame(response.json().get('data', []))

        if is_preseason_pull and not df.empty:
            preserve = {'teamId', 'teamFullName', 'seasonId'}
            for col in df.columns:
                if col not in preserve and pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = 0

        if not df.empty and 'teamId' in df.columns and 'teamFullName' in df.columns:
            priority = [c for c in ['teamId', 'teamFullName', 'gamesPlayed', 'wins', 'losses', 'otLosses', 'points', 'pointPct', 'goalsFor', 'goalsAgainst', 'shotsForPerGame', 'shotsAgainstPerGame', 'powerPlayPct', 'penaltyKillPct'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in set(priority)]
            df = df[priority + other_cols]
            sort_keys = [c for c in ['points', 'winsInRegulation', 'regulationAndOtWins', 'teamId'] if c in df.columns]
            sort_asc = [c == 'teamId' for c in sort_keys]
            df = df.sort_values(by=sort_keys, ascending=sort_asc).reset_index(drop=True)

        if verbose:
            print(df)

        export_path = os.path.dirname(file_path)
        os.makedirs(export_path, exist_ok=True)
        df.to_csv(os.path.join(export_path, filename), index=False)
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

    return


# Function to scrape raw historical data from Natural Stat Trick
def scrape_historical_team_data(start_year, end_year, projection_year, season_state, check_preexistence, verbose):
    for year in range(start_year, end_year+1):
        filename = f'{year-1}-{year}_team_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Team Data', filename)

        if check_preexistence == True:
            if os.path.exists(file_path):
                if verbose:
                    print(f'{filename} already exists in the following directory: {file_path}')
                continue

        if projection_year != year or season_state != 'PRESEASON':
            url = f'https://www.naturalstattrick.com/teamtable.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&rate=n&team=all&loc=B&gpf=410&fd=&td='
            df = pd.read_html(io.StringIO(requests.get(url, headers=NST_HEADERS).text))[0]
            df = df.iloc[:, 1:]
        else:
            response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
            data = response.json()
            df = pd.DataFrame(data['data'])
            team_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'team_metadata.csv'))
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

    return

# Function to aggregate historical player bios for all players in the engine database
def aggregate_player_bios(skaters, check_preexistence, verbose):
    if skaters:
        filename = 'skater_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Skaters', filename)
        bios_dir = os.path.join(os.path.dirname(file_path), 'Historical Skater Bios')
        name_col = 'skaterFullName'
    else:
        filename = 'goalie_bios.csv'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Player Bios', 'Goaltenders', filename)
        bios_dir = os.path.join(os.path.dirname(file_path), 'Historical Goaltender Bios')
        name_col = 'goalieFullName'

    if check_preexistence and os.path.exists(file_path):
        if verbose:
            print(f'{filename} already exists in the following directory: {file_path}')
        return

    files = sorted(f for f in os.listdir(bios_dir) if f.endswith('.csv'))
    dataframes = []
    for file in files:
        df = pd.read_csv(os.path.join(bios_dir, file))
        if 'playerId' not in df.columns:
            continue
        df['Last Season'] = file[:9]
        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)

    if skaters:
        stat_cols = ['gamesPlayed', 'goals', 'assists', 'points']
        epoch_cols = ['epochGP', 'epochGoals', 'epochAssists', 'epochPoints']
    else:
        stat_cols = ['gamesPlayed', 'wins', 'losses', 'ties']
        epoch_cols = ['epochGP', 'epochWins', 'epochLosses', 'epochTies']

    epoch_df = combined_df.groupby('playerId', as_index=False)[stat_cols].sum()
    epoch_df.rename(columns=dict(zip(stat_cols, epoch_cols)), inplace=True)

    combined_df.sort_values(by=['birthDate', 'Last Season'], na_position='first', inplace=True)
    combined_df.drop_duplicates(subset='playerId', keep='last', inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    insert_pos = list(combined_df.columns).index(stat_cols[0])
    combined_df.drop(columns=stat_cols, inplace=True)
    combined_df = combined_df.merge(epoch_df, on='playerId', how='left')
    non_epoch = [c for c in combined_df.columns if c not in epoch_cols]
    combined_df = combined_df[non_epoch[:insert_pos] + epoch_cols + non_epoch[insert_pos:]]

    if skaters:
        combined_df.sort_values(by=['epochPoints', 'epochGoals', 'epochAssists', 'playerId'], ascending=[False, False, False, True], inplace=True)
    else:
        combined_df.sort_values(by=['epochWins', 'playerId'], ascending=[False, True], inplace=True)
    combined_df = combined_df.reset_index(drop=True)

    export_path = os.path.dirname(file_path)
    os.makedirs(export_path, exist_ok=True)
    combined_df.to_csv(os.path.join(export_path, filename), index=False)
    if verbose:
        print(f'{filename} has been downloaded to the following directory: {export_path}')

    return combined_df

# Scrape team data from NHL API
def scrape_teams(projection_year, check_preexistence, verbose):

    filename = f'team_metadata.csv'
    file_path = os.path.dirname(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', filename))

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

        # Fetch division data from standings API
        standings_response = requests.get('https://api-web.nhle.com/v1/standings/now')
        standings_data = standings_response.json()
        division_map = {
            entry['teamName']['default']: entry['divisionName']
            for entry in standings_data['standings']
        }

        # Construct teams df
        df = pd.DataFrame(teams_data['data'])[['id', 'fullName', 'triCode']]
        df.columns = ['TeamID', 'Team Name', 'Abbreviation']
        df['Active'] = df['TeamID'].apply(lambda x: x in distinct_team_ids)
        df['Division'] = df['Team Name'].map(division_map)
        df = df.sort_values('TeamID').reset_index(drop=True)

        if verbose:
            print(df)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, filename))

# Scrape schedule data from NHL API
def scrape_games(projection_year, check_preexistence, verbose):

    filename = f'{projection_year-1}-{projection_year}_game_schedule.csv'
    file_path = os.path.dirname(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', filename))

    if check_preexistence == True:
        if os.path.isfile(file_path):
            return
    else:
        response = requests.get(f'https://api.nhle.com/stats/rest/en/game?cayenneExp=season={projection_year-1}{projection_year}')
        data = response.json()

        df = pd.DataFrame(data['data'])
        # join with team data to get team names
        team_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'team_metadata.csv'))
        df = df.merge(team_data[['TeamID', 'Team Name']], left_on='homeTeamId', right_on='TeamID', how='left')
        df = df.merge(team_data[['TeamID', 'Team Name']], left_on='visitingTeamId', right_on='TeamID', how='left')

        df = df[df['gameType'] == 2][['id', 'easternStartTime', 'gameNumber', 'gameStateId', 'period', 'homeTeamId', 'visitingTeamId', 'Team Name_x', 'Team Name_y', 'homeScore', 'visitingScore']]
        df.columns = ['GameID', 'Time (EST)', 'Game Number', 'Game State', 'Period', 'Home Id', 'Visiting Id', 'Home Team', 'Visiting Team', 'Home Score', 'Visiting Score']
        df['Period'] = df['Period'].fillna(0).astype(int)
        df = df.sort_values(by='GameID')
        df = df.reset_index(drop=True)

        if verbose:
            print(df)

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, filename))

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

def fetch_current_team_stats_from_nhl_api(verbose=False):
    url = "https://api.nhle.com/stats/rest/en/team/summary?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22wins%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22teamId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=50&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20252026%20and%20seasonId%3E=20252026"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        teams_data = data.get('data', [])
        if not teams_data:
            if verbose:
                print("No current team stats data returned from NHL API.")
            return pd.DataFrame()
        
        current_stats = []
        for team in teams_data:
            current_stats.append({
                'team': team.get('teamFullName'),
                'current_wins': team.get('wins'),
                'current_losses': team.get('losses'),
                'current_otl': team.get('otLosses'),
                'current_goals_for': team.get('goalsFor'),
                'current_goals_against': team.get('goalsAgainst'),
                'current_pp_pct': team.get('powerPlayPct'),
                'current_pk_pct': team.get('penaltyKillPct'),
                'current_points': team.get('points'),
            })
        
        df = pd.DataFrame(current_stats)
        return df
    
    except Exception as e:
        if verbose:
            print(f"Error fetching current team stats from NHL API: {e}")
        return pd.DataFrame()

def update_metadata(state, params):

    metadata_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    if state == 0:
        [start_time, projection_year, simulations] = params
        metadata = {
            'startTimestamp': start_time,
            'endTimestamp': None,
            'engineRunTime': None,
            'projectionYear': projection_year,
            'monteCarloSimulations': simulations
        }

    elif state == 1:
        [end_time, engine_run_time] = params
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['endTimestamp'] = end_time
        metadata['engineRunTime'] = engine_run_time

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    except IOError as e:
        print(f"Error updating metadata: {e}")

    return

_TABLE_DELETE_CONDITIONS = {
    'team_projections':   ('points',   'gt', -1),
    'player_projections': ('points',   'gt', -1),
    'game_projections':   ('game_id',  'gt', -1),
}

_TABLE_REQUIRED_COLUMNS = {
    'team_projections':   ['abbrev', 'team', 'points'],
    'player_projections': ['player_id', 'player', 'points'],
    'game_projections':   ['game_id', 'datetime'],
    'site_config':        ['id', 'datetime'],
}

def push_to_supabase(table_name, year, verbose=False):
    if table_name not in _TABLE_REQUIRED_COLUMNS:
        raise ValueError(f"Unknown table '{table_name}'. Must be one of: {list(_TABLE_REQUIRED_COLUMNS)}")
    load_dotenv()
    SUPABASE_URL = os.getenv('REACT_APP_SUPABASE_PROJ_URL')
    SUPABASE_KEY = os.getenv('REACT_APP_SUPABASE_ANON_KEY')
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    try:
        session = supabase.auth.sign_in_with_password({"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})
    except:
        supabase.auth.sign_up(credentials={"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})
        session = supabase.auth.sign_in_with_password({"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})

    if table_name == 'team_projections':
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(year), 'Teams', f'{year}_team_projections.csv')
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
            'Presidents': 'presidents_trophy_prob',
            'Playoffs': 'playoff_prob',
            'Pts_90PI_low': 'pts_90pi_low',
            'Pts_90PI_high': 'pts_90pi_high',
            'P_60Pts': 'p_60pts',
            'P_70Pts': 'p_70pts',
            'P_80Pts': 'p_80pts',
            'P_90Pts': 'p_90pts',
            'P_100Pts': 'p_100pts',
            'P_110Pts': 'p_110pts',
            'P_120Pts': 'p_120pts'
        }
        df.rename(columns=rename_dict, inplace=True)
        df['logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['abbrev'] + '_dark.svg'
        df['stanley_cup_prob'] = 0.03125
        
        team_colors_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'nhl_team_colors.csv')
        team_colors_df = pd.read_csv(team_colors_path)
        df = df.merge(team_colors_df[['abbrev', 'primary_color']], on='abbrev', how='left')
        
        current_team_stats_df = fetch_current_team_stats_from_nhl_api(verbose=verbose)
        df = df.merge(current_team_stats_df, on='team', how='left')

        df['gf%'] = (df['goals_for'] / (df['goals_for'] + df['goals_against']) * 100)
        df['offense_score'] = ((df['goals_for'] - df['goals_for'].min()) / (df['goals_for'].max() - df['goals_for'].min()) * 100).round(0).astype(int)
        df['defense_score'] = ((df['goals_against'].max() - df['goals_against']) / (df['goals_against'].max() - df['goals_against'].min()) * 100).round(0).astype(int)
        df['overall_score'] = (((df['gf%'] - df['gf%'].min()) / (df['gf%'].max() - df['gf%'].min())) * 100).round(0).astype(int)
        df = df.drop(columns=['gf%'])

    elif table_name == 'player_projections':
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(year), 'Skaters', f'{year}_skater_projections.csv')
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
            'ArtRoss': 'art_ross',
            'Rocket': 'rocket',
            'Goals_90PI_low': 'goals_90pi_low',
            'Goals_90PI_high': 'goals_90pi_high',
            'Assists_90PI_low': 'assists_90pi_low',
            'Assists_90PI_high': 'assists_90pi_high',
            'Points_90PI_low': 'points_90pi_low',
            'Points_90PI_high': 'points_90pi_high',
            'P_10G': 'p_10g',
            'P_20G': 'p_20g',
            'P_30G': 'p_30g',
            'P_40G': 'p_40g',
            'P_50G': 'p_50g',
            'P_60G': 'p_60g',
            'P_25A': 'p_25a',
            'P_50A': 'p_50a',
            'P_75A': 'p_75a',
            'P_100A': 'p_100a',
            'P_50P': 'p_50p',
            'P_75P': 'p_75p',
            'P_100P': 'p_100p',
            'P_125P': 'p_125p',
            'P_150P': 'p_150p'
        }
        df.rename(columns=rename_dict, inplace=True)
        df['position'] = df['position'].apply(lambda x: 'RW' if x == 'R' else ('LW' if x == 'L' else x))
        df['logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['team'] + '_dark.svg'

        # fill in missing values
        df['goals_90pi_low'] = df['goals_90pi_low'].fillna(0)
        df['goals_90pi_high'] = df['goals_90pi_high'].fillna(0)
        df['assists_90pi_low'] = df['assists_90pi_low'].fillna(0)
        df['assists_90pi_high'] = df['assists_90pi_high'].fillna(0)
        df['points_90pi_low'] = df['points_90pi_low'].fillna(0)
        df['points_90pi_high'] = df['points_90pi_high'].fillna(0)

        # merge in team names
        team_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'team_metadata.csv'))
        team_data = team_data[['Abbreviation', 'Team Name']]
        team_data.rename(columns={'Abbreviation': 'team', 'Team Name': 'team_name'}, inplace=True)
        df = df.merge(team_data, on='team', how='left')


        df['atoi'] = (df['TOI'] / df['games']).fillna(0)
        df['goals_per60'] = ((df['goals'] / df['TOI']) * 60).fillna(0)
        df['assists_per60'] = ((df['assists'] / df['TOI']) * 60).fillna(0)
        df = df.drop(columns=['TOI'])
        df = df.dropna(subset=['logo'])
        df = df.drop_duplicates(subset=['player_id', 'team', 'position'])

    elif table_name == 'game_projections':
        file_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(year), 'Games', f'{year}_game_projections.csv')
        df = pd.read_csv(file_path)
        df = df.drop(df.columns[0], axis=1)
        rename_dict = {
            'GameID': 'game_id',
            'DatetimeEST': 'datetime',
            'Date': 'date',
            'TimeEST': 'time',
            'Home Team': 'home_name',
            'Home Abbreviation': 'home_abbrev',
            'Home Score': 'home_score',
            'Home Win': 'home_prob',
            'Visiting Team': 'visitor_name',
            'Visiting Abbreviation': 'visitor_abbrev',
            'Visiting Score': 'visitor_score',
            'Visitor Win': 'visitor_prob',
            'Overtime': 'overtime_prob',
        }

        df.rename(columns=rename_dict, inplace=True)
        df['home_prob'] = df['home_prob'].apply(lambda x: 1.0 if x == 'True' else 0.0 if x == 'False' else x)
        df['visitor_prob'] = df['visitor_prob'].apply(lambda x: 1.0 if x == 'True' else 0.0 if x == 'False' else x)
        df['overtime_prob'] = df['overtime_prob'].apply(lambda x: 1.0 if x == 'True' else 0.0 if x == 'False' else x)
        df['time_str'] = pd.to_datetime(df['time'].astype(str), format='%H:%M:%S').dt.strftime('%I:%M %p').astype(str)
        df['time_str'] = df['time_str'].apply(lambda x: x[1:] if x.startswith('0') else x)
        df['home_logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['home_abbrev'] + '_dark.svg'
        df['visitor_logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['visitor_abbrev'] + '_dark.svg'

        # add team records and ranks from NHL API
        try:
            standings_response = requests.get('https://api-web.nhle.com/v1/standings/now')
            standings_data = standings_response.json()
            
            standings_list = []
            for team in standings_data['standings']:
                abbrev = team['teamAbbrev']['default']
                wins = team['wins']
                losses = team['losses']
                ot_losses = team['otLosses']
                rank = team['leagueSequence']
                
                if rank == 1:
                    rank_suffix = 'st'
                elif rank == 2:
                    rank_suffix = 'nd'
                elif rank == 3:
                    rank_suffix = 'rd'
                else:
                    rank_suffix = 'th'
                
                standings_list.append({
                    'abbrev': abbrev,
                    'record': f'{wins}-{losses}-{ot_losses}',
                    'rank': f'{rank}{rank_suffix}'
                })
            
            standings_df = pd.DataFrame(standings_list)
            
            df = df.merge(standings_df[['abbrev', 'record', 'rank']], left_on='home_abbrev', right_on='abbrev', how='left')
            df = df.rename(columns={'record': 'home_record', 'rank': 'home_rank'})
            df = df.drop(columns=['abbrev'])
            df = df.merge(standings_df[['abbrev', 'record', 'rank']], left_on='visitor_abbrev', right_on='abbrev', how='left')
            df = df.rename(columns={'record': 'visitor_record', 'rank': 'visitor_rank'})
            df = df.drop(columns=['abbrev'])
            
            if verbose:
                print(f"Updated standings from NHL API for {len(standings_list)} teams")
                
        except Exception as e:
            if verbose:
                print(f"Failed to fetch standings from NHL API, using fallback: {e}")
            
            # fallback to default values if API fails
            df['home_record'] = '0-0-0'
            df['home_rank'] = '1st'
            df['visitor_record'] = '0-0-0'
            df['visitor_rank'] = '1st'

    elif table_name == 'site_config':
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        end_timestamp = metadata['endTimestamp']
        end_datetime = datetime.fromtimestamp(end_timestamp)
        df = pd.DataFrame([{'id': 1, 'datetime': end_datetime.isoformat()}])

    if df.empty:
        raise ValueError(f"DataFrame for '{table_name}' is empty; aborting push.")
    required = _TABLE_REQUIRED_COLUMNS[table_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame for '{table_name}' missing columns: {missing}")

    data_to_insert = df.to_dict(orient='records')

    if verbose:
        print(df)
        print(data_to_insert)

    if table_name == 'site_config':
        try:
            upsert_response = supabase.table(table_name).upsert(data_to_insert).execute()
            print(f"Successfully upserted {len(data_to_insert)} records into '{table_name}'.")
            return None, upsert_response
        finally:
            supabase.auth.sign_out()

    col, op, val = _TABLE_DELETE_CONDITIONS[table_name]

    delete_response = None
    insert_response = None
    try:
        backup = supabase.table(table_name).select('*').execute().data
        delete_response = getattr(supabase.table(table_name).delete(), op)(col, val).execute()
        try:
            insert_response = supabase.table(table_name).insert(data_to_insert).execute()
            print(f"Successfully inserted {len(data_to_insert)} records into '{table_name}'.")
        except Exception as insert_err:
            print(f"Insert failed for '{table_name}': {insert_err}. Attempting rollback...")
            if backup:
                try:
                    batch_size = 500
                    for i in range(0, len(backup), batch_size):
                        supabase.table(table_name).insert(backup[i:i + batch_size]).execute()
                    print(f"Rollback successful: restored {len(backup)} rows to '{table_name}'.")
                except Exception as rollback_err:
                    raise RuntimeError(
                        f"Insert failed AND rollback failed for '{table_name}'. "
                        f"Insert error: {insert_err}. Rollback error: {rollback_err}. "
                        f"Table may be empty."
                    ) from rollback_err
            raise RuntimeError(
                f"Insert failed for '{table_name}': {insert_err}. Rollback successful."
            ) from insert_err
    finally:
        supabase.auth.sign_out()

    return delete_response, insert_response