import os
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
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=False)
    combined_df.drop_duplicates(subset='Player', keep='last', inplace=True)
    combined_df = combined_df[combined_df['Date of Birth'] != '-']
    combined_df = combined_df.reset_index(drop=True)

    export_path = os.path.dirname(file_path)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    combined_df.to_csv(os.path.join(export_path, filename), index=True)
    if verbose:
        print(f'{filename} has been downloaded to the following directory: {export_path}')

    return combined_df