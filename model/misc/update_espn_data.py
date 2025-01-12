import os
import requests
import unidecode
import pandas as pd

def pull_espn_data(update_scrape, limit, download_files, verbose):
    
    if update_scrape:
        url = "https://site.web.api.espn.com/apis/common/v3/sports/hockey/nhl/statistics/byathlete"

        params = {
            "region": "us",
            "lang": "en",
            "contentorigin": "espn",
            "isqualified": "false",
            "page": 1,
            "limit": limit,
            "sort": "offensive:points:desc",
            "category": "skaters"
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()

            player_data = []
            for athlete in data.get("athletes", []):
                athlete_info = {
                    "Player": athlete.get('athlete', {}).get('displayName'),
                    "EspnId": athlete.get('athlete', {}).get('id'),
                    "EspnHeadshot": athlete.get('athlete', {}).get('headshot', {}).get('href'),
                }
                player_data.append(athlete_info)

            espn_df = pd.DataFrame(player_data)
            if verbose:
                print(espn_df)

            if download_files:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'espn_data.csv')
                espn_df.to_csv(save_path, index=True)

            return espn_df
        else:
            print(f"Failed to fetch data: {response.status_code}")

    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fetch_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'espn_data.csv')
        espn_df = pd.read_csv(fetch_path)
        espn_df = espn_df.drop(espn_df.columns[0], axis=1)
        return espn_df

def replace_names_espn(name):
    if not isinstance(name, str):
        name = str(name)
    
    name_replacement_dict = {
        'è': 'e',
        'é': 'e',
        'ü': 'u',
        'ö': 'o',
        'ä': 'a',
        'mathew': 'matt',
        'alexander': 'alex',
        'patrick': 'pat',
        'janis': 'jj',
        'johnny': 'john',
        'nate': 'nathan',
    }

    for original, replacement in name_replacement_dict.items():
        name = name.replace(original, replacement)
    return name

def add_espn_to_player_bios(espn_df, download_files, verbose):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fetch_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv')
    skater_bios_df = pd.read_csv(fetch_path)
    skater_bios_df = skater_bios_df.drop(skater_bios_df.columns[0], axis=1)

    # strip and lowercase names for comparison
    skater_bios_df['playerStrippedEspn'] = skater_bios_df['Player'].str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names_espn)
    espn_df['playerStrippedEspn'] = espn_df['Player'].str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names_espn)

    # merge to update headshots for existing players
    merged_df = pd.merge(skater_bios_df, espn_df[['playerStrippedEspn', 'EspnHeadshot']], how='outer', on='playerStrippedEspn')
    if verbose:
        print('Updated Player Bios:')
        print(merged_df)
    if download_files:
        save_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv')
        merged_df.to_csv(save_path, index=True)

    # check for failed merges
    failed_df = pd.merge(skater_bios_df, espn_df[['playerStrippedEspn', 'EspnHeadshot']], how='right', on='playerStrippedEspn')
    failed_df = failed_df[failed_df['PlayerID'].isnull()]
    if verbose:
        print('Failed Merges:')
        print(failed_df)
    if download_files:
        save_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'espn_failed_merge.csv')
        failed_df.to_csv(save_path, index=True)

def main():
    espn_df = pull_espn_data(update_scrape=False, limit=1000, download_files=True, verbose=True)
    add_espn_to_player_bios(espn_df=espn_df, download_files=True, verbose=True)

if __name__ == '__main__':
    main()