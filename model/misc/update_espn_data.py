import os
import requests
import unidecode
import pandas as pd

def pull_espn_data(limit, download_files, verbose):
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

def replace_names(name):
    if not isinstance(name, str):
        name = str(name)
    
    name_replacement_dict = {
        'Jr.': 'Jr',
    }

    for original, replacement in name_replacement_dict.items():
        name = name.replace(original, replacement)
    return name

def update_player_bios(espn_df):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fetch_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'skater_bios.csv')
    skater_bios_df = pd.read_csv(fetch_path)
    skater_bios_df = skater_bios_df.drop(skater_bios_df.columns[0], axis=1)

    skater_bios_df['playerStrippedEspn'] = skater_bios_df['Player'].str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names)
    espn_df['playerStrippedEspn'] = espn_df['Player'].str.replace(' ', '').str.replace('.', '').str.replace('-', '').str.replace('\'', '').str.lower().apply(replace_names)

    # merge
    merged_df = pd.merge(skater_bios_df, espn_df, how='outer', on='playerStrippedEspn')
    print(merged_df)
    save_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'test.csv')
    merged_df.to_csv(save_path, index=True)
    quit()

def main():
    espn_df = pull_espn_data(limit=1000, download_files=True, verbose=True)
    update_player_bios(espn_df)

if __name__ == '__main__':
    main()