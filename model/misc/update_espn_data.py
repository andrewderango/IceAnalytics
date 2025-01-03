import os
import requests
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
                "Name": athlete.get('athlete', {}).get('displayName'),
                "Id": athlete.get('athlete', {}).get('id'),
                "Headshot": athlete.get('athlete', {}).get('headshot', {}).get('href'),
            }
            player_data.append(athlete_info)

        df = pd.DataFrame(player_data)
        if verbose:
            print(df)

        if download_files:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(current_dir, '..', '..', 'engine_data', 'Player Bios', 'Skaters', 'espn_data.csv')
            df.to_csv(save_path, index=True)

        return df
    else:
        print(f"Failed to fetch data: {response.status_code}")

def main():
    df = pull_espn_data(limit=1000, download_files=True, verbose=True)

if __name__ == '__main__':
    main()