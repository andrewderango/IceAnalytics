# Could use https://github.com/Zmalski/NHL-API-Reference/issues/32

# or this: https://api-web.nhle.com/v1/player/8478402/landing

### CHANGE IT SUCH THAT IT REQUESTS URL FROM ISSUE 32, CREATES DATA STRUCTURE, UPDATES DF FROM THAT DATA STRUCTURE (only 1 call)

import requests

def fix_teams(df):
    for index, row in df.iterrows():
        player_id = row['PlayerID']  # Assuming the column name that stores player IDs is 'Player_ID'
        url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
            data = response.json()
            
            # Assuming the JSON structure contains a 'team' key at some level (adjust as needed)
            team_name = data['currentTeamAbbrev']  # Adjust the key path according to the actual JSON structure
            print(row['Player'], team_name)
            df.at[index, 'Team_2'] = team_name  # Update the 'Team_2' column with the obtained team name
            
        except requests.RequestException as e:
            print(f"Error fetching data for player ID {player_id}: {e}")
    
    print(df)
    return df