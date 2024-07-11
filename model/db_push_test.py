import os
import time
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Initialize Supabase client
load_dotenv()
SUPABASE_URL = os.getenv('REACT_APP_SUPABASE_PROJ_URL')
SUPABASE_KEY = os.getenv('REACT_APP_SUPABASE_ANON_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    session = supabase.auth.sign_in_with_password({"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})
except:
    supabase.auth.sign_up(credentials={"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})
    session = supabase.auth.sign_in_with_password({"email": os.getenv('SUPABASE_EMAIL'), "password": os.getenv('SUPABASE_PASSWORD')})

def read_csv_and_push_to_db(csv_file_path: str, table_name: str):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
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

    # Convert DataFrame to list of dictionaries
    data_to_insert = df.to_dict(orient='records')

    print(data_to_insert)
    
    # Insert data into the database
    try:
        delete_response = supabase.table(table_name).delete().gt('points', 0).execute()
        insert_response = supabase.table(table_name).insert(data_to_insert).execute()
        print(f"Successfully inserted {len(data_to_insert)} records into '{table_name}' table.")
    except Exception as e:
        print(f"An error occurred: {e}")

csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Teams', '2025_team_aggregated_projections.csv')
table_name = "team-projections"
read_csv_and_push_to_db(csv_file_path, table_name)

supabase.auth.sign_out()