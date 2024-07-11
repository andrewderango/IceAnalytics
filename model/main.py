import time
import pandas as pd
from model_training import *
from model_inference import *
from scraper_functions import *

def main():
    start_time = time.time()
    PROJECTION_YEAR = 2025
    season_state = get_season_state(PROJECTION_YEAR)

    # # Scrape or fetch player data
    # scrape_historical_player_data(2008, 2024, True, False, PROJECTION_YEAR, season_state, True, False)
    # scrape_historical_player_data(2008, 2024, False, False, PROJECTION_YEAR, season_state, True, False)
    # scrape_historical_player_data(2008, 2024, True, True, PROJECTION_YEAR, season_state, True, False)
    # scrape_historical_player_data(2008, 2024, False, True, PROJECTION_YEAR, season_state, True, False)
    # scrape_historical_player_data(2025, 2025, True, False, PROJECTION_YEAR, season_state, False, True)
    # scrape_historical_player_data(2025, 2025, False, False, PROJECTION_YEAR, season_state, False, True)
    # scrape_historical_player_data(2025, 2025, True, True, PROJECTION_YEAR, season_state, False, True)
    # scrape_historical_player_data(2025, 2025, False, True, PROJECTION_YEAR, season_state, False, True)
    # scrape_nhlapi_data(2008, 2024, False, PROJECTION_YEAR, season_state, True, False)
    # scrape_nhlapi_data(2008, 2024, True, PROJECTION_YEAR, season_state, True, False)
    # scrape_nhlapi_data(2025, 2025, False, PROJECTION_YEAR, season_state, False, True)
    # scrape_nhlapi_data(2025, 2025, True, PROJECTION_YEAR, season_state, False, True)
    # aggregate_player_bios(True, False, False)
    # aggregate_player_bios(False, False, False)

    # # Scrape or fetch team data
    # scrape_historical_team_data(2008, 2024, PROJECTION_YEAR, season_state, True, False)
    # scrape_historical_team_data(2025, 2025, PROJECTION_YEAR, season_state, False, True)
    # scrape_teams(PROJECTION_YEAR, True, False)
    # scrape_games(PROJECTION_YEAR, False, True)

    # Train models
    atoi_model_data = train_atoi_model(PROJECTION_YEAR, False, False)
    goal_model = train_goal_model(PROJECTION_YEAR, False, False)
    a1_model = train_a1_model(PROJECTION_YEAR, False, False)
    a2_model = train_a2_model(PROJECTION_YEAR, False, False)
    ga_model = train_ga_model(PROJECTION_YEAR, False, False)

    # Make player inferences
    player_stat_df = pd.DataFrame()
    player_stat_df = atoi_model_inference(PROJECTION_YEAR, player_stat_df, atoi_model_data, True, False)
    player_stat_df = goal_model_inference(PROJECTION_YEAR, player_stat_df, goal_model, True, False)
    player_stat_df = a1_model_inference(PROJECTION_YEAR, player_stat_df, a1_model, True, False)
    player_stat_df = a2_model_inference(PROJECTION_YEAR, player_stat_df, a2_model, True, False)
    # player_stat_df = player_stat_df.sort_values(by='Gper1kChunk', ascending=False)
    player_stat_df['iGoals'] = player_stat_df['Gper1kChunk']/500 * player_stat_df['ATOI'] * 82
    player_stat_df['iPoints'] = (player_stat_df['Gper1kChunk']+player_stat_df['A1per1kChunk']+player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI'] * 82
    player_stat_df = player_stat_df.sort_values(by='iPoints', ascending=False)
    player_stat_df = player_stat_df.reset_index(drop=True)
    # print(player_stat_df.to_string())

    # Make team inferences
    team_stat_df = pd.DataFrame()
    team_stat_df = ga_model_inference(PROJECTION_YEAR, team_stat_df, ga_model, True, False)
    team_stat_df = team_stat_df.sort_values(by='GA/GP', ascending=False)
    # print(team_stat_df.to_string())

    # Simulate season
    simulate_season(PROJECTION_YEAR, 10, True, True, True)

    # push_to_supabase("team-projections", True)
    # push_to_supabase("player-projections", True)

    print(f"Runtime: {time.time()-start_time:.3f} seconds")

if __name__ == "__main__":
    main()