import time
import pandas as pd
from model_training import *
from model_inference import *
from scraper_functions import *

def main():
    start_time = time.time()
    projection_year = 2024

    # Scrape or fetch player data
    scrape_historical_player_data(2008, 2024, True, False, True, False)
    scrape_historical_player_data(2008, 2024, False, False, True, False)
    scrape_historical_player_data(2008, 2024, True, True, True, False)
    scrape_historical_player_data(2008, 2024, False, True, True, False)
    scrape_nhlapi_data(2008, 2024, False, True, False)
    scrape_nhlapi_data(2008, 2024, True, True, False)
    aggregate_player_bios(True, True, False)
    aggregate_player_bios(False, True, False)

    # Scrape or fetch team data
    scrape_historical_team_data(2008, 2024, True, False)
    scrape_teams(True, False)
    scrape_games(projection_year, True, False)

    # Train models
    atoi_model_data = train_atoi_model(projection_year, False, False)
    goal_model = train_goal_model(projection_year, False, False)
    a1_model = train_a1_model(projection_year, False, False)
    a2_model = train_a2_model(projection_year, False, False)
    ga_model = train_ga_model(projection_year, False, False)

    # Make player inferences
    player_stat_df = pd.DataFrame()
    player_stat_df = atoi_model_inference(projection_year, player_stat_df, atoi_model_data, True, False)
    player_stat_df = goal_model_inference(projection_year, player_stat_df, goal_model, True, False)
    player_stat_df = a1_model_inference(projection_year, player_stat_df, a1_model, True, False)
    player_stat_df = a2_model_inference(projection_year, player_stat_df, a2_model, True, False)

    # Make team inferences
    team_stat_df = pd.DataFrame()
    team_stat_df = ga_model_inference(projection_year, team_stat_df, ga_model, True, False)

    # Simulate seasons
    simulate_season(projection_year, 100, True, True, True)

    print(f"Runtime: {time.time()-start_time:.3f} seconds")

# temporary main function for updating data pipeline and projections
def main_updater():
    start_time = time.time()
    projection_year = 2024

    # Scrape or fetch player data
    scrape_historical_player_data(2008, 2023, True, False, True, False)
    scrape_historical_player_data(2008, 2023, False, False, True, False)
    scrape_historical_player_data(2008, 2023, True, True, True, False)
    scrape_historical_player_data(2008, 2023, False, True, True, False)
    scrape_historical_player_data(2024, 2024, True, False, False, True)
    scrape_historical_player_data(2024, 2024, False, False, False, True)
    scrape_historical_player_data(2024, 2024, True, True, False, True)
    scrape_historical_player_data(2024, 2024, False, True, False, True)
    scrape_nhlapi_data(2008, 2023, False, True, False)
    scrape_nhlapi_data(2008, 2023, True, True, False)
    scrape_nhlapi_data(2024, 2024, False, False, True)
    scrape_nhlapi_data(2024, 2024, True, False, True)
    aggregate_player_bios(True, True, False)
    aggregate_player_bios(False, True, False)

    # Scrape or fetch team data
    scrape_historical_team_data(2008, 2024, True, False)
    scrape_historical_team_data(2024, 2024, False, True)
    scrape_teams(True, False)
    scrape_games(projection_year, False, True)

    # Train models
    atoi_model_data = train_atoi_model(projection_year, False, False)
    goal_model = train_goal_model(projection_year, False, False)
    a1_model = train_a1_model(projection_year, False, False)
    a2_model = train_a2_model(projection_year, False, False)
    ga_model = train_ga_model(projection_year, False, False)

    # Make player inferences
    player_stat_df = pd.DataFrame()
    player_stat_df = atoi_model_inference(projection_year, player_stat_df, atoi_model_data, True, False)
    player_stat_df = goal_model_inference(projection_year, player_stat_df, goal_model, True, False)
    player_stat_df = a1_model_inference(projection_year, player_stat_df, a1_model, True, False)
    player_stat_df = a2_model_inference(projection_year, player_stat_df, a2_model, True, False)
    # player_stat_df = player_stat_df.sort_values(by='Gper1kChunk', ascending=False)
    # print(player_stat_df)

    # Make team inferences
    team_stat_df = pd.DataFrame()
    team_stat_df = ga_model_inference(projection_year, team_stat_df, ga_model, True, False)
    # team_stat_df = team_stat_df.sort_values(by='GA/GP', ascending=False)
    # print(team_stat_df)

    # Simulate season
    simulate_season(projection_year, 10, True, True, True)

    print(f"Runtime: {time.time()-start_time:.3f} seconds")

if __name__ == "__main__":
    main()