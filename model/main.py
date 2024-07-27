import time
import pandas as pd
from model_training import *
from model_inference import *
from scraper_functions import *
from sim_engine import *

def main():
    start_time = time.time()
    PROJECTION_YEAR = 2025
    season_state = get_season_state(PROJECTION_YEAR)

    # # Scrape or fetch player data
    # scrape_historical_player_data(start_year=2008, end_year=2024, skaters=True, bios=False, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_historical_player_data(start_year=2008, end_year=2024, skaters=True, bios=False, on_ice=True, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_historical_player_data(start_year=2008, end_year=2024, skaters=False, bios=False, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_historical_player_data(start_year=2008, end_year=2024, skaters=True, bios=True, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_historical_player_data(start_year=2008, end_year=2024, skaters=False, bios=True, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_historical_player_data(start_year=2025, end_year=2025, skaters=True, bios=False, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_historical_player_data(start_year=2025, end_year=2025, skaters=True, bios=False, on_ice=True, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_historical_player_data(start_year=2025, end_year=2025, skaters=False, bios=False, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_historical_player_data(start_year=2025, end_year=2025, skaters=True, bios=True, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_historical_player_data(start_year=2025, end_year=2025, skaters=False, bios=True, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_nhlapi_data(start_year=2008, end_year=2024, bios=False, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_nhlapi_data(start_year=2008, end_year=2024, bios=False, on_ice=True, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_nhlapi_data(start_year=2008, end_year=2024, bios=True, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_nhlapi_data(start_year=2025, end_year=2025, bios=False, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_nhlapi_data(start_year=2025, end_year=2025, bios=False, on_ice=True, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_nhlapi_data(start_year=2025, end_year=2025, bios=True, on_ice=False, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # aggregate_player_bios(skaters=True, check_preexistence=False, verbose=False)
    # aggregate_player_bios(skaters=False, check_preexistence=False, verbose=False)

    # # Scrape or fetch team data
    # scrape_historical_team_data(start_year=2008, end_year=2024, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=True, verbose=False)
    # scrape_historical_team_data(start_year=2025, end_year=2025, projection_year=PROJECTION_YEAR, season_state=season_state, check_preexistence=False, verbose=True)
    # scrape_teams(projection_year=PROJECTION_YEAR, check_preexistence=True, verbose=False)
    # scrape_games(projection_year=PROJECTION_YEAR, check_preexistence=False, verbose=True)

    # Train models
    atoi_model_data = train_atoi_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    goal_model = train_goal_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    a1_model = train_a1_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    a2_model = train_a2_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    team_ga_model = train_ga_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    skater_xga_model = train_skater_xga_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    skater_ga_model = train_skater_ga_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)

    # Make player inferences
    player_stat_df = pd.DataFrame()
    player_stat_df = atoi_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, atoi_model_data=atoi_model_data, download_file=True, verbose=False)
    player_stat_df = goal_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, goal_model=goal_model, download_file=True, verbose=False)
    player_stat_df = a1_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, a1_model=a1_model, download_file=True, verbose=False)
    player_stat_df = a2_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, a2_model=a2_model, download_file=True, verbose=False)
    player_stat_df = skater_xga_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, skater_xga_model=skater_xga_model, download_file=True, verbose=False)
    player_stat_df = skater_ga_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, skater_ga_model=skater_ga_model, download_file=True, verbose=False)
    player_stat_df['iGoals'] = player_stat_df['Gper1kChunk']/500 * player_stat_df['ATOI'] * 82
    player_stat_df['iPoints'] = (player_stat_df['Gper1kChunk']+player_stat_df['A1per1kChunk']+player_stat_df['A2per1kChunk'])/500 * player_stat_df['ATOI'] * 82
    player_stat_df = player_stat_df.sort_values(by='iPoints', ascending=False)
    player_stat_df = player_stat_df.reset_index(drop=True)
    player_stat_df = fix_teams(player_stat_df)
    player_stat_df['iG/60'] = player_stat_df['iGoals']/player_stat_df['ATOI']/82*60
    # print(player_stat_df[['PlayerID', 'Player', 'Position', 'Team', 'Age', 'ATOI', 'Gper1kChunk', 'A1per1kChunk', 'A2per1kChunk', 'iGoals', 'iPoints']].to_string())
    # print(player_stat_df[['PlayerID', 'Player', 'Position', 'Team', 'Age', 'ATOI', 'Gper1kChunk', 'iGoals', 'iG/60']].to_string())
    # print(player_stat_df.info())

    # Make team inferences
    team_stat_df = pd.DataFrame()
    team_stat_df = team_ga_model_inference(projection_year=PROJECTION_YEAR, team_stat_df=team_stat_df, player_stat_df=player_stat_df, team_ga_model=team_ga_model, download_file=True, verbose=False)
    # team_stat_df = team_stat_df.sort_values(by='Agg GA/GP', ascending=False)
    # print(team_stat_df.to_string())

    # Simulate season
    simulate_season(PROJECTION_YEAR, 15, True, True, True)

    # Push the simulation results to Supabase
    # push_to_supabase("team-projections", False)
    # push_to_supabase("player-projections", False)

    print(f"Runtime: {time.time()-start_time:.3f} seconds")

if __name__ == "__main__":
    main()