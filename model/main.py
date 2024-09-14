import time
import pandas as pd
from scraper_functions import *
from model_training import *
from model_inference import *
from model_bootstrap import *
from projection_engine import *

def main():
    start_time = time.time()
    PROJECTION_YEAR = 2025
    SIMULATIONS = 1
    # season_state = get_season_state(PROJECTION_YEAR)

    # Update metadata.json
    update_metadata(state=0, params=[start_time, PROJECTION_YEAR, SIMULATIONS])

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
    atoi_model = train_atoi_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    gp_model_data = train_gp_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    goal_model = train_goal_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    a1_model = train_a1_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    a2_model = train_a2_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    team_ga_model = train_ga_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    skater_xga_model = train_skater_xga_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)
    skater_ga_model = train_skater_ga_model(projection_year=PROJECTION_YEAR, retrain_model=False, verbose=False)

    # Make player inferences
    player_stat_df = pd.DataFrame()
    player_stat_df = atoi_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, atoi_model=atoi_model, download_file=False, verbose=False)
    player_stat_df = gp_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, gp_model_data=gp_model_data, download_file=False, verbose=False)
    player_stat_df = goal_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, goal_model=goal_model, download_file=False, verbose=False)
    player_stat_df = a1_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, a1_model=a1_model, download_file=False, verbose=False)
    player_stat_df = a2_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, a2_model=a2_model, download_file=False, verbose=False)
    player_stat_df = savitzky_golvay_calibration(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df)
    player_stat_df = skater_xga_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, skater_xga_model=skater_xga_model, download_file=False, verbose=False)
    player_stat_df = skater_ga_model_inference(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, skater_ga_model=skater_ga_model, download_file=False, verbose=False)

    # Bootstrap player inferences
    bootstrap_df = pd.DataFrame()
    bootstrap_df = bootstrap_atoi_inferences(projection_year=PROJECTION_YEAR, bootstrap_df=bootstrap_df, retrain_model=False, download_file=True, verbose=False)
    bootstrap_df = bootstrap_gp_inferences(projection_year=PROJECTION_YEAR, bootstrap_df=bootstrap_df, retrain_model=False, download_file=True, verbose=False)
    bootstrap_df = bootstrap_goal_inferences(projection_year=PROJECTION_YEAR, bootstrap_df=bootstrap_df, retrain_model=False, download_file=True, verbose=False)
    bootstrap_df = bootstrap_a1_inferences(projection_year=PROJECTION_YEAR, bootstrap_df=bootstrap_df, retrain_model=False, download_file=True, verbose=False)
    bootstrap_df = bootstrap_a2_inferences(projection_year=PROJECTION_YEAR, bootstrap_df=bootstrap_df, retrain_model=False, download_file=True, verbose=False)
    display_inferences(projection_year=PROJECTION_YEAR, player_stat_df=player_stat_df, bootstrap_df=bootstrap_df, inference_state='TOTAL', download_file=True, verbose=True)

    # Make team inferences
    team_stat_df = pd.DataFrame()
    team_stat_df = team_ga_model_inference(projection_year=PROJECTION_YEAR, team_stat_df=team_stat_df, player_stat_df=player_stat_df, team_ga_model=team_ga_model, download_file=True, verbose=False)
    # team_stat_df = team_stat_df.sort_values(by='Agg GA/GP', ascending=False)
    # print(team_stat_df.to_string())

    # Run projection engine and simulate season
    run_projection_engine(projection_year=PROJECTION_YEAR, simulations=SIMULATIONS, download_files=True, verbose=True) ###

    # Push the simulation results to Supabase
    # push_to_supabase(table_name="team-projections", projection_year=PROJECTION_YEAR, verbose=True)
    # push_to_supabase(table_name="player-projections", projection_year=PROJECTION_YEAR, verbose=True)
    # push_to_supabase(table_name="game-projections", projection_year=PROJECTION_YEAR, verbose=True)

    print(f"Runtime: {time.time()-start_time:.3f} seconds")
    update_metadata(state=1, params=[time.time(), time.time()-start_time])

    return

if __name__ == "__main__":
    main()