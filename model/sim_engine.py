import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_training import *
from scraper_functions import *
from scipy.stats import poisson

def simulate_game(gameid, home_team_abbrev, home_roster, home_defence_score, visiting_team_abbrev, visitor_roster, visitor_defence_score, player_scoring_dict, team_scoring_dict, game_scoring_dict, a1_probability, a2_probability, verbose):

    # determine active rosters
    home_fwds = home_roster[home_roster['PosFD'] == 'F']
    home_dfcs = home_roster[home_roster['PosFD'] == 'D']
    selected_forwards = home_fwds.sample(n=12, weights=home_fwds['GPprb'], replace=False)
    selected_defensemen = home_dfcs.sample(n=6, weights=home_dfcs['GPprb'], replace=False)
    home_active_roster = pd.concat([selected_forwards, selected_defensemen])
    visitor_fwds = visitor_roster[visitor_roster['PosFD'] == 'F']
    visitor_dfcs = visitor_roster[visitor_roster['PosFD'] == 'D']
    selected_forwards = visitor_fwds.sample(n=12, weights=visitor_fwds['GPprb'], replace=False)
    selected_defensemen = visitor_dfcs.sample(n=6, weights=visitor_dfcs['GPprb'], replace=False)
    visitor_active_roster = pd.concat([selected_forwards, selected_defensemen])

    # set initial score to 0-0
    home_score = 0
    visitor_score = 0

    # increment games played for each player on active roster in game scoring dict
    for player_id in home_active_roster['PlayerID']:
        player_scoring_dict[player_id][0] += 1
    for player_id in visitor_active_roster['PlayerID']:
        player_scoring_dict[player_id][0] += 1

    # Calculate the weighted averages
    home_weighted_avg = np.average(home_active_roster['Pper1kChunk'], weights=home_active_roster['ATOI'])/1000 * 5/(1+a1_probability+a2_probability)
    visitor_weighted_avg = np.average(visitor_active_roster['Pper1kChunk'], weights=visitor_active_roster['ATOI'])/1000 * 5/(1+a1_probability+a2_probability)

    # adjust for home ice advantage ###
    home_weighted_avg *= 1.025574015
    visitor_weighted_avg *= 0.9744259847

    # adjust for team defence
    home_weighted_avg *= visitor_defence_score
    visitor_weighted_avg *= home_defence_score

    # determining scorers and assisters
    home_scorer_ids = home_active_roster.sample(n=11, replace=True, weights=home_active_roster['Gper1kChunk']*home_active_roster['ATOI'])['PlayerID'].values
    visitor_scorer_ids = visitor_active_roster.sample(n=11, replace=True, weights=visitor_active_roster['Gper1kChunk']*visitor_active_roster['ATOI'])['PlayerID'].values
    home_assist_ids = home_active_roster.sample(n=22, replace=True, weights=home_active_roster['Aper1kChunk']*home_active_roster['ATOI'])['PlayerID'].values
    visitor_assist_ids = visitor_active_roster.sample(n=22, replace=True, weights=visitor_active_roster['Aper1kChunk']*visitor_active_roster['ATOI'])['PlayerID'].values

    for chunk in range(120):
        rng = random.uniform(0, 1)
        if rng < home_weighted_avg and home_score <= 10: # home goal
            try:
                scorer_id = home_scorer_ids[home_score]
                a1_id = home_assist_ids[home_score]
                a2_id = home_assist_ids[home_score + 10]
            except IndexError: # score more than 10 goals
                scorer_id = home_active_roster.sample(n=1, replace=True, weights=home_active_roster['Gper1kChunk']*home_active_roster['ATOI'])['PlayerID'].values[0]
                a1_id = home_active_roster.sample(n=1, replace=True, weights=home_active_roster['Aper1kChunk']*home_active_roster['ATOI'])['PlayerID'].values[0]
                a2_id = home_active_roster.sample(n=1, replace=True, weights=home_active_roster['Aper1kChunk']*home_active_roster['ATOI'])['PlayerID'].values[0]
            home_score += 1
        elif rng > 1 - visitor_weighted_avg and visitor_score <= 10: # visitor goal
            try:
                scorer_id = visitor_scorer_ids[visitor_score]
                a1_id = visitor_assist_ids[visitor_score]
                a2_id = visitor_assist_ids[visitor_score + 10]
            except IndexError: # score more than 10 goals
                scorer_id = visitor_active_roster.sample(n=1, replace=True, weights=visitor_active_roster['Gper1kChunk']*visitor_active_roster['ATOI'])['PlayerID'].values[0]
                a1_id = visitor_active_roster.sample(n=1, replace=True, weights=visitor_active_roster['Aper1kChunk']*visitor_active_roster['ATOI'])['PlayerID'].values[0]
                a2_id = visitor_active_roster.sample(n=1, replace=True, weights=visitor_active_roster['Aper1kChunk']*visitor_active_roster['ATOI'])['PlayerID'].values[0]
            visitor_score += 1
        else:
            continue # no goal occurs in chunk; advance to next chunk

        player_scoring_dict[scorer_id][1] += 1

        # Assign assists
        if random.uniform(0, 1) < a1_probability:
            player_scoring_dict[a1_id][2] += 1

            if random.uniform(0, 1) < a2_probability:
                player_scoring_dict[a2_id][2] += 1

    if home_score > visitor_score:
        team_scoring_dict[home_team_abbrev][0] += 1
        team_scoring_dict[visiting_team_abbrev][1] += 1
        game_scoring_dict[gameid][2] = 0
    elif home_score < visitor_score:
        team_scoring_dict[visiting_team_abbrev][0] += 1
        team_scoring_dict[home_team_abbrev][1] += 1
        game_scoring_dict[gameid][2] = 0
    else:
        # overtime
        rng = random.uniform(0, 1)
        home_weighted_avg_ot = home_weighted_avg/(home_weighted_avg + visitor_weighted_avg)
        if rng < home_weighted_avg_ot: # home goal
            scorer_id = home_scorer_ids[home_score]
            a1_id = home_assist_ids[home_score]
            a2_id = home_assist_ids[home_score + 10]
            home_score += 1
            team_scoring_dict[home_team_abbrev][0] += 1
            team_scoring_dict[visiting_team_abbrev][2] += 1
        else: # visitor goal
            scorer_id = visitor_scorer_ids[visitor_score]
            a1_id = visitor_assist_ids[visitor_score]
            a2_id = visitor_assist_ids[visitor_score + 10]
            visitor_score += 1
            team_scoring_dict[visiting_team_abbrev][0] += 1
            team_scoring_dict[home_team_abbrev][2] += 1

        player_scoring_dict[scorer_id][1] += 1

        # Assign assists
        if random.uniform(0, 1) < a1_probability:
            player_scoring_dict[a1_id][2] += 1

            if random.uniform(0, 1) < a2_probability:
                player_scoring_dict[a2_id][2] += 1

        game_scoring_dict[gameid][2] = 1

    # add gf and ga to team scoring dict
    team_scoring_dict[home_team_abbrev][3] += home_score
    team_scoring_dict[home_team_abbrev][4] += visitor_score
    team_scoring_dict[visiting_team_abbrev][3] += visitor_score
    team_scoring_dict[visiting_team_abbrev][4] += home_score

    # add score to game scoring dict
    game_scoring_dict[gameid][0] = home_score
    game_scoring_dict[gameid][1] = visitor_score

    return player_scoring_dict, team_scoring_dict, game_scoring_dict

def simulate_season(projection_year, projection_strategy, simulations, resume_season, download_files, verbose):
    # load dfs
    schedule_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', f'{projection_year-1}-{projection_year}_game_schedule.csv'), index_col=0)
    schedule_df['Home Win'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else row['Home Score'] > row['Visiting Score'], axis=1)
    schedule_df['Visitor Win'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else row['Home Score'] < row['Visiting Score'], axis=1)
    schedule_df['Overtime'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else (True if row['Period'] > 3 else False), axis=1)
    metaprojection_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters', f'{projection_year}_skater_metaprojections.csv'), index_col=0)
    metaprojection_df['Aper1kChunk'] = metaprojection_df['A1per1kChunk'] + metaprojection_df['A2per1kChunk']
    metaprojection_df['Pper1kChunk'] = metaprojection_df['Gper1kChunk'] + metaprojection_df['Aper1kChunk']
    teams_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'), index_col=0)
    team_metaproj_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Teams', f'{projection_year}_team_projections.csv'), index_col=0)
    if resume_season == True:
        existing_skater_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))
        existing_skater_df['Assists'] = existing_skater_df['First Assists'] + existing_skater_df['Second Assists']
        existing_team_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Team Data', f'{projection_year-1}-{projection_year}_team_data.csv'))
        existing_team_df = existing_team_df.rename(columns={'W': 'Wins', 'L': 'Losses', 'GA': 'Goals Against', 'GF': 'Goals For'})

    # add team abbreviations to schedule
    schedule_df = schedule_df.merge(teams_df, left_on='Home Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Home Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID'])
    schedule_df = schedule_df.merge(teams_df, left_on='Visiting Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Visiting Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID', 'Active_x', 'Active_y'])

    # configure skater monte carlo projection df
    monte_carlo_skater_proj_df = metaprojection_df[['PlayerID', 'Player', 'Position', 'Team', 'Age']].copy()
    monte_carlo_skater_proj_df = monte_carlo_skater_proj_df.assign(Games_Played=0, Goals=0, Assists=0)
    monte_carlo_skater_proj_df.rename(columns={'Games_Played': 'Games Played'}, inplace=True)
    
    # configure team monte carlo projection df
    monte_carlo_team_proj_df = teams_df[['Team Name', 'Abbreviation']].copy()
    active_teams = pd.concat([schedule_df['Home Team'], schedule_df['Visiting Team']])
    monte_carlo_team_proj_df = monte_carlo_team_proj_df[monte_carlo_team_proj_df['Team Name'].isin(active_teams)]
    monte_carlo_team_proj_df = monte_carlo_team_proj_df.assign(Wins=0, Losses=0, OTL=0, Goals_For=0, Goals_Against=0)
    monte_carlo_team_proj_df.rename(columns={'Team Name': 'Team', 'Goals_For': 'Goals For', 'Goals_Against': 'Goals Against'}, inplace=True)

    # configure games monte carlo projection df
    monte_carlo_game_proj_df = schedule_df[['GameID', 'Time (EST)', 'Home Team', 'Home Abbreviation', 'Home Score', 'Visiting Team', 'Visiting Abbreviation', 'Visiting Score']].copy()
    monte_carlo_game_proj_df = monte_carlo_game_proj_df.assign(
        Date = pd.to_datetime(monte_carlo_game_proj_df['Time (EST)']).dt.date, 
        TimeEST = pd.to_datetime(monte_carlo_game_proj_df['Time (EST)']).dt.time, 
        Home_Win = 0,
        Visitor_Win = 0,
        Overtime = 0
    )
    monte_carlo_game_proj_df.rename(columns={'Home_Win': 'Home Win', 'Visitor_Win': 'Visitor Win', 'Time (EST)': 'DatetimeEST'}, inplace=True)
    monte_carlo_game_proj_df = monte_carlo_game_proj_df[['GameID', 'DatetimeEST', 'Date', 'TimeEST', 'Home Team', 'Home Abbreviation', 'Visiting Team', 'Visiting Abbreviation', 'Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']]

    if resume_season == True:
        # move stats from current season to player_scoring_dict and team_scoring_dict
        core_player_scoring_dict = existing_skater_df.dropna(subset=['PlayerID']).fillna(0).set_index('PlayerID')[['GP', 'Goals', 'Assists']].T.to_dict('list')
        temp_df = pd.merge(existing_team_df, monte_carlo_team_proj_df[['Team', 'Abbreviation']], on='Team', how='left')
        temp_df.loc[temp_df['Team'] == 'Montreal Canadiens', 'Abbreviation'] = 'MTL'
        temp_df.loc[temp_df['Team'] == 'St Louis Blues', 'Abbreviation'] = 'STL'
        temp_df.loc[temp_df['Team'] == 'Arizona Coyotes', 'Abbreviation'] = 'UTA'
        core_team_scoring_dict = temp_df.fillna(0).set_index('Abbreviation')[['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against']].T.to_dict('list')
        core_game_scoring_dict = schedule_df.fillna(0).set_index('GameID')[['Home Score', 'Visiting Score', 'Overtime']].T.to_dict('list')

        # players that haven't played yet
        for player_id in monte_carlo_skater_proj_df['PlayerID']:
            if player_id not in core_player_scoring_dict:
                core_player_scoring_dict[player_id] = [0, 0, 0]

        # teams that haven't played yet
        for team_abbreviation in monte_carlo_team_proj_df['Abbreviation']:
            if team_abbreviation not in core_team_scoring_dict:
                core_team_scoring_dict[team_abbreviation] = [0, 0, 0, 0, 0]
    else:
        # create game scoring dictionary
        core_player_scoring_dict = {} # {player_id: [games, goals, assists]}
        for player_id in monte_carlo_skater_proj_df['PlayerID']:
            core_player_scoring_dict[player_id] = [0, 0, 0]

        # create team scoring dictionary
        core_team_scoring_dict = {} # {team_abbreviation: [wins, losses, ot_losses, goals_for, goals_against]}
        for team_abbreviation in monte_carlo_team_proj_df['Abbreviation']:
            core_team_scoring_dict[team_abbreviation] = [0, 0, 0, 0, 0]

    # determine active rosters. we convert the data from the dataframe to dictionary because the lookup times are faster (O(n) vs O(1))
    team_rosters = {}
    metaprojection_df.loc[metaprojection_df['Team'] == 'ARI', 'Team'] = 'UTA' ### temporary fix for ARI
    for team_abbreviation in monte_carlo_team_proj_df['Abbreviation']:
        team_roster = metaprojection_df[metaprojection_df['Team'] == team_abbreviation].copy()
        team_roster['PosFD'] = team_roster['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
        team_rosters[team_abbreviation] = team_roster

    # compute team defence scores
    defence_scores = {}
    avg_ga = team_metaproj_df['Normalized GA/GP'].mean()
    for team in monte_carlo_team_proj_df['Team']:
        if team == 'Montréal Canadiens':
            lookup_team = 'Montreal Canadiens'
        elif team == 'St. Louis Blues':
            lookup_team = 'St Louis Blues'
        elif team == 'Utah Hockey Club': ### temp
            lookup_team = 'Arizona Coyotes'
        else:
            lookup_team = team

        defence_scores[team] = 1 + (team_metaproj_df[team_metaproj_df['Team'] == lookup_team]['Normalized GA/GP'].values[0] - avg_ga)/avg_ga

    a1_probability = 0.9438426454 # probability of a goal having a primary assistor ###
    a2_probability = 0.7916037451 # probability of a goal with a primary assistor also having a secondary assistor ###

    # check projection strategy
    if projection_strategy.upper() == 'MONTE CARLO':
        pass
    elif projection_strategy.upper() in ['INFERENCE', 'INFERENCE_ONLY']:
        generate_game_inferences(projection_year, schedule_df, team_rosters, defence_scores, a1_probability, a2_probability, download_files=True, verbose=False)
        if projection_strategy.upper() == 'INFERENCE_ONLY':
            return
    else:
        raise ValueError(f"Invalid projection strategy '{type}'. Please choose either 'MONTE CARLO' or 'INFERENCE'.")

    # create dataframe to store individual simulation results
    skater_simulations_df = pd.DataFrame(columns=['Simulation', 'PlayerID', 'Player', 'Position', 'Team', 'Age', 'Games Played', 'Goals', 'Assists', 'Points'])
    team_simulations_df = pd.DataFrame(columns=['Simulation', 'Abbreviation', 'Team', 'Wins', 'Losses', 'OTL', 'Points', 'Goals For', 'Goals Against'])
    game_simulations_df = pd.DataFrame(columns=['Simulation', 'GameID', 'DatetimeEST', 'Date', 'TimeEST', 'Home Team', 'Home Abbreviation', 'Visiting Team', 'Visiting Abbreviation', 'Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime'])

    # run simulations
    for simulation in tqdm(range(simulations), desc='Simulating Seasons'):
        # initialize stat storing dictionaries
        player_scoring_dict = copy.deepcopy(core_player_scoring_dict)
        team_scoring_dict = copy.deepcopy(core_team_scoring_dict)
        game_scoring_dict = copy.deepcopy(core_game_scoring_dict)

        # for index, row in tqdm(schedule_df.iterrows(), total=schedule_df.shape[0]):
        for index, row in schedule_df.iterrows():
            if resume_season == True and row['Game State'] == 7:
                continue
        
            gameid = row['GameID']
            home_abbrev = row['Home Abbreviation']
            visiting_abbrev = row['Visiting Abbreviation']

            player_scoring_dict, team_scoring_dict, game_scoring_dict = simulate_game(gameid, home_abbrev, team_rosters[home_abbrev], defence_scores[row['Home Team']], visiting_abbrev, team_rosters[visiting_abbrev], defence_scores[row['Visiting Team']], player_scoring_dict, team_scoring_dict, game_scoring_dict, a1_probability, a2_probability, verbose)

        # add player scoring dict stats to monte_carlo_skater_proj_df
        player_scoring_df = pd.DataFrame.from_dict(player_scoring_dict, orient='index', columns=['Games Played', 'Goals', 'Assists'])
        monte_carlo_skater_proj_df.set_index('PlayerID', inplace=True)
        monte_carlo_skater_proj_df[['Games Played', 'Goals', 'Assists']] = player_scoring_df
        monte_carlo_skater_proj_df.reset_index(inplace=True)

        # add team scoring dict stats to monte_carlo_team_proj_df
        team_scoring_df = pd.DataFrame.from_dict(team_scoring_dict, orient='index', columns=['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against'])
        monte_carlo_team_proj_df.set_index('Abbreviation', inplace=True)
        monte_carlo_team_proj_df[['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against']] = team_scoring_df
        monte_carlo_team_proj_df.reset_index(inplace=True)

        # add game scoring dict stats to monte_carlo_team_proj_df
        game_scoring_df = pd.DataFrame.from_dict(game_scoring_dict, orient='index', columns=['Home Score', 'Visiting Score', 'Overtime'])
        monte_carlo_game_proj_df.set_index('GameID', inplace=True)
        monte_carlo_game_proj_df[['Home Score', 'Visiting Score', 'Overtime']] = game_scoring_df
        monte_carlo_game_proj_df.reset_index(inplace=True)

        # calculate points for skaters, sort
        monte_carlo_skater_proj_df['Points'] = monte_carlo_skater_proj_df['Goals'] + monte_carlo_skater_proj_df['Assists']
        monte_carlo_skater_proj_df = monte_carlo_skater_proj_df.sort_values(by=['Points', 'Goals', 'Assists'], ascending=False)
        monte_carlo_skater_proj_df.reset_index(drop=True, inplace=True)
        monte_carlo_skater_proj_df.index += 1

        # calculate points for teams, sort
        monte_carlo_team_proj_df['Points'] = monte_carlo_team_proj_df['Wins']*2 + monte_carlo_team_proj_df['OTL']
        monte_carlo_team_proj_df = monte_carlo_team_proj_df.sort_values(by=['Points', 'Wins', 'Goals For', 'Goals Against'], ascending=False)
        monte_carlo_team_proj_df.reset_index(drop=True, inplace=True)
        monte_carlo_team_proj_df.index += 1

        # calculate win/loss for games
        monte_carlo_game_proj_df['Home Win'] = monte_carlo_game_proj_df.apply(lambda row: 1 if row['Home Score'] > row['Visiting Score'] else 0, axis=1)
        monte_carlo_game_proj_df['Visitor Win'] = monte_carlo_game_proj_df.apply(lambda row: 1 if row['Home Score'] < row['Visiting Score'] else 0, axis=1)

        if resume_season == True:
            monte_carlo_skater_proj_df[['Games Played', 'Goals', 'Assists', 'Points']] = monte_carlo_skater_proj_df[['Games Played', 'Goals', 'Assists', 'Points']].astype(int)
            monte_carlo_team_proj_df[['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against', 'Points']] = monte_carlo_team_proj_df[['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against', 'Points']].astype(int)
            monte_carlo_game_proj_df[['Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']] = monte_carlo_game_proj_df[['Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']].astype(int)

        # print results from an individual simulation
        # print(monte_carlo_skater_proj_df.head(10))
        # print(monte_carlo_team_proj_df.head(10))
        # print(monte_carlo_game_proj_df.head(10))

        # store results from an individual simulation
        monte_carlo_skater_proj_df['Simulation'] = simulation + 1
        skater_simulations_df = pd.concat([skater_simulations_df, monte_carlo_skater_proj_df])
        monte_carlo_team_proj_df['Simulation'] = simulation + 1
        team_simulations_df = pd.concat([team_simulations_df, monte_carlo_team_proj_df])
        monte_carlo_game_proj_df['Simulation'] = simulation + 1
        game_simulations_df = pd.concat([game_simulations_df, monte_carlo_game_proj_df])
    
    skater_simulations_df = skater_simulations_df.set_index(['Simulation', 'PlayerID', 'Player', 'Position', 'Team', 'Age'])
    team_simulations_df = team_simulations_df.set_index(['Simulation', 'Abbreviation', 'Team'])
    game_simulations_df = game_simulations_df.set_index(['Simulation', 'GameID', 'DatetimeEST', 'Date', 'TimeEST', 'Home Team', 'Home Abbreviation', 'Visiting Team', 'Visiting Abbreviation'])
    
    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        skater_simulations_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_monte_carlo_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_monte_carlo_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_skater_monte_carlo_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_simulations_df.to_csv(os.path.join(export_path, f'{projection_year}_team_monte_carlo_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_monte_carlo_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_team_monte_carlo_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Games')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        game_simulations_df.to_csv(os.path.join(export_path, f'{projection_year}_game_monte_carlo_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_game_monte_carlo_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_game_monte_carlo_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

    skater_aggregated_df = skater_simulations_df.groupby(level=[1, 2, 3, 4, 5]).mean()
    skater_aggregated_df = skater_aggregated_df.reset_index()
    skater_aggregated_df = skater_aggregated_df.sort_values(by=['Points', 'Goals', 'Assists'], ascending=False)
    skater_aggregated_df = skater_aggregated_df.reset_index(drop=True)
    skater_aggregated_df.index += 1
    print(skater_aggregated_df)

    team_aggregated_df = team_simulations_df.groupby(level=[1, 2]).mean()
    team_aggregated_df = team_aggregated_df.reset_index()
    team_aggregated_df = team_aggregated_df.sort_values(by=['Points', 'Wins', 'Goals For'], ascending=False)
    team_aggregated_df = team_aggregated_df.reset_index(drop=True)
    team_aggregated_df.index += 1
    print(team_aggregated_df)

    game_aggregated_df = game_simulations_df.groupby(level=[1, 2, 3, 4, 5, 6, 7, 8]).mean()
    game_aggregated_df = game_aggregated_df.reset_index()
    game_aggregated_df.index += 1
    print(game_aggregated_df)

    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        skater_aggregated_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_aggregated_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_aggregated_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_skater_aggregated_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_aggregated_df.to_csv(os.path.join(export_path, f'{projection_year}_team_aggregated_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_aggregated_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_team_aggregated_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        if projection_strategy.upper() == 'MONTE CARLO':
            export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Games')
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            game_aggregated_df.to_csv(os.path.join(export_path, f'{projection_year}_game_aggregated_projections.csv'), index=True)
            if verbose:
                print(f'{projection_year}_game_aggregated_projections.csv has been downloaded to the following directory: {export_path}')
                file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_game_aggregated_projections.csv'))/1000000
                print(f'\tFile size: {file_size} MB')

def generate_game_inferences(projection_year, schedule_df, team_rosters, defence_scores, a1_probability, a2_probability, download_files, verbose=False):

    # configure games game_aggregated_df
    game_aggregated_df = schedule_df[['GameID', 'Time (EST)', 'Home Team', 'Home Abbreviation', 'Home Score', 'Visiting Team', 'Visiting Abbreviation', 'Visiting Score']].copy()
    game_aggregated_df = game_aggregated_df.assign(
        Date = pd.to_datetime(game_aggregated_df['Time (EST)']).dt.date, 
        TimeEST = pd.to_datetime(game_aggregated_df['Time (EST)']).dt.time, 
        Home_Win = 0,
        Visitor_Win = 0,
        Overtime = 0
    )
    game_aggregated_df.rename(columns={'Home_Win': 'Home Win', 'Visitor_Win': 'Visitor Win', 'Time (EST)': 'DatetimeEST'}, inplace=True)
    game_aggregated_df = game_aggregated_df[['GameID', 'DatetimeEST', 'Date', 'TimeEST', 'Home Team', 'Home Abbreviation', 'Visiting Team', 'Visiting Abbreviation', 'Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']]

    # initialize lists to store results
    home_score_list = []
    visitor_score_list = []
    home_prob_list = []
    visitor_prob_list = []
    overtime_list = []

    # loop through each game
    # for index, row in game_aggregated_df.iterrows():
    for index, row in tqdm(game_aggregated_df.iterrows(), total=game_aggregated_df.shape[0], desc="Generating Game Inferences"):
        # fetch rosters
        home_roster = team_rosters[row['Home Abbreviation']]
        visitor_roster = team_rosters[row['Visiting Abbreviation']]

        # Calculate the weighted averages
        home_weighted_avg = np.average(home_roster['Pper1kChunk'], weights=home_roster['ATOI']*home_roster['GPprb'])/1000 * 5/(1+a1_probability+a2_probability)
        visitor_weighted_avg = np.average(visitor_roster['Pper1kChunk'], weights=visitor_roster['ATOI']*visitor_roster['GPprb'])/1000 * 5/(1+a1_probability+a2_probability)

        # adjust for home ice advantage ###
        home_weighted_avg *= 1.025574015
        visitor_weighted_avg *= 0.9744259847

        # adjust for team defence
        home_weighted_avg *= defence_scores[row['Visiting Team']]
        visitor_weighted_avg *= defence_scores[row['Home Team']]

        # determine expected goals and probabilities
        home_score, visitor_score, home_prob, visitor_prob, overtime = compute_poisson_probabilities(home_weighted_avg, visitor_weighted_avg)

        # save results to dataframe
        home_score_list.append(home_score)
        visitor_score_list.append(visitor_score)
        home_prob_list.append(home_prob)
        visitor_prob_list.append(visitor_prob)
        overtime_list.append(overtime)

        if verbose:
            print(index+1, row['GameID'], row['Home Abbreviation'], row['Visiting Abbreviation'], home_score, visitor_score, home_prob, visitor_prob, overtime)

    game_aggregated_df['Home Score'] = home_score_list
    game_aggregated_df['Visiting Score'] = visitor_score_list
    game_aggregated_df['Home Win'] = home_prob_list
    game_aggregated_df['Visitor Win'] = visitor_prob_list
    game_aggregated_df['Overtime'] = overtime_list

    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', 'Games')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        game_aggregated_df.to_csv(os.path.join(export_path, f'{projection_year}_game_aggregated_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_game_aggregated_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_game_aggregated_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

# function to compute poisson probabilities of winning and overtime
def compute_poisson_probabilities(home_weighted_avg, visitor_weighted_avg, chunks=120, max_goals=10):
    home_score = home_weighted_avg * chunks
    visitor_score = visitor_weighted_avg * chunks
    home_prob, visitor_prob, overtime = 0, 0, 0

    for home_goals in range(max_goals):
        for visitor_goals in range(max_goals):
            prob_goals_home = poisson.pmf(home_goals, home_score)
            prob_goals_visitor = poisson.pmf(visitor_goals, visitor_score)

            if home_goals > visitor_goals:
                home_prob += prob_goals_home * prob_goals_visitor
            elif home_goals < visitor_goals:
                visitor_prob += prob_goals_home * prob_goals_visitor
            else:
                overtime += prob_goals_home * prob_goals_visitor

    # normalize
    home_prob /= (home_prob + visitor_prob)
    visitor_prob /= (visitor_prob - home_prob*visitor_prob/(home_prob-1))

    # scale
    home_prob = 1/2+(home_prob-0.5)*0.83543279
    visitor_prob = 1/2+(visitor_prob-0.5)*0.83543279

    return home_score, visitor_score, home_prob, visitor_prob, overtime