import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_training import *
from scraper_functions import *
from scipy.stats import poisson
import copy

def run_projection_engine(projection_year, simulations, download_files, verbose):

    # load dfs
    schedule_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', f'{projection_year-1}-{projection_year}_game_schedule.csv'), index_col=0)
    schedule_df['Home Win'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else row['Home Score'] > row['Visiting Score'], axis=1)
    schedule_df['Visitor Win'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else row['Home Score'] < row['Visiting Score'], axis=1)
    schedule_df['Overtime'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else (True if row['Period'] > 3 else False), axis=1)
    metaprojection_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters', f'{projection_year}_skater_metaprojections.csv'), index_col=0)
    metaprojection_df['Aper1kChunk'] = metaprojection_df['A1per1kChunk'] + metaprojection_df['A2per1kChunk']
    metaprojection_df['Pper1kChunk'] = metaprojection_df['Gper1kChunk'] + metaprojection_df['Aper1kChunk']
    teams_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'nhlapi_team_data.csv'), index_col=0)
    team_metaproj_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Teams', f'{projection_year}_team_projections.csv'), index_col=0)
    existing_skater_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))
    existing_skater_df['Assists'] = existing_skater_df['First Assists'] + existing_skater_df['Second Assists']
    # existing_skater_df['ATOI'] = (existing_skater_df['TOI'].fillna(0) / existing_skater_df['GP'].fillna(0)).fillna(0)
    existing_team_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Historical Team Data', f'{projection_year-1}-{projection_year}_team_data.csv'))
    existing_team_df = existing_team_df.rename(columns={'W': 'Wins', 'L': 'Losses', 'GA': 'Goals Against', 'GF': 'Goals For'})

    # add team abbreviations to schedule
    schedule_df = schedule_df.merge(teams_df, left_on='Home Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Home Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID'])
    schedule_df = schedule_df.merge(teams_df, left_on='Visiting Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Visiting Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID', 'Active_x', 'Active_y'])

    # configure skater projection df
    skater_proj_df = copy.deepcopy(metaprojection_df[['PlayerID', 'Player', 'Position', 'Team', 'Age']])
    skater_proj_df = skater_proj_df.assign(Games_Played=0, Goals=0, Assists=0)
    skater_proj_df.rename(columns={'Games_Played': 'Games Played'}, inplace=True)
    
    # configure team projection df
    team_proj_df = copy.deepcopy(teams_df[['Team Name', 'Abbreviation']])
    active_teams = pd.concat([schedule_df['Home Team'], schedule_df['Visiting Team']])
    team_proj_df = team_proj_df[team_proj_df['Team Name'].isin(active_teams)]
    team_proj_df = team_proj_df.assign(Wins=0, Losses=0, OTL=0, Goals_For=0, Goals_Against=0)
    team_proj_df.rename(columns={'Team Name': 'Team', 'Goals_For': 'Goals For', 'Goals_Against': 'Goals Against'}, inplace=True)

    # configure games projection df
    game_proj_df = copy.deepcopy(schedule_df[['GameID', 'Time (EST)', 'Home Team', 'Home Abbreviation', 'Home Score', 'Visiting Team', 'Visiting Abbreviation', 'Visiting Score']])
    game_proj_df = game_proj_df.assign(
        Date = pd.to_datetime(game_proj_df['Time (EST)']).dt.date, 
        TimeEST = pd.to_datetime(game_proj_df['Time (EST)']).dt.time, 
        Home_Win = 0,
        Visitor_Win = 0,
        Overtime = 0
    )
    game_proj_df.rename(columns={'Home_Win': 'Home Win', 'Visitor_Win': 'Visitor Win', 'Time (EST)': 'DatetimeEST'}, inplace=True)
    game_proj_df = game_proj_df[['GameID', 'DatetimeEST', 'Date', 'TimeEST', 'Home Team', 'Home Abbreviation', 'Visiting Team', 'Visiting Abbreviation', 'Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']]

    # move stats from current season to player_scoring_dict and team_scoring_dict
    core_player_scoring_dict = existing_skater_df.dropna(subset=['PlayerID']).fillna(0).set_index('PlayerID')[['GP', 'TOI', 'Goals', 'Assists']].T.to_dict('list')
    temp_df = pd.merge(existing_team_df, team_proj_df[['Team', 'Abbreviation']], on='Team', how='left')
    temp_df.loc[temp_df['Team'] == 'Montreal Canadiens', 'Abbreviation'] = 'MTL'
    temp_df.loc[temp_df['Team'] == 'St Louis Blues', 'Abbreviation'] = 'STL'
    temp_df.loc[temp_df['Team'] == 'Arizona Coyotes', 'Abbreviation'] = 'UTA'
    core_team_scoring_dict = temp_df.fillna(0).set_index('Abbreviation')[['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against']].T.to_dict('list')
    core_game_scoring_dict = schedule_df.fillna(0).set_index('GameID')[['Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']].T.to_dict('list')

    # players that haven't played yet
    for player_id in skater_proj_df['PlayerID']:
        if player_id not in core_player_scoring_dict:
            core_player_scoring_dict[player_id] = [0, 0, 0, 0]

    # teams that haven't played yet
    for team_abbreviation in team_proj_df['Abbreviation']:
        if team_abbreviation not in core_team_scoring_dict:
            core_team_scoring_dict[team_abbreviation] = [0, 0, 0, 0, 0]

    # determine active rosters. we convert the data from the dataframe to dictionary because the lookup times are faster (O(n) vs O(1))
    team_rosters = {}
    metaprojection_df.loc[metaprojection_df['Team'] == 'ARI', 'Team'] = 'UTA' ### temporary fix for ARI
    for team_abbreviation in team_proj_df['Abbreviation']:
        team_roster = copy.deepcopy(metaprojection_df[metaprojection_df['Team'] == team_abbreviation])
        team_roster['PosFD'] = team_roster['Position'].apply(lambda x: 'D' if x == 'D' else 'F')
        team_rosters[team_abbreviation] = team_roster

    # compute team defence scores
    defence_scores = {}
    avg_ga = team_metaproj_df['Normalized GA/GP'].mean()
    for team in team_proj_df['Team']:
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
    player_scoring_dict, team_scoring_dict, game_scoring_dict = generate_game_inferences(core_player_scoring_dict, core_team_scoring_dict, core_game_scoring_dict, schedule_df, team_rosters, defence_scores, a1_probability, a2_probability)

    # add player scoring dict stats to skater_proj_df, sort by points
    indexed_player_scoring_df = pd.DataFrame.from_dict(player_scoring_dict, orient='index', columns=['Games Played', 'TOI', 'Goals', 'Assists'])
    skater_proj_df.set_index('PlayerID', inplace=True)
    skater_proj_df[['Games Played', 'TOI', 'Goals', 'Assists']] = indexed_player_scoring_df
    skater_proj_df['Points'] = skater_proj_df['Goals'] + skater_proj_df['Assists']
    skater_proj_df = skater_proj_df.sort_values(by=['Points', 'Goals', 'Assists'], ascending=False)
    skater_proj_df.reset_index(inplace=True)
    skater_proj_df.index += 1
    if verbose:
        print(skater_proj_df)

    # add team scoring dict stats to team_proj_df, sort by points
    indexed_team_scoring_df = pd.DataFrame.from_dict(team_scoring_dict, orient='index', columns=['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against'])
    team_proj_df.set_index('Abbreviation', inplace=True)
    team_proj_df[['Wins', 'Losses', 'OTL', 'Goals For', 'Goals Against']] = indexed_team_scoring_df
    team_proj_df['Points'] = team_proj_df['Wins']*2 + team_proj_df['OTL']
    team_proj_df = team_proj_df.sort_values(by=['Points', 'Wins', 'Goals For', 'Goals Against'], ascending=False)
    team_proj_df.reset_index(inplace=True)
    team_proj_df.index += 1
    if verbose:
        print(team_proj_df)

    # add game scoring dict stats to game_proj_df
    indexed_game_scoring_df = pd.DataFrame.from_dict(game_scoring_dict, orient='index', columns=['Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime'])
    game_proj_df.set_index('GameID', inplace=True)
    game_proj_df[['Home Score', 'Visiting Score', 'Home Win', 'Visitor Win', 'Overtime']] = indexed_game_scoring_df
    game_proj_df.reset_index(inplace=True)
    game_proj_df.index += 1
    if verbose:
        print(game_proj_df)


    # generate player uncertainty-based projections via monte carlo engine
    skater_proj_df = player_monte_carlo_engine(skater_proj_df, core_player_scoring_dict, projection_year, simulations, download_files, verbose)

    # generate team uncertainty-based projections via monte carlo engine
    team_proj_df = team_monte_carlo_engine(team_proj_df, core_team_scoring_dict, projection_year, simulations, download_files, verbose)
    
    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        skater_proj_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_skater_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_proj_df.to_csv(os.path.join(export_path, f'{projection_year}_team_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_team_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Games')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        game_proj_df.to_csv(os.path.join(export_path, f'{projection_year}_game_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_game_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_game_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

def generate_game_inferences(core_player_scoring_dict, core_team_scoring_dict, core_game_scoring_dict, schedule_df, team_rosters, defence_scores, a1_probability, a2_probability):    

    player_scoring_dict = copy.deepcopy(core_player_scoring_dict)
    team_scoring_dict = copy.deepcopy(core_team_scoring_dict)
    game_scoring_dict = copy.deepcopy(core_game_scoring_dict)

    # loop through each game
    for index, row in tqdm(schedule_df.iterrows(), total=schedule_df.shape[0], desc="Generating Game Inferences"):
        if row['Game State'] == 7:
            continue

        # fetch rosters
        home_roster = team_rosters[row['Home Abbreviation']]
        visitor_roster = team_rosters[row['Visiting Abbreviation']]

        # calculate the weighted averages
        home_weighted_avg = np.average(home_roster['Pper1kChunk'], weights=home_roster['ATOI']*home_roster['GPprb'])/1000 * 5/(1+a1_probability+a2_probability)
        visitor_weighted_avg = np.average(visitor_roster['Pper1kChunk'], weights=visitor_roster['ATOI']*visitor_roster['GPprb'])/1000 * 5/(1+a1_probability+a2_probability)
        home_scoring_dotproduct = (home_roster['Gper1kChunk'] * home_roster['ATOI'] * home_roster['GPprb']).sum()
        visitor_scoring_dotproduct = (visitor_roster['Gper1kChunk'] * visitor_roster['ATOI'] * visitor_roster['GPprb']).sum()
        home_assisting_dotproduct = (home_roster['Aper1kChunk'] * home_roster['ATOI'] * home_roster['GPprb']).sum()
        visitor_assisting_dotproduct = (visitor_roster['Aper1kChunk'] * visitor_roster['ATOI'] * visitor_roster['GPprb']).sum()

        # adjust for home ice advantage ###
        home_weighted_avg *= 1.025574015
        visitor_weighted_avg *= 0.9744259847

        # adjust for opponent defence
        home_weighted_avg *= defence_scores[row['Visiting Team']]
        visitor_weighted_avg *= defence_scores[row['Home Team']]

        # compute game level projections
        home_score, visitor_score, home_prob, visitor_prob, overtime = compute_poisson_game_probabilities(home_weighted_avg, visitor_weighted_avg) ### r&d: poisson vs alternative efficient game inference methods

        # update team scoring dict
        team_scoring_dict[row['Home Abbreviation']][0] += home_prob
        team_scoring_dict[row['Home Abbreviation']][1] += visitor_prob - (overtime * visitor_prob)
        team_scoring_dict[row['Home Abbreviation']][2] += overtime * visitor_prob
        team_scoring_dict[row['Home Abbreviation']][3] += home_score
        team_scoring_dict[row['Home Abbreviation']][4] += visitor_score
        team_scoring_dict[row['Visiting Abbreviation']][0] += visitor_prob
        team_scoring_dict[row['Visiting Abbreviation']][1] += home_prob - (overtime * home_prob)
        team_scoring_dict[row['Visiting Abbreviation']][2] += overtime * home_prob
        team_scoring_dict[row['Visiting Abbreviation']][3] += visitor_score
        team_scoring_dict[row['Visiting Abbreviation']][4] += home_score

        # update game scoring dict
        game_scoring_dict[row['GameID']] = [home_score, visitor_score, home_prob, visitor_prob, overtime]

        for home_index, home_row in home_roster.iterrows():
            player_id = home_row['PlayerID']
            player_goal_ratio = (home_row['Gper1kChunk']*home_row['ATOI']*home_row['GPprb'])/home_scoring_dotproduct
            player_assist_ratio = (home_row['Aper1kChunk']*home_row['ATOI']*home_row['GPprb'])/home_assisting_dotproduct * (a1_probability + a2_probability)
            player_scoring_dict[player_id][0] += home_row['GPprb']
            player_scoring_dict[player_id][1] += home_row['ATOI']
            player_scoring_dict[player_id][2] += ((home_row['Gper1kChunk'] / 1000 * 2 * home_row['ATOI'])*0.897806 + (player_goal_ratio * home_weighted_avg * 120)*0.102194) * home_row['GPprb'] ###
            player_scoring_dict[player_id][3] += ((home_row['Aper1kChunk'] / 1000 * 2 * home_row['ATOI'])*0.897806 + (player_assist_ratio * home_weighted_avg * 120)*0.102194) * home_row['GPprb'] ###

        for visitor_index, visitor_row in visitor_roster.iterrows():
            player_id = visitor_row['PlayerID']
            player_goal_ratio = (visitor_row['Gper1kChunk']*visitor_row['ATOI']*visitor_row['GPprb'])/visitor_scoring_dotproduct
            player_assist_ratio = (visitor_row['Aper1kChunk']*visitor_row['ATOI']*visitor_row['GPprb'])/visitor_assisting_dotproduct
            player_scoring_dict[player_id][0] += visitor_row['GPprb']
            player_scoring_dict[player_id][1] += visitor_row['ATOI']
            player_scoring_dict[player_id][2] += ((visitor_row['Gper1kChunk'] / 1000 * 2 * visitor_row['ATOI'])*0.897806 + (player_goal_ratio * visitor_weighted_avg * 120)*0.102194) * visitor_row['GPprb'] ###
            player_scoring_dict[player_id][3] += ((visitor_row['Aper1kChunk'] / 1000 * 2 * visitor_row['ATOI'])*0.897806 + (player_assist_ratio * visitor_weighted_avg * 120)*0.102194) * visitor_row['GPprb'] ###

    return player_scoring_dict, team_scoring_dict, game_scoring_dict

# function to compute poisson probabilities of winning and overtime
def compute_poisson_game_probabilities(home_weighted_avg, visitor_weighted_avg, chunks=120, max_goals=10):
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

    return home_score, visitor_score, home_prob, visitor_prob, overtime

# Generate player uncertainty-based projections via monte carlo engine
def player_monte_carlo_engine(skater_proj_df, core_player_scoring_dict, projection_year, simulations, download_files, verbose):

    # create monte_carlo_player_df
    monte_carlo_player_df = copy.deepcopy(skater_proj_df)
    existing_scoring_dict = copy.deepcopy(core_player_scoring_dict)

    # extract bootstrap_df from CSV
    bootstrap_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Skaters', f'{projection_year}_skater_bootstraps.csv'), index_col=0)
    bootstrap_df['PlayerID'] = bootstrap_df['PlayerID'].astype(int)
    bootstrap_df['Aper1kChunk'] = bootstrap_df['A1per1kChunk'] + bootstrap_df['A2per1kChunk'] # variances are additive when independent; approximate as independent for simplicity
    bootstrap_df.drop(columns=['A1per1kChunk', 'A2per1kChunk'], inplace=True)
    bootstrap_df.rename(columns={'ATOI': 'vATOI', 'GP': 'vGames Played', 'Gper1kChunk': 'vGper1kChunk', 'Aper1kChunk': 'vAper1kChunk'}, inplace=True)

    # convert monte carlo player df to rates
    monte_carlo_player_df['ATOI'] = monte_carlo_player_df['TOI'] / monte_carlo_player_df['Games Played']
    monte_carlo_player_df['Gper1kChunk'] = monte_carlo_player_df['Goals'] / monte_carlo_player_df['TOI'] * 500
    monte_carlo_player_df['Aper1kChunk'] = monte_carlo_player_df['Assists'] / monte_carlo_player_df['TOI'] * 500
    monte_carlo_player_df.drop(columns=['Goals', 'Assists', 'TOI', 'Points'], inplace=True)

    # left join bootstrap_df into monte_carlo_player_df
    monte_carlo_player_df = monte_carlo_player_df.merge(bootstrap_df[['PlayerID', 'vGames Played', 'vATOI', 'vGper1kChunk', 'vAper1kChunk']], on='PlayerID', how='left')

    # loop through each row and simulate
    simulation_results = []
    for index, row in tqdm(monte_carlo_player_df.iterrows(), total=monte_carlo_player_df.shape[0], desc="Simulating Player Seasons"):
        sim_data = {'PlayerID': row['PlayerID']}
        curr_gp = existing_scoring_dict[row['PlayerID']][0]
        curr_toi = existing_scoring_dict[row['PlayerID']][1]
        curr_g = existing_scoring_dict[row['PlayerID']][2]
        curr_a = existing_scoring_dict[row['PlayerID']][3]
        for sim in range(simulations):
            sim_gp = min(max(np.random.normal(row['Games Played'], np.sqrt(row['vGames Played'])), curr_gp), 82)
            if sim_gp == 0: 
                sim_ATOI = 0
                sim_Gper1kChunk = 0
                sim_Aper1kChunk = 0
            else: 
                sim_ATOI = max(np.random.normal(row['ATOI'], np.sqrt(row['vATOI'])), curr_toi/sim_gp)
                sim_Gper1kChunk = max(np.random.normal(row['Gper1kChunk'], np.sqrt(row['vGper1kChunk'])), curr_g/(sim_ATOI*sim_gp/500))
                sim_Aper1kChunk = max(np.random.normal(row['Aper1kChunk'], np.sqrt(row['vAper1kChunk'])), curr_a/(sim_ATOI*sim_gp/500))
            sim_data[f'{sim+1}_goals'] = sim_Gper1kChunk * sim_ATOI * sim_gp / 500
            sim_data[f'{sim+1}_assists'] = sim_Aper1kChunk * sim_ATOI * sim_gp / 500
            sim_data[f'{sim+1}_points'] = sim_data[f'{sim+1}_goals'] + sim_data[f'{sim+1}_assists']
        simulation_results.append(sim_data)

    simulation_df = pd.DataFrame(simulation_results)
    monte_carlo_player_df = pd.concat([monte_carlo_player_df.set_index('PlayerID'), simulation_df.set_index('PlayerID')], axis=1).reset_index()

    ### find a way to efficiently store KDE curves for player cards (IceAnalytics v2). probably will want to store ~100 pts from the KDE curve then re-construct it in React.

    if verbose:
        print('Computing Monte Carlo Player Awards...')

    # art ross calculation
    monte_carlo_player_df['ArtRoss'] = 0
    for sim in range(1, simulations + 1):
        max_points_player = monte_carlo_player_df.loc[monte_carlo_player_df[f'{sim}_points'].idxmax(), 'PlayerID']
        monte_carlo_player_df.loc[monte_carlo_player_df['PlayerID'] == max_points_player, 'ArtRoss'] += 1
    monte_carlo_player_df['ArtRoss'] /= simulations

    # rocket richard calculation
    monte_carlo_player_df['Rocket'] = 0
    for sim in range(1, simulations + 1):
        max_goals_player = monte_carlo_player_df.loc[monte_carlo_player_df[f'{sim}_goals'].idxmax(), 'PlayerID']
        monte_carlo_player_df.loc[monte_carlo_player_df['PlayerID'] == max_goals_player, 'Rocket'] += 1
    monte_carlo_player_df['Rocket'] /= simulations

    if verbose:
        print('Computing Monte Carlo Player Prediction Intervals...')
    
    # for each player, find the 90% prediction interval for goals, assists, and points
    monte_carlo_player_df['Goals_90PI_low'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].quantile(0.05, axis=1)
    monte_carlo_player_df['Goals_90PI_high'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].quantile(0.95, axis=1)
    monte_carlo_player_df['Assists_90PI_low'] = monte_carlo_player_df[[f'{sim}_assists' for sim in range(1, simulations + 1)]].quantile(0.05, axis=1)
    monte_carlo_player_df['Assists_90PI_high'] = monte_carlo_player_df[[f'{sim}_assists' for sim in range(1, simulations + 1)]].quantile(0.95, axis=1)
    monte_carlo_player_df['Points_90PI_low'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].quantile(0.05, axis=1)
    monte_carlo_player_df['Points_90PI_high'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].quantile(0.95, axis=1)

    if verbose:
        print('Computing Monte Carlo Player Statistical Benchmarks...')

    # goal benchmark probabilities
    monte_carlo_player_df['P_10G'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 9.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_20G'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 19.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_30G'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 29.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_40G'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 39.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_50G'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 49.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_60G'] = monte_carlo_player_df[[f'{sim}_goals' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 59.5).sum() / simulations, axis=1)

    # assist benchmark probabilities
    monte_carlo_player_df['P_25A'] = monte_carlo_player_df[[f'{sim}_assists' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 24.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_50A'] = monte_carlo_player_df[[f'{sim}_assists' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 49.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_75A'] = monte_carlo_player_df[[f'{sim}_assists' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 74.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_100A'] = monte_carlo_player_df[[f'{sim}_assists' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 99.5).sum() / simulations, axis=1)

    # point benchmark probabilities
    monte_carlo_player_df['P_50P'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 49.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_75P'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 74.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_100P'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 99.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_125P'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 124.5).sum() / simulations, axis=1)
    monte_carlo_player_df['P_150P'] = monte_carlo_player_df[[f'{sim}_points' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 149.5).sum() / simulations, axis=1)

    # download monte_carlo_player_df to CSV (very large file; contains all player simulation data)
    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), 'test')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        monte_carlo_player_df.to_csv(os.path.join(export_path, 'full_monte_carlo_skater_data.csv'), index=True)
        if verbose:
            print(f'full_monte_carlo_skater_data.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, 'full_monte_carlo_skater_data.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

    # join calculated probabilities to skater_proj_df
    skater_proj_df = skater_proj_df.merge(monte_carlo_player_df[['PlayerID', 'ArtRoss', 'Rocket']], on='PlayerID', how='left')
    skater_proj_df = skater_proj_df.merge(monte_carlo_player_df[['PlayerID', 'Goals_90PI_low', 'Goals_90PI_high', 'Assists_90PI_low', 'Assists_90PI_high', 'Points_90PI_low', 'Points_90PI_high']], on='PlayerID', how='left')
    skater_proj_df = skater_proj_df.merge(monte_carlo_player_df[['PlayerID', 'P_10G', 'P_20G', 'P_30G', 'P_40G', 'P_50G', 'P_60G']], on='PlayerID', how='left')
    skater_proj_df = skater_proj_df.merge(monte_carlo_player_df[['PlayerID', 'P_25A', 'P_50A', 'P_75A', 'P_100A']], on='PlayerID', how='left')
    skater_proj_df = skater_proj_df.merge(monte_carlo_player_df[['PlayerID', 'P_50P', 'P_75P', 'P_100P', 'P_125P', 'P_150P']], on='PlayerID', how='left')

    return skater_proj_df

# Generate team uncertainty-based projections via monte carlo engine
def team_monte_carlo_engine(team_proj_df, core_team_scoring_dict, projection_year, simulations, download_files, verbose):
    
    # create monte_carlo_team_df
    monte_carlo_team_df = copy.deepcopy(team_proj_df)
    existing_scoring_dict = copy.deepcopy(core_team_scoring_dict)

    # remove goal stats from team stat dict
    for team in existing_scoring_dict:
        existing_scoring_dict[team] = existing_scoring_dict[team][:-2]

    # extract schedule_df from CSV, filter out games that have already been played
    schedule_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Projections', str(projection_year), 'Games', f'{projection_year}_game_projections.csv'), index_col=0)
    schedule_df.drop(columns=['DatetimeEST', 'TimeEST'], inplace=True)
    schedule_df = schedule_df[schedule_df['Home Win'] != 'True']
    schedule_df = schedule_df[schedule_df['Home Win'] != 'False']

    # get team divisions
    team_divisions_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'engine_data', 'Team Data', 'divisions.csv'))
    monte_carlo_team_df = monte_carlo_team_df.merge(team_divisions_df[['Abbreviation', 'Division']], on='Abbreviation', how='left')

    # loop through monte carlo simulations
    simulation_results = []
    for sim in tqdm(range(simulations), desc="Monte Carlo Team Simulations"):
        sim_team_scoring_dict = copy.deepcopy(existing_scoring_dict)
        
        # loop through each game
        for index, row in schedule_df.iterrows():
            home_team = row['Home Abbreviation']
            visitor_team = row['Visiting Abbreviation']
            home_prob = float(row['Home Win'])
            visitor_prob = float(row['Visitor Win'])
            ot_prob = float(row['Overtime'])

            home_reg_prob = home_prob * (1 - ot_prob)
            visitor_reg_prob = visitor_prob * (1 - ot_prob)
            home_ot_prob = home_prob * ot_prob
            visitor_ot_prob = visitor_prob * ot_prob

            r = np.random.random()

            # Determine the game outcome based on the random draw
            if r < home_reg_prob:
                sim_team_scoring_dict[home_team][0] += 1
                sim_team_scoring_dict[visitor_team][1] += 1
            elif r < home_reg_prob + visitor_reg_prob:
                sim_team_scoring_dict[visitor_team][0] += 1
                sim_team_scoring_dict[home_team][1] += 1
            elif r < home_reg_prob + visitor_reg_prob + home_ot_prob:
                sim_team_scoring_dict[home_team][0] += 1
                sim_team_scoring_dict[visitor_team][2] += 1
            else:
                sim_team_scoring_dict[visitor_team][0] += 1
                sim_team_scoring_dict[home_team][2] += 1

        # store simulation results
        sim_team_points_dict = {}
        for team in sim_team_scoring_dict:
            sim_team_points_dict[team] = sim_team_scoring_dict[team][0]*2 + sim_team_scoring_dict[team][2]
        simulation_results.append(sim_team_points_dict)

    simulations_df = pd.DataFrame(simulation_results).T
    simulations_df.columns = [f'{sim+1}_Pts' for sim in range(simulations)]

    # join to monte_carlo_team_df
    monte_carlo_team_df = monte_carlo_team_df.merge(simulations_df, left_on='Abbreviation', right_index=True, how='left')

    if verbose:
        print('Computing Monte Carlo Team Standings Odds...')

    # presidents trophy calculation
    monte_carlo_team_df['Presidents'] = 0
    for sim in tqdm(range(1, simulations + 1), desc="Monte Carlo Presidents Trophy Odds"):
        max_points_team = monte_carlo_team_df.loc[monte_carlo_team_df[f'{sim}_Pts'].idxmax(), 'Abbreviation']
        monte_carlo_team_df.loc[monte_carlo_team_df['Abbreviation'] == max_points_team, 'Presidents'] += 1
    monte_carlo_team_df['Presidents'] /= simulations

    # playoff odds calculation
    monte_carlo_team_df['Playoffs'] = 0
    for sim in tqdm(range(1, simulations + 1), desc="Monte Carlo Playoff Odds"):

        # get top 3 teams from each division
        division_teams = monte_carlo_team_df[monte_carlo_team_df['Division'] == 'Atlantic'].nlargest(3, f'{sim}_Pts')['Abbreviation'].tolist()
        division_teams += monte_carlo_team_df[monte_carlo_team_df['Division'] == 'Metropolitan'].nlargest(3, f'{sim}_Pts')['Abbreviation'].tolist()
        division_teams += monte_carlo_team_df[monte_carlo_team_df['Division'] == 'Central'].nlargest(3, f'{sim}_Pts')['Abbreviation'].tolist()
        division_teams += monte_carlo_team_df[monte_carlo_team_df['Division'] == 'Pacific'].nlargest(3, f'{sim}_Pts')['Abbreviation'].tolist()
        
        # get top 2 wildcard teams from each conference
        remaining_teams = monte_carlo_team_df[~monte_carlo_team_df['Abbreviation'].isin(division_teams)]
        wildcard_teams = remaining_teams[(remaining_teams['Division'] == 'Atlantic') | (remaining_teams['Division'] == 'Metropolitan')].nlargest(2, f'{sim}_Pts')['Abbreviation'].tolist()
        wildcard_teams += remaining_teams[(remaining_teams['Division'] == 'Central') | (remaining_teams['Division'] == 'Pacific')].nlargest(2, f'{sim}_Pts')['Abbreviation'].tolist()

        # add to playoff odds
        for team in division_teams + wildcard_teams:
            monte_carlo_team_df.loc[monte_carlo_team_df['Abbreviation'] == team, 'Playoffs'] += 1

    monte_carlo_team_df['Playoffs'] /= simulations

    if verbose:
        print('Computing Monte Carlo Team Prediction Intervals...')

    # for each team, find the 90% prediction interval for points
    monte_carlo_team_df['Pts_90PI_low'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].quantile(0.05, axis=1)
    monte_carlo_team_df['Pts_90PI_high'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].quantile(0.95, axis=1)

    if verbose:
        print('Computing Monte Carlo Team Statistical Benchmarks...')

    # goal benchmark probabilities
    monte_carlo_team_df['P_60Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 59.5).sum() / simulations, axis=1)
    monte_carlo_team_df['P_70Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 69.5).sum() / simulations, axis=1)
    monte_carlo_team_df['P_80Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 79.5).sum() / simulations, axis=1)
    monte_carlo_team_df['P_90Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 89.5).sum() / simulations, axis=1)
    monte_carlo_team_df['P_100Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 99.5).sum() / simulations, axis=1)
    monte_carlo_team_df['P_110Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 109.5).sum() / simulations, axis=1)
    monte_carlo_team_df['P_120Pts'] = monte_carlo_team_df[[f'{sim}_Pts' for sim in range(1, simulations + 1)]].apply(lambda x: (x >= 119.5).sum() / simulations, axis=1)

    # download monte_carlo_team_df to CSV (very large file; contains all player simulation data)
    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), 'test')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        monte_carlo_team_df.to_csv(os.path.join(export_path, 'full_monte_carlo_team_data.csv'), index=True)
        if verbose:
            print(f'full_monte_carlo_team_data.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, 'full_monte_carlo_team_data.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

    # join to team_proj_df
    team_proj_df = team_proj_df.merge(monte_carlo_team_df[['Abbreviation', 'Presidents', 'Playoffs']], on='Abbreviation', how='left')
    team_proj_df = team_proj_df.merge(monte_carlo_team_df[['Abbreviation', 'Pts_90PI_low', 'Pts_90PI_high']], on='Abbreviation', how='left')
    team_proj_df = team_proj_df.merge(monte_carlo_team_df[['Abbreviation', 'P_60Pts', 'P_70Pts', 'P_80Pts', 'P_90Pts', 'P_100Pts', 'P_110Pts', 'P_120Pts']], on='Abbreviation', how='left')

    return team_proj_df