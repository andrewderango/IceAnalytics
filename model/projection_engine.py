import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_training import *
from scraper_functions import *
from scipy.stats import poisson

def run_projection_engine(projection_year, simulations, download_files, verbose):

    # load dfs
    schedule_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', f'{projection_year-1}-{projection_year}_game_schedule.csv'), index_col=0)
    schedule_df['Home Win'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else row['Home Score'] > row['Visiting Score'], axis=1)
    schedule_df['Visitor Win'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else row['Home Score'] < row['Visiting Score'], axis=1)
    schedule_df['Overtime'] = schedule_df.apply(lambda row: None if row['Game State'] == 1 else (True if row['Period'] > 3 else False), axis=1)
    metaprojection_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters', f'{projection_year}_skater_metaprojections.csv'), index_col=0)
    metaprojection_df['Aper1kChunk'] = metaprojection_df['A1per1kChunk'] + metaprojection_df['A2per1kChunk']
    metaprojection_df['Pper1kChunk'] = metaprojection_df['Gper1kChunk'] + metaprojection_df['Aper1kChunk']
    teams_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Team Data', 'nhlapi_team_data.csv'), index_col=0)
    team_metaproj_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Teams', f'{projection_year}_team_projections.csv'), index_col=0)
    existing_skater_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Skater Data', f'{projection_year-1}-{projection_year}_skater_data.csv'))
    existing_skater_df['Assists'] = existing_skater_df['First Assists'] + existing_skater_df['Second Assists']
    # existing_skater_df['ATOI'] = (existing_skater_df['TOI'].fillna(0) / existing_skater_df['GP'].fillna(0)).fillna(0)
    existing_team_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Historical Team Data', f'{projection_year-1}-{projection_year}_team_data.csv'))
    existing_team_df = existing_team_df.rename(columns={'W': 'Wins', 'L': 'Losses', 'GA': 'Goals Against', 'GF': 'Goals For'})

    # add team abbreviations to schedule
    schedule_df = schedule_df.merge(teams_df, left_on='Home Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Home Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID'])
    schedule_df = schedule_df.merge(teams_df, left_on='Visiting Team', right_on='Team Name', how='left')
    schedule_df = schedule_df.rename(columns={'Abbreviation': 'Visiting Abbreviation'})
    schedule_df = schedule_df.drop(columns=['Team Name', 'TeamID', 'Active_x', 'Active_y'])

    # configure skater projection df
    skater_proj_df = metaprojection_df[['PlayerID', 'Player', 'Position', 'Team', 'Age']].copy()
    skater_proj_df = skater_proj_df.assign(Games_Played=0, Goals=0, Assists=0)
    skater_proj_df.rename(columns={'Games_Played': 'Games Played'}, inplace=True)
    
    # configure team projection df
    team_proj_df = teams_df[['Team Name', 'Abbreviation']].copy()
    active_teams = pd.concat([schedule_df['Home Team'], schedule_df['Visiting Team']])
    team_proj_df = team_proj_df[team_proj_df['Team Name'].isin(active_teams)]
    team_proj_df = team_proj_df.assign(Wins=0, Losses=0, OTL=0, Goals_For=0, Goals_Against=0)
    team_proj_df.rename(columns={'Team Name': 'Team', 'Goals_For': 'Goals For', 'Goals_Against': 'Goals Against'}, inplace=True)

    # configure games projection df
    game_proj_df = schedule_df[['GameID', 'Time (EST)', 'Home Team', 'Home Abbreviation', 'Home Score', 'Visiting Team', 'Visiting Abbreviation', 'Visiting Score']].copy()
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
        team_roster = metaprojection_df[metaprojection_df['Team'] == team_abbreviation].copy()
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
    
    if download_files:
        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Skaters')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        skater_proj_df.to_csv(os.path.join(export_path, f'{projection_year}_skater_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_skater_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_skater_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Teams')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        team_proj_df.to_csv(os.path.join(export_path, f'{projection_year}_team_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_team_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_team_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

        export_path = os.path.join(os.path.dirname(__file__), '..', 'Sim Engine Data', 'Projections', str(projection_year), 'Games')
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        game_proj_df.to_csv(os.path.join(export_path, f'{projection_year}_game_projections.csv'), index=True)
        if verbose:
            print(f'{projection_year}_game_projections.csv has been downloaded to the following directory: {export_path}')
            file_size = os.path.getsize(os.path.join(export_path, f'{projection_year}_game_projections.csv'))/1000000
            print(f'\tFile size: {file_size} MB')

def generate_game_inferences(core_player_scoring_dict, core_team_scoring_dict, core_game_scoring_dict, schedule_df, team_rosters, defence_scores, a1_probability, a2_probability):    

    player_scoring_dict = core_player_scoring_dict.copy()
    team_scoring_dict = core_team_scoring_dict.copy()
    game_scoring_dict = core_game_scoring_dict.copy()

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
        home_score, visitor_score, home_prob, visitor_prob, overtime = compute_poisson_probabilities(home_weighted_avg, visitor_weighted_avg) ### r&d: poisson vs alternative efficient game inference methods

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
            player_scoring_dict[player_id][2] += ((home_row['Gper1kChunk'] / 1000 * 2 * home_row['ATOI'])*0.897806 + (player_goal_ratio * home_weighted_avg * 120)*0.102194) ###
            player_scoring_dict[player_id][3] += ((home_row['Aper1kChunk'] / 1000 * 2 * home_row['ATOI'])*0.897806 + (player_assist_ratio * home_weighted_avg * 120)*0.102194) ###

        for visitor_index, visitor_row in visitor_roster.iterrows():
            player_id = visitor_row['PlayerID']
            player_goal_ratio = (visitor_row['Gper1kChunk']*visitor_row['ATOI']*visitor_row['GPprb'])/visitor_scoring_dotproduct
            player_assist_ratio = (visitor_row['Aper1kChunk']*visitor_row['ATOI']*visitor_row['GPprb'])/visitor_assisting_dotproduct
            player_scoring_dict[player_id][0] += visitor_row['GPprb']
            player_scoring_dict[player_id][1] += visitor_row['ATOI']
            player_scoring_dict[player_id][2] += ((visitor_row['Gper1kChunk'] / 1000 * 2 * visitor_row['ATOI'])*0.897806 + (player_goal_ratio * visitor_weighted_avg * 120)*0.102194) ###
            player_scoring_dict[player_id][3] += ((visitor_row['Aper1kChunk'] / 1000 * 2 * visitor_row['ATOI'])*0.897806 + (player_assist_ratio * visitor_weighted_avg * 120)*0.102194) ###

    return player_scoring_dict, team_scoring_dict, game_scoring_dict

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

    return home_score, visitor_score, home_prob, visitor_prob, overtime