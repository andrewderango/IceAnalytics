import pandas as pd
import numpy as np

# Function to generate a DataFrame for each simulation
def simulate_season(player_data, simulation_num):
    # Add the simulation number to the 'Goals' column for each player
    player_data['Goals'] = np.array(player_data['Goals']) + simulation_num
    df = pd.DataFrame(player_data, columns=['PlayerID', 'Player', 'Position', 'Team', 'Age', 'Games Played', 'Goals', 'Assists', 'Points'])
    return df

# Sample data for demonstration
player_data = {
    'PlayerID': [1, 2, 3],
    'Player': ['Player1', 'Player2', 'Player3'],
    'Position': ['Forward', 'Midfielder', 'Defender'],
    'Team': ['TeamA', 'TeamB', 'TeamC'],
    'Age': [25, 28, 30],
    'Games Played': [30, 30, 30],
    'Goals': [20, 10, 5],
    'Assists': [10, 5, 3],
    'Points': [30, 15, 8]
}

# Create an empty DataFrame
simulations_df = pd.DataFrame(columns=['Simulation', 'PlayerID', 'Player', 'Position', 'Team', 'Age', 'Games Played', 'Goals', 'Assists', 'Points'])

# Number of simulations
num_simulations = 5

# Perform simulations
for i in range(1, num_simulations + 1):
    # Generate player data for simulation i
    simulated_data = simulate_season(player_data.copy(), i)
    
    # Set simulation number as a column
    simulated_data['Simulation'] = i

    print(simulated_data)
    
    # Append to the outer DataFrame
    simulations_df = pd.concat([simulations_df, simulated_data])

# Set 'Simulation', 'PlayerID', and 'Player' as indices after all data has been appended
simulations_df = simulations_df.set_index(['Simulation', 'PlayerID', 'Player', 'Position', 'Team', 'Age'])

print(simulations_df)

# Filter data for 'PlayerID' == 1 and calculate the average for each simulation
player1_data = simulations_df.xs(1, level='PlayerID')
average_goals = player1_data['Goals'].groupby(level='Simulation').mean()
print(average_goals)

# export simulations_df
# simulations_df.to_csv('simulations.csv')