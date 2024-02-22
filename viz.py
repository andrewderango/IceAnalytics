import os
import pandas as pd
from urllib.request import urlopen
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

player_id = 8478402
monte_carlo_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters', f'2024_skater_monte_carlo_projections.csv'), index_col=0)
aggregated_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Projections', 'Skaters', f'2024_skater_aggregated_projections.csv'), index_col=0)
player_bios = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Player Bios', 'Skaters', 'skater_bios.csv'), index_col=0)

print(monte_carlo_df)
print(aggregated_df)
print(player_bios)

# print player name, position, team, age from aggregated_df given player_id
player_name = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Player'].values[0]
player_position = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Position'].values[0]
player_team = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Team'].values[0]
player_age = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Age'].values[0]
print(f"Player: {player_name}, Position: {player_position}, Team: {player_team}, Age: {player_age}")

# get player projected stats from aggregated_df
player_gp = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Games Played'].values[0]
player_g = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Goals'].values[0]
player_a = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Assists'].values[0]
player_p = aggregated_df.loc[aggregated_df['PlayerID'] == player_id, 'Points'].values[0]

# headshot and team logo from player_bios
player_headshot = player_bios.loc[player_bios['PlayerID'] == player_id, 'Headshot'].values[0]
team_logo = player_bios.loc[player_bios['PlayerID'] == player_id, 'Team Logo'].values[0]
print(f"Headshot: {player_headshot}")
print(f"Team Logo: {team_logo}")

# Create a new figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Add title to the figure
fig.suptitle('Player Information', fontsize=16)

# Add text to the figure
for ax in axs:
    ax.axis('off')
axs[0].text(0.5, 0.9, f"Player: {player_name}", ha='center', va='center', size=16, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
axs[0].text(0.5, 0.8, f"Position: {player_position}", ha='center', va='center', size=16, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
axs[0].text(0.5, 0.7, f"Team: {player_team}", ha='center', va='center', size=16, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
axs[0].text(0.5, 0.6, f"Age: {player_age}", ha='center', va='center', size=16, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))

box_properties = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5)

axs[0].text(0.1, 0.5, f"GP: {player_gp}", ha='center', va='center', size=14, bbox=box_properties)
axs[0].text(0.3, 0.5, f"G: {player_g}", ha='center', va='center', size=14, bbox=box_properties)
axs[0].text(0.5, 0.5, f"A: {player_a}", ha='center', va='center', size=14, bbox=box_properties)
axs[0].text(0.7, 0.5, f"P: {player_p}", ha='center', va='center', size=14, bbox=box_properties)

# URL of the image
url = 'https://assets.nhle.com/mugs/nhl/20232024/EDM/8478402.png'

# Open the URL and read the image
with urlopen(url) as url:
    s = url.read()
    img = mpimg.imread(BytesIO(s), format='PNG')

# Add border around the player headshot image
axs[1].imshow(img)
axs[1].axis('off')  # to hide the axis
axs[1].set_title('Player Headshot', fontsize=16)
axs[1].set_aspect('equal')
axs[1].spines['top'].set_visible(True)
axs[1].spines['right'].set_visible(True)
axs[1].spines['bottom'].set_visible(True)
axs[1].spines['left'].set_visible(True)

plt.tight_layout()
plt.show()