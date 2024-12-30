import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import anderson, norm
from statsmodels.graphics.gofplots import qqplot

# read the CSV file
df = pd.read_csv('test.csv')

# extract point projection columns
point_columns = [col for col in df.columns if '_points' in col]

# filter for a specific player
player = 'Connor McDavid'
player_df = df[df['Player'] == player].copy()
print(player_df)

# get the point projections for the specific player
list = player_df[point_columns].values.tolist()[0]
print(list)
print(len(list))
print(min(list))

# Convert list to NumPy array
array = np.array(list)

# Perform hypothesis test to see if the distribution is normal (Anderson-Darling Test)
result = anderson(array)
print(result)

# Step 5: Plot the KDE distribution for the specific player
sns.kdeplot(array, fill=True, label='KDE')

# Calculate mean and standard deviation
mean = np.mean(array)
std_dev = np.std(array)

# Generate values for the normal distribution
x = np.linspace(min(array), max(array), 100)
normal_dist = norm.pdf(x, mean, std_dev)

# Plot the normal distribution
plt.plot(x, normal_dist, label='Normal Distribution', linestyle='--')
plt.title(f'Point Projections for {player}')
plt.xlabel('Points')
plt.ylabel('Density')
plt.legend()
plt.show()