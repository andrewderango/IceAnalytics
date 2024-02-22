import pandas as pd
import numpy as np

# Create a MultiIndex
index = pd.MultiIndex.from_product([[437289, 43278, 432378, 34283], ['Team1', 'Team2', 'Team3', 'Team4'], [_ for _ in range(5)]], names=['PlayerID', 'Team', 'Simulation'])

# Create a DataFrame with a MultiIndex
df = pd.DataFrame(index=index, columns=['Goals', 'Assists'])

# Set values in the DataFrame
for idx in df.index:
    df.loc[idx] = [np.random.randint(0, 10), np.random.randint(0, 10)]

print(df)