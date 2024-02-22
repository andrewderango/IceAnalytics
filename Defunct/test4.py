import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame(columns=['PlayerID', 'Goals', 'Assists'])

# Fill with sample data
data = [
    {'PlayerID': 1, 'Goals': 10, 'Assists': 5},
    {'PlayerID': 2, 'Goals': 15, 'Assists': 7},
    {'PlayerID': 3, 'Goals': 12, 'Assists': 8},
    {'PlayerID': 4, 'Goals': 17, 'Assists': 9},
    {'PlayerID': 5, 'Goals': 14, 'Assists': 6},
]
df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

# Set the index
df.set_index('PlayerID', inplace=True)

# add simulation number to index
df = df.reindex(pd.MultiIndex.from_product([df.index, [_ for _ in range(5)]], names=['PlayerID', 'Simulation']))

print(df)