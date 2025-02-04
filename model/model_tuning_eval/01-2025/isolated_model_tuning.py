import optuna
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('model/model_tuning_eval/01-2025/offence_train_data.csv')

# Define the feature columns and target variable
feature_cols = ['Y-3 Gper1kChunk', 'Y-2 Gper1kChunk', 'Y-1 Gper1kChunk', 'Y-3 xGper1kChunk', 'Y-2 xGper1kChunk', 'Y-1 xGper1kChunk', 'Y-3 SHper1kChunk', 'Y-2 SHper1kChunk', 'Y-1 SHper1kChunk', 'Y-3 iCFper1kChunk', 'Y-2 iCFper1kChunk', 'Y-1 iCFper1kChunk', 'Y-3 RAper1kChunk', 'Y-2 RAper1kChunk', 'Y-1 RAper1kChunk', 'Y-0 Age', 'Position']
target = 'Y-0 Gper1kChunk'

# Select features and target
X = df[feature_cols]
y = df[target]

# Preprocess: standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter optimization space for SVR
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
    
    # Create the SVR model with suggested parameters
    svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    
    # Perform 10-fold cross-validation and return the average MSE or MAE
    mse = cross_val_score(svr, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    mae = cross_val_score(svr, X_scaled, y, cv=10, scoring='neg_mean_absolute_error')
    
    # We can either use MSE or MAE; let's use MSE in this case
    return mse.mean()

# Create the study and optimize
study = optuna.create_study(direction='minimize')  # Minimize MSE
study.optimize(objective, n_trials=50)

# Print the best trial and corresponding hyperparameters
print(f"Best trial: {study.best_trial.params}")
print(f"Best MSE: {-study.best_trial.value}")