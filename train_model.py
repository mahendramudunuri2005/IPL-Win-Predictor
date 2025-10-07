import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r'C:\Users\hp\Documents\IPL-Win-Predictor-ML-Project-main\IPL  Predictor\ipl_data.csv')

# Calculate 'wickets' from 'wickets_out'
df['wickets'] = 10 - df['wickets_out']

# Rename 'target' to 'total_runs_x'
df.rename(columns={'target': 'total_runs_x'}, inplace=True)

# Select features for training (match the model's expected input)
X = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']]

# Encode categorical variables
encoder = LabelEncoder()
for column in ['batting_team', 'bowling_team', 'city']:
    X[column] = encoder.fit_transform(X[column])

# Prepare target variable (assuming you have a 'winning_team' column to predict)
# For simplicity, let's assume we want to predict if the batting team wins (1) or loses (0)
df['result'] = df.apply(lambda row: 1 if row['batting_team'] == row['winning_team'] else 0, axis=1)
y = df['result']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")