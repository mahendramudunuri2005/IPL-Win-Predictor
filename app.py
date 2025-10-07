from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


# Load trained model with error handling
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except (EOFError, FileNotFoundError):
    model = None
    print("Error: model.pkl is either missing or corrupted. Retrain the model!")

# Define teams and cities
teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Punjab Kings', 'Delhi Capitals','Sunrisers Hyderabad','Gujarat Titans','Lucknow Super Giants',
         'Rajasthan Royals' ]
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi', 'Chennai']

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", teams=teams, cities=cities)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return "Model is missing or corrupted. Please retrain and save a valid model.pkl."

    try:
        batting_team = request.form["batting_team"]
        bowling_team = request.form["bowling_team"]
        selected_city = request.form["selected_city"]
        target = int(request.form["target"])
        score = int(request.form["score"])
        overs = float(request.form["overs"])
        wickets_out = int(request.form["wickets_out"])

        # Validate Input
        if overs > 20 or score > target or wickets_out > 10:
            return "Invalid input values. Please enter realistic match conditions."

        # Feature Calculations
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets_out
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_data = pd.DataFrame({
            'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
            'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
            'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]
        })

        # Encode categorical features
        encoder = LabelEncoder()
        for col in ['batting_team', 'bowling_team', 'city']:
            input_data[col] = encoder.fit_transform(input_data[col])

        # Predict Win Probabilities
        result = model.predict_proba(input_data)
        win_prob = round(result[0][1] * 100, 2)
        loss_prob = round(result[0][0] * 100, 2)

        return render_template("result.html", win_prob=win_prob, loss_prob=loss_prob,
                               batting_team=batting_team, bowling_team=bowling_team)
    except Exception as e:
        return f"Error processing prediction: {str(e)}"

if __name__ == "__main__":
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))
