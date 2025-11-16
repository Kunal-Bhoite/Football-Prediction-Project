from flask import Flask, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    try:
        # Execute the Python file and capture its output
        output = subprocess.check_output(['python', 'C:\\Users\\Kunal Bhoite\\OneDrive\\Desktop\\Football project\\new football wining team prediction 1 (2)\\new football wining team prediction 1\\xboost.py'])
        # Convert byte-like output to string
        output_str = output.decode('utf-8')
        return render_template('result.html', output=output_str)
    except Exception as e:
        error_message = f"Error executing prediction: {e}"
        return render_template('error.html', error=error_message)

##############################

import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import LabelEncoder


# Load the trained XGBoost model
model = joblib.load('xgboost_model3.joblib')

# Load the dataset and get unique team names
data = pd.read_csv('updated_dataset1.csv')  # Replace 'your_dataset_path.csv' with the actual path to your dataset file
team_names = pd.concat([data['home_team'], data['away_team']]).unique()
team_name_to_id = {name: i for i, name in enumerate(team_names)}

# Convert 'month' column to numerical representation using a mapping
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_to_num = {month: num + 1 for num, month in enumerate(months)}

# Initialize LabelEncoder for categorical variables
label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([data['tournament'], data['city'], data['country'], data['Continent']]))

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    try:
        # Get the form data from the request
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        tournament = request.form['tournament']
        city = request.form['city']
        country = request.form['country']
        neutral = int(request.form['neutral'])
        year = int(request.form['year'])
        month = request.form['month']
        day = int(request.form['day'])
        continent = request.form['continent']
        max_temp = int(request.form['max_temp'])
        min_temp = int(request.form['min_temp'])
        home_team_rank = int(request.form['home_team_rank'])
        away_team_rank = int(request.form['away_team_rank'])

        # Create a DataFrame from the form data
        data = pd.DataFrame({
            'home_team': [home_team],
            'away_team': [away_team],
            'tournament': [tournament],
            'city': [city],
            'country': [country],
            'neutral': [neutral],
            'year': [year],
            'month': [month],
            'day': [day],
            'Continent': [continent],
            'MaximumTemp_Month': [max_temp],
            'MinimumTemp_Month': [min_temp],
            'HomeTeamRank': [home_team_rank],
            'AwayTeamRank': [away_team_rank]
        })

        # Convert categorical variables to numerical using label encoding
        data['tournament'] = label_encoder.transform(data['tournament'])
        data['city'] = label_encoder.transform(data['city'])
        data['country'] = label_encoder.transform(data['country'])
        data['Continent'] = label_encoder.transform(data['Continent'])

        # Convert 'neutral' column to numerical (TRUE: 1, FALSE: 0)
        data['neutral'] = data['neutral'].astype(int)

        # Map team names to numerical representations
        data['home_team'] = team_name_to_id[home_team]
        data['away_team'] = team_name_to_id[away_team]

        # Convert 'month' column to numerical representation using a mapping
        data['month'] = month_to_num[month]

        # Make predictions on the new data
        prediction = model.predict(data)[0]
        print(prediction)

        # Convert the prediction to the corresponding match result
        match_result = 'Away Team Wins' if prediction == 0 else 'Home Team Wins' if prediction ==  1 else 'Draw'
        '''if prediction == 0:
            match_result = 'Home Team Wins'
        elif prediction == 1:
            match_result = 'Away Team wins'
        else:
            match_result = 'Draw'''

        return render_template('predict1.html', prediction=match_result)

    except Exception as e:
        return render_template('predict1.html', error=str(e))



if __name__ == '__main__':
    app.run(debug=True)
