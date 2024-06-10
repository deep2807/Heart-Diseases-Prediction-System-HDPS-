from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File is not a CSV'}), 400

    # Save the uploaded CSV file
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Load the CSV file
    data = pd.read_csv(filepath)

    # Check for required columns
    required_columns = ['Age','Sex','Chest pain type','BP','Cholesterol','FBS over 120','EKG results','Max HR','Exercise angina','ST depression','Slope of ST','Number of vessels fluro','Thallium','Heart Disease']
    if not all(col in data.columns for col in required_columns):
        return jsonify({'error': 'CSV file does not contain the required columns'}), 400

    # Separate the features (X) and target variable (y)
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Randomly select an input from the test set for demonstration
    random_input = X_test.iloc[0].to_dict()

    # Make prediction on the randomly selected input
    new_prediction = model.predict([list(random_input.values())])
    prediction_result = "The person is likely to have heart disease." if new_prediction[0] == 1 else "The person is not likely to have heart disease."

    response = {
        'accuracy': accuracy,
        'prediction': prediction_result,
        'random_input': random_input
    }
    return jsonify(response)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
