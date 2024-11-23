from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

# Load the saved model and scaler
model = joblib.load("C:/Users/Nitin/loan/xgboost_loan_model.pkl")
scaler = joblib.load("C:/Users/Nitin/loan/scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    # Render the HTML form for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Invalid input, JSON body is required"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {str(e)}"}), 400

    # Ensure the input data contains all required features
    features = [
        "person_age", "person_gender", "person_education", "person_income",
        "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "previous_loan_defaults_on_file"
    ]
    try:
        # Validate and extract features
        input_array = []
        for feature in features:
            if feature not in input_data:
                return jsonify({"error": f"Missing feature in input: {feature}"}), 400
            input_array.append(float(input_data[feature]))  # Ensure numerical type
        input_array = np.array(input_array).reshape(1, -1)

        # Scale the input data
        scaled_input = scaler.transform(input_array)

        # Get the prediction from the model
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)[0][1]

        # Convert float32 to Python float for JSON compatibility
        probability = float(probability)

        # Prepare response
        response = {
            "loan_status": "Approved" if prediction[0] == 1 else "Rejected",
            "approval_probability": probability
        }
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Suppress favicon.ico requests

if __name__ == '__main__':
    app.run(debug=True)
