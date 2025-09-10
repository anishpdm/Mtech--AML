from flask import Flask, request, jsonify
import pandas as pd
import sklearn.linear_model as lm
import numpy as np
from flask_cors import CORS # Import CORS


# Load Dataset and Train Model
dataset = pd.read_csv("data.csv")
x = dataset[["population"]]
y = dataset[["profit"]]
model = lm.LinearRegression()
model.fit(x, y)

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()
        population_value = data['population']

        # Ensure the input is a valid number
        if not isinstance(population_value, (int, float)):
            return jsonify({"error": "Invalid input. 'population' must be a number."}), 400

        # Create a DataFrame for the single prediction
        input_data = pd.DataFrame([[population_value]], columns=["population"])
        
        # Make the prediction
        result = model.predict(input_data)
        predicted_profit = round(result[0][0], 2)

        return jsonify({"predicted_profit": predicted_profit})

    except KeyError:
        return jsonify({"error": "Missing 'population' key in the request body."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # You can change the port as needed
    app.run(debug=True, port=5001)