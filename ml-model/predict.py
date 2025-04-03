import sys
import json
import joblib

# Load the trained model
model = joblib.load('models/mental_health_model.pkl')

# Read input data from command-line arguments
input_data = json.loads(sys.argv[1])

# Convert input data to a 2D array (required by scikit-learn)
input_array = [list(input_data.values())]

# Make prediction
prediction = model.predict(input_array).tolist()

# Return the prediction as JSON
print(json.dumps(prediction[0]))