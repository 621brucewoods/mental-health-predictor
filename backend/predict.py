import sys
import json
import joblib
import numpy as np
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore")

try:
    # Load model (ensure this path is correct)
    model = joblib.load('./models/mental_health_model.pkl')
    
    # Parse input data
    input_data = json.loads(sys.argv[1])
    
    # Convert to numpy array in correct order
    feature_order = [
        'Age', 'Number of Children', 'Physical Activity Level',
        'Employment Status', 'Income', 'Alcohol Consumption',
        'Dietary Habits', 'Sleep Patterns', 'History of Mental Illness',
        'History of Substance Abuse', 'Family History of Depression',
        'Chronic Medical Conditions', 'Marital_Single', 'Marital_Widowed',
        'Education_Bachelor\'s Degree', 'Education_High School',
        'Education_Master\'s Degree', 'Education_PhD',
        'Smoking_Former', 'Smoking_Non-smoker'
    ]
    
    features = np.array([[float(input_data[col]) for col in feature_order]])
    
    # Predict
    prediction = model.predict(features)[0]
    print(int(prediction))  # Ensure integer output for Node.js

except Exception as e:
    print(f"ERROR: {str(e)}", file=sys.stderr)
    sys.exit(1)