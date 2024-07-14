import numpy as np
import pickle

# Load the saved model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

# Load the model
loaded_model = load_model('ML DRIVER\model.npz')

# Function to preprocess user inputs
def preprocess_inputs(inputs):
    # Assuming inputs is a dictionary with keys: 'Engine rpm', 'Lub oil', 'Fuel pressure', 'Coolant pressure', 'Lub oil temp', 'Coolant temp', and 'Temp_diff'
    # Preprocessing steps may include scaling/normalizing inputs
    # For simplicity, we'll just convert inputs to a numpy array
    return np.array([inputs['Engine rpm'], inputs['Lub oil'], inputs['Fuel pressure'], inputs['Coolant pressure'], inputs['Lub oil temp'], inputs['Coolant temp'], inputs['Temp_diff']])

# Function to predict driver behavior
def predict_driver_behavior(model, inputs):
    # Preprocess inputs
    processed_inputs = preprocess_inputs(inputs)
    # Forward pass through the model
    z1 = np.dot(processed_inputs, model['weights1']) + model['bias1']
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(a1, model['weights2']) + model['bias2']
    output = 1 / (1 + np.exp(-z2))
    # Analyze the output and provide driver behavior analysis
    if output > 0.5:
        return "Warning: Risky driving behavior detected!"
    else:
        return "No risky driving behavior detected."

# Take user inputs for all fields
user_inputs = {
    'Engine rpm': float(input("Enter Engine rpm: ")),
    'Lub oil': float(input("Enter Lub oil: ")),
    'Fuel pressure': float(input("Enter Fuel pressure: ")),
    'Coolant pressure': float(input("Enter Coolant pressure: ")),
    'Lub oil temp': float(input("Enter Lub oil temp: ")),
    'Coolant temp': float(input("Enter Coolant temp: ")),
    'Temp_diff': float(input("Enter Temp_diff: "))
}

# Predict driver behavior based on user inputs
result = predict_driver_behavior(loaded_model, user_inputs)
print(result)
