import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the model parameters
def load_model(filename='model.npz'):
    npzfile = np.load(filename)
    W1 = npzfile['W1']
    b1 = npzfile['b1']
    W2 = npzfile['W2']
    b2 = npzfile['b2']
    return W1, b1, W2, b2

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to make predictions
def predict(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2 > 0.5

# Function to standardize user input
def standardize_input(user_input, scaler):
    user_input = np.array(user_input).reshape(1, -1)
    return scaler.transform(user_input)

# Load the dataset to fit the scaler
data = pd.read_csv('engine_data.csv')
X = data.drop('Engine Condition', axis=1).values

# Fit the scaler on the dataset
scaler = StandardScaler().fit(X)

# Load the model parameters
W1, b1, W2, b2 = load_model()

# Take user input for the fields
print("Please enter the following details:")
engine_rpm = float(input("Engine RPM: "))
lub_oil_pressure = float(input("Lub Oil Pressure: "))
fuel_pressure = float(input("Fuel Pressure: "))
coolant_pressure = float(input("Coolant Pressure: "))
lub_oil_temp = float(input("Lub Oil Temperature: "))
coolant_temp = float(input("Coolant Temperature: "))

# Collect the input into a list
user_input = [engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]

# Standardize the user input
user_input_standardized = standardize_input(user_input, scaler)

# Make prediction
prediction = predict(user_input_standardized, W1, b1, W2, b2)

# Display the result
if prediction:
    print("Risky engine condition detected! Please take necessary precautions.")
else:
    print("Engine condition is good.")
