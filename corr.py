import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('engine_data.csv')

# Features and labels
X = data.drop('Engine Condition', axis=1).values
y = data['Engine Condition'].values.reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load the model parameters
def load_model(filename='model.npz'):
    npzfile = np.load(filename)
    W1 = npzfile['W1']
    b1 = npzfile['b1']
    W2 = npzfile['W2']
    b2 = npzfile['b2']
    return W1, b1, W2, b2

W1, b1, W2, b2 = load_model()

# Prediction function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2 > 0.5

def provide_feedback(predictions):
    feedback = np.where(predictions, "Risky engine condition detected! Please take necessary precautions.", "Engine condition is good.")
    return feedback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        final_features = scaler.transform(final_features)
        prediction = predict(final_features, W1, b1, W2, b2)
        feedback = provide_feedback(prediction)
        return render_template('index.html', prediction_text=feedback[0])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
