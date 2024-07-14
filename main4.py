import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Step 1: Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

data = load_data('engine_data.csv')

# Step 2: Data Preprocessing
def preprocess_data(df):
    df = df.dropna()  # Drop rows with missing values
    return df

preprocessed_data = preprocess_data(data)

# Step 3: Feature Engineering
def feature_engineering(df):
    df['Temp_diff'] = df['Coolant temp'] - df['lub oil temp']
    return df[['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp', 'Temp_diff']]

features = feature_engineering(preprocessed_data)
labels = preprocessed_data['Engine Condition']  # Assuming 'Engine Condition' is the target label

# Step 4: Data Normalization
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Step 5: Handle Imbalanced Data
smote = SMOTE(random_state=45)
X_res, y_res = smote.fit_resample(features_normalized, labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=45)

# Step 6: Model Training with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=45)
model.fit(X_train, y_train)

# Step 7: Model Evaluation
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y, predictions))
    print(f'ROC-AUC: {roc_auc_score(y, predictions)}')

evaluate_model(model, X_test, y_test)

# Step 8: Feedback Mechanism
def provide_feedback(predictions):
    for i, pred in enumerate(predictions):
        if pred == 1:
            print(f"Warning: Risky driving behavior detected at index {i}")

predictions = model.predict(X_test)
provide_feedback(predictions)

# Save Model
def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {file_path}')

save_model(model, 'trained_model_rf.pkl')

# Load Model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

loaded_model = load_model('trained_model_rf.pkl')

# Verify the loaded model
evaluate_model(loaded_model, X_test, y_test)

# Plot the feature importances
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances", fontsize=16)
    bars = plt.bar(range(len(importances)), importances[indices], align="center", color='skyblue', edgecolor='black')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add numerical labels to each bar
    for bar, importance in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{importance:.2f}', ha='center', va='bottom', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()

feature_names = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp', 'Temp_diff']
plot_feature_importances(model, feature_names)
