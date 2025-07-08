!pip install pillow
!pip install gradio
!pip install seaborn

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from io import BytesIO
from PIL import Image

# Sample Heart Disease Dataset with glucose, BMI, age, and family history
data = {
    'age': [63, 67, 67, 37, 41, 56, 57, 45, 46, 56],
    'sex': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'glucose': [150, 200, 190, 110, 120, 160, 170, 180, 130, 140],
    'bmi': [25.5, 28.2, 26.4, 24.0, 23.5, 27.6, 29.0, 24.8, 26.2, 24.3],
    'family_history': [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    'cp': [1, 4, 4, 3, 2, 1, 1, 1, 1, 1],
    'trestbps': [145, 160, 120, 130, 130, 140, 130, 130, 130, 130],
    'chol': [233, 286, 229, 250, 204, 236, 235, 240, 226, 236],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'restecg': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'thalach': [150, 108, 129, 187, 172, 178, 160, 168, 162, 148],
    'exang': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'oldpeak': [2.3, 3.5, 1.4, 0.6, 0.6, 0.7, 0.6, 0.4, 1.5, 0.2],
    'slope': [3, 3, 2, 1, 1, 1, 1, 1, 1, 1],
    'ca': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'thal': [2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
    'target': [0, 1, 1, 0, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# Prepare feature and target variables
X = df.drop(columns=['target'])
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Save the trained model and scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to predict heart disease and include accuracy results
def predict_heart_disease(age, sex, glucose, bmi, family_history, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Map 'Yes'/'No' to 1/0 for binary variables
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0
    sex = 1 if sex == "Male" else 0

    # Prepare input for prediction
    input_data = np.array([[age, sex, glucose, bmi, family_history, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_data_scaled = scaler.transform(input_data)

    # Predict the risk of heart disease
    prediction = model.predict(input_data_scaled)
    if prediction == 1:
        prediction_result = "Prediction: High Risk of Heart Disease"
        recommendation = "You may be at risk for heart disease. Please consult with a healthcare provider for further tests."
    else:
        prediction_result = "Prediction: Low Risk of Heart Disease"
        recommendation = "You are at a lower risk for heart disease, but maintaining a healthy lifestyle is always recommended."

    # Return prediction, recommendation, and accuracy results
    return prediction_result, recommendation, f"Training Accuracy: {train_accuracy * 100:.2f}%", f"Testing Accuracy: {test_accuracy * 100:.2f}%"

# Function to generate a heatmap of correlations
def generate_heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")

    # Save the plot to a BytesIO object and return as an image using PIL
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# Function to generate feature importance plot
def generate_feature_importance():
    plt.figure(figsize=(10, 6))
    feature_importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importances)

    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance in Predicting Heart Disease")

    # Save the plot to a BytesIO object and return as an image using PIL
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# Gradio Interface for Heart Disease Prediction
inputs = [
    gr.Slider(minimum=18, maximum=100, value=63, label="Age"),
    gr.Radio(choices=["Male", "Female"], value="Male", label="Sex"),
    gr.Slider(minimum=0, maximum=300, value=150, label="Glucose Levels (mg/dL)"),
    gr.Slider(minimum=15, maximum=50, value=25.5, label="Body Mass Index (BMI)"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Family History of Heart Disease"),
    gr.Slider(minimum=1, maximum=4, value=1, label="Chest Pain Type (cp)"),
    gr.Slider(minimum=90, maximum=200, value=145, label="Resting Blood Pressure (trestbps)"),
    gr.Slider(minimum=200, maximum=400, value=233, label="Serum Cholesterol (chol)"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Fasting Blood Sugar > 120 mg/dl (fbs)"),
    gr.Slider(minimum=0, maximum=2, value=2, label="Resting Electrocardiographic Results (restecg)"),
    gr.Slider(minimum=150, maximum=220, value=150, label="Max Heart Rate Achieved (thalach)"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Exercise Induced Angina (exang)"),
    gr.Slider(minimum=0, maximum=6, value=2.3, label="Depression Induced by Exercise (oldpeak)"),
    gr.Slider(minimum=1, maximum=3, value=3, label="Slope of the Peak Exercise ST Segment (slope)"),
    gr.Slider(minimum=0, maximum=3, value=0, label="Number of Major Vessels Colored by Fluoroscopy (ca)"),
    gr.Slider(minimum=3, maximum=7, value=2, label="Thalassemia (thal)")
]

outputs = [
    gr.Textbox(label="Heart Disease Prediction"),
    gr.Textbox(label="Recommendation"),
    gr.Textbox(label="Training Accuracy"),
    gr.Textbox(label="Testing Accuracy"),
    gr.Image(label="Correlation Heatmap"),
    gr.Image(label="Feature Importance Plot")
]

# Define the Gradio app layout
app = gr.Interface(
    fn=lambda age, sex, glucose, bmi, family_history, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal: (
        *predict_heart_disease(age, sex, glucose, bmi, family_history, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal),
        generate_heatmap(),
        generate_feature_importance()
    ),
    inputs=inputs,
    outputs=outputs,
    title="Heart Disease Risk Prediction",
    description="An application to predict heart disease risk based on health metrics. "
                "Provides prediction, recommendation, and visualizations of data correlations and feature importance."
)

# Launch the app
app.launch()

