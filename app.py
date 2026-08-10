import streamlit as st
import pandas as pd
import joblib


model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('feature_columns.pkl')



st.title("Heart Disease Prediction by Saisish")
st.markdown("Provide the following details:")


age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure",
    80,
    200,
    120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    ["No", "Yes"]
)

restecg = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T Abnormality",
        "LV Hypertrophy"
    ]
)

max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    0.0,
    10.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    raw_data = {

        'age': age,

        'sex': 1 if sex == "Male" else 0,

        'cp': (
            0 if chest_pain_type == "Typical Angina"
            else 1 if chest_pain_type == "Atypical Angina"
            else 2 if chest_pain_type == "Non-Anginal Pain"
            else 3
        ),

        'trestbps': resting_blood_pressure,

        'chol': cholesterol,

        'fbs': 1 if fbs == "Yes" else 0,

        'restecg': (
            0 if restecg == "Normal"
            else 1 if restecg == "ST-T Abnormality"
            else 2
        ),

        'thalach': max_heart_rate,

        'exang': 1 if exang == "Yes" else 0,

        'oldpeak': oldpeak,

        'slope': (
            0 if st_slope == "Upsloping"
            else 1 if st_slope == "Flat"
            else 2
        ),

        'ca': ca,

        'thal': (
            0 if thal == "Normal"
            else 1 if thal == "Fixed Defect"
            else 2
        )
    }


    input_data = pd.DataFrame([raw_data])
    input_data = input_data[columns]


    # Check for missing values
    if input_data.isnull().sum().sum() > 0:

        st.error("Some input values are missing.")

    else:

        # Scale input
        scaled_input_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(scaled_input_data)[0]


        # Display result
        if prediction == 1:

            st.error(
                "The patient is at high risk of heart disease."
            )

        else:

            st.success(
                "The patient is at low risk of heart disease."
            )
    
    
            
            



