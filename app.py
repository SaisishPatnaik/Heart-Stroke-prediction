import streamlit as st
import pandas as pd
import joblib

# LOAD TRAINED MODEL AND PREPROCESSING

model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("feature_columns.pkl")


# PAGE TITLE
st.title("Heart Stroke Prediction Application")
st.markdown("### Enter the patient's details below")

# USER INPUTS
age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

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
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
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
    min_value=60,
    max_value=220,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
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
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thalassemia",
    [0, 1, 2, 3]
)


# PREDICTION

if st.button("Predict"):

    raw_data = {

        "age": age,

        "sex": 1 if sex == "Male" else 0,

        "cp": (
            0 if chest_pain_type == "Typical Angina"
            else 1 if chest_pain_type == "Atypical Angina"
            else 2 if chest_pain_type == "Non-Anginal Pain"
            else 3
        ),

        "trestbps": resting_blood_pressure,

        "chol": cholesterol,

        "fbs": 1 if fbs == "Yes" else 0,

        "restecg": (
            0 if restecg == "Normal"
            else 1 if restecg == "ST-T Abnormality"
            else 2
        ),

        "thalach": max_heart_rate,

        "exang": 1 if exang == "Yes" else 0,

        "oldpeak": oldpeak,

        "slope": (
            0 if st_slope == "Upsloping"
            else 1 if st_slope == "Flat"
            else 2
        ),

        "ca": ca,

        "thal": thal
    }




    input_data = pd.DataFrame([raw_data])
    input_data = input_data[columns]


  

    if input_data.isnull().sum().sum() > 0:

        st.error("Some input values are missing.")

        st.write(input_data)

    else:


        scaled_input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]


        # Display result

        if prediction == 1:

            st.error(
                "🔴 High Risk: The model predicts a higher likelihood of heart disease."
            )

        else:

            st.success(
                "🟢 Low Risk: The model predicts a lower likelihood of heart disease."
            )
    
    
            
            



