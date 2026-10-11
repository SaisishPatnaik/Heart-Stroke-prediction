Live Link : https://heart-stroke-prediction-9dhvusagv4jsldkrtgmeks.streamlit.app/

# Heart Disease Prediction App

A machine learning web application that predicts the likelihood of heart disease based on patient health parameters. Built with Python, Scikit-learn, and Streamlit, this project demonstrates how machine learning can be integrated into an interactive web application.


## Features

-  **Heart Disease Prediction** — Predicts the likelihood of heart disease using a trained K-Nearest Neighbors (KNN) classifier.
-  **Interactive Interface** — Enter health parameters through a simple Streamlit interface.
-  **Feature Processing** — Standardizes numerical inputs using Scikit-learn's StandardScaler.
-  **Instant Predictions** — Generates predictions using a pre-trained machine learning model.
-  **Dark-Themed UI** — Provides a clean interface for entering patient information.
-  **Saved Model Artifacts** — Loads the trained model, scaler, and feature configuration from serialized files.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Scikit-learn | Machine learning and preprocessing |
| Pandas | Dataset manipulation |
| NumPy | Numerical computations |
| Joblib | Saving and loading model artifacts |

## Machine Learning Model

The application uses the **K-Nearest Neighbors (KNN)** classification algorithm to classify patient data based on patterns learned from a heart disease dataset.

### Workflow

1. **Data Collection:** Load the heart disease dataset from `heart.csv`.
2. **Data Preprocessing:** Prepare the input features for model training.
3. **Feature Scaling:** Apply StandardScaler to standardize the numerical features.
4. **Model Training:** Train a KNN classifier on the prepared dataset.
5. **Model Persistence:** Save the trained model and preprocessing artifacts using Joblib.
6. **Prediction:** Process user inputs and generate a prediction through the Streamlit application.

## 📋 Input Features Explained

The application uses the following health parameters:

| Feature | Description |
|---|---|
| **Age** | Patient's age in years. |
| **Sex** | Biological sex as encoded in the dataset. |
| **Chest Pain Type** | Category of chest pain experienced by the patient. |
| **Resting Blood Pressure** | Blood pressure measured at rest, in mmHg. |
| **Cholesterol** | Blood cholesterol level, measured in mg/dL. |
| **Fasting Blood Sugar** | Indicates whether fasting blood sugar exceeds the dataset's specified threshold. |
| **Resting ECG** | Results of an electrocardiogram recorded at rest. |
| **Maximum Heart Rate** | Maximum heart rate achieved during an exercise test, measured in beats per minute. |
| **Exercise-Induced Angina** | Indicates whether angina occurs during exercise. |
| **Oldpeak (ST Depression)** | ST-segment depression measured during exercise relative to rest. |
| **ST Slope** | Direction of the peak exercise ST segment: upsloping, flat, or downsloping. |
| **Number of Major Vessels** | Number of major blood vessels visualized using the imaging procedure represented in the dataset. |
| **Thalassemia** | Encoded result of a thalassemia-related cardiac test in the dataset. |

## Project Structure

```text
Heart-Stroke-prediction/
│
├── .devcontainer/          # Development container configuration
├── README.md               # Project documentation
├── app.py                  # Streamlit application
├── heart.py                # Additional Python script
├── heart.csv               # Heart disease dataset
├── knn_model.pkl            # Trained KNN model
├── scaler.pkl               # Fitted StandardScaler
├── feature_columns.pkl      # Model feature configuration
└── requirements.txt         # Python dependencies
```

##  Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/SaisishPatnaik/Heart-Stroke-prediction.git
cd Heart-Stroke-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal. By default, Streamlit uses:

`http://localhost:8501`

##  Model Files

The application uses saved artifacts to generate predictions without requiring model training every time it starts.

- `knn_model.pkl` — Trained KNN classifier.
- `scaler.pkl` — Fitted feature scaler.
- `feature_columns.pkl` — Expected model input feature names and ordering.

Keep these files consistent with the version of the model and preprocessing pipeline used by the application.

##  Learning Outcomes

This project demonstrates practical experience with:

- Machine learning classification using KNN.
- Data preprocessing and feature standardization.
- Loading and using serialized machine learning models.
- Building interactive applications with Streamlit.
- Integrating a machine learning pipeline with a user interface.
- Organizing a Python machine learning project for GitHub.

## Disclaimer

This project is intended for educational and demonstration purposes only. Its predictions are not medical diagnoses and should not be used as the basis for medical decisions. Always consult a qualified healthcare professional for medical advice and evaluation.

## Author
**Saisish Patnaik**

- GitHub: [@SaisishPatnaik](https://github.com/SaisishPatnaik)


