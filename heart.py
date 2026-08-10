import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib


df = pd.read_csv('heart.csv')

df_new = df.copy()
df_new = df_new.astype(int)


x = df_new.drop('target', axis=1)
y = df_new['target']


train_x, test_x, train_y, test_y = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

scaled_x = scaler.fit_transform(train_x)
scaled_test_x = scaler.transform(test_x)


models = {
    # 'logistic_regression': LogisticRegression(),
    # 'svm': SVC(),
    # 'naive_bayes': GaussianNB(),
    # 'decision_tree': DecisionTreeClassifier(),

    'knn': KNeighborsClassifier()
}



result = []

for name, current_model in models.items():

    current_model.fit(scaled_x, train_y)

    pred = current_model.predict(scaled_test_x)

    accuracy = accuracy_score(test_y, pred)

    f1 = f1_score(test_y, pred)

    result.append({
        'model': name,
        'accuracy': round(accuracy, 3),
        'f1_score': round(f1, 3)
    })


print(result)

# Save KNN Model
knn_model = models['knn']

joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(x.columns.tolist(), 'feature_columns.pkl')

print("Model saved successfully!")






