#making the model using lightGBM and trainig using Tabular data
import pandas as pd
import lightgbm as lg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib


try:
    df = pd.read_csv("home/shobhit/AI-ML/Unified mentor/Heart_disease_ML/Heart Disease/dataset.csv")

except FileNotFoundError:
    print("some error while listing the file\nMay be file location is incorrect")
    exit()

if 'target' not in df.columns:
    print("Error: 'target' is not available homie or you mispelled it buddy")
    exit()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)


scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)



Model = lg.LGBMClassifier(random_state=69)
print("WOOHOO!!! We are training the model")
Model.fit(X_train_Scaled, y_train)
print("Model Trained")

y_pred=Model.predict(X_test_Scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
model_filename = 'heart_disease_model.joblib'
scaler_filename = 'scaler.joblib'

print(f"\nSaving the model now: {model_filename}...")
joblib.dump(Model, model_filename)
print("Model saved_successfully. !!! ")