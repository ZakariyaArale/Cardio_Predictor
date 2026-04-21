#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from google.colab import files
uploaded = files.upload()

#Read csv
heart_csv_path = "cardio_train.csv"
heart_data = pd.read_csv(heart_csv_path, sep=";")

#Clean data

#Remove physiologically implausible blood pressure values
heart_data = heart_data[heart_data["ap_hi"] > heart_data["ap_lo"]]
heart_data = heart_data[(heart_data["ap_hi"] >= 80) & (heart_data["ap_hi"] <= 250)]
heart_data = heart_data[(heart_data["ap_lo"] >= 40) & (heart_data["ap_lo"] <= 180)]

#Remove extreme heights/weights
heart_data = heart_data[(heart_data["height"] >= 120) & (heart_data["height"] <= 230)]
heart_data = heart_data[(heart_data["weight"] >= 35) & (heart_data["weight"] <= 250)]

#Add helpful columns to the csv (feature engineering)
heart_data["age_years"] = heart_data["age"] / 365
heart_data["BMI"] = heart_data["weight"] / ((heart_data["height"]/100) ** 2)
heart_data["pulse_pressure"] = heart_data["ap_hi"] - heart_data["ap_lo"]
heart_data["hypertension"] = ((heart_data["ap_hi"] >= 140) | (heart_data["ap_lo"] >= 90)).astype(int)

#Target: Create y
y = heart_data["cardio"]

#Train: Create X
X = heart_data.drop(columns=["cardio", "id", "age"])

#Split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, stratify=y)

#Model
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

#Predictions
results = {}

for name, model in models.items():

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    probs = (
        model.predict_proba(val_X)[:, 1]
        if hasattr(model, "predict_proba")
        else preds
    )

    results[name] = {
        "accuracy": accuracy_score(val_y, preds),
        "auc": roc_auc_score(val_y, probs)
    }

    print(f"\n{name}")
    print("Accuracy:", round(results[name]["accuracy"], 4))
    print("ROC-AUC:", round(results[name]["auc"], 4))


#Accuracy
plt.figure()
plt.bar(results.keys(), [r["accuracy"] for r in results.values()])
plt.title("Model Accuracy Comparison")
plt.ylim(0.5, 1.0)
plt.xticks(rotation=15)
plt.show()

#Auc
plt.figure()
plt.bar(results.keys(), [r["auc"] for r in results.values()])
plt.title("Model ROC-AUC Comparison")
plt.ylim(0.5, 1.0)
plt.xticks(rotation=15)
plt.show()