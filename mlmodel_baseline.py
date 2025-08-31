#!/usr/bin/env python3
#05/21/2025
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Read csv
heart_csv_path = "cardio_train.csv"
heart_data = pd.read_csv(heart_csv_path, sep=";")

# Target: Create y
y = heart_data.cardio

# Train: Create X
X = heart_data.drop(columns=["cardio"]) 

# Split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Baseline Model - always predict most frequent class
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(train_X, train_y)

# Predict
baseline_predictions = dummy_model.predict(val_X)

# Evaluate
print("Accuracy:", accuracy_score(val_y, baseline_predictions))
print("\nConfusion Matrix:\n", confusion_matrix(val_y, baseline_predictions))
print("\nClassification Report:\n", classification_report(val_y, baseline_predictions))