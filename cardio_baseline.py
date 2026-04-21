#!/usr/bin/env python3
#05/21/2025
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#Read csv
heart_csv_path = "cardio_train.csv"
heart_data = pd.read_csv(heart_csv_path, sep=";")

#Target: Create y
y = heart_data.cardio

#Train: Create X
X = heart_data.drop(columns=["cardio"]) 

#Split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Model
rand_forest_model = RandomForestClassifier(max_leaf_nodes=100, random_state=1)
rand_forest_fit = rand_forest_model.fit(train_X, train_y)

#Predict
predictions_heart = rand_forest_model.predict(val_X)

#Evaluate
print("Accuracy:", accuracy_score(val_y, predictions_heart)) 
print("\nConfusion Matrix:\n", confusion_matrix(val_y, predictions_heart))
print("\nClassification Report:\n", classification_report(val_y, predictions_heart))

# Visualize Confusion Matrix
cm = confusion_matrix(val_y, predictions_heart)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)  # optional: nicer color
plt.title("Confusion Matrix - Cardiovascular Risk Prediction")
plt.show()
