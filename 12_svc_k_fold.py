import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("D:/GCET/Machine Learning/Datasets/framingham.csv")

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
model = SVC(kernel='rbf')

# K-Fold Cross Validation (k=5)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Get predictions using cross-validation
y_pred = cross_val_predict(model, X_scaled, y, cv=kf)

# Evaluate performance
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

print("\nModel Performance (K-Fold Cross Validation)")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - SVM (K-Fold)")
plt.show()