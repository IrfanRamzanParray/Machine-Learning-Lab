import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)

# Add target column
df["species"] = iris.target

#Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:, :-1])

scaled_df = pd.DataFrame(
    scaled_features,
    columns=iris.feature_names
)

X = df.iloc[:, :-1]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# accuracy = model.score(X_test, y_test)
# Make predictions
y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("Confusion Matrix - Logistic Regression (Iris Dataset)")
plt.show()

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(precision)
print(recall)
print(f1)