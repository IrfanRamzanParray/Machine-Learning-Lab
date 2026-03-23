import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.impute import SimpleImputer

# 1. Load dataset
#df = pd.read_csv(r"D:\GCET\Machine Learning\Datasets\framingham.csv")   # replace with your dataset file
df = pd.read_csv("D:/GCET/Machine Learning/Datasets/framingham.csv")
#df = df.dropna()  # drop entire row
df = df.fillna(df.mean())  #replace the null with mean



# 2. Display first few rows
print(df.head())

# 3. Separate features and target
X = df.iloc[:, :-1]   # all columns except last
y = df.iloc[:, -1]    # target column

# imputer = SimpleImputer(strategy='mean') 
# X = imputer.fit_transform(X)


# 4. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Create Logistic Regression model
model = LogisticRegression(max_iter=200)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Performance")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 10. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Logistic Regression")
plt.show()