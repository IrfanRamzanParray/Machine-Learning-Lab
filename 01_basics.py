import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
# print("hello")

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)

# Add target column
df["species"] = iris.target

print("First five rows:\n",df.head())

print("Shape of the dataset: ",df.shape)

mean_values = np.mean(df.iloc[:, :-1], axis=0)
median_values = np.median(df.iloc[:, :-1], axis=0)
std_values = np.std(df.iloc[:, :-1], axis=0)

print("Mean values:\n", mean_values)
print("Median values: ",median_values)
print("Standard deviation:\n",std_values)

print(df.iloc[:, :-1].corr());

