import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

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

print(scaled_df.head())