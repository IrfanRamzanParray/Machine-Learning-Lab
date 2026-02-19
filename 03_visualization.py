import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df["species"] = iris.target


# Histogram
plt.figure()
plt.hist(df["sepal length (cm)"], bins=20)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Sepal Length")
plt.show()


#Scatter plot
plt.figure()
plt.scatter(
    df["sepal length (cm)"],
    df["petal length (cm)"],
    c=df["species"]
)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal vs Petal Length")
plt.show()

#Box Plot
plt.figure()
df.iloc[:, :-1].boxplot()
plt.title("Boxplot of Features")
#plt.xticks(rotation=45)
plt.show()
