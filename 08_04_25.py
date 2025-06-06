#naivnaya bayessovskaya klassifikatsiya  -bistro deshevo i serdito
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.native_bayes import GaussiaNB

iris = sns.load_dataset("iris")
print(iris.head())

data = iris[["sepal_lenght", "petal_lenght","species"]]

#setose versicolor

data_df = data[(data["species"] == "setosa") |(data["species"] == "versicolor")]

X = data_df[["sepal_length","petal_length"]]
Y = data_df["species"]
model = GaussiaNB()
model.fit(X,Y)
print(model.theta_[0])
print(model.theta_[1])
print(model.var_[0])
print(model.var_[1])

theta0 = model.theta_[0]
theta1 = model.theta_[1]
var0 = model.var_[0]
var1 = model.var_[1]
z1 = 1/(np.pi *((var0 *var1)**0.5 )*np.exp(-0.5*((X1_p - theta0)**2 / var0)))

data_df_seposa = data_df[data_df["species" == "setosa"]]
data_df_versicolor = data_df[data_df["species" == "versicolor"]]

x1_p = np.linspace(min(data_df[["sepal_length"]]),max(data_df[["sepal_length"]]),100)
x2_p = x1_p = np.linspace(min(data_df[["sepal_length"]]),max(data_df[["sepal_length"]]),100)

X1_p, X2_p = np.meshgrid(x1_p,x2_p)
X_p = pd.DataFrame(np.vstack([X1_p.ravel(),X2_p.ravel()].T,columns = ["sepal_lenght","petal_lenght"]))

y_p = model.predict(X_p)

fig = plt.figure()

ax = plt.axes(projection = "3d")
plt.contour3D(X1_p, X2_p,z1)
plt.contour3D(X1_p, X2_p,z2)