#SVC
# Метод опорных векторов: классификация и регрессия.
# (SCM — support vector machine)

# Разделяющая классификация.
# Выбирается линия с максимальным отступом.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import plotly.express as px

iris = sns.load_dataset("iris")
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print(iris.head())

# Добавляем параметр sepal_width
data = iris[["sepal_length", "petal_length","sepal_width", "species"]]
data_df = data[(data["species"] == "virginica") |
               (data["species"] == "versicolor")]

# Включаем sepal_width в признаки
X = data_df[["sepal_length", "petal_length", "sepal_width"]]
y = data_df["species"]

data_df_virginica = data_df[data_df["species"] == "virginica"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

# Вычисляем среднее значение sepal_width для визуализации
mean_sepal_width = data_df["sepal_width"].mean()


ax.scatter(
    data_df_virginica["sepal_length"],
    data_df_virginica["petal_length"],
    data_df_virginica["sepal_width"]
)
ax.scatter(
    data_df_versicolor["sepal_length"],
    data_df_versicolor["petal_length"],
    data_df_versicolor["sepal_width"]
)

model = SVC(kernel='linear', C=1)
model.fit(X, y)


ax.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    model.support_vectors_[:, 2],
    s=400,
    facecolor='none',
    edgecolors='black'
)

x1_p = np.linspace(
    min(data_df["sepal_length"]),
    max(data_df["sepal_length"]), 100
)
x2_p = np.linspace(
    min(data_df["petal_length"]),
    max(data_df["petal_length"]), 100
)
x3_p = np.linspace(
    min(data_df["sepal_width"]),
    max(data_df["sepal_width"]), 100
)

X1_p, X2_p, X3_p = np.meshgrid(x1_p, x2_p, x3_p)


X_p = pd.DataFrame(
    np.column_stack([X1_p.ravel(), X2_p.ravel(), X3_p.ravel()]),
    columns=["sepal_length", "petal_length", "sepal_width"]
)

y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_virginica = X_p[X_p["species"] == "virginica"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

ax.scatter(
    X_p_virginica["sepal_length"],
    X_p_virginica["petal_length"],
    X_p_virginica["sepal_width"],alpha=0.1
)
ax.scatter(
    X_p_versicolor["sepal_length"],
    X_p_versicolor["petal_length"],
    X_p_versicolor["sepal_width"],alpha=0.1
)

plt.show()


#PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
iris = sns.load_dataset("iris")


data = iris[["petal_width", "petal_length","sepal_width", "species"]]
data_v = data[data["species"] == "versicolor"]
data_v = data_v.drop(columns=["species"])
data_vi = data[data["species"] == "virginica"]
data_vi = data_vi.drop(columns=["species"])

X = data_v["petal_width"]
Y = data_v["petal_length"]
Z = data_v["sepal_width"]

Xi = data_vi["petal_width"]
Yi = data_vi["petal_length"]
Zi = data_vi["sepal_width"]


ax.scatter(X, Y, Z)
ax.scatter(Xi, Yi, Zi)

p = PCA(n_components=3)
p.fit(data_v)
X_p = p.transform(data_v)
pi = PCA(n_components=3)
pi.fit(data_vi)
X_pi = pi.transform(data_v)


ax.scatter(p.mean_[0], p.mean_[1],p.mean_[2])

ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[0][0] *
     np.sqrt(p.explained_variance_[0])],
    [p.mean_[1], p.mean_[1] + p.components_[0][1] *
     np.sqrt(p.explained_variance_[0])],
    [p.mean_[2], p.mean_[2] + p.components_[0][2] *
     np.sqrt(p.explained_variance_[0])]
)

ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[1][0] *
     np.sqrt(p.explained_variance_[1])],
    [p.mean_[1], p.mean_[1] + p.components_[1][1] *
     np.sqrt(p.explained_variance_[1])],
    [p.mean_[2], p.mean_[2] + p.components_[1][2] *
     np.sqrt(p.explained_variance_[1])]
)

ax.scatter(pi.mean_[0], pi.mean_[1],pi.mean_[2])

ax.plot(
    [pi.mean_[0], pi.mean_[0] + pi.components_[0][0] *
     np.sqrt(pi.explained_variance_[0])],
    [pi.mean_[1], pi.mean_[1] + pi.components_[0][1] *
     np.sqrt(pi.explained_variance_[0])],
    [pi.mean_[2], pi.mean_[2] + pi.components_[0][2] *
     np.sqrt(pi.explained_variance_[0])]
)

ax.plot(
    [pi.mean_[0], pi.mean_[0] + pi.components_[1][0] *
     np.sqrt(pi.explained_variance_[1])],
    [pi.mean_[1], pi.mean_[1] + pi.components_[1][1] *
     np.sqrt(pi.explained_variance_[1])],
    [pi.mean_[2], pi.mean_[2] + pi.components_[1][2] *
     np.sqrt(pi.explained_variance_[1])]
)

plt.show()

#k mean

# Наивная байесовская классификация
# Набор моделей, которые предлагают быстрые и простые алгоритмы классификации

# Гауссовский наивный байесовский классификатор
# Допущение состоит в том, что ! данные всех категорий взяты из простого нормального распределения !

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

iris = sns.load_dataset("iris")
# print(iris.head())

# sns.pairplot(iris, hue = 'species')

data = iris[["sepal_length", "petal_length", "sepal_width", "species"]]

data_df = data[(data["species"] == "versicolor") | (data["species"] == "virginica")]

X = data_df[["sepal_length", "petal_length", "sepal_width"]]
y = data_df["species"]

model = GaussianNB()
model.fit(X, y)



theta0 = model.theta_[0]
var0 = model.var_[0]
theta1 = model.theta_[1]
var1 = model.var_[1]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]
data_df_virginica = data_df[data_df["species"] == "virginica"]

ax.scatter(data_df_seposa["sepal_length"], data_df_seposa["petal_length"],data_df_seposa["sepal_width"])
ax.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"],data_df_versicolor["sepal_width"])

x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)
x3_p = np.linspace(min(data_df["sepal_width"]), max(data_df["sepal_width"]), 100)

X1_p, X2_p, X3_p = np.meshgrid(x1_p, x2_p, x3_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel(), X3_p.ravel()]).T, columns=["sepal_length", "petal_length","sepal_width"]
)

print(X_p.head())

y_p = model.predict(X_p)

X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]
X_p_virginica = X_p[X_p["species"] == "virginica"]

print(X_p.head())

ax.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"],X_p_setosa["sepal_width"], alpha=0.4)
ax.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"],X_p_versicolor["sepal_width"], alpha=0.4)
ax.scatter(X_p_virginica["sepal_length"], X_p_virginica["petal_length"],X_p_virginica["sepal_width"], alpha=0.4)

# sns.pairplot(data_df, hue = 'species')




plt.show()