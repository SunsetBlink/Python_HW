# Переобучение присуще всем чё-то там...

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")

# print(iris.head())

sns.pairplot(iris, hue="species")

species_int = []
for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
# print(species_int_df.head())

data = iris[["sepal_length", "petal_length"]]
data["species"] = species_int

# print(data.head())
# print(data.shape)

data_versicolor = data[data["species"] == 2]
data_virginica = data[data["species"] == 3]

print(data_versicolor.shape)
print(data_virginica.shape)

data_versicolor_A = data_versicolor.iloc[:25, :]
data_versicolor_B = data_versicolor.iloc[25:, :]

data_virginica_A = data_virginica.iloc[:25, :]
data_virginica_B = data_virginica.iloc[:25, :]

data_df_A = pd.concat([data_virginica_A, data_versicolor_A],
                      ignore_index=True)
data_df_B = pd.concat([data_virginica_B, data_versicolor_B],
                      ignore_index=True)

x1_p = np.linspace(min(data["sepal_length"]), max(data["sepal_length"]))
x2_p = np.linspace(min(data["petal_length"]), max(data["petal_length"]))

X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns=["sepal_length", "petal_length"]
)

fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')

max_depth = [1, 3, 5, 7]

X = data_df_A[["sepal_length", "petal_length"]]
y = data_df_A["species"]

j = 0
for md in max_depth:
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)

    ax[0, j].scatter(data_virginica_A["sepal_length"],
                     data_virginica_A["petal_length"])
    ax[0, j].scatter(data_versicolor_A["sepal_length"],
                     data_versicolor_A["petal_length"])

    y_p = model.predict(X_p)

    ax[0, j].contourf(
        X1_p,
        X2_p,
        y_p.reshape(X1_p.shape),
        alpha=0.4,
        levels=2,
        cmap="rainbow",
        zorder=1,
    )
    j += 1

X = data_df_B[["sepal_length", "petal_length"]]
y = data_df_B["species"]

j = 0
for md in max_depth:
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)

    ax[1, j].scatter(data_virginica_B["sepal_length"],
                     data_virginica_B["petal_length"])
    ax[1, j].scatter(data_versicolor_B["sepal_length"],
                     data_versicolor_B["petal_length"])

    y_p = model.predict(X_p)

    ax[1, j].contourf(
        X1_p,
        X2_p,
        y_p.reshape(X1_p.shape),
        alpha=0.4,
        levels=2,
        cmap="rainbow",
        zorder=1,
    )
    j += 1

plt.show()
#second file
# Ансамблевые методы. В основе — идея объединения нескольких переобученных (!)
# моделей для уменьшения эффекта переобучения. Это называется баггинг (bagging).
# Баггинг усредняет результаты —> ...

# ...

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import random_sample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

iris = sns.load_dataset("iris")

# print(iris.head())

sns.pairplot(iris, hue="species")

species_int = []
for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
# print(species_int_df.head())

data = iris[["sepal_length", "petal_length","species"]]
data["species"] = species_int

data_setosa = data[data["species"] == "setosa"]
data_versicolor = data[data["species"] == 2]
data_virginica = data[data["species"] == 3]

x_p = pd.DataFrame(np.linspace(np.min(data_setosa["sepal_length"]), np.max(data_setosa["sepal_length"]),100))


# max_depth = [1, 3, 5, 7]

md = 6

X = pd.DataFrame(data_setosa["sepal_length"],columns = ["sepal_length"])
y = data["species"]

model1 = RandomForestRegressor(n_estimators=20)
model1.fit(X, y)

y_p1 = model1.predict(X_p)

ax[0].contourf(
        X1_p,
        X2_p,
        y_p1.reshape(X1_p.shape),
        alpha=0.4,
        levels=2,
        cmap="rainbow",
        zorder=1,
)

# Bagging

model2 = DecisionTreeClassifier(max_depth=md)
b = BaggingClassifier(model2, n_estimators=20, max_samples=0.8, random_state=1)

b.fit(X, y)

y_p2 = b.predict(X_p)

ax[1].contourf(
        X1_p,
        X2_p,
        y_p1.reshape(X1_p.shape),
        alpha=0.4,
        levels=2,
        cmap="rainbow",
        zorder=1,
)

plt.show()

#random Forest
model3 = RandomForestClassifier(max_depth=md,n_estimators=20, max_samples=0.8, random_state=1)

model3.fit(X, y)

y_p2 = b.predict(X_p)

ax[1].contourf(
        X1_p,
        X2_p,
        y_p1.reshape(X1_p.shape),
        alpha=0.4,
        levels=2,
        cmap="rainbow",
        zorder=1,
)

plt.show()
#regression with random forest
#dostoinstva - predict bistro. rasparallelivaniye, veroyatnost, neparam model
#nedosatki - slozhno interpritirovat'

#third file
#metod glavnikh component (PCA) - algoritm bez uchitilya, isp dlya ponizheniya razmernosti, smis - viyavleniye zavisimosti mezhdu priznakami
#kachestvennaya ocenka zavisimosti (glavniye osi)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import random_sample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")

# print(iris.head())

sns.pairplot(iris, hue="species")



data = iris[["petal_width", "petal_length","species"]]

data_v = data[data["species"] == "versicolor"]


# max_depth = [1, 3, 5, 7]

Y = data_v["petal_lenght"]
X = data_v["petal_width"]
from sklearn.decomposition import PCA
p = PCA(n_components=2)
p.fit(data_v)
X_p = p.transform(data_v)
X_p_new = p.inverse_transform(X_p)
print(p.components_)
print(p.explained_variance_)
print(p.mean_)
plt.scatter(p.mean_[0], p.mean_[1])
plt.plot([p.mean_[0], p.mean_[0] + p.components_[0][0]*np.sqrt(p.explained_variance_[0])],[p.mean_[1], p.mean_[1] + p.components_[1][0]*np.sqrt(p.explained_variance_[0])])
plt.plot([p.mean_[0], p.mean_[0] + p.components_[0][0]*np.sqrt(p.explained_variance_[1])],[p.mean_[1], p.mean_[1] + p.components_[1][1]*np.sqrt(p.explained_variance_[1])])