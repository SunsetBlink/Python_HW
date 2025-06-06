

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

#metod opornikh vectorov - classification and regression, razdelyaushaya classificatsiya

iris = sns.load_dataset("iris")

data = iris[["sepal_lenght" , "petal_lenght" , "species"]]
data_df = data[(data["species" == "setosa"]) | (data["species"] == "versicolor")]

data_df_seposa = data_df[data_df["species" == "setosa"]]
data_df_versicolor = data_df[data_df["species" == "versicolor"]]

x = data_df[["sepal_lenght", "petal_lenght"]]
y = data_df["species"]

model = SVC(kernel = "linear", C=10000)
model.fit(x,y)

#HOMEWORK
x2 = x.iloc[:-2]
y2 = y.iloc[:-2]

model2 = SVC(kernel = "linear", C=10000)
model2.fit(x2,y2)

print(model2.support_vectors_ == model.support_vectors_)


plt.scatter (model.support_vectors_[:,0],model.support_vectors_[:,1], s = 400)

x1_p = np.linspace(min(data_df[["sepal_length"]]),max(data_df[["sepal_length"]]),100)
x2_p = x1_p = np.linspace(min(data_df[["sepal_length"]]),max(data_df[["sepal_length"]]),100)
X1_p, X2_p = np.meshgrid(x1_p,x2_p)

y_p = model.predict(X_p)

X_p = pd.DataFrame(np.vstack[x1_p.ravel(),x2_p.ravel()].T, columns = ["sepal_lenght","petal_lenght"])
X_p["species"] = y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["versicolor"] == "versicolor"]

#esli danniye perecrivautsa to idealnoi graniti net i sush parametr razmitiya, menshe C, otstup bolee razmitiy
C_value = [[10000,1000,100,10],[1,0.1,0.01]]
fig,ax = plt.subplots(2,4)

#derevya resheniy i sluchayniye lesa, neparam algoritm

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