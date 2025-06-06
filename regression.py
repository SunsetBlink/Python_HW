#обучение с учителем - классификация(дискретное) и регрессия(непрерывное
#без учителя - выявление структцры немаркированных данных(кластеризация), делится на кластеризацию и понижение размерности(более сжатое представление данных)
#частичное обучение - не все данные промаркированы
#обучение с подкреплением - улучшает взаимодействие с данными при наградах

import seaborn as sns

iris = sns.load_dataset("iris")

print(iris.head())

print(type(iris.values))

print(iris.values.shape)

print(iris.columns)

#строки - отдельные объекты(образцы)
#столбцы - признаки
#целевой массив - массив меток(одномерный, длина - ч-ло образцов)

#процесс построения системы машинного обучения
#1 - предварительная обработка (выбор и масштабирование признаков, понижение размерности, выборка образцов)
#2 - обучение(выбор модели, перекрестная проверка, измерение эффективности, оптимизация гиперпараметров(внешние параметры)
#3 - оценка и формирование финальной модели
#4 - использование модели

#scikit-learn
#1 - класс модели, 2 - выбор гиперпараметров, 3 - матрица признаков и целевоцй массив, 4 - обучение модели fit(), predict()

#линейная регрессия - обучение с учителем
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(1)
x = 10*np.random.rand(50)

y = 2*x + np.random.rand(50)

plt.scatter(x,y)
plt.show()
X=x[:,np.newaxis]
model = LinearRegression()
model.fit(X,y)

print(model.coef_)

#применение к новым данным

xfit = np.linspace(0,10,5)
yfit = model.predict(xfit[:,np.newaxis])
#25.03.25
from sklearn.datasets import make_regression
features,target = make_regression(n_samples=7, n_features=1,n_informative=1, noise=1, random_state=1)
print(features.head())
print(target.head())
model = LinearRegression.fit(features,target)

plt.scatter(features,target)
x = np.linspace(features.min(),features.max(),100)
plt.plot(x,model.coef[0]*x + model.intercept_)
plt.show()
data = np.array[
    [1,5],
    [2,7],
    [3,7],
    [4,10],
    [5,11],
    [6,14],
    [7,17],
    [8,19],
    [9,22],
    [10,28]
]
x = data[:,0]
y = data[:,1]
n = len(x)
w_1 = (sum(x[i]*y[i] for i in range(n))*n - sum(x)*sum(y))/(n*sum(x**2) - n*sum(x)**2)
w_0 = sum(y)/n - w_1*sum(x)/n

print(w_1,w_0)
x_1 = np.vstack([x,np.ones(len(x))]).T
w = inv(x_1.transpose() @ x_1) @ (x_1.transpose @ y)
print(w)
Q,R = np.qr(x_1)
w = inv(R).dot(Q.transpose()).dot(y)

def f(x):
    return (x-3)**2 +4
def df(x):
    return 2*(x-3)
x = np.linspace(-10,10,100)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.NullLocator(0.5))
#plt.plot(x,f(x))
plt.plot(x,df(x))
plt.grid()
plt.show()

L = 0.001
iterations =100_000
x = random.randint(0,5)
for i in range(iterations):
    d_x = dx_f(x)
    x -= L*d_x
print(x,f(x))

w1 = 0.
w0 = 0.
for i in range(iterations):
    D_w0 = 2*sum((y[i]-w0 - w1*x[i])for i in range(n))
    D_w1 = 2 * sum(x[i]*(-y[i]-w0-w1*x[i])   for i in range(n))
    w1 -= L *D_w1
    w0 -= L* D_w0

w1 = np.linspace(-10,10,100)
w0 = w1
def E(w1,w0,x,y):
    return sum((y[i]-(w0+w1*x))**2 for i in range(len(x)))

W1,W0=np.meshgrid(w1,w0)
EW=E(W1,W0,x,y)
ax = plt.axes(projection='3d')
as.plot_surface(W1,W0,EW)
w1_fit = 2.4
w0_fit = 0.8
E_fit = E(w1_fit, w0_fit , x , y)
ax.scatter3D(w1_fit, w0_fit, E_fit, color='cyan')

#01.04.25
#gradientniy spusk
from statistics import LinearRegression

x = data [:,0]
y = data [:,1]
n = len(x)
L=0.001
sample_size = 1
for i in range(iterations):
    idx = np.random.choice(n,sample_size,replace = False)
    D_w0 = 2*sum(-y[i]+w0+w1*x[i] for i in range(n))
    D_w1 = 2*sum(((x[i]*(-y[i]+w0+w1*x[i]))) for i in range(n))
    w1 -= L*D_w1
    w0 -= L*D_w0
#vnosim smeshenie i boremsya s pereobucheniyem
#ocenka promaha
data_df = pd.DataFrame(data)
print(data_df.corr(method = 'pearson'))
print(data_df[1].values[::-1])
#obuchaushie i testovie viborki
#nabor dannih delitsa na obuchaushyu 2 chasti i testovyu  1 chast
X = data_df.values[:,0]
Y = data_df.values[:,1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3)
model = LinearRegression()
model.fit(X_train,Y_train)
r = model.score(X_test,Y_test)
print(r)
#perekrestnaya validatsiya - model 3 raza obuchaut i 3 raza testiryut
kfold = KFold(n_splits=3,random_state=1,shuffle=True)
results = cross_val_score(model,X,Y,cv = kfold)
print(results)
print(results.mean(),results.std())
#poelementno tozhe mozhno, mozhno sluchainuyu validatsiu
#validatsionnaya viborka
data_df = pd.read_csv('multiple_independent_variable_linear.csv')
X = data_df.values[:,:-1]
Y = data_df.values[:,-1]
model = LinearRegression().fit(X,Y)
print(model.coef_, model.intercept_)
ax = plt.axes(projection="3d")
ax.scatter3D(x1,x2,y)

x1_=np.linspace(min(x1),max(x1),100)
x2_=np.linspace(min(x2),max(x2),100)

X1_, X2_ = np.meshgrid(x1_,x2_)
Y_=model.intercept_ + model.coef_[0]*X1_ + model.coef_[1]*X2_
ax.plot_surface(X1_,X2_,Y_,camp = "greys",alpha=0.1)
plt.show()