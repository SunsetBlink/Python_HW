
#6 lesson
# 1. Сценарий
# 2. Командная оболочка (интерпретатор) IPython
# 3. Jupyter
# Во всех этих случаях matplotlib используется немного по-разному
from inspect import markcoroutinefunction

# 1.
# plt.show() — запускается только один раз (обычно в самом конце)
# Figure

import matplotlib.pyplot as plt
import numpy as np


#
# fig = plt.figure
# plt.plot(x, np.sin(x)) # Ничего не вышло, ибо нет plt.show()
# plt.plot(x, np.cos(x)) # Если это поставить после show, то косинуса мы не увидим
# plt.show()
#
# # 2. Если используем IPython
# # %matplotlib
# # import matplotlib.pyplot as plt
# # plt.plot(...) будет открывать окно графика и можно в нём работать
# # plt.draw() используется достаточно редко
# # Точка с запятой после plt.plot — ?
#
# # 3. Используя Jupyter:
# # %matplotlib inline — в блокнот добавляется статическая картинка
# # %matplotlib notebook — в блокнот добавляются интерактивные графики
#
# fig.savefig('saved_images.png')

# print (fig.canvas_get_supported_filetypes())

# Два способа вывода графиков:
# а) MatLab-подобный стиль;
# б) Объектно-ориентированный стиль.

x = np.linspace(0, 10, 100)

# ML-стиль
# plt.figure()
# plt.subplot(2, 1, 1) # Две строки, одна колонка (когда нужно несколько картинок за раз), первая строка
# plt.plot(x, np.sin(x))
#
# plt.subplot(2, 1, 2) # Снова две строки и одна колонка, но уже вторая строчка
# plt.plot(x, np.cos(x))

# ОО-стиль
# fig: Figure, ax: Axes
#fig, ax = plt.subplots(4)
#ax[0].plot(x, np.cos(x),color = 'blue') # Верхний элемент по оси
#ax[1].plot(x, np.cos(x)-1) # Нижний элемент по оси

# fig: plt.Figure — контейнер, содержащий все объекты (СК, тексты, метки),
# ax: Axes — система координат (прямоугольник, деления, метки)



#цвета линий color (mb rgb, FFOOEE ili zaranee vbitie cian)
#solib line - '-'
#dashed line - '--'
#dashed dot - '-.'
#dotted - ':'

#ax[0].plot(x,np.sin(x))
#ax[1].plot(x,np.sin(x))
#ax[2].plot(x,np.sin(x))
#ax[3].plot(x,np.sin(x),'^',markeredgewidth=2)
#ax[1].set_xlim(-2,12)
#ax[1].set_ylim(0,1)
#ax[3].autoscale(tight=True)
#plt.title("sinus")
#plt.xlabel("xlabel")
#plt.ylabel("Sin(X)")
#plt.subplots_adjust(hspace = 0.5)


rng = np.random.default_rng(0)

colors = rng.random(30)
sizes = 30* rng.random(30)

#plt.scatter(x,np.sin(x),marker='o',c=colors,s=sizes)
#plt.colorbar()
#x = np.linspace(0, 10, 50)
#dy = 0.4
#y = np.sin(x) + dy * np.random.randn(50)
#plt.errorbar(x,y,yerr=dy)
#plt.fill_between(x,y-dy,y+dy,color='red',alpha=0.4)
#plt.show()
def f(x,y):
    return np.sin(x**2+y**2)

x = np.linspace(0,5,50)
y = np.linspace(0,5,40)
X,Y=np.meshgrid(x,y)
Z = f(X,Y)
#plt.contour(X,Y,Z)
c = plt.contourf(X,Y,Z)
plt.clabel(c)
plt.imshow(Z,extent = [0,5,0,5],interpolation='gaussian',origin='upper')
plt.colorbar()
plt.show()
#7 lesson
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
from datetime import datetime
fig, ax = plt.subplots(2,3,sharex='col')
for i in range(2):
    for j in range(3):
        ax[i,j].text(0.5,0.5,str((i,j)),fontsize=15, ha = 'center',sharey='row')

grid = plt.GridSpec(2,3)
plt.subplot(grid[0,0])
plt.subplot(grid[0,1:])
plt.subplot(grid[1,:2])
plt.subplot(grid[1,2])

mean = [0,0]
cov = [[1,1],[1,2]]

rng = np.random.default_rng(1)
#x,y = rng.multivariate_normal(mean = mean, cov = cov,3000).T
fig = plt.figure()
grid = plt.Gridspec(4,4,hspace=0.2,wspace=0.2)
main_ax = fig.add_subplot(grid[:-1,1:])
y_hist = fig.add_subplot(grid[:-1,0],yticklabel=[],sharey=main_ax,)
x_hist = fig.add_subplot(grid[-1,1:],xticklabel=[],sharex=main_ax)
main_ax.plot(x,y,'ok',markersize=3)

y_hist.hist(y, 40,orientation='horizontal',color='gray',histtype='stepfilled')
x_hist.hist(x, 40,orientation='vertical',color='gray',histtype='step')

births = pd.read_csv('./births-1969.csv')
births.index = pd.to_datetime(10000*births.year+100*births.month+births.day)
print(births.head())
births_by_date = births.pivot_table('births',[births.index.year,births.index.month,births.index.day])
print(births_by_date.head())
births_by_date.index = [datetime(1969,month,day) for (month,day) in births_by_date.index]

fig,ax = plt.subplots()
births_by_date.plot(ax=ax)
style = dict(size=10,color='gray')
ax.text('1969-01-01',5500,'New year',**style)
ax.text('1969-00-01',5500,'knowlenge day',ha = 'right')
ax.set(title='birthing in 1969',elabel = 'number of births')
ax.xaxis.set_major_formatter(plt.NullFormatter())
fig = plt.figure()
ax1 = plt.axes()
ax2 = plt.axes([0.4,0.3,0.2,0.1])
ax1.set_xlim(0,2)
ax1.text(0.6,0.8,'data1 (0.6,0.8), transform=ax1.transData')
ax2.text(0.6,0.8,'data1 (0.6,0.8), transform=ax2.transData')

ax1.text(0.5,0.1,'data1 (0.5,0.1), transform=ax1.transAxes')
ax2.text(0.5,0.1,'data1 (0.5,0.1), transform=ax2.transAxes')

ax1.text(0.2,0.2,'data1 (0.2,0.2), transform=ax1.transFigure')
ax2.text(0.2,0.2,'data1 (0.2,0.2), transform=ax2.transFigure')

x = np.linspace(0,20,1000)
ax.plot(x,np.cos(x))
ax.axis('equal')
ax.annotate('loc max',xy = (6.28,1),xytext(10,4),arrowprops=dict(facecolor='red'))

fig,ax = plt.subplots(4,4,sharex=True,sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(5))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
x = np.random.randn(1000)

plt.hist(x)
fig = plt.figure(facecolor='gray')
ax = plt.axes(facecolor='green')
plt.grid(color='w',linestyle='solid')
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
plt.show()
#8 lesson
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
# 3D dots slf lines
#fig = plt.figure()
ax = plt.axes(projection = '3d')

#z = np.linspace(0,15,1000)
#y = np.cos(z)
#x = np.sin(z)
#ax.plot3D(x,y,z,'red')
#z2 = 15*np.random.random(100)
#y2 = np.sin(z2)+ 0.1*np.random.random(100)
#x2 = np.cos(z2)+ 0.1*np.random.random(100)
#ax.scatter3D(x2,y2,z2,c = z2,cmap = 'Greens')
def f(x,y):
    return np.sin(np.sqrt(x**2 + y**2))
x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)
#ax.contour3D(X,Y,Z,40)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(45)
#ax.scatter3D(x,Y,X,c=Z,cmap = 'Greens')
#carcas
#ax.plot_wireframe(X,Y,Z)
#surface
#ax.plot_surface(X,Y,Z,cmap = 'viridis',edgecolor = 'none')

r = np.linspace(0,6,20)
theta = np.linspace (-0.9*np.pi,0.8*np.pi,40)
R,Theta = np.meshgrid(r,theta)
X = r*np.sin(Theta)
Y = r*np.cos(Theta)
Z = f(X,Y)
ax.plot_surface(X,Y,Z,rstride = 1,cmap = 'viridis',edgecolor = 'none')
theta = 2*np.pi + np.random.random(1000)
r = 6*np.random.random(1000)
x = r*np.sin(theta)
y = r*np.cos(theta)
z = f(x,y)
ax.scatter3D(x,y,z,c=z,cmap='viridis')
ax.plot_trisurf(x,y,z,cmap = 'viridis')

#seaborn

data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size = 2000)
data = pd.DataFrame(data,columns=['x','y'])
print(data.head())

plt.hist(data['x'],alpha=0.5)
plt.hist(data['y'],alpha=0.5)
fig = plt.figure()
#sns.kdeplot(data=data,shade = True)
iris = sns.load_dataset('iris')
print(iris.head())
#sns.pairplot(iris,hue='species',height=2.5)
tips = sns.load_dataset('tips')
print(iris.head())

grid = sns.FacetGrid(tips)
grid = sns.FacetGrid(tips,col='sex',row='day',hue='time')
grid.map(plt.hist,"tip",bins=np.linspace(0,40,15))

#sns.catplot(data = tips,x='day',y='total_bill')
sns.jointplot(data=tips,x='day',y='total_bill',kind='hex')
planets=sns.load_dataset('planets')
sns.catploat(data=planets,x='year',kind='count',hue='methos',order='magnitude')
#heat map
tips_corr = [['total_bill','tip','size']]
sns.heatmap(tips_corr.corr(),cmap='RdBu_r',annot=True,vmin=-1,vmax=1)

sns.scatterplot(data=tips,x='total_bill',y='tip')
#lin graph
sns.lineplot(data=tips,x='total_bill',y='tip')
#svodnaya diagramma
sns.jointplot(data=tips,x='total_bill',y='ti')
sns.barplot(data=tips,x = 'total_bill',y = 'tips')
sns.pointplot(data=tips,x = 'total_bill',y = 'tips')
sns.boxplot(data=tips,x = 'total_bill',y = 'tips')
sns.violinplot()(data=tips,x = 'total_bill',y = 'tips')

plt.show()