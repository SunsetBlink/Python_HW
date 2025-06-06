import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#1
#fig,ax = plt.subplots(1,1)
#x = [2,5,10,15,20]
#y1 = [1,7,4,5,11]
#y2 = [4,3,1,9,12]
#ax.plot(x,y1,'red',ls = '-',label='line1')
#ax.scatter(x,y1,color = 'red')
#ax.plot(x,y2,'green',ls = '-.',label='line2')
#ax.scatter(x,y2,color = 'green')
#ax.legend()

#2
#grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
#ax1 = plt.subplot(grid[0,:2])
#ax2 = plt.subplot(grid[1,0])
#ax3 = plt.subplot(grid[1,1])
#x = [1,2,3,4,5]
#y1 = [1,7,6,4,5]
#y2 = [9,4,2,4,9]
#y3 = [-7,-4,2,-4,-7]
#ax1.plot(x,y1)
#ax2.plot(x,y2)
#ax3.plot(x,y3)

#3
#x = np.linspace(-5,5,11)
#def y(x):
#    return x**2
#fig,ax = plt.subplots(1,1)
#ax.plot(x,y(x))
#ax.annotate("min", xy=(0, 0), xytext=(0, 10),arrowprops=dict(facecolor='green', shrink=0.05))

#4
#fig,ax = plt.subplots(1,1)
#array = np.random.rand(7, 7)*10
#x = np.linspace(0,7,7)
#y = x
#ax.imshow(array)

#5
#x = np.linspace(0,5,1000)
#x1 = x
#y1 = np.zeros(len(x))
#y = np.cos(np.pi*x)
#fig,ax = plt.subplots(1,1)
#ax.plot(x,y,color='red')
#ax.fill_between(x,y, np.zeros_like(y),color='blue')

#6
#fig,ax = plt.subplots(1,1)
#ax.set_ylim(-0.5,1)
#x1 = np.linspace(0,0.75,100)
#y1 = np.cos(np.pi*x1)
#x2 = np.linspace(1.25,2.75,100)
#y2 = np.cos(np.pi*x2)
#ax.plot(x1,y1,color = 'blue')
#ax.plot(x2,y2,color='blue')
#ax.plot(x2+2,y2,color='blue')
#ax.set_ylim(-1,1)

#7
#fig,ax = plt.subplots(1,3)
#x = np.arange(7)
#y = x
#ax[0].grid()
#ax[1].grid()
#ax[2].grid()
#ax[0].step(x, y,color = 'green')
#ax[0].scatter(x,y,color = 'green')
#ax[1].step(x, y,where='mid',color = 'green')
##ax[1].scatter(x,y,color = 'green')
#ax[2].step(x, y,where='post',color = 'green')
#ax[2].scatter(x,y,color = 'green')

#8
#x = np.linspace(0,10,100)
#y1 = -(x**2-10*x)/3
#y2 = -(x**2-10*x)
#y3 = -(x**2-15*x)
#fig,ax = plt.subplots(1,1)
#ax.plot(x,y1)
#ax.plot(x,y2)
#ax.plot(x,y3)
#ax.fill_between(x,y3,color='green')
#ax.fill_between(x,y2,color='orange')
#ax.fill_between(x,y1,color='cyan')

#9
#vals = [24, 17, 53, 21, 35]
#labels = ["Ford", "Toyota", "BMV", "AUDI", "Jaguar"]
#fig, ax = plt.subplots()
#ax.pie(vals, labels=labels,explode=(0,0,0.1,0,0))
#ax.axis("equal")

#10
vals = [24, 17, 53, 21, 35]
labels = ["Ford", "Toyota", "BMV", "AUDI", "Jaguar"]
fig, ax = plt.subplots()
ax.pie(vals, labels=labels, wedgeprops=dict(width=0.5))
plt.show()