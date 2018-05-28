import numpy as np
from math import *

xdat = np.array([0.4,0.5,0.6,0.7,0.8])
ydat = np.array([0.1,0.25,0.5,0.75,0.85])

xbar = sum(xdat)/len(xdat)
ybar = sum(ydat)/len(ydat)

print(np.dot(xdat-xbar,ydat)/np.sum((xdat-xbar)**2))
print(ybar-np.dot(xdat-xbar,ydat)/np.sum((xdat-xbar)**2)*xbar)
