#khai bĂ¡o thÆ° viá»‡n
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
# height (cm)
y = np.array([[14.2,11.7,6.4,2.1,4.8,8.1,15.4,9.8]]).T
# weight (kg)
X = np.array([[17.5,15.6,9.8,5.3,7.9,10	,19.2,13.1]]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([4.5,20,1.5,16])
plt.xlabel('interest rate (lĂ£i suáº¥t)')
plt.ylabel('inflationary (láº¡m phĂ¡t)')
plt.show()
# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(2, 15.5, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([4.5,20,1.5,16])
plt.xlabel('muc lai suat')
plt.ylabel('ti le lam phat')
plt.show()
y = w_1*11 + w_0

print( u'du doan ti le lam phat khi muc lai suat la 11: ' , y )