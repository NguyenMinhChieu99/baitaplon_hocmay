#khai bĂ¡o thÆ° viá»‡n
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
# height (cm)
Y = np.array([[14.2,11.7,6.4,2.1,4.8,8.1,15.4,9.8]]).T
# weight (kg)
x = np.array([[17.5,15.6,9.8,5.3,7.9,10	,19.2,13.1]]).T
# Visualize data 
plt.plot(Y, x, 'ro')
plt.axis([4.5,20,1.5,16])
plt.xlabel('interest rate (lĂ£i suáº¥t)')
plt.ylabel('inflationary (láº¡m phĂ¡t)')
plt.show()
# Building Xbar 
one = np.ones((Y.shape[0], 1))#khai bĂ¡o  máº£ng one
Xbar = np.concatenate((one, Y), axis = 1)
# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)#tĂ­nh A
b = np.dot(Xbar.T, x)#tĂ­nh b
w = np.dot(np.linalg.pinv(A), b)#tĂ­nh w dá»±a vĂ o nghá»‹ch Ä‘áº£o ma tráº­n A
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
y0 = np.linspace(2, 15.5, 2)
x0 = w_0 + w_1*y0
#váº½ cĂ¡c Ä‘Æ°á»ng
plt.plot(Y.T, x.T, 'ro')     # data 
plt.plot(y0, x0)               # the fitting line
plt.axis([4.5,20,1.5,16])
plt.xlabel('muc lai suat')
plt.ylabel('ti le lam phat')
plt.show()
x = w_1*11 + w_0
#in má»©c láº¡m phĂ¡t vs lĂ£i suáº¥t 11
print( u'du doan ti le lam phat khi muc lai suat la 11: ' , x )