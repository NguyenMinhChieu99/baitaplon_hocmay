#khai bĂ¡o thÆ° viá»‡n
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T# Visualize data 
#biá»ƒu diá»…n dltrĂªn Ä‘á»“ thá»‹
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
# Building Xbar 
one = np.ones((X.shape[0], 1)) #khai bĂ¡o  máº£ng one 
Xbar = np.concatenate((one, X), axis = 1)
# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar) #tĂ­nh A
b = np.dot(Xbar.T, y) #tĂ­nh b
w = np.dot(np.linalg.pinv(A), b)#tĂ­nh w dá»±a vĂ o nghá»‹ch Ä‘áº£o ma tráº­n A
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0 
#váº½ cĂ¡c Ä‘Æ°á»ng
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
y = w_1*178 + w_0
#in ra ng cĂ³ ccao 159cm
print( u'Predict weight of person with height 159 cm: %.2f (kg), real number: 52 (kg)'  %(y) )