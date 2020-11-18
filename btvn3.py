#khai bĂ¡o thÆ° viá»‡n
import numpy as np #tĂ­nh toĂ¡n máº£ng
import pandas as pd #Ä‘á»c file csv
import matplotlib.pyplot as plt #váº½ Ä‘á»“ thá»‹
from sklearn import linear_model, datasets#cĂ³ sáºµn dl,sd Ä‘á»ƒ láº¥y kq
from sklearn.model_selection import train_test_split#chia tĂ¡ch Ä‘á»ƒ thá»­ train vĂ  test
from sklearn.metrics import accuracy_score
#load dá»¯ liá»‡u
data = pd.read_csv('winequality-red.csv',encoding='utf-8', sep=';')#Äá»c tá»‡p giĂ¡ trá»‹ (csv) vĂ o Data
one = np.ones((data.shape[0],1))#khai bĂ¡o  máº£ng one cĂ³ Ä‘á»™ dĂ i báº±ng Ä‘á»™ dĂ i máº£ng data = 15
data.insert(loc=0, column='A', value=one)#chĂ¨n cá»™t A vĂ o data vá»‹ trĂ­
data_x = data[["A","density"]]#Ä‘á»c dl máº­t Ä‘á»™
data_y = data["alcohol"]#Ä‘á»c dl ná»“ng Ä‘á»™
# x = [1,máº­t Ä‘á»™]; w=[w0, w1]
#f(x) = y=x*w = [1,máº­t Ä‘á»™] * [w0,w1] = 1*w0 + máº­t Ä‘á»™*w1 = ná»“ng Ä‘á»™
#output train => w1, w0
#tĂ¡ch training vĂ  test sets
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=500)#hĂ m train_test_split tráº£ vá» 4 hĂ m
#mĂ´ Ä‘un thá»±c hiá»‡n tĂ­nh toĂ¡n tuyáº¿n tĂ­nh
regr = linear_model.LinearRegression(fit_intercept=False)#dá»¯ liá»‡u cÄƒn giá»¯a Ä‘á»ƒ tĂ­nh toĂ¡n Ä‘á»™ lá»‡ch #fit_intercept=false for calculating the bias
regr.fit(X_train, y_train) #sau Ä‘Ă³ Ä‘i training
Y_pred = regr.predict(X_test) #Ä‘i test #dá»± Ä‘oĂ¡n nhĂ£n
#plt.scatter(data.density, data.alcohol)
#plt.plot(truyá»n Ä‘á»‘i sá»‘)
plt.plot(data.density, data.alcohol, 'ro-')#váº½ Ä‘á»“thá»‹ cháº¥m trĂ²n
#plt.scatter(data.density, data.alcohol, color='black')
plt.plot(X_test.density, Y_pred, color='blue')#váº½ Ä‘á»“ thá»‹ Ä‘g mĂ u xanh
w_0 = regr.coef_[0]
w_1 = regr.coef_[1]
x0 = np.linspace(0.99, 1.004, 2)
y0 = w_0 + w_1*x0
plt.plot(x0,y0, color='pink')#váº½ Ä‘á»“ thá»‹ Ä‘g há»“i uy tuyáº¿n tĂ­nh
plt.show()