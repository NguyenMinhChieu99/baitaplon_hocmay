#thĂªm thÆ° viá»‡n
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score	

#khá»Ÿi táº¡o dá»¯ liá»‡u
iris = datasets.load_iris() #khá»Ÿi táº¡o dá»¯ liá»‡u máº«u
iris #In dá»¯ liá»‡u máº«u trĂªn terminal
iris_X = iris.data   #GĂ¡n giĂ¡ trá»‹ data cá»§a dá»¯ liá»‡u máº«u cho biáº¿n iris_X 
iris_y = iris.target #GĂ¡n giĂ¡ trá»‹ target cá»§a giá»¯ liá»‡u máº«u cho biáº¿n iris_Y
print ("sá»‘ lá»›p : %d" %len(np.unique(iris_y))) #hiá»ƒn thá»‹ sá»‘ lá»›p
print ("sá»‘ Ä‘iá»ƒm dá»¯ liá»‡u: %d" %len(iris_y)) #hiá»ƒn thá»‹ sá»‘ Ä‘iá»ƒm dá»¯ liá»‡u

#hiá»ƒn thá»‹ thá»­ 1 sá»‘ dá»¯ liá»‡u
X0 = iris_X[iris_y == 0,:] #tĂ¡ch thĂ nh class 0 tá»« tá»‡p dá»¯ liá»‡u máº«u
print ("\ncĂ¡c Ä‘iá»ƒm máº«u tá»« lá»›p 0:\n", X0[:5,:]) # hiá»ƒn thá»‹ class 0

X1 = iris_X[iris_y == 1,:] #tĂ¡ch thĂ nh class 1 tá»« tá»‡p dá»¯ liá»‡u máº«u
print ("\ncĂ¡c Ä‘iá»ƒm máº«u tá»« lá»›p 1:\n", X1[:5,:]) # hiá»ƒn thá»‹ class 1

X2 = iris_X[iris_y == 2,:] #tĂ¡ch thĂ nh class 2 tá»« tá»‡p dá»¯ liá»‡u máº«u
print ("\ncĂ¡c Ä‘iá»ƒm máº«u tá»« lá»›p 2:\n", X2[:5,:]) # hiá»ƒn thá»‹ class 2

#tĂ¡ch táº­p test tá»« táº­p training báº¡n Ä‘áº§u
X_train, X_test, Y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

#in kĂ­ch thÆ°á»›c táº­p training vĂ  táº­p test
print ("kĂ­ch thÆ°á»›c táº­p training: %d" %len(Y_train)) #Sá»‘ lÆ°á»£ng training
print ("kĂ­ch thÆ°á»›c táº­p test    : %d" %len(y_test)) #sá»‘ lÆ°á»£ng test

#xĂ©t thá»­ TH k = 1 tĂ¬m lĂ¢n cáº­n cĂ³ nhĂ£n gáº§n nháº¥t Ä‘á»ƒ Ä‘oĂ¡n nhĂ£n cá»§a Ä‘iá»ƒm nĂ y
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2) #táº¡o 1 model vá»›i k = 1 vĂ  khoáº£ng cĂ¡ch euclidian
clf.fit(X_train, Y_train) #training
Y_pred = clf.predict(X_test) #training
	
#in thá»­ káº¿t quáº£
print ("káº¿t quáº£ cho 20 Ä‘iá»ƒm dá»¯ liá»‡u test:")
print ("nhĂ£n dá»± Ä‘oĂ¡n   :", Y_pred[20:40])
print ("káº¿t quáº£ thá»±c táº¿:", y_test[20:40])

#Ä‘Ă¡nh giĂ¡ sá»± chĂ­nh xĂ¡c cá»§a TH k = 1
print ("sá»± chĂ­nh xĂ¡c cá»§a 1NN: %d" %(100*accuracy_score(y_test, Y_pred))," %")

#xĂ©t TH k = 10 tĂ¬m lĂ¢n cáº­n cĂ³ nhĂ£n gáº§n nháº¥t Ä‘á»ƒ Ä‘oĂ¡n nhĂ£n cá»§a Ä‘iá»ƒm nĂ y
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2) #táº¡o 1 model vá»›i k = 10 vĂ  khoáº£ng cĂ¡ch euclidian
clf.fit(X_train, Y_train) #training #
Y_pred = clf.predict(X_test) #training
#in thá»­ káº¿t quáº£
print ("Print results for 20 test data points:")#in kq dá»± Ä‘oĂ¡n vĂ  thá»±c táº¿ Ä‘á»ƒ so sĂ¡nh
print ("Predicted labels: ", Y_pred[20:40])# y_lĂ  má»t máº£ngláº¥y tá»« 20 Ä‘Ă©n 39- nhĂ£n dá»± Ä‘oĂ¡n
print ("Ground truth    : ", y_test[20:40])# nhĂ£n thá»±c táº¿
#Ä‘Ă¡nh giĂ¡ sá»± chĂ­nh xĂ¡c cá»§a TH k = 10 vĂ  viá»‡c bá» phiáº¿u
print ("Accuracy of 10NN with major voting: %d" %(100*accuracy_score(y_test, Y_pred))," %")
#Ä‘Ă¡nh trá»ng sá»‘ cho cĂ¡c Ä‘iá»ƒm lĂ¢n cáº­n
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = "distance")#táº¡o 1 model vá»›i k = 10 vĂ  khoáº£ng cĂ¡ch euclidian
clf.fit(X_train, Y_train) #training
Y_pred = clf.predict(X_test) #training
#Ä‘Ă¡nh giĂ¡ sá»± chĂ­nh xĂ¡c cá»§a TH k = 10 
print ("Accuracy of 10NN (1/distance weights): %d" %(100*accuracy_score(y_test, Y_pred))," %")

#dá»‹nh nghÄ©a hĂ m tĂ­nh khoáº£ng cĂ¡ch 
def myweight(distances):
    sigma2 = .5 #cĂ³ thá»ƒ thay Ä‘á»•i con sá»‘ nĂ y
    return np.exp(-distances**2/sigma2) #tráº£ vá» giĂ¡ trá»‹ cá»§a hĂ m

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, Y_train) #training
Y_pred = clf.predict(X_test) #training
#Ä‘Ă¡nh giĂ¡ sá»± chĂ­nh xĂ¡c cá»§a TH k = 10 
print ("Accuracy of 10NN (customized weights): %d" %(100*accuracy_score(y_test, Y_pred))," %")