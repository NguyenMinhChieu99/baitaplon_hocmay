import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv("Boston.csv")

#giĂ¡(medv)
X = data.values[: , 14]

#cĂ¡c thuá»™c tĂ­nh cĂ²n láº¡i trá»« 
Y = data.drop("medv", axis=1)

regr = linear_model.LinearRegression()
# Táº¡o model
regr.fit(Y, X)

# Há»‡ sá»‘ há»“i quy
print(pd.DataFrame({"Name":Y.columns,"Há»‡ sĂ´":regr.coef_}).sort_values(by='Há»‡ sĂ´') )
# Sai sá»‘
print(regr.intercept_)