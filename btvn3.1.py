#khai bĂ¡o thÆ° viá»‡n
import numpy as np
import pandas as pd
	
from sklearn import linear_model
	
wine = pd.read_csv("winequality-red.csv", sep=";")
wine.head
	
clf = linear_model.LinearRegression()
	 
# Táº¡o dataframe chá»‰ chá»©a data lĂ m biáº¿n giáº£i thĂ­ch
wine_except_quality = wine.drop("quality", axis=1)
X = wine_except_quality

# Sá»­ dá»¥ng quality lĂ m biáº¿n má»¥c tiĂªu
Y = wine['quality']	 
# Táº¡o model
clf.fit(X, Y)
# Há»‡ sá»‘ há»“i quy
print(pd.DataFrame({"Name":wine_except_quality.columns,
                    "Coefficients":clf.coef_}).sort_values('Coefficients') )
