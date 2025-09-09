import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

dataset= pd.read_csv("data.csv")
x= dataset[["population"]]
y= dataset[["profit"]]
plt.scatter(x,y)
plt.xlabel("population")
plt.ylabel("profit")
plt.show()
model = lm.LinearRegression()
model.fit(x,y)
input=pd.DataFrame([[35000]], columns=["population"])
result=model.predict( input  )
print( round(result[0][0] , 2))
