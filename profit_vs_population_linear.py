import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

#Load Dataset
dataset= pd.read_csv("data.csv")
x= dataset[["population"]]
y= dataset[["profit"]]

#Training
model = lm.LinearRegression()
model.fit(x,y)
print("Intercept", model.intercept_)
print("Coefficient", model.coef_)
input=pd.DataFrame([[35000]], columns=["population"])
result=model.predict( input  )
print( round(result[0][0] , 2))

#Visualisation
plt.scatter(x,y)
plt.plot(x,model.predict(x), color="red")
plt.xlabel("population")
plt.ylabel("profit")
plt.show()

#Evaluation

y_pred= model.predict(x)
print("MSE : ", mean_squared_error(y,y_pred))
print("RMSE : ", np.sqrt(mean_squared_error(y,y_pred)))


