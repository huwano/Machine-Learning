import pandas as pd
import numpy as np
from sklearn import linear_model

cars = pd.read_csv('./Data/cars.data', header=None, sep=r'\s+')

y = cars.iloc[:,0].values
X = cars.iloc[:,[3,4]].values

reg = linear_model.LinearRegression()
reg.fit(X,y)

print(f"Intercept: {reg.intercept_}")
print(f"Coefficients: {reg.coef_}")
print(f"Correlation coefficient: {np.corrcoef(reg.predict(X),y)[0, 1]}")

mse = sum((reg.predict(X) - y)**2) / len(y)
print(f"Mean Squared Error: {mse}")