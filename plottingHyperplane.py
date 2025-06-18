from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Load data and train the model
try:
    # Assuming the data file has no header and columns are separated by whitespace
    cars = pd.read_csv('./Data/cars.data', header=None, sep=r'\s+')
    cars.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    # Drop rows with missing 'horsepower' values, which are marked with '?'
    cars = cars[cars.horsepower != '?']
    cars.horsepower = cars.horsepower.astype(float)
except FileNotFoundError:
    print("Error: 'Data/cars.data' not found. Please ensure the file exists in the correct directory.")
    exit()


# Define features (X) and target (y)
y = cars['mpg'].values
X = cars[['horsepower', 'weight']].values

# Fit linear regression model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# --- Plotting ---

# Create a figure and a 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the actual data points
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o', label='Actual data')

ax.set_xlabel('Horsepower')
ax.set_ylabel('Weight')
ax.set_zlabel('MPG')
ax.set_title('Regression Hyperplane for MPG Prediction')

# Create a meshgrid to plot the hyperplane
x_surf = np.arange(X[:, 0].min() - 10, X[:, 0].max() + 10, 10)
y_surf = np.arange(X[:, 1].min() - 100, X[:, 1].max() + 100, 100)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)

# Predict z values (MPG) for the meshgrid to form the plane
exog = pd.DataFrame({'horsepower': x_surf.ravel(), 'weight': y_surf.ravel()})
z_surf = reg.predict(exog).reshape(x_surf.shape)

# Plot the surface of the hyperplane
ax.plot_surface(x_surf, y_surf, z_surf, color='blue', alpha=0.4, label='Regression Plane')

plt.legend()
plt.show()