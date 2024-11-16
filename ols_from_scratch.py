# Importing libraries
import numpy as np
import pandas as pd

# Creating a synthetic dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [20000, 25000, 30000, 35000, 40000, 50000, 55000, 60000, 65000, 70000]
}

# Converting it into pandas dataframe
import pandas as pd
df = pd.DataFrame(data)
df.head()

# Separating X and Y / dependent and independant valriable.
X = df['Experience']
y = df['Salary']

# Finding mean of X and Y

X_mean = np.mean(X)
y_mean = np.mean(y)

#The formula for OLS method is    (sumation of (x - x_bar)) X (sumation of (y - y_bar)) / (sumation of (x - x_bar)square)
# Calculate the slope (m) and intercept (b)

# Calculating the numerator
numerator = np.sum((X - X_mean) * (y - y_mean))

# Calculating the denomenator
denominator = np.sum((X - X_mean) ** 2)

# Calculating Slope
m = numerator / denominator

# Calculating intercept
b = y_mean - m * X_mean


print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Creating a predict function
def predict(x):
    return m * x + b

# Test the model with a new value
x_test = 6
y_pred = predict(x_test)
print(f"Predicted value for x={x_test}: {y_pred}")

# Checking error
def error(y_true, y_pred):
  error = y_true - y_pred
  return error

y_true = 6
print(error(y_true, y_pred))

# Assembling the code
class Regression:
  #Solving the OLS problem
  def train_model(self, X_array, y_array):
    self.X_mean = np.mean(X_array)
    self.y_mean = np.mean(y_array)
    self.numerator = np.sum((X_array - X_mean) * (y_array - y_mean))
    self.denominator = np.sum((X_array - X_mean) ** 2)
    self.m = numerator / denominator
    self.b = self.y_mean - self.m * self.X_mean
    print(f"Slope (m): {self.m}")
    print(f"Intercept (b): {self.b}")

  def predict(self, new_data):
    return self.m * new_data + self.b

  def error(self, y_true, y_pred):
    error = y_true - y_pred
    error_score = error / y_true
    return error_score

linear_regression = Regression()

# Checking if it works
linear_regression.train_model(X, y)

# Predicting
y_pred = linear_regression.predict(9)
print(y_pred)

#Checking error
y_actual = 65000
error_score = linear_regression.error(y_actual, y_pred)
print(error_score)