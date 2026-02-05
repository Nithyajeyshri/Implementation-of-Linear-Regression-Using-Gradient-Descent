![WhatsApp Image 2026-02-04 at 4 02 22 PM](https://github.com/user-attachments/assets/6f0cc3f4-174c-4a07-b864-723814602e3a)# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. .Collect the dataset of input and output.

2.Load and preprocess the data.

3.Train the Simple Linear Regression model using the training data.

4.Use the trained model to predict the marks for new input values.

## Program:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("/content/score_updated.csv")
display(df.head(10))

# Visualize data
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()

x = df.iloc[:, 0:1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
lr = LinearRegression()
lr.fit(X_train, Y_train)

# Correcting variable case and generating predictions
print("X_train samples:", X_train.head())
print("Y_train samples:", Y_train.head())

# Generate predictions needed for metrics
y_pred = lr.predict(X_test)

# Regression line plot
plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title('Regression Line')
plt.show()

print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

# Metrics (fixed function name and defined y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)
```
## Output:

![WhatsApp Image 2026-02-04 at 4 02 22 PM](https://github.com/user-attachments/assets/7530ee06-6de4-4d15-8ccf-45963bce2693)
![WhatsApp Image 2026-02-04 at 4 02 22 PM (1)](https://github.com/user-attachments/assets/f1a521ad-c07e-442d-bd53-708b019d9086)
![WhatsApp Image 2026-02-04 at 4 02 22 PM (2)](https://github.com/user-attachments/assets/4a12a29f-3394-4e96-97ea-65fb68cfd5d1)
![WhatsApp Image 2026-02-04 at 4 02 22 PM (3)](https://github.com/user-attachments/assets/7f75cada-37ea-4c3e-98e7-f0dc88cecf45)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
