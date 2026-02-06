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

data = pd.read_csv("50_Startups.csv")
data = data.iloc[:, [0, 4]]
data.columns = ["Population", "Profit"]

data["Population"] = (data["Population"] - data["Population"].mean()) / data["Population"].std()

plt.scatter(data["Population"], data["Profit"])
plt.xlabel("Scaled Population")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()

def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    return (1/(2*m)) * np.sum((h - y)**2)

m = len(data)
X_raw = data["Population"].values.reshape(m, 1)
X = np.append(np.ones((m, 1)), X_raw, axis=1)
y = data["Profit"].values.reshape(m, 1)
theta = np.zeros((2, 1))

print("Initial Cost:", computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.T, (predictions - y))
        theta -= alpha * (1/m) * error
        j_history.append(computeCost(X, y, theta))
    return theta, j_history

theta, j_history = gradientDescent(X, y, theta, 0.01, 1500)

print(f"Model: h(x) = {round(theta[0,0],2)} + {round(theta[1,0],2)}x")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\\Theta)$")
plt.title("Cost Function Reduction")
plt.show()

plt.scatter(data["Population"], data["Profit"])
x_line = np.linspace(data["Population"].min(), data["Population"].max(), 100)
y_line = theta[0,0] + theta[1,0] * x_line
plt.plot(x_line, y_line, color="r")
plt.xlabel("Scaled Population")
plt.ylabel("Profit ($10,000)")
plt.title("Linear Regression Fit")
plt.show()
```
## Output:
<img width="749" height="613" alt="Screenshot 2026-02-06 150339" src="https://github.com/user-attachments/assets/b8da75f8-4709-4e7d-a930-f1a9db306c4a" />
<img width="701" height="570" alt="Screenshot 2026-02-06 150355" src="https://github.com/user-attachments/assets/bbc8cb83-79d0-4664-8b09-5a43f34af0ba" />
<img width="747" height="568" alt="Screenshot 2026-02-06 150433" src="https://github.com/user-attachments/assets/055164a5-720e-4450-af7d-d1760eedd395" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
