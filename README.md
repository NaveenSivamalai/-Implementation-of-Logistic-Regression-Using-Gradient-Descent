# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NAVEEN S
RegisterNumber:  212222110030
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data= np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:,[0,1]]
y = data[:,2]


X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))

  plt.plot()
X_plot =np.linspace(-10, 10, 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()


def costFunction(theta,X,y):
  h = sigmoid(np.dot(X, theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)


X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2  ])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)


def cost(theta, X, y):
  h= sigmoid(np.dot(X, theta))
  J = - (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
  return J

  def gradient(theta, X, y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h - y) / X.shape[0]
  return grad

  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),
                        method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta, X, y):
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0], 1)), X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()



  plotDecisionBoundary(res.x, X, y)


  prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
print(prob)


def predict(theta, X):
  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
  prob = sigmoid(np.dot(X_train, theta))
  return (prob >= 0.5).astype(int)


  np.mean(predict(res.x, X) == y)
```

## Output:
# ARRAY VALUE OF X:

![270337529-c0e7ccf6-596f-426c-9ec6-bb6c44db5279](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/24edb823-701a-47cc-8939-22ca7f3fb258)


# ARRAY VALUE OF Y:

![270337621-18c2caf9-6cb8-425e-af9d-2cc5793aa16c](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/cdffea9f-02f6-45cb-9433-bcdddd4932e2)


# EXAM 1- SCORE GRAPH:
![270337765-fd6cb900-697f-4525-8c84-19890bc8af11](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/f8bb6690-c03c-4f0b-bb9a-e4c9f19ab89a)



# SIGMOID FUNCTION GRAPH:
![270338011-5a24f549-2334-4e2b-bfdb-ba3f42db8bb9](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/d579b428-fcbc-4759-9574-fb6942e23851)



# X_TRAIN GRAD VALUE:
![270338152-2fc97b75-229a-4531-bd54-1db63c0d96a9](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/136fbd0b-d736-4141-adb5-7f07b49f3817)



# Y_ TRAIN GRAD VALUE:
![270338294-49c4267f-7764-44d8-8e59-e384d19405d9](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/33951971-b85e-45b2-bdd1-9f32764b431c)



# PRINT RES.X
![270338417-9b6c0ec7-6ba9-4f37-aaf3-05435a68e7d3](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/92f3578e-a273-44c3-99be-c187bd0c62c7)



# DECISION BOUNDARY - GRAPH FOR EXAM SCORE:
![270338537-4514c07c-d7f7-41d6-b6b5-7a7b632edcea](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/7ad2dd3a-f0b7-4adc-9639-862e566082a8)


# PROBABILITY VALUE:
![270338671-a8af9b5d-e5ef-4d64-9cff-78fd37b38ce6](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/310585fa-9600-4d7b-b290-c23b98f6840a)



# PREDICTION VALUE OF MEAN:

![270338777-2e5fe8ee-4633-47b8-83fa-e0583fb12d20](https://github.com/NaveenSivamalai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123792574/3a997f20-9629-494b-9e38-7fc6b02a7c13)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

