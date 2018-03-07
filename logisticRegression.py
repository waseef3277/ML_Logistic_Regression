from __future__ import print_function
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# scale larger positive and values to between -1,1 depending on the largest
# value in the data
min_max_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
df = pd.read_csv('data.csv', header=0)

df.columns = ["grade1", "grade2", "label"]

x = df["label"].map(lambda x: float(x.rstrip(';')))

# formats the input data into two arrays, one of independant variables
# and one of the dependant variable
X = df[["grade1","grade2"]]
X = np.array(X)
X = min_max_scalar.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

# creating testing and training set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)


def Sigmoid(z):
    G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
    return G_of_Z


def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i]*theta[i]
    return Sigmoid(z)

def Cost_Function(X, Y, theta, m):
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        hi = Hypothesis(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-hi)
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    print('cost is ', J)
    return J

def Cost_Function_Derivative(X, Y, theta, j, m, alpha):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = Hypothesis(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha)/float(m)
    J = constant * sumErrors
    return J

def Gradient_Descent(X, Y, theta, m, alpha):
    new_theta = []
    constant = alpha/m
    for j in range(len(theta)):
        CFDerivative = Cost_Function_Derivative(X, Y, theta, j, m, alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta


def Logistic_Regression(X, Y, alpha, theta, num_iters):
    m = len(Y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X, Y, theta, m, alpha)
        theta = new_theta
        Cost_Function(X, Y, theta, m)
        print('theta ', theta)
        print('cost is ', Cost_Function(X, Y, theta, m))


initial_theta = [0, 0]
alpha = 0.1
iterations = 5000
Logistic_Regression(X, Y, alpha, initial_theta, iterations)