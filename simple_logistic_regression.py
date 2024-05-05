import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights, lambda_reg):
    m = len(y)
    h = sigmoid(X @ weights)
    cost = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
    reg_cost = (lambda_reg / (2 * m)) * np.sum(weights ** 2)
    return cost + reg_cost
def compute_gradient(X, y, weights, lambda_reg):
    m = len(y)
    h = sigmoid(X @ weights)
    gradient = np.dot(X.T, (h - y)) / m
    gradient += (lambda_reg / m) * weights
    return gradient


def logistic_regression(X, y, lr, num_iterations, lambda_val):
    _, n = X.shape
    weights = np.zeros(n)

    for i in range(num_iterations):
        weights -= lr * compute_gradient(X, y, weights, lambda_val)
        if i % 100 == 0:
            cost = compute_cost(X, y, weights, lambda_val)
            print(f"Iteration {i}: Cost {cost}")

    return weights

def predict(X, weights):
    probabilities = sigmoid(X @ weights)
    return (probabilities >= 0.5).astype(int)  

def accuracy_score(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    accuracy = correct / len(y_true)
    return accuracy

d = 500
train_n = 2000
test_n = 500
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = ((np.sign(X_train.dot(a_true)) + 1) / 2).flatten()
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = ((np.sign(X_test.dot(a_true)) + 1) / 2).flatten()

num_iterations = 500
lambda_val = 0

weights = logistic_regression(X_train, y_train, lr=0.1, num_iterations=num_iterations, lambda_val=lambda_val)

y_pred = predict(X_test, weights)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
