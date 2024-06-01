import numpy as np
from client import Client


class LogisticRegressionClient:
    def __init__(self, X, y, lr=0.01, lambda_val=0.2):
        self.X = X
        self.y = y
        self.lr = lr
        self.lambda_val = lambda_val
        _, n = X.shape
        self.weights = np.zeros(n)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, weights, lambda_reg):
        m = len(y)
        h = self.sigmoid(X @ weights)
        cost = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
        reg_cost = (lambda_reg / (2 * m)) * np.sum(weights**2)
        return cost + reg_cost

    def compute_gradient(self, X, y, weights, lambda_reg):
        m = len(y)
        h = self.sigmoid(X @ weights)
        gradient = np.dot(X.T, (h - y)) / m
        gradient += (lambda_reg / m) * weights
        return gradient

    def update_weights(self, weights):
        self.weights = weights["logistic_weights"]

    def train_local(self, num_iterations):
        for _ in range(num_iterations):
            self.weights -= self.lr * self.compute_gradient(
                self.X, self.y, self.weights, self.lambda_val
            )

    def get_weights(self):
        return {"logistic_weights": self.weights}

    def predict(self, X):
        probabilities = self.sigmoid(X @ self.weights)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        test_accuracy = 1 - np.mean(y_pred != y_test)
        return test_accuracy


num_clients = 5

d = 500
train_n = 2000
test_n = 500
X_train = np.random.normal(0, 1, size=(train_n, d))
a_true = np.random.normal(0, 1, size=(d, 1))
y_train = ((np.sign(X_train.dot(a_true)) + 1) / 2).flatten()
X_test = np.random.normal(0, 1, size=(test_n, d))
y_test = ((np.sign(X_test.dot(a_true)) + 1) / 2).flatten()

X_train_chunks = np.split(X_train, num_clients)
y_train_chunks = np.split(y_train, num_clients)

server_host = "192.168.192.231"

server_port = 5000

clients = []

for i in range(num_clients):
    model = LogisticRegressionClient(X_train_chunks[i], y_train_chunks[i])
    client = Client(
        server_host, server_port, model, X_train_chunks[i], y_train_chunks[i]
    )
    client.run()
    clients.append(client)

for client in clients:
    client.wait()

for client in clients:
    print("train accuracy", client.model.evaluate(X_train, y_train))
    print("test accuracy", client.model.evaluate(X_test, y_test))
