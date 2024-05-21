import numpy as np
import matplotlib.pyplot as plt
from cryptographpy_helper import generate_cyclic_group, hkdf
import secrets

# np.random.seed(42)

# Dataset
d = 500
train_n = 2001
test_n = 500
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = ((np.sign(X_train.dot(a_true)) + 1) / 2).flatten()
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = ((np.sign(X_test.dot(a_true)) + 1) / 2).flatten()

# Constants
num_iterations = 50
num_clients = 3
epochs = 50
lambda_bits = 128
alpha = 1
epsilon = 1

# we scale everything by the rounding factor because modular operations
# are numerically instable on floats in python this is equivalent to rounding
# each float to the 6th digit 
rounding_factor = 1e6 

class LogisticRegressionClient():
  def __init__(self, X, y, lr = 0.01, lambda_val = 0.2):
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
      reg_cost = (lambda_reg / (2 * m)) * np.sum(weights ** 2)
      return cost + reg_cost
  def compute_gradient(self, X, y, weights, lambda_reg):
      m = len(y)
      h = self.sigmoid(X @ weights)
      gradient = np.dot(X.T, (h - y)) / m
      gradient += (lambda_reg / m) * weights
      return gradient

  def update_weights(self, weights):
    self.weights = weights

  def train_local(self, num_iterations):
    for _ in range(num_iterations):
      self.weights -= self.lr * self.compute_gradient(self.X, self.y, self.weights, self.lambda_val)

  def predict(self, X):
    probabilities = self.sigmoid(X @ self.weights)
    return (probabilities >= 0.5).astype(int)  

  def evaluate(self, X_test, y_test):
    y_pred = self.predict(X_test)
    test_accuracy = 1 - np.mean(y_pred != y_test)
    return test_accuracy

class LogisticRegressionServer():
  def __init__(self, initial_weights = None):
    self.weights = initial_weights

  def sigmoid(self, z):
    return 1 / (1 + np.exp(np.float128(-z)))

  def update_weights(self, weights):
    self.weights = weights

  def predict(self, X):
    probabilities = self.sigmoid(X @ self.weights)
    return (probabilities >= 0.5).astype(int)  
  
  def evaluate(self, X_test, y_test):
    y_pred = self.predict(X_test)
    test_accuracy = 1 - np.mean(y_pred != y_test)
    return test_accuracy

X_train_chunks = np.split(X_train, num_clients)
y_train_chunks = np.split(y_train, num_clients)

clients = []
for i in range(num_clients):
    clients.append(LogisticRegressionClient(X_train_chunks[i], y_train_chunks[i]))

server = LogisticRegressionServer()

(p, q, g) = generate_cyclic_group(lambda_bits)

print(f"Group G of order q={q} modulo p={p} with generator g={g}")

scale = 2 / (num_clients * len(X_train_chunks[0]) * alpha * epsilon)
noise = np.random.laplace(scale=scale, size=num_clients)
# noise *= rounding_factor
# noise = np.rint(noise).astype(int)

print ("doing diffie helman key exchange")

# secret_keys = [[secrets.randbelow(q) for _ in range(num_clients)] for _ in range(num_clients)]
# public_keys = [[pow(g, secret_keys[i][j], p) for j in range(num_clients)] for i in range(num_clients)]

secret_keys = [secrets.randbelow(q) for _ in range(num_clients)]
public_keys = [pow(g, secret_keys[i], p) for i in range(num_clients)]

common_keys = [[None]*num_clients for _ in range(num_clients)]
final_keys = [[None]*num_clients for _ in range(num_clients)]

for i in range(num_clients):
    for j in range(num_clients):
        c_ij = pow(public_keys[j], secret_keys[i], p)
        common_keys[i][j] = c_ij.to_bytes(lambda_bits,  byteorder='big')
        final_keys[i][j] = hkdf(b'', c_ij.to_bytes(lambda_bits,  byteorder='big'), b'', lambda_bits // 8)

print ("train models")
for _ in range(epochs):

  local_weights = []
  for i in range(num_clients):
    # print("train accuracy:", clients[i].evaluate(clients[i].X, clients[i].y))
    clients[i].train_local(num_iterations=num_iterations)
    local_weight = clients[i].weights.copy()
    local_weight += 1
    local_weight *= rounding_factor
    local_weight = np.rint(local_weight).astype(int)
    # local_weight += noise[i]
    local_weights.append(local_weight)

  # print("local weights: ", local_weights)

  # local_weights = np.array(local_weights)
  # original_weights = np.mean(local_weights, axis=0)
  # print(np.mean(local_weights, axis=0))
  # min_weight = abs(np.min(local_weights))
  # min_weight = 1
  # print(min_weight)
  # local_weights += min_weight
  # local_weights *= rounding_factor
  # local_weights = np.rint(local_weights).astype(int)

  encrypted_weights = []

  for i in range(num_clients):
      sum_r_ij = sum(int.from_bytes(final_keys[i][j], 'big') for j in range(i+1, num_clients))
      sum_r_ki = sum(int.from_bytes(final_keys[i][k], 'big') for k in range(i))
      encrypted_weight = (local_weights[i] + sum_r_ij - sum_r_ki) % p
      encrypted_weights.append(encrypted_weight)

  aggregate_weights = sum(encrypted_weights) % p 

  # print(aggregate_weights)

  aggregate_weights /= num_clients
  aggregate_weights /= rounding_factor
  aggregate_weights -= 1
  aggregate_weights = aggregate_weights.astype('float64')

  # print(aggregate_weights)

  server.update_weights(aggregate_weights)

  print("train accuracy", server.evaluate(X_train, y_train))
  print("test accuracy", server.evaluate(X_test, y_test))

  for i in range(num_clients):
    clients[i].update_weights(aggregate_weights)


baseline = LogisticRegressionClient(X_train, y_train, lr=0.1)
baseline.train_local(num_iterations=num_iterations * epochs)

print("baseline train accuracy", baseline.evaluate(X_train, y_train))
print("baseline test accuracy", baseline.evaluate(X_test, y_test))

