import secrets
import numpy as np
from cryptographpy_helper import hkdf, generate_shares, aes_encrypt, aes_decrypt
from node import Node

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

class Client:
  def __init__(self, server_host, server_port, client_port, X, y):
    self.server_host = server_host
    self.server_port = server_port
    self.client_port = client_port
    self.X = X
    self.y = y
    self.model = LogisticRegressionClient(X, y)
    self.node = Node(client_port, receiver_handler=self.handle_message)
    self.cyclic_group_params = None
    self.node_id = -1
    self.shared_keys = {}
    self.secret_shares = {}
    self.node_set = set()

    self.cur_epoch = 0

    # We should probably get this from the server but hard code for now
    self.num_iterations = 100
    self.lambda_bits = 128

  def run(self):
    self.node.start()
    self.connect()

  def wait(self):
    self.node.t1.join()
    self.node.t2.join()

  def connect(self):
    self.node.connect_to_node(self.server_host, self.server_port)

  def train_model(self):
    print("training model for epoch", self.cur_epoch)
    self.model.train_local(self.num_iterations)

    scale = 2 / (self.num_clients * len(self.X))
    noise = np.random.laplace(scale=scale)

    encrypted_weight = self.model.weights + noise
    encrypted_weight += 1
    encrypted_weight *= 1e6
    encrypted_weight = np.rint(encrypted_weight).astype(int)

    (p, _, _) = self.cyclic_group_params

    for node_id in self.node_set:
      key = self.shared_keys[node_id]
      key_val = int.from_bytes(key, 'big')
      if node_id > self.node_id:
        encrypted_weight = (encrypted_weight + key_val) % p
      else:
        encrypted_weight = (encrypted_weight - key_val) % p

    server_conn = self.node.outward_connections[0]
    self.node.send_message(({"weights": encrypted_weight.tolist()}), server_conn)
    self.cur_epoch += 1

  def handle_message(self, conn, msg):

    if "group_params" in msg:
      self.threshold = msg['group_params'][-1]
      self.cyclic_group_params = msg['group_params'][:3]
      self.diffie_helman_exchange()

    if "public_keys" in msg:
      (p, _, _) = self.cyclic_group_params
      public_keys = msg["public_keys"]
      self.num_clients = len(public_keys)
      for (node_id, pub_key) in public_keys:
        
        if pub_key == self.pub_key:
          self.node_id = node_id
        else:
          self.node_set.add(node_id)
          common_key = pow(pub_key, self.secret_key, p)
          final_key = hkdf(b'', common_key.to_bytes(self.lambda_bits,  byteorder='big'), b'', self.lambda_bits // 8)
          self.shared_keys[node_id] = final_key

      self.shamirs_secret_exchange()

    if "encrypted_secret_shares" in msg:
      encrypted_secret_shares = msg["encrypted_secret_shares"]

      self.node_set = set()
      self.num_clients = len(encrypted_secret_shares) + 1
      for (node_id, encrypted_share) in encrypted_secret_shares:
        self.node_set.add(node_id)
        decrypted_share = aes_decrypt(encrypted_share, self.shared_keys[node_id])
        decrypted_share = int.from_bytes(decrypted_share, 'big')
        self.secret_shares[node_id] = decrypted_share 

    if "dropout_set" in msg:
      self.node_set -= set(msg["dropout_set"])
      self.num_clients = len(self.node_set) + 1

    if "aggregate_weights" in msg:
      aggregate_weights = np.array(msg["aggregate_weights"]).astype('float64')
      aggregate_weights /= (self.num_clients)
      aggregate_weights /= 1e6
      aggregate_weights -= 1
      aggregate_weights = aggregate_weights.astype('float64')
      self.model.update_weights(aggregate_weights)

    if "recover_secrets" in msg:
      nodes_to_recover = msg["recover_secrets"]
      shares_to_send = []
      for (node_id) in nodes_to_recover:
        shares_to_send.append((node_id, (self.node_id+1, self.secret_shares[node_id])))

      server_conn = self.node.outward_connections[0]
      self.node.send_message(({"decrypted_secret_shares": shares_to_send}), server_conn)

    
    if "status" in msg:
      if msg["status"] == "stop":
        self.node.stop()
      elif msg["status"] == "continue":
        self.train_model()


  def diffie_helman_exchange(self):
    (p, q, g) = self.cyclic_group_params

    self.secret_key = secrets.randbelow(q)
    self.pub_key = pow(g, self.secret_key, p)

    server_conn = self.node.outward_connections[0]
    self.node.send_message(({"public_key": self.pub_key}), server_conn)

  def shamirs_secret_exchange(self):
    (p, _, _) = self.cyclic_group_params

    total_clients = max(self.node_id + 1, np.max(list(self.node_set)) + 1)
    secret_key_shares = generate_shares(self.secret_key, total_clients, self.threshold, p)
    self.secret_shares[self.node_id] = secret_key_shares[self.node_id][1]
    encrypted_shares = []
    for (node_id, shared_key) in self.shared_keys.items():
      encrypted = aes_encrypt(secret_key_shares[node_id][1].to_bytes(self.lambda_bits, byteorder='big'), shared_key)
      encrypted_shares.append((self.node_id, node_id, encrypted))

    server_conn = self.node.outward_connections[0]
    self.node.send_message(({"encrypted_secret_shares": encrypted_shares}), server_conn)

num_clients = 5

d = 1000
train_n = 2000
test_n = 500
X_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = ((np.sign(X_train.dot(a_true)) + 1) / 2).flatten()
X_test = np.random.normal(0,1, size=(test_n,d))
y_test = ((np.sign(X_test.dot(a_true)) + 1) / 2).flatten()

X_train_chunks = np.split(X_train, num_clients)
y_train_chunks = np.split(y_train, num_clients)

server_host = '192.168.192.231'
server_port = 5000

clients = []

for i in range(num_clients):
  client = Client(server_host, server_port, server_port + i + 1, X_train_chunks[i], y_train_chunks[i])
  client.run()
  clients.append(client)

for client in clients:
  client.wait()

for client in clients:
  print("train accuracy", client.model.evaluate(X_train, y_train))
  print("test accuracy", client.model.evaluate(X_test, y_test))

