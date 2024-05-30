import secrets
import numpy as np
from cryptographpy_helper import hkdf, generate_shares, aes_encrypt, aes_decrypt
from node import Node



class Client:
  def __init__(self, server_host, server_port, model, X, y):
    self.server_host = server_host
    self.server_port = server_port
    self.X = X
    self.y = y
    self.model = model
    self.node = Node(None, receiver_handler=self.handle_message)
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
    print(f"I am running at server port {self.server_port}")
    self.node.start()
    self.connect()

  def wait(self):
    self.node.t2.join()

  def connect(self):
    self.node.connect_to_node(self.server_host, self.server_port)

  def train_model(self):
    # if (self.node_id == 0):
    #   self.node.stop()
    #   return
    print("training model for epoch", self.cur_epoch)
    self.model.train_local(self.num_iterations)

    scale = 2 / (self.num_clients * len(self.X))
    noise = np.random.laplace(scale=scale)

    model_weights = self.model.get_weights()
    

    for model_param in model_weights.keys():

      encrypted_weight = model_weights[model_param] + noise
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

      model_weights[model_param] = list(encrypted_weight)
      
    server_conn = self.node.outward_connections[0]
    self.node.send_message(({"weights": model_weights}), server_conn)
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
      aggregate_weight_dict = msg["aggregate_weights"]
      for key in aggregate_weight_dict.keys():
        aggregate_weight = np.array(aggregate_weight_dict[key]).astype('float64')
        aggregate_weight /= (self.num_clients)
        aggregate_weight /= 1e6
        aggregate_weight -= 1
        aggregate_weight = aggregate_weight.astype('float64')
        aggregate_weight_dict[key] = aggregate_weight

      self.model.update_weights(aggregate_weight_dict)

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


