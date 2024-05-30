import numpy as np
from cryptographpy_helper import generate_cyclic_group, recover_secret, hkdf
from functools import partial
from node import Node
import threading

class Server:
    def __init__(self, port, num_clients, num_epochs):
      self.port = port
      self.num_clients = num_clients
      self.num_epochs = num_epochs
      self.cur_epoch = 0
      self.connection_count = 0
      self.connection_dict = {}
      self.node_set = set()
      self.new_node_set = set()
      self.public_keys = {}
      self.encrypted_weights = []
      self.encrypted_secret_shares = []
      self.decrypted_secret_shares = []
      self.node = Node(port, connection_handler=self.handle_connection, receiver_handler=self.handle_message)

      self.lambda_bits = 128
      p, q, g = generate_cyclic_group(self.lambda_bits)

      self.cyclic_group_params = (p, q, g)
      self.threshold = self.num_clients // 2 + 1

    def run(self):
      self.node.start()

    def aggregate_weights(self):
      p, _, _ = self.cyclic_group_params
      aggregate_weights = (sum(self.encrypted_weights) % p)
      aggregate_weights = aggregate_weights.astype('int')


      for conn in self.node_set:
        self.node.send_message(({"aggregate_weights": aggregate_weights.tolist()}), conn)

      self.cur_epoch += 1
      if (self.cur_epoch == self.num_epochs):
        for conn in self.node_set:
          self.node.send_message(({"status": "stop"}), conn)
        self.node.stop()
      else:
        self.encrypted_weights = []
        self.decrypted_secret_shares = []
        self.encrypted_secret_shares = []
        for conn in self.node_set:
          self.node.send_message(({"status": "continue"}), conn)

    def send_pub_keys(self):
      self.node_set = self.new_node_set
      self.new_node_set = set()
      for conn in self.node_set:
        self.node.send_message(({"public_keys": list(self.public_keys.items())}), conn)

    def handle_secret_shares(self):
      self.node_set = self.new_node_set
      self.new_node_set = set()
      # flatten list of lists
      encrypted_secret_shares = sum(self.encrypted_secret_shares, [])
      
      shares_dict = {}

      for conn in self.node_set:
        shares_dict[self.connection_dict[conn]] = []

      for (from_id, to_id, share) in encrypted_secret_shares:
        if (to_id) in shares_dict:
          shares_dict[to_id].append((from_id, share))

      for conn in self.node_set:
        node_id = self.connection_dict[conn]
        self.node.send_message(({"encrypted_secret_shares": shares_dict[node_id]}), conn)
        self.node.send_message(({"status": "continue"}), conn)

    def recover_weights(self):
      dropout_nodes = self.node_set - self.new_node_set
      dropout_set = list(map(lambda conn: self.connection_dict[conn], dropout_nodes))
      self.node_set = self.new_node_set
      self.new_node_set = set()
      for conn in self.node_set:
        self.node.send_message(({"dropout_set": dropout_set}), conn)
        self.node.send_message(({"recover_secrets": dropout_set}), conn)

    def delayed_func(self, function_to_run):
      if not self.event.wait(timeout=3):
        function_to_run()

    def handle_connection(self, conn):
      self.node_set.add(conn)
      self.connection_dict[conn] = self.connection_count
      self.connection_count += 1
      if (len(self.node_set) == self.num_clients):
        for conn in self.node_set:
          self.node.send_message(({"group_params": self.cyclic_group_params + (self.threshold, )}), conn)

    def handle_message(self, conn, msg):

      if "public_key" in msg:
        pub_key = msg['public_key']
        node_id = self.connection_dict[conn]

        # self.public_keys.append((node_id, pub_key))
        self.public_keys[node_id] = pub_key
        self.new_node_set.add(conn)

        if (len(self.public_keys) == self.threshold):
          self.thread = threading.Thread(target=partial(self.delayed_func, self.send_pub_keys))
          self.event = threading.Event()
          self.thread.start()


        if (len(self.public_keys) == len(self.node_set)):
          self.event.set()
          self.send_pub_keys()

      if "encrypted_secret_shares" in msg:
        self.encrypted_secret_shares.append(msg['encrypted_secret_shares'])
        self.new_node_set.add(conn)

        if (len(self.encrypted_secret_shares) == self.threshold):
          self.thread = threading.Thread(target=partial(self.delayed_func, self.handle_secret_shares))
          self.event = threading.Event()
          self.thread.start()

        if (len(self.encrypted_secret_shares) == len(self.node_set)):
          self.event.set()
          self.handle_secret_shares()

      if "weights" in msg:
        encrypted_weight = np.array(msg["weights"])
        self.encrypted_weights.append(encrypted_weight)
        self.new_node_set.add(conn)

        if len(self.encrypted_weights) == self.threshold:
          self.thread = threading.Thread(target=partial(self.delayed_func, self.recover_weights))
          self.event = threading.Event()
          self.thread.start()
        
        if (len(self.encrypted_weights) == len(self.node_set)):
          self.event.set()
          self.node_set = self.new_node_set
          self.new_node_set = set()
          self.aggregate_weights()

      if "decrypted_secret_shares" in msg:
        self.decrypted_secret_shares.append(msg['decrypted_secret_shares'])
        shares_dict = {}
        if (len(self.decrypted_secret_shares) == self.threshold):
          for secret_shares in self.decrypted_secret_shares:
            for (node_id, share) in secret_shares:
              if node_id not in shares_dict:
                shares_dict[node_id] = []
              shares_dict[node_id].append(share)

          p, _, _ = self.cyclic_group_params
          cur_node_ids = set(map(lambda conn: self.connection_dict[conn], self.node_set))
          for (node_id, shares) in shares_dict.items():
            recovered_secret_key = recover_secret(shares, p)   
            for conn in self.node_set:
              pub_id = self.connection_dict[conn]
              pub_key = self.public_keys[pub_id]
              common_key = pow(pub_key, recovered_secret_key, p)
              final_key = hkdf(b'', common_key.to_bytes(self.lambda_bits,  byteorder='big'), b'', self.lambda_bits // 8)
              key_val = int.from_bytes(final_key, 'big')
              if (pub_id > node_id):
                self.encrypted_weights.append(key_val)
              elif (pub_id < node_id):
                self.encrypted_weights.append(-key_val)
          self.aggregate_weights()


if __name__ == '__main__':
    port = 5000
    num_clients = 5
    num_epochs = 50
    server = Server(port, num_clients, num_epochs)
    server.run()

