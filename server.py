import numpy as np
from cryptographpy_helper import generate_cyclic_group
from node import Node

class Server:
    def __init__(self, port, num_clients, num_epochs):
      self.port = port
      self.num_clients = num_clients
      self.num_epochs = num_epochs
      self.cur_epoch = 0
      self.connections = []
      self.node = Node(port, receiver_handler=self.handle_message)
      self.encrypted_weights = []

      lambda_bits = 128
      p, q, g = generate_cyclic_group(lambda_bits)

      self.cyclic_group_params = (p, q, g)

    def run(self):
      self.node.start()

    def begin_setup(self):
      for i in range(self.num_clients):
        conn = self.node.received_connections[i]
        self.node.send_message(({"node_list": self.connections}), conn)
        self.node.send_message(({"group_params": self.cyclic_group_params}), conn)

    def handle_message(self, conn, msg):
      if "connection_info" in msg:
        (host, port) = msg["connection_info"]
        self.connections.append((host, port))
        if len(self.connections) == self.num_clients:
          self.begin_setup()

      if "weights" in msg:
        encrypted_weight = np.array(msg["weights"])
        self.encrypted_weights.append(encrypted_weight)
        if len(self.encrypted_weights) == self.num_clients:
          p, _, _ = self.cyclic_group_params
          aggregate_weights = (sum(self.encrypted_weights) % p)
          aggregate_weights = aggregate_weights.astype('int')

          for i in range(self.num_clients):
            client_conn = self.node.received_connections[i]
            self.node.send_message(({"aggregate_weights": aggregate_weights.tolist()}), client_conn)

          self.cur_epoch += 1
          if (self.cur_epoch == self.num_epochs):
            for i in range(self.num_clients):
              client_conn = self.node.received_connections[i]
              self.node.send_message(({"status": "stop"}), client_conn)
          else:
            self.encrypted_weights = []
            for i in range(self.num_clients):
              client_conn = self.node.received_connections[i]
              self.node.send_message(({"status": "continue"}), client_conn)
            

if __name__ == '__main__':
    port = 5000
    num_clients = 3
    num_epochs = 50
    server = Server(port, num_clients, num_epochs)
    server.run()

