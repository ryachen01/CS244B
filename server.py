import numpy as np
from cryptographpy_helper import generate_cyclic_group, recover_secret, hkdf
from functools import partial
from node import Node
import threading


def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()

    for key, value in dict2.items():
        if key in merged_dict:
            if not isinstance(merged_dict[key], list):
                merged_dict[key] = [merged_dict[key]]
            merged_dict[key].append(value)
        else:
            merged_dict[key] = value

    return merged_dict


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
        self.encrypted_weights = {}
        self.encrypted_secret_shares = []
        self.decrypted_secret_shares = []
        self.node = Node(
            port,
            connection_handler=self.handle_connection,
            receiver_handler=self.handle_message,
        )

        self.lambda_bits = 128
        p, q, g = generate_cyclic_group(self.lambda_bits)

        self.cyclic_group_params = (p, q, g)
        self.threshold = self.num_clients // 2 + 1

    def run(self):
        self.node.start()

    def aggregate_weights(self):
        p, _, _ = self.cyclic_group_params

        for key in self.encrypted_weights.keys():
            encrypted_weight = self.encrypted_weights[key]
            aggregate_weights = sum(encrypted_weight) % p
            aggregate_weights = np.array(aggregate_weights).astype("int")
            self.encrypted_weights[key] = aggregate_weights.tolist()

        for conn in self.node_set:
            self.node.send_message(
                ({"aggregate_weights": self.encrypted_weights}), conn
            )

        self.cur_epoch += 1
        if self.cur_epoch == self.num_epochs:
            for conn in self.node_set:
                self.node.send_message(({"status": "stop"}), conn)
            self.node.stop()
        else:
            self.encrypted_weights = {}

            for conn in self.node_set:
                self.node.send_message(({"status": "continue"}), conn)

    def send_pub_keys(self):
        self.node_set = self.new_node_set
        self.new_node_set = set()
        for conn in self.node_set:
            self.node.send_message(
                ({"public_keys": list(self.public_keys.items())}), conn
            )

    def handle_secret_shares(self, message):
        self.node_set = self.new_node_set
        self.new_node_set = set()
        # flatten list of lists
        encrypted_secret_shares = sum(self.encrypted_secret_shares, [])

        shares_dict = {}

        for conn in self.node_set:
            shares_dict[self.connection_dict[conn]] = []

        for from_id, to_id, share in encrypted_secret_shares:
            if (to_id) in shares_dict:
                shares_dict[to_id].append((from_id, share))

        for conn in self.node_set:
            node_id = self.connection_dict[conn]
            self.node.send_message(({message: shares_dict[node_id]}), conn)
            # self.node.send_message(({"status": "continue"}), conn)

        self.encrypted_secret_shares = []

    def recover_weights(self):
        dropout_nodes = self.node_set - self.new_node_set
        dropout_set = list(map(lambda conn: self.connection_dict[conn], dropout_nodes))
        self.node_set = self.new_node_set
        self.new_node_set = set()
        for conn in self.node_set:
            # self.node.send_message(({"dropout_set": dropout_set}), conn)
            self.node.send_message(({"recover_secrets": dropout_set}), conn)

    def delayed_func(self, function_to_run):
        if not self.event.wait(timeout=3):
            function_to_run()

    def handle_connection(self, conn):
        self.node_set.add(conn)
        self.connection_dict[conn] = self.connection_count
        self.connection_count += 1
        if len(self.node_set) == self.num_clients:
            for conn in self.node_set:
                self.node.send_message(
                    ({"group_params": self.cyclic_group_params + (self.threshold,)}),
                    conn,
                )

    def handle_message(self, conn, msg):
        if "public_key" in msg:
            pub_key = msg["public_key"]
            node_id = self.connection_dict[conn]

            self.public_keys[node_id] = pub_key
            self.new_node_set.add(conn)

            if len(self.public_keys) == self.threshold:
                self.thread = threading.Thread(
                    target=partial(self.delayed_func, self.send_pub_keys)
                )
                self.event = threading.Event()
                self.thread.start()

            if len(self.public_keys) == len(self.node_set):
                self.event.set()
                self.send_pub_keys()

        if "encrypted_secret_shares" in msg:
            self.encrypted_secret_shares.append(msg["encrypted_secret_shares"])
            self.new_node_set.add(conn)

            if len(self.encrypted_secret_shares) == self.threshold:
                self.thread = threading.Thread(
                    target=partial(
                        self.delayed_func,
                        partial(self.handle_secret_shares, "encrypted_secret_shares"),
                    )
                )
                self.event = threading.Event()
                self.thread.start()

            if len(self.encrypted_secret_shares) == len(self.node_set):
                self.event.set()
                self.handle_secret_shares("encrypted_secret_shares")

        if "encrypted_double_mask" in msg:
            self.encrypted_secret_shares.append(msg["encrypted_double_mask"])
            self.new_node_set.add(conn)

            if len(self.encrypted_secret_shares) == self.threshold:
                self.thread = threading.Thread(
                    target=partial(
                        self.delayed_func,
                        partial(self.handle_secret_shares, "encrypted_double_mask"),
                    )
                )
                self.event = threading.Event()
                self.thread.start()

            if len(self.encrypted_secret_shares) == len(self.node_set):
                self.event.set()
                self.handle_secret_shares("encrypted_double_mask")

        if "weights" in msg:

            weight_dict = msg["weights"]
            for key in weight_dict.keys():
                weight_dict[key] = np.array(weight_dict[key])
            self.encrypted_weights = merge_dicts(self.encrypted_weights, weight_dict)
            self.new_node_set.add(conn)

            if len(self.new_node_set) == self.threshold:
                self.thread = threading.Thread(
                    target=partial(self.delayed_func, self.recover_weights)
                )
                self.event = threading.Event()
                self.thread.start()

            if len(self.new_node_set) == len(self.node_set):
                self.decrypted_secret_shares = []
                self.event.set()
                self.recover_weights()

        if "decrypted_secret_shares" in msg:

            self.decrypted_secret_shares.append(msg["decrypted_secret_shares"])
            shares_dict = {}
            if len(self.decrypted_secret_shares) == self.threshold:
                for secret_shares in self.decrypted_secret_shares:
                    for node_id, share in secret_shares:
                        if node_id not in shares_dict:
                            shares_dict[node_id] = []
                        shares_dict[node_id].append(share)

                p, _, _ = self.cyclic_group_params
                active_nodes = list(
                    map(lambda conn: self.connection_dict[conn], self.node_set)
                )
                for node_id, shares in shares_dict.items():
                    recovered_secret_key = recover_secret(shares, p)
                    if node_id in active_nodes:
                        final_key = hkdf(
                            b"",
                            recovered_secret_key.to_bytes(
                                self.lambda_bits, byteorder="big"
                            ),
                            b"",
                            self.lambda_bits // 8,
                        )
                        key_val = int.from_bytes(final_key, "big")
                        key_dict = {}
                        for key in self.encrypted_weights.keys():
                            key_dict[key] = [-key_val]

                        self.encrypted_weights = merge_dicts(
                            self.encrypted_weights, key_dict
                        )
                    else:
                        for conn in self.node_set:
                            pub_id = self.connection_dict[conn]
                            pub_key = self.public_keys[pub_id]
                            common_key = pow(pub_key, recovered_secret_key, p)
                            final_key = hkdf(
                                b"",
                                common_key.to_bytes(self.lambda_bits, byteorder="big"),
                                b"",
                                self.lambda_bits // 8,
                            )
                            key_val = int.from_bytes(final_key, "big")
                            if pub_id > node_id:

                                key_dict = {}
                                for key in self.encrypted_weights.keys():
                                    key_dict[key] = [key_val]

                                self.encrypted_weights = merge_dicts(
                                    self.encrypted_weights, key_dict
                                )
                            elif pub_id < node_id:
                                key_dict = {}
                                for key in self.encrypted_weights.keys():
                                    key_dict[key] = [key_val]
                                self.encrypted_weights = merge_dicts(
                                    self.encrypted_weights, -key_dict
                                )
                self.aggregate_weights()


if __name__ == "__main__":
    port = 5005
    num_clients = 1
    num_epochs = 1
    server = Server(port, num_clients, num_epochs)
    server.run()
    server.node.t2.join()
