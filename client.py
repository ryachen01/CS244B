import secrets
import numpy as np
from cryptographpy_helper import hkdf, generate_shares, aes_encrypt, aes_decrypt
from node import Node
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import warnings
import heapq
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import federated_logistic_regression
from cryptographpy_helper import generate_cyclic_group, hkdf
import secrets

GLOBAL_RANDOM_SEED = 244
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)
rounding_factor = 1e6
# Suppress warnings
warnings.filterwarnings("ignore")
client_id = '732c5cf894e54e68b3b406f8dbd93cc9'
client_secret = '5bd65999851645be9906a38a53e6a21f'


class SpotifyClient():
    def __init__(self, num_samples=50):
        # model setup
        self.num_samples = num_samples
        self.numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.model_kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        self.log_reg = None
        # data setup
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,  # Your client ID
                                                            client_secret=client_secret,  # Your client secret
                                                            redirect_uri='http://localhost:8888/callback',
                                                            scope='user-top-read',
                                                            open_browser=False))
        top_tracks = self.sp.current_user_top_tracks(limit=50)['items']
        top_tracks_ids = [track['id'] for track in top_tracks]
        top_tracks_features = self.sp.audio_features(top_tracks_ids)
        features_df = pd.DataFrame(top_tracks_features)
        self.features_df_numeric = features_df[self.numeric_features]
        self.features_df_numeric.columns = self.numeric_features
        self.model_kmeans = KMeans(n_clusters=5, random_state=0)
        self.dataset = None
        self.log_reg = None

    def run_knn(self):
        self.model_kmeans.fit(self.features_df_numeric.to_numpy())
        self.centroids = self.model_kmeans.cluster_centers_

    def get_features_of_outside_songs(self):
        # Fetch and preprocess the song catalog from which we will select songs to recommend users
        playlist_ids = ['37i9dQZF1DX0kbJZpiYdZl',
                        '37i9dQZEVXbNG2KDcFcKOF',
                        '37i9dQZEVXbLiRSasKsNU9',
                        '37i9dQZF1DX0XUsuxWHRQd',
                        '37i9dQZF1DX1lVhptIYRda',
                        '37i9dQZF1DWVqJMsgEN0F4',
                        '37i9dQZF1DX10zKzsJ2jva',
                        '37i9dQZF1DX4dyzvuaRJ0n',
                        '37i9dQZF1DX8uc99HoZBLU',
                        '37i9dQZF1DXao0JEaClQq9',
                        '37i9dQZF1DWZJmo7mlltU6',
                        '37i9dQZF1DX4SBhb3fqCJd',
                        '37i9dQZF1DWUileP28ODwg',
                        '37i9dQZF1DX9tPFwDMOaN1',
                        '37i9dQZF1DX4FcAKI5Nhzq',
                        '37i9dQZF1E8OX5RkYXtx51',
                        '37i9dQZF1DWU0r6G8OGirN',
                        '37i9dQZF1DXd3AhRYJnfcl',
                        '2peJPuYDQJMsmEpjqMALnl',
                        '37i9dQZF1DWY0DyDKedRYY',
                        '37i9dQZF1DWWjGdmeTyeJ6',
                        '37i9dQZF1DXd0ZFXhY0CRF',
                        '37i9dQZF1DWZgauS5j6pMv',
                        '37i9dQZF1DX3LyU0mhfqgP',
                        '37i9dQZF1DWX0o6sD1a6P5']
        tracks = []
        song_catalog_features = []
        song_catalog_names = []
        song_catalog_artists = []

        for playlist_id in playlist_ids:
            results = self.sp.playlist_tracks(playlist_id)
            cur_tracks = results['items']
            while results['next']:
                results = self.sp.next(results)
                cur_tracks.extend(results['items'])
            tracks.extend(cur_tracks)
            # Fetch the song ids, names and artists
            song_catalog_ids = [track['track']['id'] for track in cur_tracks]
            cur_song_catalog_names = [track['track']['name']
                                      for track in cur_tracks]
            song_catalog_names.extend(cur_song_catalog_names)
            cur_song_catalog_artists = [
                track['track']['artists'][0]['name'] for track in cur_tracks]
            song_catalog_artists.extend(cur_song_catalog_artists)
            cur_song_catalog_features = self.sp.audio_features(
                song_catalog_ids)
            song_catalog_features.extend(cur_song_catalog_features)
            # TODO: Temp Fix due to Spotify maximum amount of items that can be returned in a search is 1000
            if len(song_catalog_features) > 950:
                break
        none_indices = [i for i, feature in enumerate(
            song_catalog_features) if feature is None]
        # print(f"Indices of None values: {none_indices}")

        # Remove elements at these indices from both lists, starting from the end
        for index in sorted(none_indices, reverse=True):
            del song_catalog_features[index]
            del song_catalog_names[index]
            del song_catalog_artists[index]
        song_catalog_df = pd.DataFrame(song_catalog_features)
        song_catalog_df['name'] = song_catalog_names
        song_catalog_df['artist'] = song_catalog_artists
        self.song_catalog_df_numeric = song_catalog_df[self.numeric_features]

    def get_common_disliked_songs(self):

        client_centroids = self.model_kmeans.cluster_centers_

        distances = []
        dislike_df = self.song_catalog_df_numeric.copy()
        for index, song_feature in self.song_catalog_df_numeric.iterrows():
            total_dist = 0
            for c in client_centroids:
                cur_dist = distance.euclidean(song_feature.values, c)
                total_dist += cur_dist
            distances.append(total_dist)
            # all_songs_dict[tuple(song_feature.values)] = total_dist
        dislike_df['distance'] = distances
        top_50_dislike_df = dislike_df.nlargest(50, 'distance')
        top_50_dislike_df = top_50_dislike_df.drop(columns=['distance'])
        top_50_dislike_df['labels'] = 0
        return top_50_dislike_df


class LogisticRegressionClient:
    def __init__(self, lr=0.01, lambda_val=0.2):
        # create a client's dataset
        spotify_client = SpotifyClient()
        likes_df = spotify_client.features_df_numeric.copy()
        likes_df['labels'] = 1
        spotify_client.run_knn()
        spotify_client.get_features_of_outside_songs()
        disliked_songs = spotify_client.get_common_disliked_songs()
        client_dataset = [likes_df, disliked_songs]
        spotify_client.dataset = pd.concat(client_dataset)
        training_data = spotify_client.dataset.copy()
        X = training_data.drop('label', axis=1)
        y = training_data['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42)
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lr = lr
        self.lambda_val = lambda_val
        _, n = X.shape
        self.weights = np.zeros(n)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, weights, lambda_reg):
        m = len(self.y)
        h = self.sigmoid(self.X @ weights)
        cost = -np.sum(self.y * np.log(h) + (1 - self.y) * np.log(1 - h)) / m
        reg_cost = (lambda_reg / (2 * m)) * np.sum(weights ** 2)
        return cost + reg_cost

    def compute_gradient(self, weights, lambda_reg):
        m = len(self.y)
        h = self.sigmoid(self.X @ weights)
        gradient = np.dot(self.X.T, (h - self.y)) / m
        gradient += (lambda_reg / m) * weights
        return gradient

    def update_weights(self, weights):
        self.weights = weights

    def train_local(self, num_iterations):
        for _ in range(num_iterations):
            self.weights -= self.lr * \
                self.compute_gradient(
                    self.X, self.y, self.weights, self.lambda_val)

    def predict(self):
        probabilities = self.sigmoid(self.X @ self.weights)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        test_accuracy = 1 - np.mean(y_pred != self.y_test)
        return test_accuracy


class Client:
    def __init__(self, server_host, server_port, client_port):
        self.server_host = server_host
        self.server_port = server_port
        self.client_port = client_port
        self.model = LogisticRegressionClient()
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
        self.node.send_message(
            ({"weights": encrypted_weight.tolist()}), server_conn)
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
                    final_key = hkdf(b'', common_key.to_bytes(
                        self.lambda_bits,  byteorder='big'), b'', self.lambda_bits // 8)
                    self.shared_keys[node_id] = final_key

            self.shamirs_secret_exchange()

        if "encrypted_secret_shares" in msg:
            encrypted_secret_shares = msg["encrypted_secret_shares"]

            self.node_set = set()
            self.num_clients = len(encrypted_secret_shares) + 1
            for (node_id, encrypted_share) in encrypted_secret_shares:
                self.node_set.add(node_id)
                decrypted_share = aes_decrypt(
                    encrypted_share, self.shared_keys[node_id])
                decrypted_share = int.from_bytes(decrypted_share, 'big')
                self.secret_shares[node_id] = decrypted_share

        if "dropout_set" in msg:
            self.node_set -= set(msg["dropout_set"])
            self.num_clients = len(self.node_set) + 1

        if "aggregate_weights" in msg:
            aggregate_weights = np.array(
                msg["aggregate_weights"]).astype('float64')
            aggregate_weights /= (self.num_clients)
            aggregate_weights /= 1e6
            aggregate_weights -= 1
            aggregate_weights = aggregate_weights.astype('float64')
            self.model.update_weights(aggregate_weights)

        if "recover_secrets" in msg:
            nodes_to_recover = msg["recover_secrets"]
            shares_to_send = []
            for (node_id) in nodes_to_recover:
                shares_to_send.append(
                    (node_id, (self.node_id+1, self.secret_shares[node_id])))

            server_conn = self.node.outward_connections[0]
            self.node.send_message(
                ({"decrypted_secret_shares": shares_to_send}), server_conn)

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
        secret_key_shares = generate_shares(
            self.secret_key, total_clients, self.threshold, p)
        self.secret_shares[self.node_id] = secret_key_shares[self.node_id][1]
        encrypted_shares = []
        for (node_id, shared_key) in self.shared_keys.items():
            encrypted = aes_encrypt(secret_key_shares[node_id][1].to_bytes(
                self.lambda_bits, byteorder='big'), shared_key)
            encrypted_shares.append((self.node_id, node_id, encrypted))

        server_conn = self.node.outward_connections[0]
        self.node.send_message(
            ({"encrypted_secret_shares": encrypted_shares}), server_conn)


num_clients = 5

server_host = '192.168.192.231'
server_port = 5000

clients = []

for i in range(num_clients):
    client = Client(server_host, server_port, server_port +
                    i + 1)
    client.run()
    clients.append(client)

for client in clients:
    client.wait()

for client in clients:
    print("train accuracy", client.model.evaluate())
    print("test accuracy", client.model.evaluate())
