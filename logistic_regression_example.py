import numpy as np
from client import Client
import numpy as np
from cryptographpy_helper import hkdf, generate_shares, aes_encrypt, aes_decrypt
from node import Node
import spotipy
import random
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
import warnings
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
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
    def __init__(self, client_index, num_samples=50):
        # model setup
        self.num_samples = num_samples
        self.numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.model_kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        self.log_reg = None
        self.features_df_numeric = pd.read_csv(f"data/client_{client_index}_playlist.csv")
        self.song_catalog_df_numeric = pd.read_csv("data/song_catalog.csv")
        self.model_kmeans = KMeans(n_clusters=5, random_state=0)
        self.dataset = None
        self.log_reg = None

    def run_knn(self):
        self.model_kmeans.fit(self.features_df_numeric.to_numpy())
        self.centroids = self.model_kmeans.cluster_centers_
        
    def get_common_disliked_songs(self):
        client_centroids = self.centroids
        distances = []
        dislike_df = self.song_catalog_df_numeric.copy()
        for index, song_feature in self.song_catalog_df_numeric.iterrows():
            total_dist = 0
            for c in client_centroids:
                cur_dist = distance.euclidean(song_feature.values, c)
                total_dist += cur_dist
            distances.append(total_dist)
        dislike_df['distance'] = distances
        top_50_dislike_df = dislike_df.nlargest(50, 'distance')
        top_50_dislike_df = top_50_dislike_df.drop(columns=['distance'])
        top_50_dislike_df['labels'] = 0
        return top_50_dislike_df

class LogisticRegressionClient():
    def __init__(self, client_index, lr=0.01, lambda_val=0.2):
        # create a client's dataset
        spotify_client = SpotifyClient(client_index)
        likes_df = spotify_client.features_df_numeric.copy()
        likes_df['labels'] = 1
        spotify_client.run_knn()
        disliked_songs = spotify_client.get_common_disliked_songs()
        client_dataset = [likes_df, disliked_songs]
        spotify_client.dataset = pd.concat(client_dataset)
        training_data = spotify_client.dataset.copy()
        X = training_data.drop('labels', axis=1)
        y = training_data['labels']
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
        return 1 / (1 + np.exp(np.float16(-z)))

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
        self.weights = weights["logistic_weights"]

    def train_local(self, num_iterations):
        for _ in range(num_iterations):
            self.weights -= self.lr * self.compute_gradient(self.weights, self.lambda_val)

    def get_weights(self):
      return {"logistic_weights": self.weights}

    def predict(self, train_flag=True):
        if train_flag:
            probabilities = self.sigmoid(self.X @ self.weights)
        else:
            probabilities = self.sigmoid(self.X_test @ self.weights)
        return (probabilities >= 0.5).astype(int)

    def evaluate(self, train_flag=True):
        if train_flag:
            y_pred = self.predict(train_flag=True)
            test_accuracy = 1 - np.mean(y_pred != self.y)
        else:
            y_pred = self.predict(train_flag=False)
            test_accuracy = 1 - np.mean(y_pred != self.y_test)
        return test_accuracy

num_clients = 5

''' Ryan's IP address '''
# server_host = '192.168.192.231'

''' Young's IP address '''
server_host = '10.34.155.96'

server_port = 2100

clients = []

for i in range(num_clients):
  model = LogisticRegressionClient(i)
  X = model.X
  y = model.y
  client = Client(server_host, server_port, model, X, y)
  print(f"Hi! I am client {i}")
  client.run()
  clients.append(client)

for client in clients:
  client.wait()


for i, client in enumerate(clients):
  print(f"I am client {i} and here are my results:")
  print("train accuracy: ", client.model.evaluate(train_flag=True))
  print("test accuracy: ", client.model.evaluate(train_flag=False))