import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
import random
import numpy as np
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
        self.num_samples = num_samples
        self.numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.model_kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        self.log_reg = None
    
    def normal_init(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, # Your client ID
                                                    client_secret=client_secret, # Your client secret
                                                    redirect_uri='http://localhost:8888/callback',
                                                    scope='user-top-read',
                                                    open_browser=False))
        top_tracks = self.sp.current_user_top_tracks(limit=self.num_samples)['items']
        top_tracks_ids = [track['id'] for track in top_tracks]
        top_tracks_features = self.sp.audio_features(top_tracks_ids)
        features_df = pd.DataFrame(top_tracks_features)
        self.features_df_numeric = features_df[self.numeric_features]
        self.features_df_numeric.columns = self.numeric_features

    # Generates randomized data for experimental purposes
    def random_init(self):
        data = {
            'danceability': np.random.uniform(0, 1, self.num_samples),
            'energy': np.random.uniform(0, 1, self.num_samples),
            'key': np.random.randint(0, 12, self.num_samples),
            'loudness': np.random.uniform(-60, 0, self.num_samples),
            'mode': np.random.randint(0, 2, self.num_samples),
            'speechiness': np.random.uniform(0, 1, self.num_samples),
            'acousticness': np.random.uniform(0, 1, self.num_samples),
            'instrumentalness': np.random.uniform(0, 1, self.num_samples),
            'liveness': np.random.uniform(0, 1, self.num_samples),
            'valence': np.random.uniform(0, 1, self.num_samples),
            'tempo': np.random.uniform(50, 200, self.num_samples),
            'duration_ms': np.random.randint(180000, 300000, self.num_samples),
            'time_signature': np.random.randint(3, 8, self.num_samples)
        }
        self.features_df_numeric = pd.DataFrame(data)
            
    def run_knn(self):
        self.model_kmeans.fit(self.features_df_numeric)
        self.centroids = self.model_kmeans.cluster_centers_
        return self.centroids
        
    def centroid_to_server(self):
        # TODO: send centroids to server over socket
        print(2)
        
    def train_log_reg(self):
        # TODO: receive epoch and num_iterations from server (??), along with common dislikes
        epochs = 25
        num_iterations = 50
        for _ in range(epochs):
            local_weights = []
            self.log_reg.train_local(num_iterations=num_iterations)
            local_weight = self.log_reg.weights
            local_weights.append(local_weight)

        local_weights = np.array(local_weights)
        min_weight = abs(np.min(local_weights))
        local_weights += min_weight
        local_weights *= rounding_factor
        local_weights = np.rint(local_weights).astype(int)
        
        # TODO: send local weights to server using socket/TCP connection


class SpotifyServer():
    def __init__(self, num_clients=10, clients=[SpotifyClient() for _ in range(10)]):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, # Your client ID
                                                    client_secret=client_secret, # Your client secret
                                                    redirect_uri='http://localhost:8888/callback',
                                                    scope='user-top-read',
                                                    open_browser=False))
        self.numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.num_clients = num_clients
        self.clients  = clients
        self.model_kmeans = KMeans(n_clusters=(int)(self.num_clients * 0.8), random_state=0, n_init="auto")
        self.log_reg = None
        self.epochs = 25
        self.num_iterations = 50
        self.lambda_bits = 256
        self.alpha = 1
        self.epsilon = 1
    
    # Fetches a repetoir or songs, gathers them into a dataframe
    def fetches_top_songs(self):
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
            cur_song_catalog_names = [track['track']['name'] for track in cur_tracks]
            song_catalog_names.extend(cur_song_catalog_names)
            cur_song_catalog_artists = [track['track']['artists'][0]['name'] for track in cur_tracks]
            song_catalog_artists.extend(cur_song_catalog_artists)
            cur_song_catalog_features = self.sp.audio_features(song_catalog_ids)
            song_catalog_features.extend(cur_song_catalog_features)

        song_catalog_df = pd.DataFrame(song_catalog_features)
        song_catalog_df['name'] = song_catalog_names
        song_catalog_df['artist'] = song_catalog_artists

        self.song_catalog_df_numeric = song_catalog_df[self.numeric_features]

    def get_common_disliked_songs(self):
        local_weights = []
        for i in range(self.num_clients):
            self.clients[i].run_knn()
            local_weight = self.clients[i].centroids
            print(f"Dimension of these weights {local_weight[0].shape}, length is {len(local_weight)}")
            local_weights.extend(local_weight)
        
        # print(f"These are the local weights {local_weights}")
        
        self.model_kmeans.fit(local_weights)
        
        updated_centroids = self.model_kmeans.cluster_centers_
        print(updated_centroids)
        
        disliked_songs_df = pd.DataFrame(updated_centroids, columns=self.numeric_features)
        disliked_songs_df['label'] = 0  # Label disliked songs with 0
        
        # Send these centroids back to the clients using socket/TCP connection
        
        return disliked_songs_df
                
    def predict_top_songs(self):
        print(2)
        
    def combine_clients_data(self):
        combined_df = pd.concat([client.features_df_numeric for client in self.clients], ignore_index=True)
        combined_df['label'] = 1  # Label client data with 1
        return combined_df
    
    def prepare_training_data(self):
        liked_songs_df = self.combine_clients_data()
        disliked_songs_df = self.get_common_disliked_songs()
        training_data = pd.concat([liked_songs_df, disliked_songs_df], ignore_index=True)
        return training_data
    
    def crypto_setup(self):
        (p, q, g) = generate_cyclic_group(self.lambda_bits)
        self.p = p
        self.q = q
        self.g = g

        # TODO: do we need to scale by len(X_train_chunks[0])?
        scale = 2 / (self.num_clients * self.alpha * self.epsilon)
        noise = np.random.laplace(scale=scale, size=self.num_clients)
        noise *= rounding_factor
        self.noise = np.rint(noise).astype(int)

        self.secret_keys = [[secrets.randbelow(q) for _ in range(self.num_clients)] for _ in range(self.num_clients)]
        self.public_keys = [[pow(g, self.secret_keys[i][j], p) for j in range(self.num_clients)] for i in range(self.num_clients)]

        self.common_keys = [[None]*self.num_clients for _ in range(self.num_clients)]
        self.final_keys = [[None]*self.num_clients for _ in range(self.num_clients)]

        for i in range(self.num_clients):
            for j in range(self.num_clients):
                c_ij = pow(self.public_keys[j][i], self.secret_keys[i][j], p)
                self.common_keys[i][j] = c_ij.to_bytes(self.lambda_bits,  byteorder='big')
                self.final_keys[i][j] = hkdf(b'', c_ij.to_bytes(self.lambda_bits,  byteorder='big'), b'', self.lambda_bits // 8)

        
''' Experiment 1: removes all server communication '''
clients = []
for i in range(50):
    client = SpotifyClient()
    client.random_init()
    print(f"{client.features_df_numeric} are the numeric features of client {i}")
    clients.append(client)
server = SpotifyServer(num_clients=50, clients=clients)
server.crypto_setup()
server.get_common_disliked_songs()
training_data = server.prepare_training_data()

# Extract features (X) and labels (y)
X = training_data.drop('label', axis=1)
y = training_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# We bypass the server aggregation and straight-up train everything on the server
server.log_reg = federated_logistic_regression.LogisticRegressionClient(X_train, y_train)
server.log_reg.train_local(num_iterations=server.num_iterations * server.epochs)
print("baseline train accuracy", server.log_reg.evaluate(X_train, y_train))
print("baseline test accuracy", server.log_reg.evaluate(X_test, y_test))

