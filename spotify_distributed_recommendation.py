import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import warnings
import random
import numpy as np

random.seed(10)

# Suppress warnings
warnings.filterwarnings("ignore")

client_id = '732c5cf894e54e68b3b406f8dbd93cc9'
client_secret = '5bd65999851645be9906a38a53e6a21f'

class SpotifyClient():
    def __init__(self, num_samples=50):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, # Your client ID
                                                    client_secret=client_secret, # Your client secret
                                                    redirect_uri='http://localhost:8888/callback',
                                                    scope='user-top-read',
                                                    open_browser=False))
        self.num_samples = num_samples
        top_tracks = self.sp.current_user_top_tracks(limit=self.num_samples)['items']
        top_tracks_ids = [track['id'] for track in top_tracks]
        top_tracks_features = self.sp.audio_features(top_tracks_ids)
        features_df = pd.DataFrame(top_tracks_features)
        numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.features_df_numeric = features_df[numeric_features]
        self.features_df_numeric.columns = numeric_features
        self.model_kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
    
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

class SpotifyServer():
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, # Your client ID
                                                    client_secret=client_secret, # Your client secret
                                                    redirect_uri='http://localhost:8888/callback',
                                                    scope='user-top-read',
                                                    open_browser=False))
        self.numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.clients  = []
        self.num_clients = 10
        for i in range(self.num_clients):
            self.clients.append(SpotifyClient())
        self.model_kmeans = KMeans(n_clusters=(int)(self.num_clients * 0.8), random_state=0, n_init="auto")
    
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
            local_weights.append(local_weight)
        
        self.model_kmeans.fit(local_weights)
        
        updated_centroids = self.model_kmeans.cluster_centers_
        print(updated_centroids)
        
        
