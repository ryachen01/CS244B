import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

client_id = '732c5cf894e54e68b3b406f8dbd93cc9'
client_secret = '5bd65999851645be9906a38a53e6a21f'

class SpotifyClient():
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, # Your client ID
                                                    client_secret=client_secret, # Your client secret
                                                    redirect_uri='http://localhost:8888/callback',
                                                    scope='user-top-read',
                                                    open_browser=False))
        top_tracks = self.sp.current_user_top_tracks(limit=50)['items']
        top_tracks_ids = [track['id'] for track in top_tracks]
        top_tracks_features = self.sp.audio_features(top_tracks_ids)
        features_df = pd.DataFrame(top_tracks_features)
        numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        self.features_df_numeric = features_df[numeric_features]
        self.features_df_numeric.columns = numeric_features
        self.model_kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
        
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
        self.clients  = []
        self.num_clients = 10
        for i in range(self.num_clients):
            self.clients.append(SpotifyClient())
        self.model_kmeans = KMeans(n_clusters=(int)(self.num_clients * 0.8), random_state=0, n_init="auto")
        
    def predict_top_songs(self):
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
                        '37i9dQZF1DX4FcAKI5Nhzq'] 
        tracks = set()
        for playlist_id in playlist_ids:
            results = self.sp.playlist_tracks(playlist_id)
            tracks.add(results['items'])
            while results['next']:
                results = self.sp.next(results)
                tracks.add(results['items'])
            
        song_catalog_ids = [track['track']['id'] for track in tracks]
        song_catalog_names = [track['track']['name'] for track in tracks]
        song_catalog_artists = [track['track']['artists'][0]['name'] for track in tracks]
        song_catalog_features = self.sp.audio_features(song_catalog_ids)
        numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
        song_catalog_df = pd.DataFrame(song_catalog_features)
        song_catalog_df['name'] = song_catalog_names
        song_catalog_df['artist'] = song_catalog_artists
        self.song_catalog_df_numeric = song_catalog_df[numeric_features]

    def get_common_disliked_songs(self):
        local_weights = []
        for i in range(self.num_clients):
            self.clients[i].run_knn()
            local_weight = self.clients[i].centroids
            local_weights.append(local_weight)
        
        self.model_kmeans.fit(local_weights)
        
        
