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

# Suppress warnings
warnings.filterwarnings("ignore")
client_id = '732c5cf894e54e68b3b406f8dbd93cc9'
client_secret = '5bd65999851645be9906a38a53e6a21f'

class SpotifyDataScraper():
    def __init__(self):
        self.numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

        # Fetch and preprocess the song catalog from which we will select songs to recommend users
        self.song_catalog = ['37i9dQZF1DX0kbJZpiYdZl',
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

        # Client's individual playlists
        self.client_playlists = [
            '6X6v4AJiRZgGEFSB3ERxc6',
            '1x8VWfbVSqnwve8vDSQrNJ',
            '7asSzG1o9Uk0gEvHYBOPmE',
            '72G67PRZ1wyB6jTqmOibST',
            '046pYQXBbos7N2LQG02Abd' # this playlist is mine again
        ]
        
        # data setup
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,  # Your client ID
                                                            client_secret=client_secret,  # Your client secret
                                                            redirect_uri='http://localhost:8888/callback',
                                                            scope='user-top-read',
                                                            open_browser=False))

    def get_song_features(self, playlist):
        tracks = []
        song_catalog_features = []
        song_catalog_names = []
        song_catalog_artists = []

        for playlist_id in playlist:
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

        # Remove elements at these indices from both lists, starting from the end
        for index in sorted(none_indices, reverse=True):
            del song_catalog_features[index]
            del song_catalog_names[index]
            del song_catalog_artists[index]
        song_catalog_df = pd.DataFrame(song_catalog_features)
        song_catalog_df['name'] = song_catalog_names
        song_catalog_df['artist'] = song_catalog_artists
        
        return song_catalog_df[self.numeric_features]

scraper = SpotifyDataScraper()
scraper.get_song_features(scraper.song_catalog).to_csv('song_catalog.csv', index=False)  # Save without index
for i, playlist in enumerate(scraper.client_playlists):
    scraper.get_song_features([playlist]).to_csv(f'client_{i}_playlist.csv', index=False)