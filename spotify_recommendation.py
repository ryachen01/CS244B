import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

client_id = '732c5cf894e54e68b3b406f8dbd93cc9'
client_secret = '5bd65999851645be9906a38a53e6a21f'

# The user needs to authenticate in order to access top tracks
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, # Your client ID
                                               client_secret=client_secret, # Your client secret
                                               redirect_uri='http://localhost:8888/callback',
                                               scope='user-top-read',
                                               open_browser=False))

# Get the user's top tracks
top_tracks = sp.current_user_top_tracks(limit=50)['items']

# Extract the track IDs
top_tracks_ids = [track['id'] for track in top_tracks]

# Fetch the audio features for the user's top tracks
top_tracks_features = sp.audio_features(top_tracks_ids)

# Convert the features into a DataFrame
features_df = pd.DataFrame(top_tracks_features)

# Train the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)

# Define a list of numeric feature names
numeric_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# Select only these columns from the DataFrame
features_df_numeric = features_df[numeric_features]

# Set feature names for the DataFrame
features_df_numeric.columns = numeric_features

# Fit the model
model_knn.fit(features_df_numeric)

# Fetch and preprocess the song catalog from which we will select songs to recommend users
# playlist_id = '37i9dQZF1DX0kbJZpiYdZl' # Change this to search other playlists for recommendations
# playlist_id = '37i9dQZF1DX4dyzvuaRJ0n'
# playlist_id = '37i9dQZF1DX0MLFaUdXnjA'
# results = sp.playlist_tracks(playlist_id)
# tracks = results['items']

# while results['next']:
#     results = sp.next(results)
#     tracks.extend(results['items'])
    
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
    results = sp.playlist_tracks(playlist_id)
    cur_tracks = results['items']
    while results['next']:
        results = sp.next(results)
        cur_tracks.extend(results['items'])
    tracks.extend(cur_tracks)
    # Fetch the song ids, names and artists
    song_catalog_ids = [track['track']['id'] for track in cur_tracks]
    cur_song_catalog_names = [track['track']['name'] for track in cur_tracks]
    song_catalog_names.extend(cur_song_catalog_names)
    cur_song_catalog_artists = [track['track']['artists'][0]['name'] for track in cur_tracks]
    song_catalog_artists.extend(cur_song_catalog_artists)
    cur_song_catalog_features = sp.audio_features(song_catalog_ids)
    song_catalog_features.extend(cur_song_catalog_features)

song_catalog_df = pd.DataFrame(song_catalog_features)
song_catalog_df['name'] = song_catalog_names
song_catalog_df['artist'] = song_catalog_artists

song_catalog_df_numeric = song_catalog_df[numeric_features]
print(f"type is {type(song_catalog_df_numeric)}")

# Use the KNN model to find similar songs for each of the user's top tracks
all_recommendations = []
for track_features in song_catalog_df_numeric.values:
    distances, indices = model_knn.kneighbors([track_features])
    all_recommendations.extend(indices.flatten())

# Aggregate and rank the recommended songs
song_counts = Counter(all_recommendations)
most_recommended_songs = song_counts.most_common()

print(model_knn.get_params())

# Print the recommended songs
for index, count in most_recommended_songs[:10]: # Number of songs to recommend (Default: 10)
    print(f"{song_catalog_df['name'][index]} by {song_catalog_df['artist'][index]}")
    