import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.oauth2 as oauth2
import json
import pandas as pd
import lyricsgenius as genius
import sys
import pprint
import os

os.chdir('/Users/luislosada/Columbia Drive/Applied Text and NLP')

##Set up your client ID and Client Secret
CLIENT_ID = "33a602d7e56540d3af86373bcba237af"
CLIENT_SECRET = "6d36561d42694d8c8747079b156e15cb"

credentials = oauth2.SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET)

token = credentials.get_access_token()
sp = spotipy.Spotify(auth=token)

uri = 'spotify:user:spotifycharts:playlist:37i9dQZEVXbMDoHDwVN2tF?si=0MkpydG_SWC5Y7loG59M9Q'
username = uri.split(':')[2]
playlist_id = uri.split(':')[4]

#print(json.dumps(results, indent=4))

list_of_playslists = ["37i9dQZF1DX0XUsuxWHRQd?si=BO56978gQqSiLl4g73P4jA","37i9dQZEVXbMDoHDwVN2tF?si=0MkpydG_SWC5Y7loG59M9Q",
"37i9dQZF1DX0MLFaUdXnjA","37i9dQZF1DWXJyjYpHunCf","37i9dQZF1DWWOrEmtp56BZ","37i9dQZF1DWSkMjlBZAZ07",
"37i9dQZF1DWUH2AzNQzWua","37i9dQZF1DX3rxVfibe1L0?si=HUmvaZpJTGGUbEJiUFr-Rw","37i9dQZF1DWSqmBTGDYngZ",
"37i9dQZF1DXdPec7aLTmlC","37i9dQZF1DX7KNKjOK0o75","37i9dQZF1DX2SK4ytI2KAZ","37i9dQZF1DX2FsCLsHeMrM",
"37i9dQZF1DWSlw12ofHcMM?si=vvuafurOTHq9aFGmx1pogw","37i9dQZF1DX9Z3vMB2b8im","37i9dQZF1DWY7IeIP1cdjF",
"37i9dQZF1DWYmmr74INQlb","37i9dQZF1DX8Kgdykz6OKj","37i9dQZF1DX1lVhptIYRda","37i9dQZF1DWWnMZKMl7SWB",
"37i9dQZF1DWY4xHQp97fN6","37i9dQZF1DX76Wlfdnj7AP","37i9dQZF1DX70RN3TfWWJh","37i9dQZF1DWUVpAXiEPK8P?si=RCJzumEeRgy9fXeoxfKx2A"
]


#Function to pull from several playlists
def df_creation(playlist_id):
    results = sp.user_playlist(username, playlist_id)
    golbal_top_50 = results['tracks']['items']
    df_t50 = pd.DataFrame(columns =['artists','name','duration_ms','popularity','playlist'])
    for i in range(len(golbal_top_50)):
        try:
            df_t50.loc[i,'artists'] = golbal_top_50[i]['track']['artists'][0]['name']
            df_t50.loc[i,'name'] = golbal_top_50[i]['track']['name']
            df_t50.loc[i,'duration_ms'] = golbal_top_50[i]['track']['duration_ms']
            df_t50.loc[i,'popularity'] = golbal_top_50[i]['track']['popularity']
            df_t50.loc[i,'playlist'] = results['name']
        except TypeError:
            df_t50.loc[i, 'artists'] = None
            df_t50.loc[i, 'name'] = None
            df_t50.loc[i, 'duration_ms'] = None
            df_t50.loc[i, 'popularity'] = None
            df_t50.loc[i, 'playlist'] = None
    return df_t50
Songs_Df = {}
for playll in list_of_playslists:
    Songs_Df[playll] = df_creation(playll)

df = pd.concat(Songs_Df).reset_index(drop=True)

df = df.mask(df.astype(object).eq('None')).dropna().reset_index(drop=True)

df2 = df.drop(columns=["playlist"]).drop_duplicates().reset_index()

df3 = pd.merge(df2,df['playlist'],how='left',left_on=df2['index'],right_index=True).drop(columns=['index'])

#Pulling Lyrics
api = genius.Genius('N9fjx5mBa9dVc6dx6fxonQ8RdH57gGZp8wCWCCzTNXa_yr6qSksEL8GBzLXKRizf')

lyrics = []
for a in range(len(df3)):
    ss = api.search_song(df3.loc[a,'name'],df3.loc[a,'artists'])
    try:
        lyrics.append(ss.lyrics)
    except AttributeError:
        lyrics.append(" ")


df3.insert(loc=5,column = 'lyrics',value = lyrics)
df4 = df3.mask(df3.astype(object).eq('None')).dropna().reset_index(drop=True)
df4.to_csv('pre-clean-dataset.csv')


