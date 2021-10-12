from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy
import random
import os
!pip install billboard.py
#!pip install lyrics-extractor
!pip install lyricsgenius
import lyricsgenius
import billboard
#from PyLyrics import *
#from lyrics_extractor import SongLyrics
import csv

chartData = open("/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/charts.csv", encoding='UTF8')
csv_reader = csv.reader(chartData)

topFiveEveryWeek = []
lyrics = []
numFromDate = 0
GCS_API_KEY = 'AIzaSyAdPtPXr0gkPe8UCj-mHpMyLgyN7-wC1zY'
GCS_ENGINE_ID = 'ce85594fbec95d4c3'
#extract_lyrics = SongLyrics(GCS_API_KEY, GCS_ENGINE_ID)

access_token = '0Kt5S0Vi88cKFCf7O03cn_Ptoxp2IL-8pC0HOAVLxDSDZyuWFEEerNLloh2pHWMB'
genius = lyricsgenius.Genius(access_token)

import re 
import urllib.request 
from bs4 import BeautifulSoup 


# Gets top ten songs from every week since 1958
numLines = 0
for line in csv_reader:
  #Skips first line, which doesn't have csv values
  if numLines == 0:
    numLines = numLines + 1
    continue
  #If song rank is 1-5, append
  if int(line[1]) in range(1, 6):
    topFiveEveryWeek.append(line)

#writeLyrics = open('/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/lyrics.csv', mode='a')
#csv_writer = csv.writer(writeLyrics)
#csv_writer.writerow(['Date', 'Title', 'Artist', 'Lyrics'])

for i in range(1, 2):
  date = topFiveEveryWeek[i][0]
  songName = topFiveEveryWeek[i][2]
  artist = topFiveEveryWeek[i][3]
  try:
    # Get song
    artistSearch = genius.search_artist(artist, max_songs=0)
    song = genius.search_song(songName, artistSearch.name)
    data = song.lyrics
    data = data.replace("\n", " \n ")
    songData = [date, songName, artist, data]
    # Write to file
    #csv_writer.writerow(songData)
  except Exception as e:
    print("Song not found: ")
    print(e)
    continue

#writeLyrics.close()
readLyrics = open('/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/lyrics.csv', mode='r')
testLyrics = csv.reader(readLyrics)
count = 0
for line in testLyrics:
  print(line)
  count = count + 1
  if(count == 11): 
    break
chartData.close()
