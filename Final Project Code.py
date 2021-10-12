from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import csv
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils as np_utils
import numpy as np
import pickle
import spacy 
from spacy.lang.en import English
import string
import sys
from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt


nlp = spacy.load("en_core_web_sm") 

# Returns songs where start <= data < end
def get_songs(start, end):
  # csv has a max field limit that is by default, greater than our data size
  # So this was added to increase that max 
  maxInt = sys.maxsize
  while True:
    try: 
      csv.field_size_limit(maxInt)
      break
    except OverflowError:
      maxInt = int(maxInt/10)

  lyrics = open('/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/lyrics.csv', mode='r')
  csv_reader = csv.reader(lyrics)
  # skip header
  next(csv_reader)

  songs = []
  count = 0
  for line in csv_reader:
    if count == 0:
      # append top song of the week
      date = line[0].split("-")
      temp = date
      year = int(date[0])
      if year >= start and year < end:
        count = count + 1
        songs.append(line)
    else:
      # skip the rest
      temp = line[0].split("-")
      if temp != date:
        count = 0
      else:
        count = 1
  return songs

# Get lyrics tokens
def get_tokens(lyrics):
  tokenList = []
  for line in lyrics:
    words = []
    temp = line.split()
    for i in range(len(temp)):
      words.append(temp[i])
    tokenList.append(words)
  return tokenList

def get_sequences(lyrics):
  tokens = lyrics.split()
  sequences = []
  for i in range(1, 51):
    seq = tokens[:i+1]
    sequences.append(seq)
  for i in range(51, len(tokens)):
    seq = tokens[i-50:i+1]
    sequences.append(seq)
  tokenizer.fit_on_texts(sequences)
  tokenSequences = tokenizer.texts_to_sequences(sequences)
  tokenCount = len(tokenizer.word_index) + 1
  return tokenSequences, tokenCount

# Define LSTM model
def create_model(tokenCount):
  model = Sequential() # Model type: Stack of layers, each has exactly 1 input and 1 output tensor
  model.add(Embedding(tokenCount, 50, input_length=50))
  model.add(LSTM(100)) 
  model.add(Dense(tokenCount))
  model.compile(loss='categorical_crossentropy') 
  model.summary()
  return model

def lyrics_to_sequences(songLyric):
  tokens = songLyric.split()
  sequences = []
  for i in range(1, 51):
    seq = tokens[:i+1]
    sequences.append(seq)
  for i in range(51, len(tokens)):
    seq = tokens[i-50:i+1]
    sequences.append(seq)
  return sequences
  
def softmax(x): #computes softmax for 1d array
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

# Generate Sentences
def generate_sentence(model, numWords, tokenizer, seed):
  result = []
  input = seed
  for i in range(numWords):
    #print(result)
    token = tokenizer.texts_to_sequences([input])
    token = pad_sequences(token, maxlen=50, padding='pre')
    prediction = model.predict(token)[0]
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))

    prediction = softmax(prediction);
    prediction = np.random.choice(len(prediction), p=prediction)

    output = ''
    for word, index in tokenizer.word_index.items():
      if index == prediction:
        output = word
        break
    input += " " + output
    result.append(output)
  return ' '.join(result)

def clean_up(lyrics, lang):
  lyrics = lyrics.lower()
  tokenizer = nlp.Defaults.create_tokenizer(lang)
  tokens = tokenizer(lyrics)
  result = []
  for token in tokens:
    if not token.is_punct: 
      result.append(token.orth_)
  return ' '.join(result)

def train_model(model, trainingData, trainLabels, tokenCount):
  # Get labels
  trainLength = 1000
  count = 0
  if trainLength > tokenCount:
    trainLabels = np_utils.to_categorical(labels, num_classes=tokenCount)
    model.train_on_batch(trainingData, trainLabels)
  else:
    # train labels in increments of trainLength
    for i in range(0, tokenCount):
      trainLabels = np_utils.to_categorical(labels[i:trainLength], num_classes=tokenCount)
      model.train_on_batch(trainingData[i:trainLength], trainLabels)
      count = i
      i = i + trainLength
    # train the leftover
    leftover = tokenCount-count
    print(leftover)
    trainLabels = np_utils.to_categorical(labels[count:tokenCount], num_classes=tokenCount)
    model.train_on_batch(trainingData[count:tokenCount], trainLabels)
    model.save("/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/test.h5")
  
if __name__ == "__main__":
  start = 2010
  end = 2020
  songs = get_songs(start, end)
  # Remove duplicates
  uniqueSongs = []
  for song in songs:
    if song not in uniqueSongs:
      uniqueSongs.append(song)
  lyrics = []
  lang = English()
  for song in uniqueSongs:
    lyrics.append(clean_up(song[3], lang))

  trainingData = []
  for song in lyrics:
    sequences = lyrics_to_sequences(song)
    for seq in sequences:
      trainingData.append(seq)

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(trainingData)
  tokenCount = len(tokenizer.word_index) + 1
  trainingData = tokenizer.texts_to_sequences(trainingData)
  trainingData = pad_sequences(trainingData, padding='pre')
  trainingData, labels = trainingData[:,:-1],trainingData[:,-1]
  
  # save tokenizer
  with open('/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/tokenizer10-20.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

  model = create_model(tokenCount);

  #for i in range(len(trainingData)):
    #data = np.array(trainingData[i])
  #trainingData = (trainingData - np.min(trainingData)) / (np.max(trainingData) - np.min(trainingData))
    #print(normalized_dataset)
    #trainingData[i] = normalized_dataset
  
  # Function that trains the model using train_on_batch
  # train_model(model, trainingData, labels, tokenCount)

  labels = np_utils.to_categorical(labels, num_classes=tokenCount)
  model.fit(trainingData, labels, epochs=5)
  
  # save model
  model.save("/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/model10-20.h5")

  for i in range(0, 10):
    genSeq = generate_sentence(model, 100, tokenizer, "together");
    print(genSeq)
    print()
