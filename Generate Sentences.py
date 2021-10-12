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
import math

def softmax(x): #computes softmax
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Generate Sentences
def generate_sentence(model, numWords, tokenizer, seed):
  result = []
  input = seed
  for i in range(numWords):
    token = tokenizer.texts_to_sequences([input])
    token = pad_sequences(token, maxlen=50, padding='pre')
    prediction = model.predict(token)[0]
    #prediction = np.argmax(model.predict(token), axis=-1)
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))

    #for i in range(len(prediction)):
    #  prediction[i] = prediction[i]/prediction.sum(axis=0,keepdims=1)
    for i in range(0, len(prediction)):
      if math.isnan(prediction[i]):
        prediction[i] = 0
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

def print_decade(start, end, numLyrics, lyricLength, seed):
  print(str(start) + "-" + str(end))

  tokenizerFile = '/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/tokenizer' + str(start) + '-' + str(end) + '.pickle'
  # loading
  with open(tokenizerFile, 'rb') as handle:
    tokenizer = pickle.load(handle)
    
  modelFile = '/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/model' + str(start) + '-' + str(end) + '.h5'
  model = load_model(modelFile)
  
  for i in range(0, numLyrics):
    genSeq = generate_sentence(model, lyricLength, tokenizer, seed);
    print(genSeq)
    print()

if __name__ == "__main__": 
  numLyrics = 10
  lyricLength = 25
  seed = "The best song"

  for start in range(60, 90, 10):
    end = start + 10
    print_decade(start, end, numLyrics, lyricLength, seed)

  start = "90"
  end = "00"
  print_decade(start, end, numLyrics, lyricLength, seed)

  for start in range(0, 20, 10):
    end = start + 10
    print_decade(start, end, numLyrics, lyricLength, seed)
