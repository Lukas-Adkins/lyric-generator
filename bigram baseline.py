from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import csv
import sys
import nltk
import random
import numpy

numSentences = 10
max_sent_len = 35
random.seed()

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
  for line in csv_reader:
    date = line[0].split("-")
    year = int(date[0])
    if year >= start and year < end:
      songs.append(line)
  return songs

# Get lyrics tokens
def get_words(songs):
  words = []
  words.append("<s>")
  for line in songs:
    song = line[3]
    temp = song.split()
    for i in range(len(temp)):
      words.append(temp[i])
    words.append("</s>")
  return words

# Get Bigrams
def get_bigrams(words):
  print("50 most frequent Bigrams")
  wordFreq = nltk.FreqDist(words)
  bigrams = nltk.bigrams(words)
  fdist = nltk.FreqDist(bigrams)
  most_common = fdist.most_common(50)
  wordCount = len(words)
  # Syntax example: {'</s> <s>': .0467}
  pairProb = {}
  for key in fdist:
    probability = fdist[key]/wordFreq[key[0]]
    pairProb[key] = probability 

  return pairProb

# Generate Sentences
def ten_bigram_sentences(pairProb):
    print("10 bigram sentences:\n")
    bigram_sentences = []
    finalProbArray = []

    #print(pairProb)
  
    for i in range(0, numSentences):
      newSentence = []
      wordProbArray = []
      randWord = '<s>'
      while len(newSentence) < max_sent_len:
        topBigrams = fiftyProbBigrams(pairProb, randWord)
        #print(topBigrams)
        bigrams = topBigrams[0]
        probs = topBigrams[1]
        randWord = random.choices(bigrams, probs, k=1)[0][1]
        randP = 0
        for z in range(0, len(bigrams)):
          if bigrams[z][1] == randWord:
            randP = probs[z]
        condProb = (randP / sum(probs))
        if(randWord == '</s>'):
          wordProbArray.append(condProb)
          break
        if(randWord != '<s>'):
          newSentence.append(randWord)
          wordProbArray.append(condProb)
      #Multiplys conditional probs
      finalProbArray.append(numpy.prod(wordProbArray))
      bigram_sentences.append(newSentence)
    print_sentences(bigram_sentences)

# Gets 50 most probable Bigrams given a starting word
def fiftyProbBigrams(pairProb, word):
  bigrams = []
  probs = []
  numBigrams = 0

  for key in pairProb:
    if numBigrams > 50:
      break
    if(key[0] == word):
      bigrams.append(key)
      probs.append(pairProb[key])
      numBigrams = numBigrams + 1
  return bigrams, probs

def print_sentences(sent_list):
    for sent in sent_list:
        for word in sent:
            print(word,end=" ")
        print("\n\n")

if __name__ == "__main__":
  print("2000-2010")
  start = 2000
  end = 2010
  songs = get_songs(start, end)
  words = get_words(songs)
  pairProb = get_bigrams(words)
  ten_bigram_sentences(pairProb)

  print("1960-1970")
  start = 1960
  end = 1970
  songs = get_songs(start, end)
  words = get_words(songs)
  pairProb = get_bigrams(words)
  ten_bigram_sentences(pairProb)
