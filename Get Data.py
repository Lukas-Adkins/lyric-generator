{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"Get Data","provenance":[],"collapsed_sections":[]},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"code","metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"tWnrx3KBHzPB","executionInfo":{"status":"ok","timestamp":1621796622830,"user_tz":420,"elapsed":7589,"user":{"displayName":"Alayne Hatcher","photoUrl":"https://lh3.googleusercontent.com/a-/AOh14GiWxb3O98EqP8uhos7JwI_9bbaamxE55qLsiCOF=s64","userId":"09404502443614207211"}},"outputId":"8014ae0d-860a-47fb-a92e-4624d8c216f2"},"source":["from google.colab import drive\n","drive.mount('/content/gdrive', force_remount=True)\n","\n","from tensorflow import keras\n","from tensorflow.keras import layers\n","from keras.models import Sequential\n","from keras.layers import Dense\n","from keras.layers import Dropout\n","from keras.layers import LSTM\n","from keras.callbacks import ModelCheckpoint\n","from keras.utils import np_utils\n","import numpy\n","import random\n","import os\n","!pip install billboard.py\n","#!pip install lyrics-extractor\n","!pip install lyricsgenius\n","import lyricsgenius\n","import billboard\n","#from PyLyrics import *\n","#from lyrics_extractor import SongLyrics\n","import csv\n","\n","chartData = open(\"/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/charts.csv\", encoding='UTF8')\n","csv_reader = csv.reader(chartData)\n","\n","topFiveEveryWeek = []\n","lyrics = []\n","numFromDate = 0\n","GCS_API_KEY = 'AIzaSyAdPtPXr0gkPe8UCj-mHpMyLgyN7-wC1zY'\n","GCS_ENGINE_ID = 'ce85594fbec95d4c3'\n","#extract_lyrics = SongLyrics(GCS_API_KEY, GCS_ENGINE_ID)\n","\n","access_token = '0Kt5S0Vi88cKFCf7O03cn_Ptoxp2IL-8pC0HOAVLxDSDZyuWFEEerNLloh2pHWMB'\n","genius = lyricsgenius.Genius(access_token)\n","\n","import re \n","import urllib.request \n","from bs4 import BeautifulSoup \n","\n","\n","# Gets top ten songs from every week since 1958\n","numLines = 0\n","for line in csv_reader:\n","  #Skips first line, which doesn't have csv values\n","  if numLines == 0:\n","    numLines = numLines + 1\n","    continue\n","  #If song rank is 1-5, append\n","  if int(line[1]) in range(1, 6):\n","    topFiveEveryWeek.append(line)\n","\n","#writeLyrics = open('/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/lyrics.csv', mode='a')\n","#csv_writer = csv.writer(writeLyrics)\n","#csv_writer.writerow(['Date', 'Title', 'Artist', 'Lyrics'])\n","\n","for i in range(1, 2):\n","  date = topFiveEveryWeek[i][0]\n","  songName = topFiveEveryWeek[i][2]\n","  artist = topFiveEveryWeek[i][3]\n","  try:\n","    # Get song\n","    artistSearch = genius.search_artist(artist, max_songs=0)\n","    song = genius.search_song(songName, artistSearch.name)\n","    data = song.lyrics\n","    data = data.replace(\"\\n\", \" \\n \")\n","    songData = [date, songName, artist, data]\n","    # Write to file\n","    #csv_writer.writerow(songData)\n","  except Exception as e:\n","    print(\"Song not found: \")\n","    print(e)\n","    continue\n","\n","#writeLyrics.close()\n","readLyrics = open('/content/gdrive/MyDrive/Colab Notebooks/CS404 Final Project/lyrics.csv', mode='r')\n","testLyrics = csv.reader(readLyrics)\n","count = 0\n","for line in testLyrics:\n","  print(line)\n","  count = count + 1\n","  if(count == 11): \n","    break\n","chartData.close()"],"execution_count":null,"outputs":[{"output_type":"stream","text":["Mounted at /content/gdrive\n","Requirement already satisfied: billboard.py in /usr/local/lib/python3.7/dist-packages (6.2.1)\n","Requirement already satisfied: beautifulsoup4>=4.4.1 in /usr/local/lib/python3.7/dist-packages (from billboard.py) (4.6.3)\n","Requirement already satisfied: requests>=2.2.1 in /usr/local/lib/python3.7/dist-packages (from billboard.py) (2.23.0)\n","Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.2.1->billboard.py) (1.24.3)\n","Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.2.1->billboard.py) (2020.12.5)\n","Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.2.1->billboard.py) (2.10)\n","Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.2.1->billboard.py) (3.0.4)\n","Requirement already satisfied: lyricsgenius in /usr/local/lib/python3.7/dist-packages (3.0.1)\n","Requirement already satisfied: requests>=2.20.0 in /usr/local/lib/python3.7/dist-packages (from lyricsgenius) (2.23.0)\n","Requirement already satisfied: beautifulsoup4>=4.6.0 in /usr/local/lib/python3.7/dist-packages (from lyricsgenius) (4.6.3)\n","Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20.0->lyricsgenius) (3.0.4)\n","Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20.0->lyricsgenius) (2.10)\n","Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20.0->lyricsgenius) (2020.12.5)\n","Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.20.0->lyricsgenius) (1.24.3)\n","Searching for songs by Perez Prado And His Orchestra...\n","\n","No results found for 'Perez Prado And His Orchestra'.\n","Song not found: \n","'NoneType' object has no attribute 'name'\n","['Date', 'Title', 'Artist', 'Lyrics']\n","['1958-08-04', 'Poor Little Fool', 'Ricky Nelson', \"I used to play around with hearts \\n That hastened at my call \\n But when I met that little girl \\n I knew that I would fall \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n She played around and teased me \\n With her carefree devil eyes \\n She'd hold me close and kiss me \\n But her heart was full of lies \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n She told me how she cared for me \\n And that we'd never part \\n And so for the very first time \\n I gave away my heart \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n The next day she was gone \\n And I knew she'd lied to me \\n She left me with a broken heart \\n And won her victory \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n Well, I'd played this game with other hearts \\n But I never thought I'd see \\n The day that someone else would play \\n Love's foolish game with me \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (Poor little fool)\"]\n","['1958-08-04', 'Splish Splash', 'Bobby Darin', 'Splish splash, I was taking a bath \\n Long about a Saturday night, yeah \\n A rub dub, just relaxing in the tub \\n Thinking everything was alright \\n  \\n Well, I stepped out the tub, put my feet on the floor \\n I wrapped the towel around me \\n And I opened the door, and then a \\n Splish, splash, I jumped back in the bath \\n Well, how was I to know there was a party going on? \\n  \\n They was a-splishing and a-splashing \\n Reeling with the feeling \\n Moving and a-grooving \\n Rocking and a-rolling, yeah \\n  \\n Bing bang, I saw the whole gang \\n Dancing on my living room rug, yeah \\n Flip flop, they was doing the bop \\n All the teens had the dancing bug \\n  \\n There was Lollipop with-a Peggy Sue \\n Good golly, Miss Molly was-a even there, too \\n A-well-a, splish splash, I forgot about the bath \\n I went and put my dancing shoes on, yeah \\n I was a rolling and a-strolling \\n Reeling with the feeling \\n Moving and a-grooving \\n Splishing and a-splashing, yeah \\n  \\n Yes, I was a-splishing and a-splashing \\n I was a-rolling and a-strolling \\n Yeah, I was a-moving and a-grooving \\n We was a-reeling with the feeling \\n We was a-rolling and a-strolling \\n Moving with the grooving \\n Splish splash, yeah \\n  \\n Mm, splishing and a-splashing, one time \\n I was a-splishing and a-splashing, ooh wee \\n I was a-moving and a-grooving, yeah \\n I was a-splishing and a-splashing']\n","['1958-08-04', 'When', 'Kalin Twins', 'When, when you smile, when you smile at me \\n Well, well I know our love will always be \\n When, when you kiss, when you kiss me right \\n I, I don\\'t want to ever say good night \\n  \\n I need you \\n I want you near me \\n I love you \\n Yes, I do and I hope you hear me \\n  \\n When, when I say, when I say \"Be mine\" \\n If, if you will I know all will be fine \\n When will you be mine? \\n  \\n (Oh, baby) \\n (I need you) \\n (I want you near me) \\n (I love you) \\n (Yes, I do and I hope you hear me when) \\n  \\n When, when you smile, when you smile at me \\n Well, well I know our love will always be \\n When, when you kiss, when you kiss me right \\n I, I don\\'t want to ever say good night \\n  \\n I need you \\n I want you near me \\n I love you \\n Yes, I do and I hope you hear me \\n When, when I say, when I say \"Be mine\" \\n If, if you will I know all will be fine \\n When will you be mine?']\n","['1958-08-11', 'Poor Little Fool', 'Ricky Nelson', \"I used to play around with hearts \\n That hastened at my call \\n But when I met that little girl \\n I knew that I would fall \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n She played around and teased me \\n With her carefree devil eyes \\n She'd hold me close and kiss me \\n But her heart was full of lies \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n She told me how she cared for me \\n And that we'd never part \\n And so for the very first time \\n I gave away my heart \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n The next day she was gone \\n And I knew she'd lied to me \\n She left me with a broken heart \\n And won her victory \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n Well, I'd played this game with other hearts \\n But I never thought I'd see \\n The day that someone else would play \\n Love's foolish game with me \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (Poor little fool)\"]\n","['1958-08-11', 'Splish Splash', 'Bobby Darin', 'Splish splash, I was taking a bath \\n Long about a Saturday night, yeah \\n A rub dub, just relaxing in the tub \\n Thinking everything was alright \\n  \\n Well, I stepped out the tub, put my feet on the floor \\n I wrapped the towel around me \\n And I opened the door, and then a \\n Splish, splash, I jumped back in the bath \\n Well, how was I to know there was a party going on? \\n  \\n They was a-splishing and a-splashing \\n Reeling with the feeling \\n Moving and a-grooving \\n Rocking and a-rolling, yeah \\n  \\n Bing bang, I saw the whole gang \\n Dancing on my living room rug, yeah \\n Flip flop, they was doing the bop \\n All the teens had the dancing bug \\n  \\n There was Lollipop with-a Peggy Sue \\n Good golly, Miss Molly was-a even there, too \\n A-well-a, splish splash, I forgot about the bath \\n I went and put my dancing shoes on, yeah \\n I was a rolling and a-strolling \\n Reeling with the feeling \\n Moving and a-grooving \\n Splishing and a-splashing, yeah \\n  \\n Yes, I was a-splishing and a-splashing \\n I was a-rolling and a-strolling \\n Yeah, I was a-moving and a-grooving \\n We was a-reeling with the feeling \\n We was a-rolling and a-strolling \\n Moving with the grooving \\n Splish splash, yeah \\n  \\n Mm, splishing and a-splashing, one time \\n I was a-splishing and a-splashing, ooh wee \\n I was a-moving and a-grooving, yeah \\n I was a-splishing and a-splashing']\n","['1958-08-11', 'When', 'Kalin Twins', 'When, when you smile, when you smile at me \\n Well, well I know our love will always be \\n When, when you kiss, when you kiss me right \\n I, I don\\'t want to ever say good night \\n  \\n I need you \\n I want you near me \\n I love you \\n Yes, I do and I hope you hear me \\n  \\n When, when I say, when I say \"Be mine\" \\n If, if you will I know all will be fine \\n When will you be mine? \\n  \\n (Oh, baby) \\n (I need you) \\n (I want you near me) \\n (I love you) \\n (Yes, I do and I hope you hear me when) \\n  \\n When, when you smile, when you smile at me \\n Well, well I know our love will always be \\n When, when you kiss, when you kiss me right \\n I, I don\\'t want to ever say good night \\n  \\n I need you \\n I want you near me \\n I love you \\n Yes, I do and I hope you hear me \\n When, when I say, when I say \"Be mine\" \\n If, if you will I know all will be fine \\n When will you be mine?']\n","['1958-08-18', 'Little Star', 'The Elegants', \"Where are you little star? \\n (Where are you?) \\n  \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n  \\n Twinkle, twinkle little star \\n How I wonder where you are \\n Wish I may, wish I might \\n Make this wish come true tonight \\n Searched all over for a love \\n You're the one I'm thinkin' of \\n  \\n Whoa oh, oh, oh, uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n  \\n Twinkle, twinkle little star \\n How I wonder where you are \\n High above the clouds somewhere \\n Send me down a love to share \\n  \\n Whoah oh, oh, oh, uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa uh, oh, oh, oh \\n Oh, there you are \\n High above \\n Oh, oh, God \\n Send me a love \\n  \\n Oh, there you are \\n Lighting up the sky \\n I need a love \\n Oh me, oh, me, oh, my \\n  \\n Twinkle twinkle little star \\n How I wonder where you are \\n Wish I may, wish I might \\n Make this wish come true tonight \\n  \\n Whoa oh, oh, oh, uh oh \\n Ratta ta ta too, ooh ooh \\n Woa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n  \\n Oh, ra, ta, ta \\n Ooh, ooh, ooh, ooh, ooh \\n Ooh, ooh, ooh, ooh, ooh \\n There you are little star\"]\n","['1958-08-18', 'My True Love', 'Jack Scott', \"I prayed to the Lord to send me a love \\n He sent me an angel from heaven above \\n The stars in the sky He placed in her eyes \\n She is my true love \\n The touch of her hand \\n (My true love, my true love) \\n  \\n Captured my soul \\n (My true love, my true love) \\n And the kiss from her lips \\n (My true love, my true love) \\n Set my heart aglow \\n (My true love, my true love) \\n  \\n And I know, from heaven \\n (My true love, my true love) \\n From heaven above \\n (My true love, my true love) \\n Came my, my true love \\n  \\n Darling I love you \\n I'll always be true \\n My prayers, they were answered \\n When the Lord sent me you \\n  \\n With, love and devotion that I never knew \\n Until the Lord above sent me you \\n And I thank the heavens \\n (My true love, my true love) \\n The heavens above \\n (My true love) \\n For sending my true love \\n (My true love)\"]\n","['1958-08-18', 'Poor Little Fool', 'Ricky Nelson', \"I used to play around with hearts \\n That hastened at my call \\n But when I met that little girl \\n I knew that I would fall \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n She played around and teased me \\n With her carefree devil eyes \\n She'd hold me close and kiss me \\n But her heart was full of lies \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n She told me how she cared for me \\n And that we'd never part \\n And so for the very first time \\n I gave away my heart \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n The next day she was gone \\n And I knew she'd lied to me \\n She left me with a broken heart \\n And won her victory \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n Well, I'd played this game with other hearts \\n But I never thought I'd see \\n The day that someone else would play \\n Love's foolish game with me \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (I was a fool, oh yeah) \\n  \\n Poor little fool, oh yeah \\n I was a fool, uh huh \\n (Oh oh, poor little fool) \\n (Poor little fool)\"]\n","['1958-08-25', 'Little Star', 'The Elegants', \"Where are you little star? \\n (Where are you?) \\n  \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n  \\n Twinkle, twinkle little star \\n How I wonder where you are \\n Wish I may, wish I might \\n Make this wish come true tonight \\n Searched all over for a love \\n You're the one I'm thinkin' of \\n  \\n Whoa oh, oh, oh, uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n  \\n Twinkle, twinkle little star \\n How I wonder where you are \\n High above the clouds somewhere \\n Send me down a love to share \\n  \\n Whoah oh, oh, oh, uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n Whoa uh, oh, oh, oh \\n Oh, there you are \\n High above \\n Oh, oh, God \\n Send me a love \\n  \\n Oh, there you are \\n Lighting up the sky \\n I need a love \\n Oh me, oh, me, oh, my \\n  \\n Twinkle twinkle little star \\n How I wonder where you are \\n Wish I may, wish I might \\n Make this wish come true tonight \\n  \\n Whoa oh, oh, oh, uh oh \\n Ratta ta ta too, ooh ooh \\n Woa oh, oh, oh uh oh \\n Ratta ta ta too, ooh ooh \\n  \\n Oh, ra, ta, ta \\n Ooh, ooh, ooh, ooh, ooh \\n Ooh, ooh, ooh, ooh, ooh \\n There you are little star\"]\n"],"name":"stdout"}]}]}