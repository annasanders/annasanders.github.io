---
layout: page
title: DTSA 5580 Network Analysis Final Project
permalink: /ms_projects/dtsa5800_tweets
---
**Important Note:** Please click on the specific html files to see the weights (hover-able) and explore node names/words more in-depth.

# Load Packages


```python
import gzip
import json
import nltk

import glob
import os
import shutil
import json
import csv
import networkx as nx
import matplotlib.pyplot as plt
try:
  import pyvis
  from pyvis.network import Network
except:
  !pip install pyvis
  import pyvis
  from pyvis import Network
from time import sleep
import nltk
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
import re
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import string
import itertools
punctuation = string.punctuation
stopwordsset = set(stopwords.words("english"))
stopwordsset.add('rt')
stopwordsset.add("'s")
from datetime import datetime
import pandas as pd

from IPython.core.display import display, HTML

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /root/nltk_data...
    [nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
    




    True



# Load Data


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

## Define Extract Functions


```python
def extract_mention(tweet, mention):
  if 'user_mentions' in tweet['entities'].keys():
    t_mentions = tweet['entities']['user_mentions']
    for i in t_mentions:
      if (i['screen_name'].upper() == mention.upper()) or (i['name'].upper() == mention.upper()):
        return 1
  else: return 0

def filter_tweets(tweets, mention):
    for tweet in tweets:
        if isinstance(tweet, (bytes, str)):
            tweet = json.loads(tweet)
        if extract_mention(tweet, mention):
            yield (tweet['user']['screen_name'], mention, tweet)

# Taken from Week 4 Lecture Notebook
#Removing urls
def removeURL(text):
  result = re.sub(r"http\S+", "", text)
  result = re.sub(r"’", "", result) # more special characters not coded as punctuation
  result = re.sub(r"“", "", result)
  result = re.sub(r"”", "", result)
  result = re.sub(r"—", "", result)
  result = re.sub(r"…", "", result)
  return result

#removes useless words such as a, an, the
def stopWords(tokenizedtext):
  goodwords = []
  for aword in tokenizedtext:
    if aword not in stopwordsset:
      goodwords.append(aword)
  return goodwords

# feature reduction. taking words and getting their roots and graphing only the root words
def lemmatizer(tokenizedtext):
  lemmawords = []
  for aword in tokenizedtext:
    aword = wn.lemmatize(aword)
    lemmawords.append(aword)
  return lemmawords

#inputs a list of tokens and returns a list of unpunctuated tokens/words
def removePunctuation(tokenizedtext):
  nopunctwords = []
  for aword in tokenizedtext:
    if aword not in punctuation:
      nopunctwords.append(aword)
  cleanedwords = []
  for aword in nopunctwords:
    aword = aword.translate(str.maketrans('', '', string.punctuation))
    cleanedwords.append(aword)

  return cleanedwords

def removesinglewords(tokenizedtext):
  goodwords = []
  for a_feature in tokenizedtext:
    if len(a_feature) > 1:
      goodwords.append(a_feature)
  return goodwords

# Adapted from Week 4 Lab
def token_counts(tweets, tagger=nltk.tag.PerceptronTagger().tag, tokenizer=nltk.TweetTokenizer().tokenize, parts_of_speech=None):
    if parts_of_speech == None:
      parts_of_speech = []
    token_dict = {}
    for tweet in tweets:
      if isinstance(tweet, (bytes, str)):
            tweet = json.loads(tweet)
      if 'full_text' in tweet.keys():
        tweet_text = tweet['full_text']
      else: tweet_text = tweet['text']
      tweet_text = removeURL(tweet_text)
      token = tokenizer(tweet_text)
      token = stopWords(token)
      token = lemmatizer(token)
      token = removePunctuation(token)
      tags = tagger(token)
      if len(tags) == 0: continue
      if len(parts_of_speech) == 0:
        for i in tags:
          if i[0] in token_dict:
            token_dict[i[0]] += 1
          else:
            token_dict[i[0]] = 1
      for i in tags:
        if i == None: continue
        elif i[1] in parts_of_speech:
          if i[0] in token_dict:
            token_dict[i[0]] += 1
          else:
            token_dict[i[0]] = 1
    return token_dict

def tweet_sentiment(tweet, tokenizer=nltk.TweetTokenizer().tokenize):
  if 'full_text' in tweet.keys():
        tweet_text = tweet['full_text']
  else: tweet_text = tweet['text']
  tweet_text = removeURL(tweet_text)
  token = tokenizer(tweet_text)
  token = stopWords(token)
  token = lemmatizer(token)
  token = removePunctuation(token)
  sentence = ' '.join(token)
  sentim_analyzer = SentimentIntensityAnalyzer()
  scores = sentim_analyzer.polarity_scores(sentence)
  return scores['compound']
```

## Extract Tweets with an @ Reference to Companies


```python
with gzip.open('drive/MyDrive/nikelululemonadidas_tweets.jsonl.gz') as f:
  lululemon = list(filter_tweets(f, 'lululemon'))

with gzip.open('drive/MyDrive/nikelululemonadidas_tweets.jsonl.gz') as g:
  nike = list(filter_tweets(g, 'nike'))

with gzip.open('drive/MyDrive/nikelululemonadidas_tweets.jsonl.gz') as h:
  adidas = list(filter_tweets(h, 'adidas'))
```


```python
print('Lululemon Tweet Mentions:', len(lululemon))
print('Nike Tweet Mentions:', len(nike))
print('Adidas Tweet Mentions:', len(adidas))
```

    Lululemon Tweet Mentions: 6168
    Nike Tweet Mentions: 118953
    Adidas Tweet Mentions: 36485
    

# Central Users

We can start by investigating the key users and their tweets. We will first create a subset of the top 100 users per segment.


```python
nike_df = pd.DataFrame(nike, columns=['user','segment', 'tweet'])
adidas_df = pd.DataFrame(adidas, columns=['user','segment', 'tweet'])
lululemon_df = pd.DataFrame(lululemon, columns=['user','segment', 'tweet'])
```


```python
nike_df['user_description'] = nike_df['tweet'].apply(lambda x: x['user']['description'])
adidas_df['user_description'] = lululemon_df['tweet'].apply(lambda x: x['user']['description'])
lululemon_df['user_description'] = lululemon_df['tweet'].apply(lambda x: x['user']['description'])

nike_df['user_description'] = nike_df['user_description'].fillna('N/A')
adidas_df['user_description'] = adidas_df['user_description'].fillna('N/A')
lululemon_df['user_description'] = lululemon_df['user_description'].fillna('N/A')
```


```python
nike_df['user'].value_counts(ascending=False).rename_axis('user').reset_index(name='counts').head(20).merge(nike_df[['user','user_description']].groupby('user',as_index=False).max(), how='left', on='user')
```





  <div id="df-a5d4bbb0-74ae-4c95-b66e-f1e6e35ffa54" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>counts</th>
      <th>user_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SneakerScouts</td>
      <td>6891</td>
      <td>The #1 source for sneaker news, release dates,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HUsBadGuys</td>
      <td>4067</td>
      <td>HU's Bad Guys #HBOW</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kaya_Alexander5</td>
      <td>720</td>
      <td>Just a girl who loves her sneakers. The sneake...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stealth783</td>
      <td>590</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>GirardisGod</td>
      <td>530</td>
      <td>My 2 Instagram accounts are (@girardisgodgram)...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ShockandAweEnt</td>
      <td>361</td>
      <td>Providing a broad range of entertainment aroun...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>vadriano2000</td>
      <td>334</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>turtlepace5</td>
      <td>321</td>
      <td>CT born, WI raised. Packers, Auntie, Coffee, S...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SSBrandon</td>
      <td>278</td>
      <td>@Nike Apostle &amp; #SNKRS VET, who pledged allegi...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>zen_masstah</td>
      <td>271</td>
      <td>SNKR head, hip hop, anti influencer, hater of ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>jadendaly</td>
      <td>247</td>
      <td>Please allow me to introduce myself: I’m a man...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DJBLUIZ</td>
      <td>212</td>
      <td>Dj/Sneakerhead 👟10.5-11 - Cowboys-Knicks-Devil...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>levibrian86</td>
      <td>207</td>
      <td>chef, private sec., vocal singing,let u know  ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>therealJCW</td>
      <td>205</td>
      <td>#SneakerScouts @SneakerScouts @ShockandAweEnt</td>
    </tr>
    <tr>
      <th>14</th>
      <td>efiorentino31</td>
      <td>203</td>
      <td>☀︎︎ ♍︎ ☽ ♓︎ ❥ sneakers &amp; makeup ♥︎</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Moonman989</td>
      <td>177</td>
      <td>#AJ1FAM</td>
    </tr>
    <tr>
      <th>16</th>
      <td>joshuajhan</td>
      <td>171</td>
      <td>🇰🇷 916 English Bulldogs Nike Jumpman sneakerhe...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>beiberlove69</td>
      <td>157</td>
      <td>it's a big hat, it's funny\nsz13 crew</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BoysRevelacion</td>
      <td>137</td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>DoBetterBB</td>
      <td>136</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a5d4bbb0-74ae-4c95-b66e-f1e6e35ffa54')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-a5d4bbb0-74ae-4c95-b66e-f1e6e35ffa54 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a5d4bbb0-74ae-4c95-b66e-f1e6e35ffa54');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-26a6ad4e-7b3d-4e66-9059-e7f09f4a6049">
  <button class="colab-df-quickchart" onclick="quickchart('df-26a6ad4e-7b3d-4e66-9059-e7f09f4a6049')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-26a6ad4e-7b3d-4e66-9059-e7f09f4a6049 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
adidas_df['user'].value_counts(ascending=False).rename_axis('user').reset_index(name='counts').head(20).merge(adidas_df[['user','user_description']].groupby('user',as_index=False).max(), how='left', on='user')
```





  <div id="df-7684075b-0759-4d10-85db-39163316b7c1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>counts</th>
      <th>user_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BoysRevelacion</td>
      <td>137</td>
      <td>test pilot for pies. no thoughts, no vibes. #b...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>zen_masstah</td>
      <td>123</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bleustar9757</td>
      <td>108</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>jajuanmharley</td>
      <td>89</td>
      <td>🏳️‍🌈Health educator and certified fitness trai...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>turtlepace5</td>
      <td>68</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TheRealMGesta</td>
      <td>68</td>
      <td>🌎  #VoteBlue2022 #NeverGOP #WeVote,WeWin\nNo u...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>restebanrf1993</td>
      <td>67</td>
      <td>Transport Writer and Editor En/Fr</td>
    </tr>
    <tr>
      <th>7</th>
      <td>KVSwitzer</td>
      <td>53</td>
      <td>likes = 🙄 / 🥰/ 🤣 / 🤬 / 🥴</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GrossAmilee</td>
      <td>39</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>natmmom</td>
      <td>35</td>
      <td>🇱🇧West Coast Phalange Supporter🇱🇧 Retvrn to Af...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>wearekrimy</td>
      <td>35</td>
      <td>So Cal born and raised. Athletic Trainer, Phys...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Moonman989</td>
      <td>34</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>12</th>
      <td>erik102079</td>
      <td>34</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FootwearNews</td>
      <td>33</td>
      <td>photo:@linksgems</td>
    </tr>
    <tr>
      <th>14</th>
      <td>golacokits</td>
      <td>31</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>15</th>
      <td>kaflickinger74</td>
      <td>30</td>
      <td>seo | snack enthusiast | photography | grandma...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>josiethewonder</td>
      <td>27</td>
      <td>Photography📸 Cane Creek Distillery🥃</td>
    </tr>
    <tr>
      <th>17</th>
      <td>DeionPatterson1</td>
      <td>27</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BPrince95</td>
      <td>26</td>
      <td>welcome to the clown show</td>
    </tr>
    <tr>
      <th>19</th>
      <td>JamieGeorge93</td>
      <td>25</td>
      <td>N/A</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7684075b-0759-4d10-85db-39163316b7c1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7684075b-0759-4d10-85db-39163316b7c1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7684075b-0759-4d10-85db-39163316b7c1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c43f1c2b-2378-49bf-83ff-6344d66336d4">
  <button class="colab-df-quickchart" onclick="quickchart('df-c43f1c2b-2378-49bf-83ff-6344d66336d4')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c43f1c2b-2378-49bf-83ff-6344d66336d4 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
lululemon_df['user'].value_counts(ascending=False).rename_axis('user').reset_index(name='counts').head(20).merge(lululemon_df[['user','user_description']].groupby('user',as_index=False).max(), how='left', on='user')
```





  <div id="df-5c69135c-09ad-4538-be1d-8dab159dd167" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>counts</th>
      <th>user_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JimAceman</td>
      <td>32</td>
      <td>Mammal, Partner, Father, Son, Brother, Friend,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MattMully</td>
      <td>31</td>
      <td>Be...excellent to eachother.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>kinseyfit</td>
      <td>24</td>
      <td>“Fit &amp; Fearless” - where we believe it’s not j...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WhatAndrewSaid</td>
      <td>23</td>
      <td>“so great looking and smart, a true Stable Gen...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MasashiKohmura</td>
      <td>20</td>
      <td>Cross-county skiing information specialized in...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>liab9845</td>
      <td>20</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>blythelia3505</td>
      <td>16</td>
      <td>$jillnauyokas</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Chrisblythe9845</td>
      <td>16</td>
      <td>Food &amp; Drink Funny Fashion Television Music Di...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DeezeFi</td>
      <td>14</td>
      <td>all I know is I don't know nothing | lindy wal...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gomerland2</td>
      <td>14</td>
      <td>Survivor and researcher as I embark in a journ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MrLeonardKim</td>
      <td>14</td>
      <td>I feel the pain in your heart and I know what ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>365yogadream</td>
      <td>13</td>
      <td>Ambassador @lululemon -Nutrition -Emergency Pr...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>C_kelly1988</td>
      <td>13</td>
      <td>⚾️ 🌳 🥾 👟 size 11-12. Cincy sports fan. #teamcr...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MattTooze</td>
      <td>13</td>
      <td>Experienced ex athlete 800m 1.57.3. sub 16 5k ...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>AFineBlogger</td>
      <td>13</td>
      <td>VP East 86th St Assoc, created https://t.co/Uz...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>cloudwhiteNFT</td>
      <td>12</td>
      <td>financial philosopher. @axieinfinity evangelis...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>lulunotify</td>
      <td>11</td>
      <td>The first monitor service for #lululemon drops...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TheSportsIndex</td>
      <td>11</td>
      <td>“THE bellwether stock market index for sports....</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sean_broedow</td>
      <td>11</td>
      <td>I’m a runner, a 2021 Legacy Nuun Ambassador, 2...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>aleman305</td>
      <td>10</td>
      <td>BJJ, MMA, CrossFit, and Olympic Weightlifting</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5c69135c-09ad-4538-be1d-8dab159dd167')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5c69135c-09ad-4538-be1d-8dab159dd167 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5c69135c-09ad-4538-be1d-8dab159dd167');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c9c72f6f-77fd-4c1c-bf51-ee50dc4d4963">
  <button class="colab-df-quickchart" onclick="quickchart('df-c9c72f6f-77fd-4c1c-bf51-ee50dc4d4963')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c9c72f6f-77fd-4c1c-bf51-ee50dc4d4963 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Just looking at the top users for each company, we can see that users who at (@) tweet Nike are often sneakerheads, or people who either collect or buy lots of sneakers. Both Adidas and Lululemon's users seem to be a more random assortment of people. Lululemon does, however, seem to be at (@) tweeted by a lot of fitness and health inclined people.


```python
top_nike = nike_df['user'].value_counts(ascending=False).rename_axis('user').reset_index(name='counts').head(100)['user'].values
top_adidas = adidas_df['user'].value_counts(ascending=False).rename_axis('user').reset_index(name='counts').head(100)['user'].values
top_lululemon = lululemon_df['user'].value_counts(ascending=False).rename_axis('user').reset_index(name='counts').head(100)['user'].values
```


```python
nt = Network('1200', '1600', directed=True, notebook=True, cdn_resources='remote')
Graph = nx.DiGraph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')
tweet_groups = [lululemon, adidas, nike]
tweet_users = [top_lululemon, top_adidas, top_nike]

for lst in range(len(tweet_groups)):
  for i in range(len(tweet_groups[lst])):
    if tweet_groups[lst][i][0] in tweet_users[lst]:
      if Graph.has_edge(tweet_groups[lst][i][0], tweet_groups[lst][i][1]):
        w = Graph.edges[tweet_groups[lst][i][0], tweet_groups[lst][i][1]]['title']
        Graph.edges[tweet_groups[lst][i][0], tweet_groups[lst][i][1]]['weight'] = ((w+1) //100)  +1
        Graph.edges[tweet_groups[lst][i][0], tweet_groups[lst][i][1]]['title'] = w+1
      else: Graph.add_edge(tweet_groups[lst][i][0], tweet_groups[lst][i][1], weight=1, title=1)

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/user_mention_pyvis.html")
```
   
```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 285
    Edges: 300
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\user_mention_pyvis.html)

We can see from the pyvis network graph that of the top 100 users for each company, the red cluster (Nike) and yellow cluster (Adidas) had more users in common than the blue cluster (Lululemon). This makes sense as both Nike and Adidas are primarily known for their shoes, while Lululemon is known for their clothing. Similarly, even when comparing clothing, which Nike and Adidas also manufacture, they primarily focus on workout clothing, while Lululemon is famous for it's yoga attire.

Similarly, when looking at the number and size of node linkages, Nike has lots of users who continually at (@) tweet them. One user, SneakerScouts, tweeted almost 7,000 times within the dataset. Comparatively, Adidas and Lululemon did not have any followers who tweeted at (@) them that much.

There do seem to be a handful of users who tweet at both Nike and Adidas.

We can now investigate the key words these top users are using in their quote tweets.


```python
lululemon_df_users = lululemon_df[lululemon_df['user'].isin(top_lululemon)]
adidas_df_users = adidas_df[adidas_df['user'].isin(top_adidas)]
nike_df_users = nike_df[nike_df['user'].isin(top_nike)]

lululemon_user_tweets = lululemon_df_users['tweet'].values
adidas_user_tweets = adidas_df_users['tweet'].values
nike_user_tweets = nike_df_users['tweet'].values

lululemon_vocab_df = pd.DataFrame(list(token_counts(list(lululemon_user_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count']) #Adjective, Nouns, and Verbs Only
adidas_vocab_df = pd.DataFrame(list(token_counts(list(adidas_user_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
nike_vocab_df = pd.DataFrame(list(token_counts(list(nike_user_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
```


```python
nt = Network('1200', '1600', directed=False, notebook=True, cdn_resources='remote')
Graph = nx.Graph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')

vocab_df = [lululemon_vocab_df, adidas_vocab_df, nike_vocab_df]
tweet_groups = ['lululemon', 'adidas', 'nike']

for i in range(len(vocab_df)):
  temp = vocab_df[i].sort_values('count', ascending=False).head(150)
  for j in range(len(temp)):
    Graph.add_edge(temp.iloc[j]['word'].lower(), tweet_groups[i], weight=int(temp.iloc[j]['count']) // 100, title=int(temp.iloc[j]['count']))

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/user_words_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 386
    Edges: 433
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\user_words_pyvis.html)


We can see that Nike in particular is referenced much more in at (@) tweets than other companies. Interestingly enough, Nike's at (@) tweets also had a higher frequency of 'RT,' or retweet, than the other companies. Nike's at (@) tweets also had a higher frequency of 'available,' which could indicate that more people are asking if shoes or other Nike products are available for purchase.

Lululemon had more emojis as frequent words than other companies. Nike's most frequent word, however, was a reference to it's most frequent at (@) tweeter, which seems to be a sneaker new site or account.

# Segmentation by Follower Count

Because the number of unique users from the entire subset of tweets is so large, we can split the twitter mentions graph into two sections: users with over 50,000 followers and users between 4,300 and 5,000. This was done to limit the number of nodes on a graph and provide a similar number of nodes between network graphs.


```python
nt = Network('1200', '1600', directed=True, notebook=True, cdn_resources='remote')
Graph = nx.DiGraph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')
tweet_groups = [lululemon, adidas, nike]

for lst in tweet_groups:
  for i in range(len(lst)):
    if lst[i][2]['user']['followers_count'] >= 50000:
      if Graph.has_edge(lst[i][0], lst[i][1]):
        w = Graph.edges[lst[i][0], lst[i][1]]['title']
        Graph.edges[lst[i][0], lst[i][1]]['weight'] = ((w+1) //2)  +1
        Graph.edges[lst[i][0], lst[i][1]]['title'] = w+1
      else: Graph.add_edge(lst[i][0], lst[i][1], weight=1, title=1)

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/high_follower_mention_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 285
    Edges: 300
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\high_follower_mention_pyvis.html)

We can see that most of the high follower count users at (@) tweet Nike more than the other brands. Adidas is (@) tweeted less and Lululemon even less so. We can also see there is a similar distribution to the number of users who at (@) tweet multiple companies. Most users who at (@) tweet two companies will (@) tweet Nike and Adidas. There are a very few amount of users who at (@) tweet all three companies.


```python
nt = Network('1200', '1600', directed=True, notebook=True, cdn_resources='remote')
Graph = nx.DiGraph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')
tweet_groups = [lululemon, adidas, nike]

for lst in tweet_groups:
  for i in range(len(lst)):
    if (lst[i][2]['user']['followers_count'] >= 4300) & (lst[i][2]['user']['followers_count'] <= 5000):
      if Graph.has_edge(lst[i][0], lst[i][1]):
        w = Graph.edges[lst[i][0], lst[i][1]]['title']
        Graph.edges[lst[i][0], lst[i][1]]['weight'] = ((w+1) //2)  +1
        Graph.edges[lst[i][0], lst[i][1]]['title'] = w+1
      else: Graph.add_edge(lst[i][0], lst[i][1], weight=1, title=1)

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/low_follower_mention_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 805
    Edges: 869
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\low_follower_mentions_pyvis.html)

We can see that the distribution of at (@) tweets from lower follower tweeters is very similar to the high follower tweeters.

We can now investigate the semantic network graphs created from these two segments.


```python
lululemon_df['followers'] = lululemon_df['tweet'].apply(lambda x: x['user']['followers_count'])
adidas_df['followers'] = adidas_df['tweet'].apply(lambda x: x['user']['followers_count'])
nike_df['followers'] = nike_df['tweet'].apply(lambda x: x['user']['followers_count'])
```


```python
lululemon_df_high_follower = lululemon_df[lululemon_df['followers'] >= 50000]
adidas_df_high_follower = adidas_df[adidas_df['followers'] >= 50000]
nike_df_high_follower = nike_df[nike_df['followers'] >= 50000]

lululemon_high_follower_tweets = lululemon_df_high_follower['tweet'].values
adidas_high_follower_tweets = adidas_df_high_follower['tweet'].values
nike_high_follower_tweets = nike_df_high_follower['tweet'].values

lululemon_vocab_df = pd.DataFrame(list(token_counts(list(lululemon_high_follower_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count']) #Adjective, Nouns, and Verbs Only
adidas_vocab_df = pd.DataFrame(list(token_counts(list(adidas_high_follower_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
nike_vocab_df = pd.DataFrame(list(token_counts(list(nike_high_follower_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
```


```python
nt = Network('1200', '1600', directed=False, notebook=True, cdn_resources='remote')
Graph = nx.Graph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')

vocab_df = [lululemon_vocab_df, adidas_vocab_df, nike_vocab_df]
tweet_groups = ['lululemon', 'adidas', 'nike']

for i in range(len(vocab_df)):
  temp = vocab_df[i].sort_values('count', ascending=False).head(150)
  for j in range(len(temp)):
    Graph.add_edge(temp.iloc[j]['word'].lower(), tweet_groups[i], weight=int(temp.iloc[j]['count']) // 100, title=int(temp.iloc[j]['count']))

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/high_follower_words_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 338
    Edges: 436
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\high_follower_words_pyvis.html)

We can see that the top 150 words associated with each of the brands from tweets with a high follower count are distributed similarly to the previous semantic network graphs. Both Nike and Adidas have more word overlap than with Lululemon. There are a few words that all three companies have in common, but all words are fairly common on twitter and when discussing a company, including rt (re-tweet), store, and online.

Interestingly, a fair amount of overlap words from Nike and Adidas are other companies (xbox, starbucks, subway, etc.). Both Adidas and Lululemon have a handful of emoji's as their most frequent words.


```python
lululemon_df_low_follower = lululemon_df[(lululemon_df['followers'] >= 4300)&(lululemon_df['followers'] <= 5000)]
adidas_df_low_follower = adidas_df[(adidas_df['followers'] >= 4300)&(adidas_df['followers'] <= 5000)]
nike_df_low_follower = nike_df[(nike_df['followers'] >= 4300)&(nike_df['followers'] <= 5000)]

lululemon_low_follower_tweets = lululemon_df_low_follower['tweet'].values
adidas_low_follower_tweets = adidas_df_low_follower['tweet'].values
nike_low_follower_tweets = nike_df_low_follower['tweet'].values

lululemon_vocab_df = pd.DataFrame(list(token_counts(list(lululemon_low_follower_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count']) #Adjective, Nouns, and Verbs Only
adidas_vocab_df = pd.DataFrame(list(token_counts(list(adidas_low_follower_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
nike_vocab_df = pd.DataFrame(list(token_counts(list(nike_low_follower_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
```


```python
nt = Network('1200', '1600', directed=False, notebook=True, cdn_resources='remote')
Graph = nx.Graph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')

vocab_df = [lululemon_vocab_df, adidas_vocab_df, nike_vocab_df]
tweet_groups = ['lululemon', 'adidas', 'nike']

for i in range(len(vocab_df)):
  temp = vocab_df[i].sort_values('count', ascending=False).head(150)
  for j in range(len(temp)):
    Graph.add_edge(temp.iloc[j]['word'].lower(), tweet_groups[i], weight=int(temp.iloc[j]['count']) // 100, title=int(temp.iloc[j]['count']))

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/low_follower_words_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 353
    Edges: 439
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\low_follwer_words_pyvis.html)

We can see that the top 150 words associated with each of the brands from tweets with a low follower count is noticeably different than previous semantic network graphs. Here, there is more overlap words between the three companies, which could indicate that there is less brand recognition for users with lower follower counts. Visually, it looks like Lululemon has the largest number of unshared words while Nike has the least.


# Sentiment Analysis

We can also attempt to classify tweets by sentiment and take a small subset of each to analyze user and word networks.


```python
# takes a while to run
lululemon_df['tweet_sentiment'] = lululemon_df['tweet'].apply(tweet_sentiment)
adidas_df['tweet_sentiment'] = adidas_df['tweet'].apply(tweet_sentiment)
nike_df['tweet_sentiment'] = nike_df['tweet'].apply(tweet_sentiment)
```


```python
lululemon_df_high_sentiment = lululemon_df[lululemon_df['tweet_sentiment'] >= 0.95]
adidas_df_high_sentiment = adidas_df[adidas_df['tweet_sentiment'] >= 0.95]
nike_df_high_sentiment = nike_df[nike_df['tweet_sentiment'] >= 0.95]


lululemon_high_sentiment_tweets = lululemon_df_high_sentiment['tweet'].values
adidas_high_sentiment_tweets = adidas_df_high_sentiment['tweet'].values
nike_high_sentiment_tweets = nike_df_high_sentiment['tweet'].values

lululemon_vocab_df = pd.DataFrame(list(token_counts(list(lululemon_high_sentiment_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count']) #Adjective, Nouns, and Verbs Only
adidas_vocab_df = pd.DataFrame(list(token_counts(list(adidas_high_sentiment_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
nike_vocab_df = pd.DataFrame(list(token_counts(list(nike_high_sentiment_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
```


```python
nt = Network('1200', '1600', directed=True, notebook=True, cdn_resources='remote')
Graph = nx.DiGraph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')
tweet_groups = [lululemon_high_sentiment_tweets, adidas_high_sentiment_tweets, nike_high_sentiment_tweets]
tweet_dest = ['lululemon','adidas','nike']

for lst in range(len(tweet_groups)):
  for i in range(len(tweet_groups[lst])):
    if Graph.has_edge(tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]):
      w = Graph.edges[tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]]['title']
      Graph.edges[tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]]['weight'] = ((w+1) //5)  +1
      Graph.edges[tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]]['title'] = w+1
    else: Graph.add_edge(tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst], weight=1, title=1)

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/high_sentiment_mention_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 221
    Edges: 223
    

## Analysis
[Html link here!]({{site.url}}\ms_projects\dtsa_5800\high_sentiment_mention_pyvis.html)

We can see that there is very little overlap between users who at (@) tweet with a high (positive) sentiment. Only Nike and Adidas share users, while Lululemon has no user's in common between Nike and Adidas. Lululemon does have the overall least amount of users with a high sentiment tweet, but this is expected because Lululemon was the smallest subset of tweets in this dataset.


```python
nt = Network('1200', '1600', directed=False, notebook=True, cdn_resources='remote')
Graph = nx.Graph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')

vocab_df = [lululemon_vocab_df, adidas_vocab_df, nike_vocab_df]
tweet_groups = ['lululemon', 'adidas', 'nike']

for i in range(len(vocab_df)):
  temp = vocab_df[i].sort_values('count', ascending=False).head(150)
  for j in range(len(temp)):
    Graph.add_edge(temp.iloc[j]['word'].lower(), tweet_groups[i], weight=int(temp.iloc[j]['count']) // 100, title=int(temp.iloc[j]['count']))

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/high_sentiment_words_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 323
    Edges: 430
    

## Analysis

[Html link here!]({{site.url}}\ms_projects\dtsa_5800\high_sentiment_words_pyvis.html)

There are a fair amount of shared words from the positive sentiment tweets. Some words were either brand names or names, but most were positive descriptors. Between all three companies, there were shared words like 'family', 'good', and 'amazing', which was not seen in the previous semantic network graphs.


```python
lululemon_df_low_sentiment = lululemon_df[lululemon_df['tweet_sentiment'] <= -0.92]
adidas_df_low_sentiment = adidas_df[adidas_df['tweet_sentiment'] <= -0.92]
nike_df_low_sentiment = nike_df[nike_df['tweet_sentiment'] <= -0.92]


lululemon_low_sentiment_tweets = lululemon_df_low_sentiment['tweet'].values
adidas_low_sentiment_tweets = adidas_df_low_sentiment['tweet'].values
nike_low_sentiment_tweets = nike_df_low_sentiment['tweet'].values

lululemon_vocab_df = pd.DataFrame(list(token_counts(list(lululemon_low_sentiment_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count']) #Adjective, Nouns, and Verbs Only
adidas_vocab_df = pd.DataFrame(list(token_counts(list(adidas_low_sentiment_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
nike_vocab_df = pd.DataFrame(list(token_counts(list(nike_low_sentiment_tweets), parts_of_speech=['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']).items()), columns=['word','count'])
```


```python
nt = Network('1200', '1600', directed=True, notebook=True, cdn_resources='remote')
Graph = nx.DiGraph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')
tweet_groups = [lululemon_low_sentiment_tweets, adidas_low_sentiment_tweets, nike_low_sentiment_tweets]
tweet_dest = ['lululemon','adidas','nike']

for lst in range(len(tweet_groups)):
  for i in range(len(tweet_groups[lst])):
    if Graph.has_edge(tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]):
      w = Graph.edges[tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]]['title']
      Graph.edges[tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]]['weight'] = ((w+1) //5)  +1
      Graph.edges[tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst]]['title'] = w+1
    else: Graph.add_edge(tweet_groups[lst][i]['user']['screen_name'], tweet_dest[lst], weight=1, title=1)

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/low_sentiment_mention_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 202
    Edges: 203
    

## Analysis

[Html link here!]({{site.url}}\ms_projects\dtsa_5800\low_sentiment_mention_pyvis.html)

For the negative sentiment tweets, only Nike and Adidas shared users who at (@) tweeted them. Lululemon only had one user, while Nike took up almost all the unique users who at (@) tweeted them a very negative sentiment (bad) tweet.


```python
nt = Network('1200', '1600', directed=False, notebook=True, cdn_resources='remote')
Graph = nx.Graph()
Graph.add_node('lululemon', size=35, group=1, color='blue')
Graph.add_node('nike', size=35, group=2, color='red')
Graph.add_node('adidas', size=35, group=3, color='yellow')

vocab_df = [lululemon_vocab_df, adidas_vocab_df, nike_vocab_df]
tweet_groups = ['lululemon', 'adidas', 'nike']

for i in range(len(vocab_df)):
  temp = vocab_df[i].sort_values('count', ascending=False).head(150)
  for j in range(len(temp)):
    Graph.add_edge(temp.iloc[j]['word'].lower(), tweet_groups[i], weight=int(temp.iloc[j]['count']) // 100, title=int(temp.iloc[j]['count']))

nt.from_nx(Graph)
nt.show_buttons(filter_=['physics'])
nt.toggle_physics(False)
nt.show('example.html')
# display(HTML('example.html'))
nt.save_graph("drive/MyDrive/low_sentiment_words_pyvis.html")
```

```python
print('Nodes:', Graph.order())
print('Edges:', Graph.size())
```

    Nodes: 260
    Edges: 303
    

## Analysis

[Html link here!]({{site.url}}\ms_projects\dtsa_5800\low_sentiment_words_pyvis.html)

For the negative sentiment tweets, there were still some shared words between companies. Interestingly enough, the only shared word between the three companies was 'service,' indicating that it was likely most users who tweeted with a negative sentiment were unhappy potentially about the service of the companies, which could potentially mean the customer service.
