---
layout: page
title: DTSA 5510 - BBC News Classification Project
permalink: /ms_projects/dtsa5510_bbcnews
---

Using Non-Negative Matrix Factorization to Train an Unsupervised Model and Comparing Results to A Supervised Model.

Code additionally available [on GitHub](https://github.com/annasanders/ms_projects/blob/main/MS%20Projects/DTSA%205510%20-%20BBC%20News%20Classification%20Project%20NMF%20and%20Supervised%20Models.ipynb).

## Importing Libraries


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import re
import string

# Modeling
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# decision tree model

# Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
```

## Load Data


```python
data_test_x = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
data_train = pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')

data = pd.concat([data_train, data_test_x])
data = data.reset_index()
```

# Data Exploration


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2225 entries, 0 to 2224
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   index      2225 non-null   int64 
     1   ArticleId  2225 non-null   int64 
     2   Text       2225 non-null   object
     3   Category   1490 non-null   object
    dtypes: int64(2), object(2)
    memory usage: 69.7+ KB
    

We can see that the whole data (combined training and testing data) has very few columns. The ArticleId is the unique identifier of the article. The Text must be the full text of the article. The Category is then the labels of the articles.


```python
print('Total Articles: ',len(data))
print('Training Observations: ', len(data_train))
print('Teest Observations: ', len(data_test_x))
print('Train Test Split: %.2f' %(len(data_test_x)/(len(data_train)+len(data_test_x))*100), '%')
```

    Total Articles:  2225
    Training Observations:  1490
    Teest Observations:  735
    Train Test Split: 33.03 %
    


```python
print('Total Categories:', len(np.unique(data_train['Category'])))
```

    Total Categories: 5
    


```python
df = pd.DataFrame(data_train['Category'].value_counts()).reset_index()
fig, ax = plt.subplots()
ax.bar(df['Category'],df['count'], color=['tab:blue','tab:orange','tab:green','tab:red', 'tab:purple'])
ax.set_ylabel('Count')
ax.set_xlabel('Category')
ax.set_title('Count of Categories')
plt.show()
```


    
![png]({{site.url}}/ms_projects/dtsa_5510/output_10_0.png)
    



```python
df = data
df['Text Count'] = df['Text'].apply(lambda x: len(x))
fig, ax = plt.subplots()
ax.hist(data['Text Count'], bins=50)
ax.set_ylabel('Articles')
ax.set_xlabel('Word Count')
ax.set_title('Histogram of Text (String) Length - All Data')
ax.vlines(df['Text Count'].quantile(0.5), 0, 525, color='black')
ax.vlines(df['Text Count'].quantile(0.25), 0, 525, color='black', linestyle='dotted')
ax.vlines(df['Text Count'].quantile(0.75), 0, 525, color='black', linestyle='dotted')
fig.show()
```


    
![png]({{site.url}}/ms_projects/dtsa_5510/output_11_0.png)
    



```python
print('Median: ', df['Text Count'].quantile(0.5))
print('Q1: ', df['Text Count'].quantile(0.25))
print('Q3: ', df['Text Count'].quantile(0.75))
```

    Median:  1965.0
    Q1:  1446.0
    Q3:  2802.0
    


```python
df = data
df['Word Count'] = df['Text'].apply(lambda x: len(x.split(sep=' ')))
fig, ax = plt.subplots()
ax.hist(data['Word Count'], bins=50)
ax.set_ylabel('Articles')
ax.set_xlabel('Word Count')
ax.set_title('Histogram of Text Word Length')
ax.vlines(df['Word Count'].quantile(0.5), 0, 575, color='black')
ax.vlines(df['Word Count'].quantile(0.25), 0, 575, color='black', linestyle='dotted')
ax.vlines(df['Word Count'].quantile(0.75), 0, 575, color='black', linestyle='dotted')
fig.show()
```


    
![png]({{site.url}}/ms_projects/dtsa_5510/output_13_0.png)
    



```python
print('Median: ', df['Word Count'].quantile(0.5))
print('Q1: ', df['Word Count'].quantile(0.25))
print('Q3: ', df['Word Count'].quantile(0.75))
```

    Median:  361.0
    Q1:  268.0
    Q3:  514.0
    


```python
df = data_train
df['Text Count'] = df['Text'].apply(lambda x: len(x))
df['Word Count'] = df['Text'].apply(lambda x: len(x.split(sep=' ')))

df[['Category','Text Count', 'Word Count']].groupby('Category').mean()
```




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
      <th>Text Count</th>
      <th>Word Count</th>
    </tr>
    <tr>
      <th>Category</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>business</th>
      <td>1983.104167</td>
      <td>357.913690</td>
    </tr>
    <tr>
      <th>entertainment</th>
      <td>1910.380952</td>
      <td>360.531136</td>
    </tr>
    <tr>
      <th>politics</th>
      <td>2617.905109</td>
      <td>484.135036</td>
    </tr>
    <tr>
      <th>sport</th>
      <td>1894.624277</td>
      <td>361.667630</td>
    </tr>
    <tr>
      <th>tech</th>
      <td>2939.291188</td>
      <td>538.122605</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc[0]['Text']
```




    'worldcom ex-boss launches defence lawyers defending former worldcom chief bernie ebbers against a battery of fraud charges have called a company whistleblower as their first witness.  cynthia cooper  worldcom s ex-head of internal accounting  alerted directors to irregular accounting practices at the us telecoms giant in 2002. her warnings led to the collapse of the firm following the discovery of an $11bn (£5.7bn) accounting fraud. mr ebbers has pleaded not guilty to charges of fraud and conspiracy.  prosecution lawyers have argued that mr ebbers orchestrated a series of accounting tricks at worldcom  ordering employees to hide expenses and inflate revenues to meet wall street earnings estimates. but ms cooper  who now runs her own consulting business  told a jury in new york on wednesday that external auditors arthur andersen had approved worldcom s accounting in early 2001 and 2002. she said andersen had given a  green light  to the procedures and practices used by worldcom. mr ebber s lawyers have said he was unaware of the fraud  arguing that auditors did not alert him to any problems.  ms cooper also said that during shareholder meetings mr ebbers often passed over technical questions to the company s finance chief  giving only  brief  answers himself. the prosecution s star witness  former worldcom financial chief scott sullivan  has said that mr ebbers ordered accounting adjustments at the firm  telling him to  hit our books . however  ms cooper said mr sullivan had not mentioned  anything uncomfortable  about worldcom s accounting during a 2001 audit committee meeting. mr ebbers could face a jail sentence of 85 years if convicted of all the charges he is facing. worldcom emerged from bankruptcy protection in 2004  and is now known as mci. last week  mci agreed to a buyout by verizon communications in a deal valued at $6.75bn.'



From the above plots and analysis, we can see that the overall categorization is fairly balanced. Most categories have at least 400 associated articles. The overall length of the text seems to be relatively small (string lengths less than 5,000 characters); however there are some outliers. Similarly, breaking the text down by assumed words (string split by sentences), most articles are less than 1,000 words. Looking at the averages of these values between categories, most seem to have around the same text and word count; however, both politics and tech have slightly more text/words per article than other categories. Looking at the first article, the text has been uncapitalized and some punctuation has been removed, like apostrophes. The first text also seems to contain many proper nouns as well.

# Data Cleaning and Text Processing

From the above results, we should modify or remove the instances of a singular 's', as the likely indicated a possessive apostrophe, which has now been removed. The letter 's' likely should not be parsed when processing text. We also want to remove potentially unnecessary words that wouldn't add value to determining the topic (category) of the article.

We can first load in the spaCy package, which is one of many python packages that do natural language processing, or turning words and sentences into vectors to relate words, sentences, and text through modeling. 

For this project, I loaded the pre-trained English language model. For the English language, spaCy has multiple models. The 'medium' model was chosen to ensure a balance of efficiency and accuracy.

Some NLP packages will allow the user or users to train the NLP pipeline or add and/or update the list of words that shouldn't be included in the text to vectorization process. We can first investigate the "en_core_web_sm" spaCy model.


```python
print(len(nlp.Defaults.stop_words), list(nlp.Defaults.stop_words))
```

    326 ['who', 'which', 'were', 'whereupon', 'of', 'when', 'beside', 'perhaps', 'seem', 'everywhere', 'few', 'even', 'him', 'hundred', 'yet', "n't", 'besides', 'four', 'anyway', 'otherwise', "'re", '’ve', 'under', 'go', 'before', 'therein', 'itself', 'nevertheless', 'wherever', 'further', 'show', 'serious', 'after', 'eleven', 'more', 'becomes', 'ours', 'front', 'why', 'whom', 'am', 'amount', 'do', 'amongst', 'five', 'side', 'because', 'fifteen', 'not', 'me', 'hers', 'own', 'every', 'put', 'above', 'anywhere', 'if', 'thereby', 'though', 'my', 'bottom', 'should', 'whoever', 'quite', 'none', 'ca', 'we', 'may', 'back', 'whose', 'toward', 'also', '’d', 'too', 'as', 'i', 'around', 'hence', 'enough', 'beyond', '‘re', 'how', 'does', "'m", 'always', 'out', 'those', '‘ll', 'less', 'into', 'another', 'no', 'now', 'cannot', '‘d', 'first', 'afterwards', 'regarding', 'since', 'others', 'but', 'upon', 'hereupon', 'only', 'namely', 'herself', '‘ve', 'behind', '’ll', 'these', 'see', 'and', 'nine', 'done', 'really', 'off', "'s", 'thus', 'seeming', 'became', 'did', 'all', 'anyone', 'seemed', 'either', 'eight', 'several', 'former', 'used', 'had', 'onto', 'nor', 'its', "'ll", 'everything', 'their', 'whether', 'third', 'still', "'ve", 'whole', 'two', 'an', 'your', 'have', 'been', 'she', 'will', 'has', 'again', 'much', 'someone', 'this', 'together', 'everyone', 'just', 'here', 'you', 'fifty', 'her', 'twelve', 'unless', 'somehow', 'on', 'being', 'elsewhere', 'else', 'then', 'until', 'must', 'indeed', 'same', 'what', 'thence', 'latter', 'could', '’re', 'yours', "'d", 'nobody', 'call', 'they', 'move', 'using', 'thru', 'moreover', 'get', 'any', 'please', 'due', 'whereby', 'seems', 'something', 'would', 'between', 'along', 'beforehand', 'sixty', 'rather', 'ten', 'nothing', 'among', 'without', 'through', 'full', 'that', 'never', 'various', 'themselves', 'in', '‘m', 'such', 'top', 'doing', 'down', 'sometime', 'are', 'mostly', 'some', 'might', 'become', 'n’t', 'whither', 'at', 'whereas', 'was', 'herein', 'can', 'towards', 'both', 'a', 'nowhere', 'three', 'wherein', 'therefore', 'from', 'be', 'each', 'make', 'other', 'take', 'hereafter', 'or', 'them', 'meanwhile', 'than', 'below', 'himself', 'via', 'to', 'next', 'very', 'almost', 'latterly', 'becoming', 'often', 'forty', 'n‘t', 'many', 'neither', 'for', 'whatever', 'six', 'anything', 'ever', 'our', 'empty', 'up', 'keep', 'one', 'he', 'within', 'although', 'yourself', 'sometimes', 'whereafter', 'however', 'us', 'where', 'it', 'alone', '‘s', 'across', 'against', 'last', 'yourselves', 'is', 'with', 'made', 'per', 'the', 'so', 'about', 'while', 'hereby', '’m', 'well', 'part', 'thereafter', '’s', 'mine', 'there', 'ourselves', 'over', 'his', 'once', 'anyhow', 'whence', 'say', 'throughout', 'during', 'already', 'thereupon', 'except', 'least', 'myself', 'give', 'twenty', 'noone', 'most', 're', 'formerly', 'by', 'name', 'whenever', 'somewhere']
    

The model has a list of 326 default stop words. In the stop words is 's, which means that it is likely better if we remove the single s in the text instead of removing the space or adding back in an apostrophe. We should similarly remove single t's, single d's, single m's, double ll's, re's, and ve's (which also exists as a stop word in the list without an apostrophe). We should also remove any instances of a double space, as that is also not helpful to keep in the text.


```python
data_train_clean = data_train.copy()
data_test_clean = data_test_x.copy()
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace(' s ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace(' s ', ' '))
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace(' t ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace(' t ', ' '))
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace(' d ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace(' d ', ' '))
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace(' m ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace(' m ', ' '))
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace(' ll ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace(' ll ', ' '))
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace(' ve ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace(' ve ', ' '))
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.replace('  ', ' '))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.replace('  ', ' '))
data_test_clean.loc[0]['Text']
```




    'qpr keeper day heads for preston queens park rangers keeper chris day is set to join preston on a month loan. day has been displaced by the arrival of simon royce who is in his second month on loan from charlton. qpr have also signed italian generoso rossi. r manager ian holloway said: some might say it a risk as he can be recalled during that month and simon royce can now be recalled by charlton. but i have other irons in the fire. i have had a yes from a couple of others should i need them.  day rangers contract expires in the summer. meanwhile holloway is hoping to complete the signing of middlesbrough defender andy davies - either permanently or again on loan - before saturday match at ipswich. davies impressed during a recent loan spell at loftus road. holloway is also chasing bristol city midfielder tom doherty.'



We can also remove the list of stop words and leftover punctuation from the text.


```python
data_train_clean['Text'] = data_train_clean['Text'].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))
data_test_clean['Text'] = data_test_clean['Text'].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))

def remove_words(text):
    doc = nlp(text)
    n_text = [word.text for word in doc if (word not in nlp.Defaults.stop_words)] #word.text forces back to string
    return ' '.join(n_text) #forces back to full text

# takes a bit of time
data_train_clean['Text'] = data_train_clean['Text'].apply(remove_words)
data_test_clean['Text'] = data_test_clean['Text'].apply(remove_words)
    
print(data_test_clean.loc[0]['Text'])
```

    qpr keeper day heads for preston queens park rangers keeper chris day is set to join preston on a month loan day has been displaced by the arrival of simon royce who is in his second month on loan from charlton qpr have also signed italian generoso rossi r manager ian holloway said some might say it a risk as he can be recalled during that month and simon royce can now be recalled by charlton but i have other irons in the fire i have had a yes from a couple of others should i need them   day rangers contract expires in the summer meanwhile holloway is hoping to complete the signing of middlesbrough defender andy davies   either permanently or again on loan   before saturday match at ipswich davies impressed during a recent loan spell at loftus road holloway is also chasing bristol city midfielder tom doherty
    


```python
data_clean = pd.concat([data_test_clean, data_train_clean])
data_clean = data_clean.reset_index()
df = data_clean

df['Word Count'] = df['Text'].apply(lambda x: len(x.split(sep=' ')))
fig, ax = plt.subplots()
ax.hist(data['Word Count'], bins=50)
ax.set_ylabel('Articles')
ax.set_xlabel('Word Count')
ax.set_title('Histogram of Text Word Length - Post Text Cleaning')
ax.vlines(df['Word Count'].quantile(0.5), 0, 575, color='black')
ax.vlines(df['Word Count'].quantile(0.25), 0, 575, color='black', linestyle='dotted')
ax.vlines(df['Word Count'].quantile(0.75), 0, 575, color='black', linestyle='dotted')
fig.show()
```


    
![png]({{site.url}}/ms_projects/dtsa_5510/output_25_0.png)
    



```python
print('Median: ', df['Word Count'].quantile(0.5))
print('Q1: ', df['Word Count'].quantile(0.25))
print('Q3: ', df['Word Count'].quantile(0.75))
```

    Median:  340.0
    Q1:  251.0
    Q3:  482.0
    

We can see that the overall removal of stop words did not significantly decrease the overall words in the text.

# Text Vectorization

We are ready to vectorize the cleaned text. The sklearn package comes with a TF-IDF function that converts a collection of raw documents, in this case, the cleaned text column from our train and  test data, and converts it into a matrix of TF-IDF features. 

TF-IDF is a measure of the importance of words to a document, adjusted for frequent words. Higher values indicate that the word is more relevant to a text. TF-IDF is similar to the bag of words method, but normalizes the word based on frequency in other texts.

With Non-Negative Matrix Factorization in mind, TF-IDF is useful because the results are strictly positive. We can then feed the created matrix into the non-negative matrix factorization function.


```python
tfidf = TfidfVectorizer(max_df = 0.95, min_df=2, max_features=2000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(np.array(data_train_clean['Text']))
```

# Non-Negative Matrix Factorization
With the TF-IDF matrix created, we can now run the non-negative matrix factorization. We know from the data that there are 5 categories, so we can set the number of components to 5. For now, we can leave all the other hyperparameters as the model's default. Since this is unsupervised learning, we can run the model on all available data and compare the results to the known labels.


```python
nmf = NMF(n_components=5, random_state=1)
nmf.fit(tfidf_matrix)
W = nmf.fit_transform(tfidf_matrix)
H = nmf.components_
```

We can now look at what words the model has found to be most important, knowing there are 5 topics in total.


```python
# code adapted from sklearn's Topic extraction with Non-negative Matrix Factorization and Latent 
# Dirichelet Allocation (see resources below)
n_top_words = 10
model_feature_df = pd.DataFrame(columns=['Topic','Word','Weight'])

for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = tfidf.get_feature_names_out()[top_features_ind]
        weights = topic[top_features_ind]
        d = {'Topic': [topic_idx]*n_top_words, 'Word':top_features, 'Weight':weights}
        d = pd.DataFrame(d)
        model_feature_df = pd.concat([model_feature_df,d])
        print('Topic: ', topic_idx, 'Words: ', top_features)
```

    Topic:  0 Words:  ['france' 'play' 'players' 'said' 'cup' 'ireland' 'wales' 'win' 'game'
     'england']
    Topic:  1 Words:  ['prime' 'minister' 'government' 'said' 'party' 'brown' 'election' 'blair'
     'labour' 'mr']
    Topic:  2 Words:  ['phones' 'software' 'users' 'digital' 'technology' 'phone' 'said' 'music'
     'mobile' 'people']
    Topic:  3 Words:  ['director' 'won' 'festival' 'films' 'actress' 'actor' 'award' 'awards'
     'best' 'film']
    Topic:  4 Words:  ['economic' '2004' 'oil' 'bank' 'market' 'sales' 'year' 'economy' 'growth'
     'said']
    

Looking at this list, it seems like the topics have found promising words! Topic 0 looks to be related to sports; as the BBC is an UK based paper, it isn't surprising that the topic has found words like 'match' and 'club,' likely talking about football (American soccer) and words like 'england' and 'wales' which are regions in the UK. Topic 1 looks to be associated with politics, while topic 2 sounds like business news. Topic 3 has words associated with mostly films, but potentially entertainment in general, and topic 4 has words associated with technology.

## Choosing the Most Likely Category
We can look at the W matrix in order to determine the category with the highest value, or the most likely category for the text/article.


```python
cat = []
for i in range(len(W)):
    cat.append(max(enumerate(W[i]), key=lambda x:x[1])[0])
    
data_train_clean['Predicted Category'] = cat
data_train_clean
```




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
      <th>ArticleId</th>
      <th>Text</th>
      <th>Category</th>
      <th>Text Count</th>
      <th>Word Count</th>
      <th>Predicted Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1833</td>
      <td>worldcom exboss launches defence lawyers defen...</td>
      <td>business</td>
      <td>1866</td>
      <td>324</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154</td>
      <td>german business confidence slides german busin...</td>
      <td>business</td>
      <td>2016</td>
      <td>348</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1101</td>
      <td>bbc poll indicates economic gloom citizens in ...</td>
      <td>business</td>
      <td>3104</td>
      <td>551</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976</td>
      <td>lifestyle governs mobile choice faster better ...</td>
      <td>tech</td>
      <td>3618</td>
      <td>692</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>917</td>
      <td>enron bosses in 168 m payout eighteen former e...</td>
      <td>business</td>
      <td>2190</td>
      <td>381</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1485</th>
      <td>857</td>
      <td>double eviction from big brother model caprice...</td>
      <td>entertainment</td>
      <td>1266</td>
      <td>237</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>325</td>
      <td>dj double act revamp chart show dj duo jk and ...</td>
      <td>entertainment</td>
      <td>3111</td>
      <td>619</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>1590</td>
      <td>weak dollar hits reuters revenues at media gro...</td>
      <td>business</td>
      <td>1370</td>
      <td>252</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>1587</td>
      <td>apple ipod family expands market apple has exp...</td>
      <td>tech</td>
      <td>3242</td>
      <td>595</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>538</td>
      <td>santy worm makes unwelcome visit thousands of ...</td>
      <td>tech</td>
      <td>1723</td>
      <td>304</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1490 rows × 6 columns</p>
</div>



We now need to associate the predicted numeric category with the best actually labeled category and relabel the predicted classes for easy metric calculations.


```python
# Code copied from Week 2 Assignment
def label_permute_compare(ytdf,yp,n=5):
    """
    ytdf: labels dataframe object
    yp: clustering label prediction output
    Returns permuted label order and accuracy. 
    Example output: (3, 4, 1, 2, 0), 0.74 
    """
    # your code here
    ytdf = np.array(ytdf).reshape(1,len(ytdf))[0]
    
    o_labels = np.unique(ytdf)
    n_labels = np.unique(yp)
    b_labels = []
    f_label = []
    u_label = []
    for i in o_labels:
        max_label_max = float('-inf')
        max_label = None
        for j in n_labels:
            count = 0
            l_count = 0
            for n in range(len(ytdf)):
                if (ytdf[n] == i) and (yp[n] == j):
                    l_count += 1
                    count += 1
                if (ytdf[n] == i) and (yp[n] != j):
                    count += 1
#             print(j, max_label_max, float(l_count/count))
            if ((l_count/count) > max_label_max):
                if j in u_label:
                    continue
                max_label_max = float(l_count/count)
                max_label = ([i,j],[l_count, count])
        u_label.append(max_label[0][1])
        b_labels.append(max_label)
    t_count = 0
    t_ccount = 0
    for i in b_labels:
        f_label.append(i[0])
        t_ccount += i[1][0]
        t_count += i[1][1]
        
    return (f_label,(t_ccount/t_count))
```


```python
pred_labels = label_permute_compare(data_train_clean['Category'].values, data_train_clean['Predicted Category'].values)
print(pred_labels)

pred_labels_lst = [None] * len(pred_labels[0])
for i in pred_labels[0]:
    pred_labels_lst[i[1]] = i[0]

data_clean_nmf1 = data_train_clean.copy()

for i in pred_labels[0]:
    data_clean_nmf1.loc[data_clean_nmf1['Predicted Category'] == i[1], 'Predicted Category'] = i[0]

data_clean_nmf1
```

    ([['business', 4], ['entertainment', 3], ['politics', 1], ['sport', 0], ['tech', 2]], 0.9154362416107382)
    




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
      <th>ArticleId</th>
      <th>Text</th>
      <th>Category</th>
      <th>Text Count</th>
      <th>Word Count</th>
      <th>Predicted Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1833</td>
      <td>worldcom exboss launches defence lawyers defen...</td>
      <td>business</td>
      <td>1866</td>
      <td>324</td>
      <td>business</td>
    </tr>
    <tr>
      <th>1</th>
      <td>154</td>
      <td>german business confidence slides german busin...</td>
      <td>business</td>
      <td>2016</td>
      <td>348</td>
      <td>business</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1101</td>
      <td>bbc poll indicates economic gloom citizens in ...</td>
      <td>business</td>
      <td>3104</td>
      <td>551</td>
      <td>business</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976</td>
      <td>lifestyle governs mobile choice faster better ...</td>
      <td>tech</td>
      <td>3618</td>
      <td>692</td>
      <td>tech</td>
    </tr>
    <tr>
      <th>4</th>
      <td>917</td>
      <td>enron bosses in 168 m payout eighteen former e...</td>
      <td>business</td>
      <td>2190</td>
      <td>381</td>
      <td>business</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1485</th>
      <td>857</td>
      <td>double eviction from big brother model caprice...</td>
      <td>entertainment</td>
      <td>1266</td>
      <td>237</td>
      <td>entertainment</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>325</td>
      <td>dj double act revamp chart show dj duo jk and ...</td>
      <td>entertainment</td>
      <td>3111</td>
      <td>619</td>
      <td>tech</td>
    </tr>
    <tr>
      <th>1487</th>
      <td>1590</td>
      <td>weak dollar hits reuters revenues at media gro...</td>
      <td>business</td>
      <td>1370</td>
      <td>252</td>
      <td>business</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>1587</td>
      <td>apple ipod family expands market apple has exp...</td>
      <td>tech</td>
      <td>3242</td>
      <td>595</td>
      <td>tech</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>538</td>
      <td>santy worm makes unwelcome visit thousands of ...</td>
      <td>tech</td>
      <td>1723</td>
      <td>304</td>
      <td>tech</td>
    </tr>
  </tbody>
</table>
<p>1490 rows × 6 columns</p>
</div>




```python
cf = confusion_matrix(data_clean_nmf1['Category'].values, data_clean_nmf1['Predicted Category'].values)

cmd = ConfusionMatrixDisplay(cf, display_labels = pred_labels_lst)
fig, ax = plt.subplots(figsize=(8,10))
cmd.plot(ax=ax)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7ab07aec71f0>




    
![png]({{site.url}}/ms_projects/dtsa_5510/output_41_1.png)
    



```python
print('Accuracy: %.3f' %(accuracy_score(data_clean_nmf1['Category'].values, data_clean_nmf1['Predicted Category'].values)))
print('Precision: %.3f' %(precision_score(data_clean_nmf1['Category'].values, data_clean_nmf1['Predicted Category'].values, average='weighted')))
print('Recall: %.3f' %(recall_score(data_clean_nmf1['Category'].values, data_clean_nmf1['Predicted Category'].values, average='weighted')))
print('F1: %.3f' %(f1_score(data_clean_nmf1['Category'].values, data_clean_nmf1['Predicted Category'].values, average='weighted')))

```

    Accuracy: 0.915
    Precision: 0.920
    Recall: 0.915
    F1: 0.915
    

The non-negative matrix factorization performs very well. With an overall accuracy of 0.915, or 92%, this is a relatively well trained model. In the confusion matrix, we see that political articles are often misclassified as business. We can also see that some articles are misclassified as sports.

We can now try various hyperparameters in NMF to find the best model.


```python
params = {'init':['random','nndsvdar'], 'solver':['mu'], 'beta_loss':['frobenius','kullback-leibler']}
pgrid_dict = list(ParameterGrid(params))

for hyp in pgrid_dict:
    nmf = NMF(n_components=5, random_state=1, init=hyp.get('init'), solver=hyp.get('solver'), beta_loss=hyp.get('beta_loss'))
    nmf.fit(tfidf_matrix)
    W = nmf.fit_transform(tfidf_matrix)

    cat = []
    for i in range(len(W)):
        cat.append(max(enumerate(W[i]), key=lambda x:x[1])[0])

    data_train_clean['Predicted Category'] = cat

    pred_labels = label_permute_compare(data_train_clean['Category'].values, data_train_clean['Predicted Category'].values)
    print(pred_labels)

    pred_labels_lst = [None] * len(pred_labels[0])
    for i in pred_labels[0]:
        pred_labels_lst[i[1]] = i[0]

    data_clean_nmf2 = data_train_clean.copy()

    for i in pred_labels[0]:
        data_clean_nmf2.loc[data_clean_nmf2['Predicted Category'] == i[1], 'Predicted Category'] = i[0]

    cf = confusion_matrix(data_clean_nmf2['Category'].values, data_clean_nmf2['Predicted Category'].values)

    cmd = ConfusionMatrixDisplay(cf, display_labels = pred_labels_lst)
    fig, ax = plt.subplots(figsize=(8,10))
    ax.set_title(str(hyp.get('init')) + ', ' + str(hyp.get('solver')) + ', ' + str(hyp.get('beta_loss')))
    cmd.plot(ax=ax)
    
    print('Combination: ', hyp)
    print('Accuracy: %.3f' %(accuracy_score(data_clean_nmf2['Category'].values, data_clean_nmf2['Predicted Category'].values)))
    print('Precision: %.3f' %(precision_score(data_clean_nmf2['Category'].values, data_clean_nmf2['Predicted Category'].values, average='weighted')))
    print('Recall: %.3f' %(recall_score(data_clean_nmf2['Category'].values, data_clean_nmf2['Predicted Category'].values, average='weighted')))
    print('F1: %.3f' %(f1_score(data_clean_nmf2['Category'].values, data_clean_nmf2['Predicted Category'].values, average='weighted')))
```

    ([['business', 1], ['entertainment', 2], ['politics', 0], ['sport', 4], ['tech', 3]], 0.8986577181208054)
    Combination:  {'beta_loss': 'frobenius', 'init': 'random', 'solver': 'mu'}
    Accuracy: 0.899
    Precision: 0.904
    Recall: 0.899
    F1: 0.897
    ([['business', 4], ['entertainment', 3], ['politics', 1], ['sport', 0], ['tech', 2]], 0.9107382550335571)
    Combination:  {'beta_loss': 'frobenius', 'init': 'nndsvdar', 'solver': 'mu'}
    Accuracy: 0.911
    Precision: 0.916
    Recall: 0.911
    F1: 0.910
    ([['business', 1], ['entertainment', 4], ['politics', 2], ['sport', 0], ['tech', 3]], 0.9449664429530201)
    Combination:  {'beta_loss': 'kullback-leibler', 'init': 'random', 'solver': 'mu'}
    Accuracy: 0.945
    Precision: 0.947
    Recall: 0.945
    F1: 0.945
    ([['business', 4], ['entertainment', 3], ['politics', 1], ['sport', 0], ['tech', 2]], 0.9496644295302014)
    Combination:  {'beta_loss': 'kullback-leibler', 'init': 'nndsvdar', 'solver': 'mu'}
    Accuracy: 0.950
    Precision: 0.952
    Recall: 0.950
    F1: 0.950
    


    
![png]({{site.url}}/ms_projects/dtsa_5510/output_44_1.png)
    



    
![png]({{site.url}}/ms_projects/dtsa_5510/output_44_2.png)
    



    
![png]({{site.url}}/ms_projects/dtsa_5510/output_44_3.png)
    



    
![png]({{site.url}}/ms_projects/dtsa_5510/output_44_4.png)
    


The highest combination ({'beta_loss': 'kullback-leibler', 'init': 'nndsvdar', 'solver': 'mu'}) had an accuracy of 0.95, or 95%. This is an improvement from the first non-negative matrix factorization model that we ran. Overall 95% accuracy is very good.

# Supervised Learning (Clustering)
We will now use the split training data into a usable train-test split and train various supervised learning models.


```python
tfidf = TfidfVectorizer(max_df = 0.95, min_df=2, max_features=2000, stop_words='english')

train, test = train_test_split(data_train_clean, test_size=0.3, shuffle=True, random_state=1)

train_x = tfidf.fit_transform((train['Text']))
test_x = tfidf.transform((test['Text']))

y_train = train['Category'].values
y_test = test['Category'].values
```


```python
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
```


```python
svc_model = LinearSVC()
knn_model = KNeighborsClassifier()
sgd_model = SGDClassifier()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
rg_model = RidgeClassifier()
lg_model = LogisticRegression()

svc_model.fit(train_x,y_train)
knn_model.fit(train_x,y_train)
sgd_model.fit(train_x,y_train)
dt_model.fit(train_x,y_train)
rf_model.fit(train_x,y_train)
rg_model.fit(train_x, y_train)
lg_model.fit(train_x, y_train)

svc_predict = svc_model.predict(test_x)
knn_predict =  knn_model.predict(test_x)
sgd_predict =  sgd_model.predict(test_x)
dt_predict = dt_model.predict(test_x)
rf_predict = rf_model.predict(test_x)
rg_predict = rg_model.predict(test_x)
lg_predict = lg_model.predict(test_x)

print('SVC Accuracy: %.3f' %(accuracy_score(y_test, svc_predict)))
print('KNN Accuracy: %.3f' %(accuracy_score(y_test, knn_predict)))
print('SGD Accuracy: %.3f' %(accuracy_score(y_test, sgd_predict)))
print('DT Accuracy: %.3f' %(accuracy_score(y_test, dt_predict)))
print('RF Accuracy: %.3f' %(accuracy_score(y_test, rf_predict)))
print('RG Accuracy: %.3f' %(accuracy_score(y_test, rg_predict)))
print('LG Accuracy: %.3f' %(accuracy_score(y_test, rg_predict)))
```

    SVC Accuracy: 0.980
    KNN Accuracy: 0.924
    SGD Accuracy: 0.980
    DT Accuracy: 0.801
    RF Accuracy: 0.957
    RG Accuracy: 0.982
    LG Accuracy: 0.982
    

From the above models, it looks like the Ridge Classifier or the Logistic Regression fit the data best with not hyperparameter changes. Most models, except for the KNN and Decision Tree model, preform better than the Non-negative Matrix Factorization. We can take a closer look at the two models below.


```python
cv_scores = cross_val_score(rg_model, train_x,y_train, cv=10, scoring='accuracy')
print('RG CV Scores: ',cv_scores)
cv_scores = cross_val_score(lg_model, train_x,y_train, cv=10, scoring='accuracy')
print('LG CV Scores: ',cv_scores)

cf = confusion_matrix(y_test, rg_predict)
cmd = ConfusionMatrixDisplay(cf, display_labels = rg_model.classes_)
fig, ax = plt.subplots(figsize=(8,10))
ax.set_title('RG Model')
cmd.plot(ax=ax)

cf = confusion_matrix(y_test, lg_predict)
cmd = ConfusionMatrixDisplay(cf, display_labels = lg_model.classes_)
fig, ax = plt.subplots(figsize=(8,10))
ax.set_title('LG Model')
cmd.plot(ax=ax)
```

    RG CV Scores:  [0.94285714 0.98095238 0.96190476 0.97115385 0.95192308 0.95192308
     0.99038462 0.96153846 0.98076923 0.97115385]
    LG CV Scores:  [0.93333333 0.98095238 0.97142857 0.98076923 0.96153846 0.95192308
     0.98076923 0.93269231 0.95192308 0.96153846]
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7ab0768d1870>




    
![png]({{site.url}}/ms_projects/dtsa_5510/output_51_2.png)
    



    
![png]({{site.url}}/ms_projects/dtsa_5510/output_51_3.png)
    


The cross validation scores for both models look good, so there does not seem to be any overfitting in the model. Looking at the confusion matrix, the LG model seems to predict entertainment models more accurately, while the RG model classifies tech articles more accurately.

For our best models, we can now test how changing the test size effects model accuracy.


```python
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
    train, test = train_test_split(data_train_clean, test_size=i, shuffle=True, random_state=1)

    train_x = tfidf.fit_transform((train['Text']))
    test_x = tfidf.transform((test['Text']))

    y_train = train['Category'].values
    y_test = test['Category'].values
    
    rg_model = RidgeClassifier()
    lg_model = LogisticRegression()
    
    rg_model.fit(train_x, y_train)
    lg_model.fit(train_x, y_train)
    
    rg_predict = rg_model.predict(test_x)
    lg_predict = lg_model.predict(test_x)
    
    print(str(i),' Test Size RG Accuracy: %.3f' %(accuracy_score(y_test, rg_predict)))
    print(str(i),' Test Size LG Accuracy: %.3f' %(accuracy_score(y_test, rg_predict)))
    
```

    0.1  Test Size RG Accuracy: 0.980
    0.1  Test Size LG Accuracy: 0.980
    0.2  Test Size RG Accuracy: 0.980
    0.2  Test Size LG Accuracy: 0.980
    0.3  Test Size RG Accuracy: 0.982
    0.3  Test Size LG Accuracy: 0.982
    0.4  Test Size RG Accuracy: 0.977
    0.4  Test Size LG Accuracy: 0.977
    0.5  Test Size RG Accuracy: 0.974
    0.5  Test Size LG Accuracy: 0.974
    0.6  Test Size RG Accuracy: 0.969
    0.6  Test Size LG Accuracy: 0.969
    0.7  Test Size RG Accuracy: 0.968
    0.7  Test Size LG Accuracy: 0.968
    

Interestingly enough, the split of 30% to training data had the highest accuracy. The accuracy didn't fall off as much as I expected, as at 70% of the data withheld for testing, the model was still more accurate than the Non-negative Matrix Factorization.


```python
params = {'penalty':['l2'], 'C':[0.5,1,1.5,2,2.5,3], 'class_weight':[None, 'balanced'], 'solver':['sag','lbfgs','saga']}
lg_model_gs = LogisticRegression(random_state=1, max_iter=500)
gs = GridSearchCV(lg_model_gs,params)
gs.fit(train_x, y_train)
```




<style>#sk-container-id-9 {color: black;background-color: white;}#sk-container-id-9 pre{padding: 0;}#sk-container-id-9 div.sk-toggleable {background-color: white;}#sk-container-id-9 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-9 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-9 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-9 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-9 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-9 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-9 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-9 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-9 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-9 div.sk-item {position: relative;z-index: 1;}#sk-container-id-9 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-9 div.sk-item::before, #sk-container-id-9 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-9 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-9 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-9 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-9 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-9 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-9 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-9 div.sk-label-container {text-align: center;}#sk-container-id-9 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-9 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-9" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(estimator=LogisticRegression(max_iter=500, random_state=1),
             param_grid={&#x27;C&#x27;: [0.5, 1, 1.5, 2, 2.5, 3],
                         &#x27;class_weight&#x27;: [None, &#x27;balanced&#x27;], &#x27;penalty&#x27;: [&#x27;l2&#x27;],
                         &#x27;solver&#x27;: [&#x27;sag&#x27;, &#x27;lbfgs&#x27;, &#x27;saga&#x27;]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(estimator=LogisticRegression(max_iter=500, random_state=1),
             param_grid={&#x27;C&#x27;: [0.5, 1, 1.5, 2, 2.5, 3],
                         &#x27;class_weight&#x27;: [None, &#x27;balanced&#x27;], &#x27;penalty&#x27;: [&#x27;l2&#x27;],
                         &#x27;solver&#x27;: [&#x27;sag&#x27;, &#x27;lbfgs&#x27;, &#x27;saga&#x27;]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-22" type="checkbox" ><label for="sk-estimator-id-22" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, random_state=1)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-23" type="checkbox" ><label for="sk-estimator-id-23" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=500, random_state=1)</pre></div></div></div></div></div></div></div></div></div></div>




```python
print(gs.best_estimator_)
print(gs.best_score_)
```

    LogisticRegression(C=3, class_weight='balanced', max_iter=500, random_state=1,
                       solver='saga')
    0.9645334928229665
    


```python
lg_model_best = LogisticRegression(C=1, class_weight='balanced', solver='sag', random_state=1, max_iter=500)
lg_model_best.fit(train_x, y_train)
y_pred = lg_model_best.predict(test_x)
print('Accuracy: %.3f' %(accuracy_score(y_test, y_pred)))
```

    Accuracy: 0.980
    

# Final Predictions
We can now predict the labels on the project's testing data set. We will Choose the Logistic Model and re-train the entire labeled data set.

Oddly enough, through multiple submissions, the model seems to overfit when fed too much training data. The lg model referenced below is actually the model fit to only 30% of the training data.


```python
final_test_x = tfidf.transform(data_test_clean['Text'])
```


```python
train_x_final = tfidf.fit_transform((data_train_clean['Text']))
y_train_final = data_train_clean['Category'].values

# lg_model = LogisticRegression(C=1, class_weight='balanced', solver='sag', random_state=1, max_iter=500)
# lg_model.fit(train_x_final, y_train_final)
final_test_pred = lg_model.predict(final_test_x)
```


```python
final_pred_df = data_test_clean['ArticleId'].copy()
final_pred_df = pd.DataFrame(final_pred_df)
final_pred_df['Category'] = final_test_pred
final_pred_df
```




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
      <th>ArticleId</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1018</td>
      <td>sport</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1319</td>
      <td>tech</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1138</td>
      <td>sport</td>
    </tr>
    <tr>
      <th>3</th>
      <td>459</td>
      <td>business</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1020</td>
      <td>sport</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>730</th>
      <td>1923</td>
      <td>business</td>
    </tr>
    <tr>
      <th>731</th>
      <td>373</td>
      <td>entertainment</td>
    </tr>
    <tr>
      <th>732</th>
      <td>1704</td>
      <td>business</td>
    </tr>
    <tr>
      <th>733</th>
      <td>206</td>
      <td>business</td>
    </tr>
    <tr>
      <th>734</th>
      <td>471</td>
      <td>politics</td>
    </tr>
  </tbody>
</table>
<p>735 rows × 2 columns</p>
</div>




```python
final_pred_df.to_csv('submission.csv', index=False)
```

The submitted predictions had a score of 0.954, which is good, but not excellent. 

![]({{site.url}}/ms_projects/dtsa_5510/kaggle_screenshot.png)

# Reference List
* [https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/](https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/)
* [https://datagy.io/python-remove-punctuation-from-string/](https://datagy.io/python-remove-punctuation-from-string/)
* [https://spacy.io/usage/spacy-101#features](https://spacy.io/usage/spacy-101#features)
* [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
* [https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py)
* [https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)
