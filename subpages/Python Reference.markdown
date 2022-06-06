---
layout: page
title: Python Quick Reference Guide
permalink: /python/python_guide
---
.color-properties {
  color: white;
  background-color: #2ecc71;
}

## Purpose

I'm putting together a list of helpful code snippets and links for using python for data analysis and data science. I'd like to think I'm not horrible at python, but I seem to keep googling the same documentation over and over.

# Table of Contents


- [Import Header](#import-header)
- [Reading Data](#reading-data)
    - [CSV](#csv)
    - [Copying Data](#copying-data)
- [Viewing Data](#viewing-data)
    - [View Table](#view-table)
    - [View Table Attributes](#view-table-attributes)
    - [View Specific Column](#view-specific-column)
    - [View Specific Row](#view-specific-row)
    - [View Specific Cell](#view-specific-cell)
    - [View Range of Rows/Columns](#view-range-of-rows/columns)
- [Formatting DataFrame](#formatting-dataframe)
    - [Axis Names](#axis-names)
    - [Set and Reset Index](#set-and-reset-index)
    - [Change Data Type](#change-data-type)
    - [Drop Columns](#drop-columns)
    - [Iterate Over Column in New Column](#iterate-over-column-in-new-column)
- [Dataframe SQL Like Functions](#dataframe-sql-like-functions)
    - [Filtering](#filtering)
    - [Aggregating](#aggregating)
    - [Merging](#merging)
    - [Joining](#joining)
- [Extracting](#extracting)
    - [Column to List](#column-to-list)
    - [Row to List](#row-to-list)
- [Helpful Functions](#helpful-functions)
    - [Strings](#strings)
    - [Vectors](#vectors)
    - [Math](#math)
    - [Dates and Times](#dates-and-times)
    - [Plots and Figures](#plots-and-figures)
    - [Test, Train, Evaluate](#test,-train,-evaluate)
- [Other Resources](#other-resources)

## Import Header


```python
# General
## Dates and Times
import datetime
import pytz
from dateutil import parser
from datetime import timezone
## String manipulation
import string
import re #regex
import json #json parsing
## Data Generation
import random

# Data Manipulation
import pandas as pd
import numpy as np
## Vector Math
from numpy.linalg import norm
## Math
import math

# Data Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly as pl
import plotly.express as px

# Data Science
import scipy as sp
import scipy.stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
## Linear Regression
from sklearn.linear_model import LinearRegression
## ARMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
## Nearest Neighbor
from sklearn.neighbors import NearestNeighbors
#import annoy #Aprox Nearest Neighbors
## NPL
#import spacy
#nlp = spacy.load('en_core_web_md')
#doc = nlp(open("txtfile.txt").read()) #Importing NPL txt file

# Scraper
import requests
from requests import get
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
```

## Reading Data

### CSV

**read_table** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.read_table.html)

**read_csv** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

**read_""**(filepath, sep, delimiter, header, names, index_col)


```python
df1 = pd.read_table('test.csv', sep=',')
df2 = pd.read_csv('test.csv')

print(df1)
```

```
      col1 col2 col3 col4     col5
    0    1    2    3    z    10.00
    1    a    b    c    Z    12.50
    2    A    B    C    z  1282.38
```  

### Copying Data


```python
df1.copy()
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
    </tr>
  </tbody>
</table>
</div>



## Viewing Data

### View Table


```python
# Print Formatting
print(df1)
```

      col1 col2 col3 col4     col5
    0    1    2    3    z    10.00
    1    a    b    c    Z    12.50
    2    A    B    C    z  1282.38
    


```python
# Top Rows
df1.head()
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
    </tr>
  </tbody>
</table>
</div>



### View Table Attributes


```python
# Column Names, Data Types
df1.columns
```




    Index(['col1', 'col2', 'col3', 'col4', 'col5'], dtype='object')




```python
# Specific Data Types
df1.dtypes
```




    col1     object
    col2     object
    col3     object
    col4     object
    col5    float64
    dtype: object




```python
# Row Count
df1.shape[0]
len(df1.index)
```




    3



### View Specific Column


```python
df1['col1']
```




    0    1
    1    a
    2    A
    Name: col1, dtype: object



### View Specific Row


```python
# Row by Index
df1.loc[[1]]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Row by Value
df1.loc[df1['col1'] == '1']
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



### View Specific Cell


```python
# Named Column by Row Index
df1['col1'].values[0]
```




    '1'



### View Range of Rows/Columns


```python
# Range of Rows
df1[0:2]
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>new3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.0</td>
      <td>1-2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.5</td>
      <td>a-b</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Range of Rows and Columns
df1[0:1][['col1','col2']]
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
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Formatting DataFrame

### Axis Names

**set_axis** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_axis.html)

**rename** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)


```python
# All column/row names
df3 = df1.copy()
df3.set_axis(['col_a', 'col_b', 'col_c', 'col_d', 'col_e'], axis=1, inplace=True) #axis=0 is x axis
print(df3.columns)
```

    Index(['col_a', 'col_b', 'col_c', 'col_d', 'col_e'], dtype='object')
    


```python
# Select column/row names
df4 = df1.rename(columns = {'col1':'col_a', 'col2':'col_b'})
print(df4.columns)
```

    Index(['col_a', 'col_b', 'col3', 'col4', 'col5'], dtype='object')
    

### Set and Reset Index

**set_index** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html)

**reset_index** - [Pandas Documentation Link](https://www.delftstack.com/howto/python-pandas/pandas-remove-index/)


```python
df3 = df3.set_index('col_a')
df3
```


```python
df3.reset_index()
df3
```

### Change Data Type

**astype** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html?highlight=astype#pandas.DataFrame.astype)


```python
# On Dataframe
df1.astype({
    'col4': 'string',
    'col5': 'int',
    'col1': 'int'
}, errors='ignore')
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282</td>
    </tr>
  </tbody>
</table>
</div>




```python
# On Column

df1['col4'].astype('string')
```




    0    z
    1    Z
    2    z
    Name: col4, dtype: string



### Drop Columns


```python
df4 = df4.drop(columns = ['col_a'])
df4
```


```python
df4 = df1.rename(columns = {'col1':'col_a'})
df5 = df4[['col3', 'col2']].copy()
df5
```

### Iterate Over Column in New Column


```python
# Example: Multiplication
df5 = df1.copy()
df5['new1'] = df5['col4'] * 10
df5['new2'] = df5['col5'] * 10
df5
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>new3</th>
      <th>new1</th>
      <th>new2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.00</td>
      <td>1-2</td>
      <td>zzzzzzzzzz</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
      <td>a-b</td>
      <td>ZZZZZZZZZZ</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
      <td>A-B</td>
      <td>zzzzzzzzzz</td>
      <td>12823.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example: Concatinate
df5['new3'] = df5['col1'] + '-' + df5['col2']
df5
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>new3</th>
      <th>new1</th>
      <th>new2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.00</td>
      <td>1-2</td>
      <td>zzzzzzzzzz</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
      <td>a-b</td>
      <td>ZZZZZZZZZZ</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
      <td>A-B</td>
      <td>zzzzzzzzzz</td>
      <td>12823.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example: Capitalizing String
str1 = df5['new1'].values[0]
print(str1.upper())
```

    ZZZZZZZZZZ
    


```python
# Example: Trying to Capitalize Column (does not work)
df5['cap'] = df5['new1'].upper()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [52], in <cell line: 2>()
          1 # Example: Trying to Capitalize Column
    ----> 2 df5['cap'] = df5['new1'].upper()
    

    File ~/opt/anaconda3/lib/python3.9/site-packages/pandas/core/generic.py:5575, in NDFrame.__getattr__(self, name)
       5568 if (
       5569     name not in self._internal_names_set
       5570     and name not in self._metadata
       5571     and name not in self._accessors
       5572     and self._info_axis._can_hold_identifiers_and_holds_name(name)
       5573 ):
       5574     return self[name]
    -> 5575 return object.__getattribute__(self, name)
    

    AttributeError: 'Series' object has no attribute 'upper'



```python
# Lambda Function
df5['cap'] = df5['new1'].apply(lambda x: x.upper())
df5
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>new3</th>
      <th>new1</th>
      <th>new2</th>
      <th>cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>z</td>
      <td>10.00</td>
      <td>1-2</td>
      <td>zzzzzzzzzz</td>
      <td>100.0</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
      <td>a-b</td>
      <td>ZZZZZZZZZZ</td>
      <td>125.0</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
      <td>A-B</td>
      <td>zzzzzzzzzz</td>
      <td>12823.8</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
  </tbody>
</table>
</div>



## Dataframe SQL Like Functions

### Filtering


```python
# Example: Filter by Column
df6 = df5.loc[df5['col5'] >= 12.5]
df6
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>new3</th>
      <th>new1</th>
      <th>new2</th>
      <th>cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
      <td>a-b</td>
      <td>ZZZZZZZZZZ</td>
      <td>125.0</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
      <td>A-B</td>
      <td>zzzzzzzzzz</td>
      <td>12823.8</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Example: Filter by Column
df6 = df5.loc[df5['new3'].str.contains(pat = r'a|B')]
df6
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>new3</th>
      <th>new1</th>
      <th>new2</th>
      <th>cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>Z</td>
      <td>12.50</td>
      <td>a-b</td>
      <td>ZZZZZZZZZZ</td>
      <td>125.0</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>B</td>
      <td>C</td>
      <td>z</td>
      <td>1282.38</td>
      <td>A-B</td>
      <td>zzzzzzzzzz</td>
      <td>12823.8</td>
      <td>ZZZZZZZZZZ</td>
    </tr>
  </tbody>
</table>
</div>



### Aggregating
[Pandas Documentation Link](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html)

**groupby** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html)

**grouper** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html), helpful for creating time series using "freq" param

**Usable Functions** - mean, median, sum


```python
# Setup
d = {'col1': range(0,42,2), 'col2': range(10,220,10), 'col3': ('t','t','t','t','t','t','t','t','t','t','f','f','f','f','f','f','f','f','f','f','f')}
df2 = pd.DataFrame(data=d)
df2
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
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10</td>
      <td>t</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>t</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>30</td>
      <td>t</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>40</td>
      <td>t</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>50</td>
      <td>t</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>60</td>
      <td>t</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12</td>
      <td>70</td>
      <td>t</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14</td>
      <td>80</td>
      <td>t</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>90</td>
      <td>t</td>
    </tr>
    <tr>
      <th>9</th>
      <td>18</td>
      <td>100</td>
      <td>t</td>
    </tr>
    <tr>
      <th>10</th>
      <td>20</td>
      <td>110</td>
      <td>f</td>
    </tr>
    <tr>
      <th>11</th>
      <td>22</td>
      <td>120</td>
      <td>f</td>
    </tr>
    <tr>
      <th>12</th>
      <td>24</td>
      <td>130</td>
      <td>f</td>
    </tr>
    <tr>
      <th>13</th>
      <td>26</td>
      <td>140</td>
      <td>f</td>
    </tr>
    <tr>
      <th>14</th>
      <td>28</td>
      <td>150</td>
      <td>f</td>
    </tr>
    <tr>
      <th>15</th>
      <td>30</td>
      <td>160</td>
      <td>f</td>
    </tr>
    <tr>
      <th>16</th>
      <td>32</td>
      <td>170</td>
      <td>f</td>
    </tr>
    <tr>
      <th>17</th>
      <td>34</td>
      <td>180</td>
      <td>f</td>
    </tr>
    <tr>
      <th>18</th>
      <td>36</td>
      <td>190</td>
      <td>f</td>
    </tr>
    <tr>
      <th>19</th>
      <td>38</td>
      <td>200</td>
      <td>f</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40</td>
      <td>210</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Aggregate Mean
df3 = df2.groupby(['col3'], axis=0).mean()
df3
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
      <th>col1</th>
      <th>col2</th>
    </tr>
    <tr>
      <th>col3</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>f</th>
      <td>30.0</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>t</th>
      <td>9.0</td>
      <td>55.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Aggregate Sum (without NaN values)
df3 = df2.groupby(['col3'], axis=0).sum().dropna()
df3
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
      <th>col1</th>
      <th>col2</th>
    </tr>
    <tr>
      <th>col3</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>f</th>
      <td>330</td>
      <td>1760</td>
    </tr>
    <tr>
      <th>t</th>
      <td>90</td>
      <td>550</td>
    </tr>
  </tbody>
</table>
</div>



### Merging
**merge** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)

### Joining
**join** - [Pandas Documentation Link](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html)

## Extracting

### Column to List


```python
# Extract First Column (two ways)
ary1 = df5['col1'].to_numpy()
ary2 = np.array(df5['col1'])
ary1, ary2
```




    (array(['1', 'a', 'A'], dtype=object), array(['1', 'a', 'A'], dtype=object))




```python
# Extract All Columns
ary1 = np.array(df5.loc[0:, :])
ary1
```




    array([['1', '2', '3', 'z', 10.0, '1-2', 'zzzzzzzzzz', 100.0,
            'ZZZZZZZZZZ'],
           ['a', 'b', 'c', 'Z', 12.5, 'a-b', 'ZZZZZZZZZZ', 125.0,
            'ZZZZZZZZZZ'],
           ['A', 'B', 'C', 'z', 1282.38, 'A-B', 'zzzzzzzzzz',
            12823.800000000001, 'ZZZZZZZZZZ']], dtype=object)



### Row to List


```python
# Extract First Row
ary1 = np.array(df5[0:1])
ary1
```


```python
# Extract All Rows
ary1 = np.array(df5[0:])
ary1
```




    array([['1', '2', '3', 'z', 10.0, '1-2', 'zzzzzzzzzz', 100.0,
            'ZZZZZZZZZZ'],
           ['a', 'b', 'c', 'Z', 12.5, 'a-b', 'ZZZZZZZZZZ', 125.0,
            'ZZZZZZZZZZ'],
           ['A', 'B', 'C', 'z', 1282.38, 'A-B', 'zzzzzzzzzz',
            12823.800000000001, 'ZZZZZZZZZZ']], dtype=object)



## Helpful Functions

### Strings

**regex** - [Python Documentation Link](https://docs.python.org/3/library/re.html)


```python
# String Replacement
str1 = 'Hey Look A Dog'
str2 = str1.replace('Dog', 'Cat')
print(str2)
```

    Hey Look A Cat
    


```python
# String Contains item
str1 = 'Hey Look A Dog'
str2 = 'Hey Look A Cat'
str3 = 'Dogs Are Cool'
str4 = 'Cats Are Cool'

print('Dog' in str1,',', 'Dog' in str2,',', 'dog' in str3,',', 'dog' in str4)
```

    True , False , False , False
    


```python
# String Contains df
d = {'col1': [1, 2, 3, 4], 'col2': [str1, str2, str3, str4]}
df2 = pd.DataFrame(data=d)
df2['tf'] = (df2['col2'].str.contains(pat = 'Dog'))
df2
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
      <th>col1</th>
      <th>col2</th>
      <th>tf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Hey Look A Dog</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Hey Look A Cat</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Dogs Are Cool</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Cats Are Cool</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Regex
print(re.search(r'^D', str1),',', re.search(r'^D', str3),','
      , re.findall(r'^D', str1),',', re.findall(r'^D', str3))
```

    None , <re.Match object; span=(0, 1), match='D'> , [] , ['D']
    

### Vectors


```python
# Vector Manipulation, Addinv Vectors to Vectors
v1 = np.array(['#1a2b3c', '#abcdef', '#000001'])
v2 = []
v3 = []

for i in v1:
    y = i.lstrip('#')
    v2.append(y)
    
    z = np.array([int(y[:2],16), int(y[2:4],16), int(y[4:6],16)])
    v3.append(z)

v2,v3
```




    (['1a2b3c', 'abcdef', '000001'],
     [array([26, 43, 60]), array([171, 205, 239]), array([0, 0, 1])])



### Math

**Math** - [Python Documentation Link](https://docs.python.org/3/library/math.html)


```python
# Numpy Vector Math
v1 = np.array([1,2,3])
v2 = np.array([2,3,4])

print(v1 * v2)
print(v2 / v1)
```

    [ 2  6 12]
    [2.         1.5        1.33333333]
    


```python
# Math functions
print(math.sqrt(v1[0]))

print(math.factorial(2))

print(math.fmod(19, 5))
```

    1.0
    2
    4.0
    

### Dates and Times

In progress

### Plots and Figures

### Test, Train, Evaluate
In progress

# Other Resources

[Nearest Neighbors with Sentences](https://github.com/aparrish/rwet/blob/master/understanding-word-vectors.ipynb)