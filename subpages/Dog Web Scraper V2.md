---
layout: page
title: Dog Web Scraper V2
permalink: /python/doggos_v2
---
## Purpose

To add onto the previously build dog web scraper.


```python
import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
```

## Web Scraper & Initial Data Table


```python
url = "https://fetchwi.org/adopt"
results = requests.get(url)
soup = BeautifulSoup(results.text, "html.parser")

names = []
tags = []
pagelinks = []

name_div = soup.find_all('div', class_="summary-content sqs-gallery-meta-container")

for container in name_div:
    name = container.find('a', class_="summary-title-link").text
    names.append(name)
    
    href = container.find('a', class_="summary-title-link")['href']
    pagelinks.append(href)
    
    div1 = container('div', class_="summary-metadata-container summary-metadata-container--below-content")
    
    for span in div1:
        tag = span('div', class_="summary-metadata summary-metadata--primary")
        
        for tag1 in tag:
            tag2 = tag1.text
            tags.append(tag2)

## Additional date column to potentially track over-time changes
date = datetime.datetime.now()

dates = [date] * len(tags)

doggos = pd.DataFrame({
    'name': names,
    'tags':tags,
    'link':pagelinks,
    'date': dates
    
})

doggos = doggos.replace(to_replace=r"\n", value="", regex=True)
doggos["link"] = doggos["link"].replace(to_replace="/doggos/", value="", regex=True)
doggos = doggos.astype({
    'name': "string",
    'tags': "string",
    'link': "string",
    'date': "object"
})
```

### Selecting Dogs that Fit my Lifestyle


```python
potential_doggos = doggos.loc[doggos["tags"].str.contains("Could live in an apartment")]

potential_doggos
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
      <th>name</th>
      <th>tags</th>
      <th>link</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Sugar</td>
      <td>Housebroken, Good in the car, Can free roam wh...</td>
      <td>sugar3</td>
      <td>2022-02-06 16:29:22.521470</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Pride</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>pride</td>
      <td>2022-02-06 16:29:22.521470</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Myla</td>
      <td>Crate trained, Good for beginner dog owner, No...</td>
      <td>myla</td>
      <td>2022-02-06 16:29:22.521470</td>
    </tr>
  </tbody>
</table>
</div>



## Second Scraper

This is a new scraper that will scrape from the specific dog page (found in the "link" column) using a for loop and appending the specific "doggo" weblink from the base URL.


```python
dog_url = []
dog_url = potential_doggos['link'].tolist()

names2 = []
dog_infos = []
tags2 = []
description = []

url = "http://fetchwi.org/doggos/"

for dogs in dog_url: 
    results = requests.get(url + dogs)
    soup = BeautifulSoup(results.text, "html.parser")

    name = soup.find('h1', class_="entry-title entry-title--large p-name").text
    names2.append(name)

    dog_info = soup.find('div', class_="blog-item-content e-content").h4.text
    dog_infos.append(dog_info)

    tag = soup.find_all('p', class_="")[0].text
    tags2.append(tag)

    desc = soup.find_all('p', class_="")[1].text
    description.append(desc)
```

### Length Checking


```python
print("names:",len(names2))
print("dog_info:",len(dog_infos))
print("tags:",len(tags2))
print("descrpt:",len(description))
```

```python
    names: 3
    dog_info: 3
    tags: 3
    descrpt: 3
```   

### Potential Dog DataTable


```python
potential_doggos_ext = pd.DataFrame({
    'name': names2,
    'dog_info': dog_infos,
    'tags': tags2,
    'description': description
})

new = potential_doggos_ext["dog_info"].str.split("|", n=3, expand=True)

potential_doggos_ext["Breed"] = new[0]
potential_doggos_ext["Sex"] = new[1]
potential_doggos_ext["Age"] = new[2]
potential_doggos_ext["Weight"] = new[3]

potential_doggos_ext = potential_doggos_ext.astype("string", errors="ignore")

potential_doggos_ext
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
      <th>name</th>
      <th>dog_info</th>
      <th>tags</th>
      <th>description</th>
      <th>Breed</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sugar</td>
      <td>English Bulldog | Female | 2 Years Old | 43 Lbs</td>
      <td>QUICK FACTS:  ✔️ Housebroken!   ✔️ Good in car...</td>
      <td>Sugar gets up around 6:30/7am in the morning f...</td>
      <td>English Bulldog</td>
      <td>Female</td>
      <td>2 Years Old</td>
      <td>43 Lbs</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pride</td>
      <td>pointer/terrier Mix | Female | 11 months Old |...</td>
      <td>QUICK FACTS:  ✔️ Good with other dogs!   ✔️ Cr...</td>
      <td>PUPDATE 4</td>
      <td>pointer/terrier Mix</td>
      <td>Female</td>
      <td>11 months Old</td>
      <td>46 Lbs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Myla</td>
      <td>german shepherd Mix | Female | 1 year old | 48...</td>
      <td>QUICK FACTS:  ✔️ Housebroken!   ✔️ Good for be...</td>
      <td>Myla is an energetic and loving German Shepher...</td>
      <td>german shepherd Mix</td>
      <td>Female</td>
      <td>1 year old</td>
      <td>48 Lbs</td>
    </tr>
  </tbody>
</table>
</div>



## Final Thoughts

This was a great way to revisit my original scraper. I learned a lot about what html containers are displayed on the "Inspect" tool of the webpage vs. what actually is scraped. It is also unfortunate that certain parts of the page were not structured to fit what the web scraper is scraping. Most of the "Updates" section is in a "p" container with a null class, which is shared by other, non-update related page elements. For that reason, I chose to only scrape the first paragraph. While many dogs will have more than one paragraph of updates, it was better to grab just the first paragraph as all dogs will have this filled out. Similarly, for dogs with multiple paragraphs in their "Updates" section, displaying all paragraphs in the data table may not be the best formatting wise. If I were to want to grab all paragraphs in the future, it wouldn't be too much more work to add another 'for' loop, string concatenate the results above the 0th container, and filter out the unrelated results.
