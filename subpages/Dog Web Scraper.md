---
layout: page
title: Dog Web Scraper
permalink: /python/doggos
---
## Purpose

To build and test a small-scale web scraper in python, using requests, Beautiful Soup to scrape html websites, as well as pandas and numpy to build and edit the created dataframe. This was particularly helpful to get familiar with reading html websites and finding the correct way to grab the data elements to report on.


```python
import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
```

## Web Scraper


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
```

    names: 54
    tags: 54
    href: 54
    

### Length Checking


```python
print("names:",len(names))
print("tags:",len(tags))
print("href:",len(pagelinks))
```

## Creating Doggo Datatable


```python
doggos = pd.DataFrame({
    'name': names,
    'tags':tags,
    'link':pagelinks,
    'date': dates
    
})

doggos = doggos.replace(to_replace=r"\n", value="", regex=True)
doggos["link"] = doggos["link"].replace(to_replace=r"^/", value="fetch.org/", regex=True)
doggos = doggos.astype({
    'name': "string",
    'tags': "string",
    'link': "string",
    'date': "object"
})

doggos
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
      <th>0</th>
      <td>Scooby</td>
      <td>Crate trained, Housebroken, Good in the car, C...</td>
      <td>fetch.org/doggos/scooby</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Clasby</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/clasby</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blue</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/blue2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Myah</td>
      <td>Good with dogs, Crate trained, Good in the car...</td>
      <td>fetch.org/doggos/myah</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Betty</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/betty</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lyla</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/lyla</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jasper</td>
      <td>Good with dogs, Crate trained, Good in the car...</td>
      <td>fetch.org/doggos/jasper</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jolie</td>
      <td>Housebroken, Crate trained, Good for beginner ...</td>
      <td>fetch.org/doggos/jolie2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Elliot</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/elliot2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Shiloh</td>
      <td>Good with dogs, Crate trained, Housebroken, Ca...</td>
      <td>fetch.org/doggos/shiloh2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Apollo</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/apollo2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sugar</td>
      <td>Housebroken, Good in the car, Can free roam wh...</td>
      <td>fetch.org/doggos/sugar3</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Marley</td>
      <td>Good with dogs, Good with cats, Crate trained,...</td>
      <td>fetch.org/doggos/marley</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ruby</td>
      <td>Good with dogs, Crate trained, Good for beginn...</td>
      <td>fetch.org/doggos/ruby</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ripley</td>
      <td>Crate trained, Housebroken, Good in the car, W...</td>
      <td>fetch.org/doggos/ripley</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Beau</td>
      <td>Good with dogs, Good in the car, Working on po...</td>
      <td>fetch.org/doggos/beau</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sonny</td>
      <td>Good with dogs, Good with older kids, Housebro...</td>
      <td>fetch.org/doggos/sonny</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Pound Cake</td>
      <td>Good with dogs, Good with cats, Good in the ca...</td>
      <td>fetch.org/doggos/pound-cake</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cheesecake</td>
      <td>Good with dogs, Good with cats, Good with kids...</td>
      <td>fetch.org/doggos/cheesecake</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Sox</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/sox</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Cajun</td>
      <td>Housebroken, Good in the car, Good running bud...</td>
      <td>fetch.org/doggos/cajun2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Dexter</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/dexter</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Herky</td>
      <td>Housebroken, Crate trained, Good in the car, W...</td>
      <td>fetch.org/doggos/herky</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Kevin</td>
      <td>Good with dogs, Crate trained, Good for beginn...</td>
      <td>fetch.org/doggos/kevin</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Jrue</td>
      <td>Good with dogs, Good with cats, Crate trained,...</td>
      <td>fetch.org/doggos/jrue</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Jackie</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/jackie</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Pride</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/pride</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Jordan</td>
      <td>Good with dogs, Good in the car, Enjoys doggy ...</td>
      <td>fetch.org/doggos/jordan</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Stitch</td>
      <td>Good with dogs, Crate trained, Housebroken, Ca...</td>
      <td>fetch.org/doggos/stitch</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Beck</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/beck</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Lucky</td>
      <td>Good with dogs after slow intros, Good with ca...</td>
      <td>fetch.org/doggos/lucky</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Birdi</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/birdi</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Major</td>
      <td>Housebroken, Can free roam when alone, Good in...</td>
      <td>fetch.org/doggos/major</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Marvin</td>
      <td>Good with kids, Good with cats, Housebroken, C...</td>
      <td>fetch.org/doggos/marvin</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Tova</td>
      <td>Good with dogs, Housebroken, Good in the car, ...</td>
      <td>fetch.org/doggos/tova</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Kuma</td>
      <td>Crate trained, Housebroken, Walks well on leas...</td>
      <td>fetch.org/doggos/kuma</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Zoey</td>
      <td>Needs slow intros to humans, Crate trained, Wa...</td>
      <td>fetch.org/doggos/zoey-ditc</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Simone</td>
      <td></td>
      <td>fetch.org/doggos/simone</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Twinkle</td>
      <td></td>
      <td>fetch.org/doggos/twinkle</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Hella</td>
      <td></td>
      <td>fetch.org/doggos/hella</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Black Forest Cake</td>
      <td></td>
      <td>fetch.org/doggos/black-forest-cake</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Alice</td>
      <td></td>
      <td>fetch.org/doggos/alice</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Amelie</td>
      <td></td>
      <td>fetch.org/doggos/amelie</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Trail</td>
      <td></td>
      <td>fetch.org/doggos/trail</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Arby</td>
      <td></td>
      <td>fetch.org/doggos/arby</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Dotty</td>
      <td></td>
      <td>fetch.org/doggos/dotty</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Twerp</td>
      <td></td>
      <td>fetch.org/doggos/twerp</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Bundt Cake</td>
      <td></td>
      <td>fetch.org/doggos/bundt-cake</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Luciano</td>
      <td></td>
      <td>fetch.org/doggos/luciano</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Luna</td>
      <td></td>
      <td>fetch.org/doggos/luna2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Roscoe</td>
      <td></td>
      <td>fetch.org/doggos/roscoe2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Spock</td>
      <td></td>
      <td>fetch.org/doggos/spock</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>52</th>
      <td>August</td>
      <td></td>
      <td>fetch.org/doggos/august</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Cupcake</td>
      <td></td>
      <td>fetch.org/doggos/cupcake</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>8</th>
      <td>Elliot</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/elliot2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Shiloh</td>
      <td>Good with dogs, Crate trained, Housebroken, Ca...</td>
      <td>fetch.org/doggos/shiloh2</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sugar</td>
      <td>Housebroken, Good in the car, Can free roam wh...</td>
      <td>fetch.org/doggos/sugar3</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Pride</td>
      <td>Good with dogs, Crate trained, Housebroken, Go...</td>
      <td>fetch.org/doggos/pride</td>
      <td>2022-01-23 16:10:22.349954</td>
    </tr>
  </tbody>
</table>
</div>



## Final Thoughts

This works fairly well! I had a bit of issues grabbing the correct tag items, but overall it works well. Potential next steps would be to find a way to set up a reoccurring email to myself with this information, or setting up a scraper that could handle the links from the filtered table above.
