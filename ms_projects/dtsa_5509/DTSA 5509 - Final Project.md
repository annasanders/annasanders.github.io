---
layout: page
title: DTSA 5509 - Natural Gas Spot Price
permalink: /ms_projects/dtsa5509_spotprice
---
Natural gas is one of the largest sources of energy production in the United States. Natural gas, like other fossil fuels, comes from dead plant and animal remains from millions of years ago, that after being buried under sand, silt, and rock, transform from heat and pressure into oil and natural gas. Natural gas from deposits are processed and then sold to the market and consumers. In addition to being sold to consumers (including residential, commercial, and industrial use), natural gas can also be used to generate electricity. Natural gas is also used to heat buildings and power gas stoves and other appliances.

The dataset for natural gas spot prices comes from the US Energy Information Administration, and details the Henry Hub spot price weekly since 1997. The Henry Hub spot price is the spot price at close from the Henry distribution hub in Erath, Louisiana. The data was extracted on a weekly rate, taken on the Friday week date.

In the past few years, the price of natural gas has risen dramatically, much of which was centered around 2022. As natural gas is such a widely used fuel source, price forecasting is helpful to predict potential future trends and plan mitigation strategies, if we assume demand is correlated to the spot price. 

This project will try to predict future natural gas spot prices using supervised machine learning algorithms. There are three main concerns with this prediction: 
1. This data is a time series data, so various trend data should be accounted for when modeling future price.
2. This data is not adjusted for inflation. We will need to adjust to ensure accurate predictions.
3. How do you predict values beyond one week?

We will first deal with inflation by calculating the inflation rate based on the current Consumer Price Index (CPI) for Energy. We can then multiply this number by the spot price to get inflation adjusted prices.

We will then split the data into a training and test set by withholding the last N weeks of data from the modeling process. We will then predict the withheld 'future' prices and calculate modeling metrics. We will repeat this process for several models and several hyperparameter combinations.

We will then re-train the model on the full dataset and compare predictions to the actual values from true future weeks. As the data was downloaded on November 17th, the future data will include the closing date of 2023-11-17 and any future weeks until this project is completed.

**Resources:**
- [https://www.eia.gov/energyexplained/natural-gas/](https://www.eia.gov/energyexplained/natural-gas/)
- [https://en.wikipedia.org/wiki/Natural_gas_in_the_United_States#Natural_gas_electricity_generation](https://en.wikipedia.org/wiki/Natural_gas_in_the_United_States#Natural_gas_electricity_generation)
- [https://www.usinflationcalculator.com/inflation/historical-inflation-rates/](https://www.usinflationcalculator.com/inflation/historical-inflation-rates/)

**Data Citation:**
Data from the EIA located [here](https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm), which is the public government website.

```
U.S. Energy Information Administration. (n.d.).  Natural Gas Spot and Futures Prices (NYMEX) . Natural Gas Futures Prices (Nymex). https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm 
```

## Import Libraries 


```python
# Libraries
import pandas as pd
import numpy as np

## Dates and Times
import pytz
from dateutil import parser
from datetime import timezone,datetime,timedelta
from dateutil.relativedelta import relativedelta
from datetime import date 

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

# Sklearn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Models

# Metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
```

## Loading and Cleaning Data


```python
natural_gas = pd.read_excel('natural_gas_us_price.xls', sheet_name='Data 1')
natural_gas = natural_gas.drop(index={0,1})
natural_gas = natural_gas.rename(columns = {'Back to Contents':'date', 'Data 1: Weekly Henry Hub Natural Gas Spot Price (Dollars per Million Btu)':'spot_price'})
natural_gas['date'] = pd.to_datetime(natural_gas['date'])
natural_gas = natural_gas.reset_index().drop(columns={'index'})
natural_gas
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
      <th>date</th>
      <th>spot_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-01-10</td>
      <td>3.79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-01-17</td>
      <td>4.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-01-24</td>
      <td>2.98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-01-31</td>
      <td>2.91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-02-07</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>2023-10-13</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>2023-10-20</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>2023-10-27</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>2023-11-03</td>
      <td>3.17</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>2023-11-10</td>
      <td>2.46</td>
    </tr>
  </tbody>
</table>
<p>1400 rows × 2 columns</p>
</div>



We can see from the preview that this dataset contains 1,400 rows, each corresponding to the Friday week date with one single spot price listed.

We can also import a secondary data source from the extracted energy specific Consumer Price Index from the Bureau of Labor Statistics ([link](https://www.bls.gov/cpi/). 

***References:***
* [https://www.officialdata.org/](https://www.officialdata.org/)

***Data Citation:***
Bureau of Labor Statistics ([link](https://www.bls.gov/cpi/)).

```
U.S. Bureau of Labor Statistics. (n.d.). CPI Home.  Consumer Price Index Search Consumer Price Index. https://www.bls.gov/cpi/ 
```


```python
energy_cpi = pd.read_excel('energy_cpi.xlsx')
energy_cpi = energy_cpi.drop(index={0,1,2,3,4,5,6,7,8,9,10}, columns={'Unnamed: 13', 'Unnamed: 14'})
energy_cpi = energy_cpi.rename(columns={'CPI for All Urban Consumers (CPI-U)':'year', 'Unnamed: 1':'01', 'Unnamed: 2':'02',
       'Unnamed: 3':'03', 'Unnamed: 4':'04', 'Unnamed: 5':'05', 'Unnamed: 6':'06', 'Unnamed: 7':'07',
       'Unnamed: 8':'08', 'Unnamed: 9':'09', 'Unnamed: 10':'10', 'Unnamed: 11':'11', 'Unnamed: 12':'12'})
energy_cpi = energy_cpi.reset_index().drop(columns={'index'})
energy_cpi
```

    C:\Users\623an\anaconda3\Lib\site-packages\openpyxl\styles\stylesheet.py:226: UserWarning: Workbook contains no default style, apply openpyxl's default
      warn("Workbook contains no default style, apply openpyxl's default")
    




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
      <th>year</th>
      <th>01</th>
      <th>02</th>
      <th>03</th>
      <th>04</th>
      <th>05</th>
      <th>06</th>
      <th>07</th>
      <th>08</th>
      <th>09</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997</td>
      <td>113.3</td>
      <td>113.1</td>
      <td>111.2</td>
      <td>110</td>
      <td>109.9</td>
      <td>112.3</td>
      <td>111.4</td>
      <td>112.5</td>
      <td>113.9</td>
      <td>111.5</td>
      <td>110.7</td>
      <td>108.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1998</td>
      <td>105.9</td>
      <td>103.2</td>
      <td>101.6</td>
      <td>101.9</td>
      <td>103.8</td>
      <td>105.7</td>
      <td>105.2</td>
      <td>103.8</td>
      <td>102.7</td>
      <td>101.3</td>
      <td>100.5</td>
      <td>98.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1999</td>
      <td>98.1</td>
      <td>97.3</td>
      <td>98.4</td>
      <td>105</td>
      <td>105.6</td>
      <td>106.8</td>
      <td>108.7</td>
      <td>111.3</td>
      <td>113.2</td>
      <td>111.6</td>
      <td>111.2</td>
      <td>112.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>112.5</td>
      <td>116.7</td>
      <td>122.2</td>
      <td>120.7</td>
      <td>121</td>
      <td>129.6</td>
      <td>129.7</td>
      <td>125.9</td>
      <td>130.6</td>
      <td>129.3</td>
      <td>129</td>
      <td>128.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>132.5</td>
      <td>132</td>
      <td>129.5</td>
      <td>133.1</td>
      <td>140.1</td>
      <td>140.5</td>
      <td>132.4</td>
      <td>129.4</td>
      <td>132.5</td>
      <td>122.1</td>
      <td>116</td>
      <td>111.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2002</td>
      <td>111.7</td>
      <td>111</td>
      <td>115.6</td>
      <td>122.2</td>
      <td>122.9</td>
      <td>124.9</td>
      <td>125.5</td>
      <td>125.8</td>
      <td>126.1</td>
      <td>125.8</td>
      <td>125.3</td>
      <td>123.3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2003</td>
      <td>127.5</td>
      <td>135.4</td>
      <td>142.6</td>
      <td>138.1</td>
      <td>134</td>
      <td>136.5</td>
      <td>136.8</td>
      <td>140.6</td>
      <td>144.6</td>
      <td>136.9</td>
      <td>133.1</td>
      <td>131.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2004</td>
      <td>137.4</td>
      <td>140.6</td>
      <td>143.1</td>
      <td>145.9</td>
      <td>154.1</td>
      <td>159.7</td>
      <td>156.3</td>
      <td>155.3</td>
      <td>154.3</td>
      <td>157.7</td>
      <td>158.6</td>
      <td>153.7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2005</td>
      <td>151.9</td>
      <td>155.2</td>
      <td>160.8</td>
      <td>170.9</td>
      <td>169.4</td>
      <td>171.4</td>
      <td>178.5</td>
      <td>186.6</td>
      <td>208</td>
      <td>204.3</td>
      <td>187.6</td>
      <td>180</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2006</td>
      <td>189.5</td>
      <td>186.4</td>
      <td>188.6</td>
      <td>201.4</td>
      <td>209.3</td>
      <td>211.3</td>
      <td>215.1</td>
      <td>214.7</td>
      <td>199.1</td>
      <td>181.3</td>
      <td>180.4</td>
      <td>185.2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2007</td>
      <td>183.567</td>
      <td>184.451</td>
      <td>196.929</td>
      <td>207.265</td>
      <td>219.071</td>
      <td>221.088</td>
      <td>217.274</td>
      <td>209.294</td>
      <td>209.637</td>
      <td>207.588</td>
      <td>219.009</td>
      <td>217.506</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2008</td>
      <td>219.465</td>
      <td>219.311</td>
      <td>230.505</td>
      <td>240.194</td>
      <td>257.106</td>
      <td>275.621</td>
      <td>280.833</td>
      <td>266.283</td>
      <td>258.02</td>
      <td>231.561</td>
      <td>189.938</td>
      <td>171.158</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2009</td>
      <td>174.622</td>
      <td>178.741</td>
      <td>177.454</td>
      <td>179.704</td>
      <td>186.909</td>
      <td>205.408</td>
      <td>201.938</td>
      <td>204.971</td>
      <td>202.243</td>
      <td>199.198</td>
      <td>204.026</td>
      <td>202.301</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2010</td>
      <td>208.026</td>
      <td>204.455</td>
      <td>209.999</td>
      <td>212.977</td>
      <td>214.363</td>
      <td>211.66</td>
      <td>212.372</td>
      <td>212.663</td>
      <td>210.003</td>
      <td>210.947</td>
      <td>211.97</td>
      <td>217.953</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2011</td>
      <td>223.266</td>
      <td>226.86</td>
      <td>242.516</td>
      <td>253.495</td>
      <td>260.376</td>
      <td>254.17</td>
      <td>252.661</td>
      <td>251.706</td>
      <td>250.48</td>
      <td>240.902</td>
      <td>238.177</td>
      <td>232.3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2012</td>
      <td>236.942</td>
      <td>242.663</td>
      <td>253.599</td>
      <td>255.736</td>
      <td>250.306</td>
      <td>244.167</td>
      <td>239.972</td>
      <td>250.306</td>
      <td>256.332</td>
      <td>250.523</td>
      <td>238.946</td>
      <td>233.473</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2013</td>
      <td>234.624</td>
      <td>248.146</td>
      <td>249.565</td>
      <td>244.757</td>
      <td>247.805</td>
      <td>251.921</td>
      <td>251.37</td>
      <td>250.011</td>
      <td>248.513</td>
      <td>238.524</td>
      <td>233.136</td>
      <td>234.542</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2014</td>
      <td>239.551</td>
      <td>242.041</td>
      <td>250.543</td>
      <td>252.717</td>
      <td>255.982</td>
      <td>259.858</td>
      <td>257.907</td>
      <td>250.951</td>
      <td>247.077</td>
      <td>234.745</td>
      <td>221.844</td>
      <td>209.785</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015</td>
      <td>192.619</td>
      <td>196.597</td>
      <td>204.731</td>
      <td>203.715</td>
      <td>214.33</td>
      <td>220.861</td>
      <td>219.852</td>
      <td>213.248</td>
      <td>201.641</td>
      <td>194.501</td>
      <td>189.267</td>
      <td>183.378</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2016</td>
      <td>180.171</td>
      <td>172.061</td>
      <td>179.017</td>
      <td>185.652</td>
      <td>192.673</td>
      <td>200.035</td>
      <td>195.94</td>
      <td>193.524</td>
      <td>195.852</td>
      <td>194.786</td>
      <td>191.402</td>
      <td>193.306</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2017</td>
      <td>199.608</td>
      <td>198.195</td>
      <td>198.597</td>
      <td>202.869</td>
      <td>203.132</td>
      <td>204.646</td>
      <td>202.554</td>
      <td>205.894</td>
      <td>215.711</td>
      <td>207.29</td>
      <td>209.383</td>
      <td>206.598</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018</td>
      <td>210.663</td>
      <td>213.519</td>
      <td>212.554</td>
      <td>218.83</td>
      <td>226.81</td>
      <td>229.137</td>
      <td>227.107</td>
      <td>226.939</td>
      <td>226.165</td>
      <td>225.757</td>
      <td>215.91</td>
      <td>205.905</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2019</td>
      <td>200.563</td>
      <td>202.74</td>
      <td>211.724</td>
      <td>222.499</td>
      <td>225.773</td>
      <td>221.373</td>
      <td>222.492</td>
      <td>216.978</td>
      <td>215.418</td>
      <td>216.351</td>
      <td>214.636</td>
      <td>212.982</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020</td>
      <td>213.043</td>
      <td>208.354</td>
      <td>199.573</td>
      <td>183.081</td>
      <td>183.076</td>
      <td>193.379</td>
      <td>197.665</td>
      <td>197.362</td>
      <td>198.858</td>
      <td>196.458</td>
      <td>194.388</td>
      <td>198.155</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2021</td>
      <td>205.273</td>
      <td>213.277</td>
      <td>225.861</td>
      <td>229.116</td>
      <td>235.339</td>
      <td>240.72</td>
      <td>244.8</td>
      <td>246.639</td>
      <td>248.228</td>
      <td>255.338</td>
      <td>259.1</td>
      <td>256.207</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2022</td>
      <td>260.653</td>
      <td>267.771</td>
      <td>298.246</td>
      <td>298.469</td>
      <td>316.761</td>
      <td>340.917</td>
      <td>325.407</td>
      <td>305.372</td>
      <td>297.343</td>
      <td>300.359</td>
      <td>292.953</td>
      <td>274.937</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2023</td>
      <td>283.33</td>
      <td>281.673</td>
      <td>279.084</td>
      <td>283.352</td>
      <td>279.816</td>
      <td>283.854</td>
      <td>284.828</td>
      <td>294.328</td>
      <td>296.004</td>
      <td>286.754</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



There are 27 rows in this data set, with columns for each month and one denoting the year.

Since the data is available monthly, we will link to the year-month date in the previous table. Below is a function to create a bit more usable table to link back to our original dataset.

We can calculate the inflation rate by by dividing historical CPIs by the most current CPI index (October 2023).


```python
# Calculating CPI
n = len(energy_cpi)
cols = list(energy_cpi.columns)
del(cols[0])

curr_cpi = energy_cpi.iloc[26,10]
print('Current Energy Consumer Price Index:', curr_cpi)

energy_cpi.iloc[26,11] = curr_cpi # Setting November 2023 CPI to October 2023

energy_inf = energy_cpi.copy()
# energy_inf = pd.DataFrame(columns=['year','month','inflation'])

for i in cols:
    energy_inf[i] = energy_inf[i]/curr_cpi
    
energy_inf = energy_inf.melt(id_vars=['year'], var_name='month', value_name='inflation')
energy_inf['month'] = energy_inf['month'].apply(lambda x: x.rjust(2,'0'))
energy_inf['year'] = energy_inf['year'].astype('str')
energy_inf['date'] = pd.to_datetime((energy_inf['year'])+'-'+(energy_inf['month'])+'-01').dt.date
energy_inf
```

    Current Energy Consumer Price Index: 286.754
    




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
      <th>year</th>
      <th>month</th>
      <th>inflation</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997</td>
      <td>01</td>
      <td>0.395112</td>
      <td>1997-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1998</td>
      <td>01</td>
      <td>0.369306</td>
      <td>1998-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1999</td>
      <td>01</td>
      <td>0.342105</td>
      <td>1999-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>01</td>
      <td>0.392322</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>01</td>
      <td>0.462069</td>
      <td>2001-01-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>319</th>
      <td>2019</td>
      <td>12</td>
      <td>0.742734</td>
      <td>2019-12-01</td>
    </tr>
    <tr>
      <th>320</th>
      <td>2020</td>
      <td>12</td>
      <td>0.691028</td>
      <td>2020-12-01</td>
    </tr>
    <tr>
      <th>321</th>
      <td>2021</td>
      <td>12</td>
      <td>0.893473</td>
      <td>2021-12-01</td>
    </tr>
    <tr>
      <th>322</th>
      <td>2022</td>
      <td>12</td>
      <td>0.95879</td>
      <td>2022-12-01</td>
    </tr>
    <tr>
      <th>323</th>
      <td>2023</td>
      <td>12</td>
      <td>NaN</td>
      <td>2023-12-01</td>
    </tr>
  </tbody>
</table>
<p>324 rows × 4 columns</p>
</div>



We can now join the two dataframes and calculate the price in terms of October 2023 inflation by multiplying the inflation column.


```python
# Join
natural_gas_inf = natural_gas.copy()
natural_gas_inf['month'] = natural_gas_inf['date'].apply(lambda x: str(x.month).rjust(2,'0'))
natural_gas_inf['year'] = natural_gas_inf['date'].apply(lambda x: str(x.year))
natural_gas_inf = natural_gas_inf.merge(energy_inf, how='left', on =['month','year'])
natural_gas_inf = natural_gas_inf.drop(columns={'date_y', 'month','year'})
natural_gas_inf = natural_gas_inf.rename(columns={'date_x':'date'})
natural_gas_inf['spot_price_inf'] = natural_gas_inf['spot_price'] * natural_gas_inf['inflation']
natural_gas_inf
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
      <th>date</th>
      <th>spot_price</th>
      <th>inflation</th>
      <th>spot_price_inf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-01-10</td>
      <td>3.79</td>
      <td>0.395112</td>
      <td>1.497475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-01-17</td>
      <td>4.19</td>
      <td>0.395112</td>
      <td>1.65552</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-01-24</td>
      <td>2.98</td>
      <td>0.395112</td>
      <td>1.177434</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-01-31</td>
      <td>2.91</td>
      <td>0.395112</td>
      <td>1.149776</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-02-07</td>
      <td>2.53</td>
      <td>0.394415</td>
      <td>0.997869</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>2023-10-13</td>
      <td>3.2</td>
      <td>1.0</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>2023-10-20</td>
      <td>2.86</td>
      <td>1.0</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>2023-10-27</td>
      <td>2.89</td>
      <td>1.0</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>2023-11-03</td>
      <td>3.17</td>
      <td>1.0</td>
      <td>3.17</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>2023-11-10</td>
      <td>2.46</td>
      <td>1.0</td>
      <td>2.46</td>
    </tr>
  </tbody>
</table>
<p>1400 rows × 4 columns</p>
</div>



Correcting for inflation is important, especially since there were large changes to the value of a US dollar over. As this data goes all the way back to 1997, we can expect for historical spot prices to be overstated, as \\$5 in 1997 would likely have less purchasing power than 2023 (i.e. \\$5 from 1997 would be equivalent to only \\$2 in 2023).

## Data Exploration


```python
print('Data Size: ', len(natural_gas_inf))
```

    Data Size:  1400
    

In this dataset, there are 1,400 rows of data. With the addition of the 'spot_price_inf', or inflation adjusted spot price, there are 2 columns of data. There only feature of the data is the date of the spot price.


```python
fig, axs = plt.subplots()
axs.plot(natural_gas_inf['date'], natural_gas_inf['spot_price'], label='Spot Price')
axs.plot(natural_gas_inf['date'], natural_gas_inf['spot_price_inf'], label='Infation Adjusted Spot Price', color='green')

axs.set_xlabel('Date')
axs.set_ylabel('Spot Price')
axs.set_title('Spot Price Over Time')
axs.legend()
```




    <matplotlib.legend.Legend at 0x1a4709cd2d0>




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_16_1.png)
    


We can see that adjusting for inflation mostly decreases prices, especially for historical data. Adjusting for inflation will hopefully not over-predict spot prices based on higher spot prices than the adjusted spot prices.


```python
mean_sp = natural_gas_inf['spot_price'].mean()
q1_so = natural_gas_inf['spot_price'].quantile(0.25)
median_sp = natural_gas_inf['spot_price'].quantile(0.5)
q3_sp = natural_gas_inf['spot_price'].quantile(0.75)

print('Spot Price')
print('Mean Price: $%.2f' %(mean_sp))
print('Q1 Price: $%.2f' %(q1_so))
print('Median Price: $%.2f' %(median_sp))
print('Q3 Price: $%.2f' %(q3_sp))
print('')

mean_sp = natural_gas_inf['spot_price_inf'].mean()
q1_so = natural_gas_inf['spot_price_inf'].quantile(0.25)
median_sp = natural_gas_inf['spot_price_inf'].quantile(0.5)
q3_sp = natural_gas_inf['spot_price_inf'].quantile(0.75)

print('Inflation Adjusted Spot Price')
print('Mean Price: $%.2f' %(mean_sp))
print('Q1 Price: $%.2f' %(q1_so))
print('Median Price: $%.2f' %(median_sp))
print('Q3 Price: $%.2f' %(q3_sp))
```

    Spot Price
    Mean Price: $4.20
    Q1 Price: $2.64
    Median Price: $3.54
    Q3 Price: $5.27
    
    Inflation Adjusted Spot Price
    Mean Price: $2.91
    Q1 Price: $1.76
    Median Price: $2.42
    Q3 Price: $3.53
    

Looking at the mean and various quantiles of prices, we also can see that the inflation adjusted spot prices are overall less than the non-inflation adjusted prices.


```python
mean_sp = natural_gas_inf['spot_price_inf'].mean()
q1_so = natural_gas_inf['spot_price_inf'].quantile(0.25)
median_sp = natural_gas_inf['spot_price_inf'].quantile(0.5)
q3_sp = natural_gas_inf['spot_price_inf'].quantile(0.75)

max_date = natural_gas_inf['date'].max()
min_date = natural_gas_inf['date'].min()


fig, axs = plt.subplots()
axs.plot(natural_gas_inf['date'], natural_gas_inf['spot_price_inf'], color='green')
axs.hlines(mean_sp, min_date, max_date, color='black', label='mean')
axs.hlines(median_sp, min_date, max_date, color='grey', label='median')
axs.hlines(q1_so, min_date, max_date, color='grey', linestyles='dashed')
axs.hlines(q3_sp, min_date, max_date, color='grey', linestyles='dashed')


axs.set_xlabel('Date')
axs.set_ylabel('Spot Price')
axs.set_title('Inflation Adjusted Spot Price Over Time')
axs.legend()
```




    <matplotlib.legend.Legend at 0x1a470d3dc90>




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_20_1.png)
    


We can plot just the spot price and the mean, median, and quartile values. As we can see from the above plot, spot price fluctuates, but tends to be relatively stable.

## Feature Engineering

As this data has only one feature, we can transform the data to add additional dimensions to aid in prediction. We can first add rolling mean prices in various increments.

It is important to ensure the rolling window does not take into account the current spot price, or else the model would be snooping on 'future' data.


```python
# natural_gas_inf = natural_gas_inf.reset_index(inplace=True)
# natural_gas_inf = natural_gas_inf.set_index('date')
natural_gas_inf = natural_gas_inf.merge(natural_gas_inf[['date','spot_price_inf']].rolling('14D' ,on='date', axis=0, min_periods=1).mean(), \
                      how='left', on='date')
natural_gas_inf = natural_gas_inf.rename(columns={'spot_price_inf_x':'spot_price_inf', 'spot_price_inf_y':'2_week_rolling_avg'})
natural_gas_inf['2_week_rolling_avg'] = natural_gas_inf['2_week_rolling_avg'].shift(1)


natural_gas_inf = natural_gas_inf.merge(natural_gas_inf[['date','spot_price_inf']].rolling('28D' ,on='date', axis=0, min_periods=1).mean(), \
                      how='left', on='date')
natural_gas_inf = natural_gas_inf.rename(columns={'spot_price_inf_x':'spot_price_inf', 'spot_price_inf_y':'4_week_rolling_avg'})
natural_gas_inf['4_week_rolling_avg'] = natural_gas_inf['4_week_rolling_avg'].shift(1)


natural_gas_inf = natural_gas_inf.merge(natural_gas_inf[['date','spot_price_inf']].rolling('56D' ,on='date', axis=0, min_periods=1).mean(), \
                      how='left', on='date')
natural_gas_inf = natural_gas_inf.rename(columns={'spot_price_inf_x':'spot_price_inf', 'spot_price_inf_y':'8_week_rolling_avg'})
natural_gas_inf['8_week_rolling_avg'] = natural_gas_inf['8_week_rolling_avg'].shift(1)

natural_gas_inf['prev_price'] = natural_gas_inf['spot_price_inf'].shift(1)
```


```python
natural_gas_inf
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
      <th>date</th>
      <th>spot_price</th>
      <th>inflation</th>
      <th>spot_price_inf</th>
      <th>2_week_rolling_avg</th>
      <th>4_week_rolling_avg</th>
      <th>8_week_rolling_avg</th>
      <th>prev_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-01-10</td>
      <td>3.79</td>
      <td>0.395112</td>
      <td>1.497475</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-01-17</td>
      <td>4.19</td>
      <td>0.395112</td>
      <td>1.65552</td>
      <td>1.497475</td>
      <td>1.497475</td>
      <td>1.497475</td>
      <td>1.497475</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-01-24</td>
      <td>2.98</td>
      <td>0.395112</td>
      <td>1.177434</td>
      <td>1.576498</td>
      <td>1.576498</td>
      <td>1.576498</td>
      <td>1.65552</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-01-31</td>
      <td>2.91</td>
      <td>0.395112</td>
      <td>1.149776</td>
      <td>1.416477</td>
      <td>1.443477</td>
      <td>1.443477</td>
      <td>1.177434</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-02-07</td>
      <td>2.53</td>
      <td>0.394415</td>
      <td>0.997869</td>
      <td>1.163605</td>
      <td>1.370052</td>
      <td>1.370052</td>
      <td>1.149776</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>2023-10-13</td>
      <td>3.2</td>
      <td>1.0</td>
      <td>3.2</td>
      <td>2.827903</td>
      <td>2.789435</td>
      <td>2.708395</td>
      <td>2.91</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>2023-10-20</td>
      <td>2.86</td>
      <td>1.0</td>
      <td>2.86</td>
      <td>3.055000</td>
      <td>2.890080</td>
      <td>2.776094</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>2023-10-27</td>
      <td>2.89</td>
      <td>1.0</td>
      <td>2.89</td>
      <td>3.030000</td>
      <td>2.928951</td>
      <td>2.808991</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>2023-11-03</td>
      <td>3.17</td>
      <td>1.0</td>
      <td>3.17</td>
      <td>2.875000</td>
      <td>2.965000</td>
      <td>2.838628</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>2023-11-10</td>
      <td>2.46</td>
      <td>1.0</td>
      <td>2.46</td>
      <td>3.030000</td>
      <td>3.030000</td>
      <td>2.909717</td>
      <td>3.17</td>
    </tr>
  </tbody>
</table>
<p>1400 rows × 8 columns</p>
</div>



We can see from the resulting dataframe that only the first row has NAN values. This is because there is no 'previous' value to average. We will exclude this row from the training set later on.

We can also add additional columns for various date dimensions. Adding in date parts, like month, quarter, and year, can help the model determine seasonal trends and yearly trends.


```python
# Adding Date Parts as Columns
natural_gas_inf['month'] = natural_gas_inf['date'].apply(lambda x: x.month)
natural_gas_inf['quarter'] = natural_gas_inf['date'].apply(lambda x: x.quarter)
natural_gas_inf['year'] = natural_gas_inf['date'].apply(lambda x: x.year)
```

We can now calculate the correlation plot between these new dimensions.


```python
natural_gas_inf.corr()
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
      <th>date</th>
      <th>spot_price</th>
      <th>inflation</th>
      <th>spot_price_inf</th>
      <th>2_week_rolling_avg</th>
      <th>4_week_rolling_avg</th>
      <th>8_week_rolling_avg</th>
      <th>prev_price</th>
      <th>month</th>
      <th>quarter</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>date</th>
      <td>1.000000</td>
      <td>-0.137307</td>
      <td>0.826097</td>
      <td>0.189714</td>
      <td>0.191673</td>
      <td>0.193627</td>
      <td>0.197814</td>
      <td>0.190223</td>
      <td>0.025185</td>
      <td>0.024067</td>
      <td>0.999312</td>
    </tr>
    <tr>
      <th>spot_price</th>
      <td>-0.137307</td>
      <td>1.000000</td>
      <td>0.195646</td>
      <td>0.896610</td>
      <td>0.863616</td>
      <td>0.850321</td>
      <td>0.826259</td>
      <td>0.869444</td>
      <td>0.031304</td>
      <td>0.029119</td>
      <td>-0.138504</td>
    </tr>
    <tr>
      <th>inflation</th>
      <td>0.826097</td>
      <td>0.195646</td>
      <td>1.000000</td>
      <td>0.554204</td>
      <td>0.558781</td>
      <td>0.562984</td>
      <td>0.566945</td>
      <td>0.555138</td>
      <td>0.031359</td>
      <td>0.028085</td>
      <td>0.825165</td>
    </tr>
    <tr>
      <th>spot_price_inf</th>
      <td>0.189714</td>
      <td>0.896610</td>
      <td>0.554204</td>
      <td>1.000000</td>
      <td>0.973856</td>
      <td>0.963913</td>
      <td>0.942210</td>
      <td>0.977401</td>
      <td>0.038471</td>
      <td>0.035941</td>
      <td>0.188354</td>
    </tr>
    <tr>
      <th>2_week_rolling_avg</th>
      <td>0.191673</td>
      <td>0.863616</td>
      <td>0.558781</td>
      <td>0.973856</td>
      <td>1.000000</td>
      <td>0.992704</td>
      <td>0.971842</td>
      <td>0.994348</td>
      <td>0.044525</td>
      <td>0.041332</td>
      <td>0.190068</td>
    </tr>
    <tr>
      <th>4_week_rolling_avg</th>
      <td>0.193627</td>
      <td>0.850321</td>
      <td>0.562984</td>
      <td>0.963913</td>
      <td>0.992704</td>
      <td>1.000000</td>
      <td>0.987607</td>
      <td>0.982706</td>
      <td>0.043695</td>
      <td>0.041666</td>
      <td>0.192059</td>
    </tr>
    <tr>
      <th>8_week_rolling_avg</th>
      <td>0.197814</td>
      <td>0.826259</td>
      <td>0.566945</td>
      <td>0.942210</td>
      <td>0.971842</td>
      <td>0.987607</td>
      <td>1.000000</td>
      <td>0.959889</td>
      <td>0.045978</td>
      <td>0.044945</td>
      <td>0.196164</td>
    </tr>
    <tr>
      <th>prev_price</th>
      <td>0.190223</td>
      <td>0.869444</td>
      <td>0.555138</td>
      <td>0.977401</td>
      <td>0.994348</td>
      <td>0.982706</td>
      <td>0.959889</td>
      <td>1.000000</td>
      <td>0.044575</td>
      <td>0.041264</td>
      <td>0.188600</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.025185</td>
      <td>0.031304</td>
      <td>0.031359</td>
      <td>0.038471</td>
      <td>0.044525</td>
      <td>0.043695</td>
      <td>0.045978</td>
      <td>0.044575</td>
      <td>1.000000</td>
      <td>0.971172</td>
      <td>-0.011785</td>
    </tr>
    <tr>
      <th>quarter</th>
      <td>0.024067</td>
      <td>0.029119</td>
      <td>0.028085</td>
      <td>0.035941</td>
      <td>0.041332</td>
      <td>0.041666</td>
      <td>0.044945</td>
      <td>0.041264</td>
      <td>0.971172</td>
      <td>1.000000</td>
      <td>-0.011849</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.999312</td>
      <td>-0.138504</td>
      <td>0.825165</td>
      <td>0.188354</td>
      <td>0.190068</td>
      <td>0.192059</td>
      <td>0.196164</td>
      <td>0.188600</td>
      <td>-0.011785</td>
      <td>-0.011849</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see, the correlation between the various rolling averages is very high, which is unsurprising. The various averages are much less correlated with the date attributes, but are the most correlated with the yearly column. This makes sense because we expect most of the significant price changes are likely observed yearly. 

Overall, the model likely only needs one rolling average value, if it's needed at all. We can start by using the 4 week rolling average and adjusting if necessary.

## Build Pipeline

We can build a pipeline to transform our categorical variables (the date parts) and the numeric variables (the four week rolling average and date). 

We first must convert the date column to an ordinal number in order for the model not to treat the date as a categorical value. We can take the ordinal number in relation to weeks since the first recorded date.


```python
# Ordinal Date
natural_gas_inf['date_ordinal'] = natural_gas_inf['date'].apply(lambda x: date.toordinal(x))
```


```python
# Drop NANs
natural_gas_inf = natural_gas_inf.dropna(axis=0)
```

We can now build the full pipeline.


```python
# Data Preprocessing
numeric_features = ['4_week_rolling_avg', 'date_ordinal']
numeric_transformer = Pipeline( # No missing values, so no need for an imputer
    steps=[('scaler', StandardScaler())]
)

categorical_features = ['month','quarter','year']
categorical_transformers = Pipeline(
    steps=[('scaler_2', StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformers, categorical_features)
    ]
)

pipeline = Pipeline(
    steps=[('preprocessor', preprocessor)]
)
```

## Train Test Split

Instead of randomly splitting the train and testing sets, we will instead choose to hold the last 20% of spot prices for testing. This mirrors the real world where the model needs to predict only future values.


```python
natural_gas_inf = natural_gas_inf.sort_values(by='date').reset_index()
train = natural_gas_inf.loc[0:len(natural_gas_inf) - (len(natural_gas_inf) * 0.2)-1]
test = natural_gas_inf.loc[len(natural_gas_inf) - (len(natural_gas_inf) * 0.2):]

print('Training Set: ', len(train))
print('Testing Set: ', len(test))
print('Split Percent: ', len(test)/(len(train)+len(test))*100,'%')
```

    Training Set:  1119
    Testing Set:  279
    Split Percent:  19.95708154506438 %
    


```python
train_y = train['spot_price_inf']
test_y = test['spot_price_inf']

train_x = pipeline.fit_transform(train[['4_week_rolling_avg', 'month','quarter','year','date_ordinal']])
test_x = pipeline.transform(test[['4_week_rolling_avg', 'month','quarter','year','date_ordinal']])

```

## Model Evaluation

We can try various regressors from sklearn to fit our training data and predict the testing values. We will use the R2 and Explained Variance for fit metrics and Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) as prediction metrics. We are able to use MAPE because there are no actual values (spot prices) that are zero. We can always expect values greater than zero for both predictions and actuals.


```python
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor #non-neg positive=True
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
```


```python
lr_model = LinearRegression(positive=True)
lr_model.fit(train_x, train_y)
lr_predict = lr_model.predict(test_x)
print('Linear Model')
print('R2 Score: ', r2_score(lr_predict, test_y))
print('Explained Variance: ', explained_variance_score(lr_predict, test_y))
print('MSE: ', mean_squared_error(lr_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(lr_predict, test_y), '%')
print('')

r_model = Ridge()
r_model.fit(train_x, train_y)
r_predict = r_model.predict(test_x)
print('Ridge Model')
print('R2 Score: ', r2_score(r_predict, test_y))
print('Explained Variance: ', explained_variance_score(r_predict, test_y))
print('MSE: ', mean_squared_error(r_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(r_predict, test_y), '%')
print('')

sgd_model = SGDRegressor()
sgd_model.fit(train_x, train_y)
sgd_predict = sgd_model.predict(test_x)
print('SGD Model')
print('R2 Score: ', r2_score(sgd_predict, test_y))
print('Explained Variance: ', explained_variance_score(sgd_predict, test_y))
print('MSE: ', mean_squared_error(sgd_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(sgd_predict, test_y), '%')
print('')

svr_model = SVR(kernel='linear')
svr_model.fit(train_x, train_y)
svr_predict = svr_model.predict(test_x)
print('SVR Model')
print('R2 Score: ', r2_score(svr_predict, test_y))
print('Explained Variance: ', explained_variance_score(svr_predict, test_y))
print('MSE: ', mean_squared_error(svr_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(svr_predict, test_y), '%')
print('')

dt_model = DecisionTreeRegressor()
dt_model.fit(train_x, train_y)
dt_predict = dt_model.predict(test_x)
print('DT Model')
print('R2 Score: ', r2_score(dt_predict, test_y))
print('Explained Variance: ', explained_variance_score(dt_predict, test_y))
print('MSE: ', mean_squared_error(dt_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(dt_predict, test_y), '%')
print('')

gbr_model = GradientBoostingRegressor()
gbr_model.fit(train_x, train_y)
gbr_predict = gbr_model.predict(test_x)
print('GBR Model')
print('R2 Score: ', r2_score(gbr_predict, test_y))
print('Explained Variance: ', explained_variance_score(gbr_predict, test_y))
print('MSE: ', mean_squared_error(gbr_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(gbr_predict, test_y), '%')
print('')

abr_model = AdaBoostRegressor()
abr_model.fit(train_x, train_y)
abr_predict = abr_model.predict(test_x)
print('ABR Model')
print('R2 Score: ', r2_score(abr_predict, test_y))
print('Explained Variance: ', explained_variance_score(abr_predict, test_y))
print('MSE: ', mean_squared_error(abr_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(abr_predict, test_y), '%')
print('')
```

    Linear Model
    R2 Score:  0.8659888839421176
    Explained Variance:  0.8659962175215565
    MSE:  0.5539276910348315
    MAPE:  0.1166230431517775 %
    
    Ridge Model
    R2 Score:  0.8653213114260554
    Explained Variance:  0.8653322119911879
    MSE:  0.555900354339198
    MAPE:  0.11804571330243194 %
    
    SGD Model
    R2 Score:  0.8653333083555871
    Explained Variance:  0.8653846962653181
    MSE:  0.5551462735997426
    MAPE:  0.11684135858066252 %
    
    SVR Model
    R2 Score:  0.8684036743031867
    Explained Variance:  0.8686272502812704
    MSE:  0.5592030489472042
    MAPE:  0.11961196883869504 %
    
    DT Model
    R2 Score:  0.7996463636341851
    Explained Variance:  0.8003998955611449
    MSE:  0.7359502963035613
    MAPE:  0.164906156229634 %
    
    GBR Model
    R2 Score:  0.7829375633679315
    Explained Variance:  0.7881995102711575
    MSE:  0.6599324656153797
    MAPE:  0.13270916177164374 %
    
    ABR Model
    R2 Score:  0.822822387279977
    Explained Variance:  0.8344717894122089
    MSE:  0.6338089192395995
    MAPE:  0.16736115879824623 %
    
    

Overall, the models performed pretty well. The Linear model, only specifying a non-negative linear model, had an R2 score of 0.866 and MAPE of 0.117. The worse performing model was the Gradient Boosted Regression model with an R2 of 0.776 and MAPE of 0.135, which is still fairly good. The best performing model was the SVR model with a linear kernel. 

We can evaluate the Linear model and the SVR model more in depth.


```python
xy_x = range(0,11,1)
xy_y = range(0,11,1)
fig, ax = plt.subplots()
ax.scatter(lr_predict, test_y)
ax.plot(xy_x, xy_y, color='black')
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_title('Linear Model: Actual vs. Predictions')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
```




    Text(0, 0.5, 'Actual')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_43_1.png)
    



```python
xy_x = range(0,11,1)
xy_y = range(0,11,1)
fig, ax = plt.subplots()
ax.scatter(svr_predict, test_y)
ax.plot(xy_x, xy_y, color='black')
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_title('Support Vector Regression Model: Actual vs. Predictions')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
```




    Text(0, 0.5, 'Actual')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_44_1.png)
    


When we plot predictions vs. actuals, we see a similar pattern: at lower prices, predictions and actuals are very close, but at higher prices, predictions and actuals differ, but the predictions are split between over predicting and under predicting.

Around the predicted value of 42 there seem to be two points very far from the line y=x (or predictions = actuals), one of which is in actuality much higher than 4.


```python
dates = test['date']

fig, ax = plt.subplots()
ax.plot(dates, test_y, label='Actuals')
ax.plot(dates, lr_predict, label='Predicted')
ax.legend()
ax.set_title('Linear Model: Actuals vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
```




    Text(0, 0.5, 'Price')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_46_1.png)
    



```python
fig, ax = plt.subplots()
ax.plot(dates, test_y, label='Actuals')
ax.plot(dates, svr_predict, label='Predicted')
ax.legend()
ax.set_title('SVR Model: Actuals vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
```




    Text(0, 0.5, 'Price')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_47_1.png)
    



```python
lr_model.coef_
```




    array([1.79160711e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           1.34136661e-03])




```python
svr_model.coef_
```




    array([[ 1.81786274, -0.02265806,  0.05588361, -0.06351241,  0.01449046]])



Plotting the predictions and actuals over time shows exactly what this model is doing. We see that the model is more or less relying soly on the 4 Week Rolling Average to make the predictions. This means the predictions are a time shifted, smoothed version of the 4 week moving average. Overall, this makes the fit fairly good, but the predictions can be wildly off, especially if there is a large spike in price, like the spike around early 2021.

## ARMA and ARIMA

Instead of making a prediction based on the 4 week moving average (or any other moving average), we can instead try to use Autoregressive Moving Average (ARMA) and Autoregressive Integrated Moving Average modeling techniques to predict this data.

***References:***
- https://www.machinelearningplus.com/time-series/time-series-analysis-python/
- https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/


```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
train_dates = pd.DatetimeIndex(train['date'].values, freq='infer')
train_values = train['spot_price_inf'].values.astype(float)
ts = pd.DataFrame(train_values, index=train_dates, columns=['spot_price_inf'])
result_mul = seasonal_decompose(ts.asfreq('W-FRI').dropna(), model='multiplicative', period=4)
result_add = seasonal_decompose(ts.asfreq('W-FRI').dropna(), model='additive', period=4)

result_mul.plot().suptitle('Multiplicative', y=1)
result_add.plot().suptitle('Additive', y=1)
```




    Text(0.5, 1, 'Additive')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_53_1.png)
    



    
![png]({{site.url}}\ms_projects\dtsa_5509\output_53_2.png)
    


We can see by the trend decomposition graphs that there does not seem to be any obvious trends, but we can still fit the ARIMA models and plot the predictions.

We can fit the model with a (1,0,1) order model, or a first order Autoregressive and Moving Average Model.


```python
train_dates = pd.DatetimeIndex(train['date'].values).to_period('W-FRI')
train_values = train['spot_price_inf'].values.astype(float)
ts = pd.Series(train_values, index=train_dates)
arma_model = ARIMA(ts, order=(1,0,1))
arma_res = arma_model.fit()
print(arma_res.summary())

fig, ax = plt.subplots()
ax.plot(test['date'], test['spot_price_inf'], label='actual')
fig = plot_predict(arma_res, start='2018-07-13', end='2023-11-10', ax=ax)
legend = ax.legend()
```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 1119
    Model:                 ARIMA(1, 0, 1)   Log Likelihood                -220.281
    Date:                Sun, 24 Dec 2023   AIC                            448.563
    Time:                        11:24:46   BIC                            468.644
    Sample:                    01-17-1997   HQIC                           456.153
                             - 06-29-2018                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.7732      0.917      3.025      0.002       0.977       4.570
    ar.L1          0.9846      0.004    265.261      0.000       0.977       0.992
    ma.L1          0.0773      0.011      6.905      0.000       0.055       0.099
    sigma2         0.0865      0.001     75.415      0.000       0.084       0.089
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):             25231.90
    Prob(Q):                              0.90   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.09   Skew:                             0.27
    Prob(H) (two-sided):                  0.42   Kurtosis:                        26.26
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


    
![png]({{site.url}}\ms_projects\dtsa_5509\output_55_1.png)
    


Unfortunately, since the ARMIA model could not determine trends, the model is very basic and the confidence interval does not capture the spikes in data around 2022-2023.

## XGBoost

We can try one more modeling technique, XGBoost. XGBoost is another ensemble method and can be used for time series forecasting.

***References:***
- https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
- https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/


```python
import xgboost
```


```python
xgb_model = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
xgb_model.fit(train_x, train_y)
xgb_predict = xgb_model.predict(test_x)

print('XGB Model')
print('R2 Score: ', r2_score(xgb_predict, test_y))
print('Explained Variance: ', explained_variance_score(xgb_predict, test_y))
print('MSE: ', mean_squared_error(xgb_predict, test_y))
print('MAPE: ', mean_absolute_percentage_error(xgb_predict, test_y), '%')
print('')
```

    XGB Model
    R2 Score:  0.3291877891073315
    Explained Variance:  0.3750278195181632
    MSE:  1.1987879121383782
    MAPE:  0.20415018788363182 %
    
    


```python
xy_x = range(0,11,1)
xy_y = range(0,11,1)
fig, ax = plt.subplots()
ax.scatter(xgb_predict, test_y)
ax.plot(xy_x, xy_y, color='black')
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_title('XGBoost Model: Actual vs. Predictions')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
```




    Text(0, 0.5, 'Actual')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_60_1.png)
    



```python
dates = test['date']

fig, ax = plt.subplots()
ax.plot(dates, test_y, label='Actuals')
ax.plot(dates, xgb_predict, label='Predicted')
ax.legend()
ax.set_title('XGBoost Model: Actuals vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
```




    Text(0, 0.5, 'Price')




    
![png]({{site.url}}\ms_projects\dtsa_5509\output_61_1.png)
    


We can see that the XGBoost model is predicting much different values from the actual value. The overall R2 and MAPE metrics for the model are much worse than previous models, which is not surprising seeing the Predicted vs. Actuals graphs.

The model seems to under-predict when the spot price is relatively high, but over-predicts when the spot price is low.

## Actual Future Spot Price Evaluation

With no consensus for which model would preform the best, we can examine the Linear, SVR, and XGBoost predictions for the future values.

A special prediction function must be created in order to predict multiple periods into the future, while taking into account previous period predictions. This will be done by iteratively predicting each time step and taking the model prediction as the 'actual' in subsequent steps.


```python
# Prediction Function (4 Week Rolling Average)
def predict_prices(period, model, pipeline, org_data):
    copy_data = org_data.copy()
    copy_data = copy_data[['date', 'spot_price', '4_week_rolling_avg','month','quarter','year', 'date_ordinal']]
    copy_data['prediction/actual'] = 'actual'
    max_date = copy_data['date'].max()
    max_date_ordinal = copy_data['date_ordinal'].max()
    date_dict = {}
    for i in range(period):
        add = (i+1)*7
        new_date_ordinal = max_date_ordinal + add
        new_date = max_date + timedelta(days=add)
        date_dict[new_date_ordinal] = new_date
    for key in date_dict:
        c_month = date_dict[key].month
        c_quarter = date_dict[key].quarter
        c_year =  date_dict[key].year
        c_4_week_avg = copy_data[(copy_data['date_ordinal'] < key) & (copy_data['date_ordinal'] >= (key-28))]['spot_price'].mean()
        d = {'date': date_dict[key], '4_week_rolling_avg':c_4_week_avg, \
             'month':c_month, 'quarter':c_quarter, 'year':c_year, 'date_ordinal':key}
        temp_df = pd.DataFrame(d, index=[0])
        test_x = pipeline.transform(temp_df)
        predict = model.predict(test_x)
        d2 = {'date': date_dict[key], '4_week_rolling_avg':c_4_week_avg, \
             'month':c_month, 'quarter':c_quarter, 'year':c_year, 'date_ordinal':key, \
              'spot_price':round(predict[0],2), 'prediction/actual':'prediction'}
        add_df = pd.DataFrame(d2, index=[0])
        copy_data = pd.concat([copy_data, add_df], ignore_index=True)
    return copy_data 
```


```python
predict_prices(2, xgb_model, pipeline, natural_gas_inf)
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
      <th>date</th>
      <th>spot_price</th>
      <th>4_week_rolling_avg</th>
      <th>month</th>
      <th>quarter</th>
      <th>year</th>
      <th>date_ordinal</th>
      <th>prediction/actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1997-01-17</td>
      <td>4.19</td>
      <td>1.497475</td>
      <td>1</td>
      <td>1</td>
      <td>1997</td>
      <td>729041</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1997-01-24</td>
      <td>2.98</td>
      <td>1.576498</td>
      <td>1</td>
      <td>1</td>
      <td>1997</td>
      <td>729048</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997-01-31</td>
      <td>2.91</td>
      <td>1.443477</td>
      <td>1</td>
      <td>1</td>
      <td>1997</td>
      <td>729055</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1997-02-07</td>
      <td>2.53</td>
      <td>1.370052</td>
      <td>2</td>
      <td>1</td>
      <td>1997</td>
      <td>729062</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997-02-14</td>
      <td>2.3</td>
      <td>1.245150</td>
      <td>2</td>
      <td>1</td>
      <td>1997</td>
      <td>729069</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>2023-10-27</td>
      <td>2.89</td>
      <td>2.928951</td>
      <td>10</td>
      <td>4</td>
      <td>2023</td>
      <td>738820</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>2023-11-03</td>
      <td>3.17</td>
      <td>2.965000</td>
      <td>11</td>
      <td>4</td>
      <td>2023</td>
      <td>738827</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>2023-11-10</td>
      <td>2.46</td>
      <td>3.030000</td>
      <td>11</td>
      <td>4</td>
      <td>2023</td>
      <td>738834</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>2023-11-17</td>
      <td>2.43</td>
      <td>2.845000</td>
      <td>11</td>
      <td>4</td>
      <td>2023</td>
      <td>738841</td>
      <td>prediction</td>
    </tr>
    <tr>
      <th>1400</th>
      <td>2023-11-24</td>
      <td>2.46</td>
      <td>2.737500</td>
      <td>11</td>
      <td>4</td>
      <td>2023</td>
      <td>738848</td>
      <td>prediction</td>
    </tr>
  </tbody>
</table>
<p>1401 rows × 8 columns</p>
</div>



We can see the output is a dataframe with actual and prediction values. While the model does not need all data, we can return the full dataframe, as it is fairly small.

We can now assess the actual week ending spot prices. We could re-download the data, but because we are only looking at a few periods, we can just copy the data directly from the Spot Price [website](https://www.eia.gov/dnav/ng/hist/rngwhhdw.htm), previously cited.


```python
y_actual = {'date':[date(2023,11,17),date(2023,11,24),date(2023,12,1),date(2023,12,8),date(2023,12,15)],
            'spot_price':[2.74,2.64,2.72,2.62,2.38], 'prediction/actual':['actual','actual','actual','actual','actual']}
y_actual_df = pd.DataFrame(y_actual)
y_actual_df['date'] = y_actual_df['date'].apply(lambda x: pd.to_datetime(x))
y_actual_df
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
      <th>date</th>
      <th>spot_price</th>
      <th>prediction/actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-11-17</td>
      <td>2.74</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-11-24</td>
      <td>2.64</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-12-01</td>
      <td>2.72</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-12-08</td>
      <td>2.62</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-12-15</td>
      <td>2.38</td>
      <td>actual</td>
    </tr>
  </tbody>
</table>
</div>



We can now predict the next 5 weeks of data for the Linear Regressor, SVR, and XGBoost model.


```python
lr_predict_df = predict_prices(5, lr_model, pipeline, natural_gas_inf)
svr_predict_df = predict_prices(5, svr_model, pipeline, natural_gas_inf)
xgb_predict_df = predict_prices(5, xgb_model, pipeline, natural_gas_inf)

lr_predict_df_full = pd.concat([lr_predict_df,y_actual_df])
svr_predict_df_full = pd.concat([svr_predict_df,y_actual_df])
xgb_predict_df_full = pd.concat([xgb_predict_df,y_actual_df])
```


```python
fig, ax = plt.subplots()
df = lr_predict_df_full[lr_predict_df_full['date'] >= pd.to_datetime(date(2023,1,1))]
ax.plot(df[df['prediction/actual']=='actual']['date'], df[df['prediction/actual']=='actual']['spot_price'], label='actual')
ax.plot(df[df['prediction/actual']=='prediction']['date'], df[df['prediction/actual']=='prediction']['spot_price'], label='prediction')
ax.legend()
ax.set_title('Linear Model: Actuals vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

y_actuals = y_actual_df['spot_price'].values
y_predict = lr_predict_df[lr_predict_df['prediction/actual']=='prediction']['spot_price'].values

print('Mean Squared Error: %.2f'%(mean_squared_error(y_actuals, y_predict)))
print('Mean Absolute Percent Error: %.2f'%(mean_absolute_percentage_error(y_actuals, y_predict)),'%')
```

    Mean Squared Error: 0.06
    Mean Absolute Percent Error: 0.08
    


    
![png]({{site.url}}\ms_projects\dtsa_5509\output_70_1.png)
    



```python
fig, ax = plt.subplots()
df = svr_predict_df_full[svr_predict_df_full['date'] >= pd.to_datetime(date(2023,1,1))]
ax.plot(df[df['prediction/actual']=='actual']['date'], df[df['prediction/actual']=='actual']['spot_price'], label='actual')
ax.plot(df[df['prediction/actual']=='prediction']['date'], df[df['prediction/actual']=='prediction']['spot_price'], label='prediction')
ax.legend()
ax.set_title('SVR Model: Actuals vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

y_actuals = y_actual_df['spot_price'].values
y_predict = svr_predict_df[svr_predict_df['prediction/actual']=='prediction']['spot_price'].values

print('Mean Squared Error: %.2f'%(mean_squared_error(y_actuals, y_predict)))
print('Mean Absolute Percent Error: %.2f'%(mean_absolute_percentage_error(y_actuals, y_predict)),'%')
```

    Mean Squared Error: 0.03
    Mean Absolute Percent Error: 0.05 %
    


    
![png]({{site.url}}\ms_projects\dtsa_5509\output_71_1.png)
    



```python
fig, ax = plt.subplots()
df = xgb_predict_df_full[xgb_predict_df_full['date'] >= pd.to_datetime(date(2023,1,1))]
ax.plot(df[df['prediction/actual']=='actual']['date'], df[df['prediction/actual']=='actual']['spot_price'], label='actual')
ax.plot(df[df['prediction/actual']=='prediction']['date'], df[df['prediction/actual']=='prediction']['spot_price'], label='prediction')
ax.legend()
ax.set_title('XGBoost Model: Actuals vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Price')

y_actuals = y_actual_df['spot_price'].values
y_predict = xgb_predict_df[xgb_predict_df['prediction/actual']=='prediction']['spot_price'].values

print('Mean Squared Error: %.2f'%(mean_squared_error(y_actuals, y_predict)))
print('Mean Absolute Percent Error: %.2f'%(mean_absolute_percentage_error(y_actuals, y_predict)),'%')
```

    Mean Squared Error: 0.11
    Mean Absolute Percent Error: 0.11 %
    


    
![png]({{site.url}}\ms_projects\dtsa_5509\output_72_1.png)
    


# Conclusion and Reflections

Overall, the SVR model did the best at predicting future prices for multiple future periods. This is not surprising given that the model performed the best against the testing data. While the first two models overpredicted the spot price, the XGBoost model underpredicted the spot price. However, the XGBoost predicted a closer price during the final period, where the spot price took a dramatic decrease, compared to the first two models.

This project was very interesting to complete. There are a lot of things to take into account with predicting time series data, including how past data should be incorporated as well as how to predict multiple periods in the future. I am happy with the skills I gained completing this project even if the overall model results could be better. 

Future improvements can be made by incorporating or testing various rolling average ranges or trying advanced time series data techniques, as the first two models are too dependent on the single rolling-average spot price column. Additionally, finding a way to output confidence intervals similar to the ARMA model would be helpful, as it's unlikely these models will predict the exact spot price.
