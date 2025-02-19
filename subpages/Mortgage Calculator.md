---
layout: page
title: 🏠 Mortgage Calculator
permalink: /python/mortgage
---
I would love to own a house one day. Will it ever happen? Who knows! This calculator was built to weight the many different options for loans, as well as the impact of other necessary and potential payments when buying a house or property. 

The calculator can handle the following:
- Mortgage Loan Cost and Down Payment
- Mortgage Loan Interest Rate
- Mortgage Loan Terms (15, 30)
- [Conventional Loans and FHA Loans](https://www.nerdwallet.com/article/mortgages/fha-loan-vs-conventional-mortgage)
- [Mortgages with PMI (LTV less than 20%)](https://www.investopedia.com/mortgage/insurance/qualified-insurance-premium/)
- [Mortgage Points (Lowering Interest Rate)](https://www.bankrate.com/mortgages/mortgage-points/)

To use the calculator, you input the following costs into the mortgage() class, which will calculate a list of values potentially relevant to deciding on whether or not to buy a house.

```
mortgage(House Cost, Down Payment, Start Date of Loan, Loan Length (years), 
Interest Rate, Property Tax Rate, Points Bought, Closing Cost Percentage, 
HOA (monthly), Utilities (monthly), Home Insurance (monthly), FHA (1-Yes, 0-No)
```

The calculator can return the following:
- A DataFrame of Principle and Interest Paid Over Time
- A DataFrame list of Relevant Values
- A Sentence on the Potential Mortgage Loan
- A Short Sentence on Monthly and Upfront Costs

See below for examples. 

This tool was intended to be a fully developed app, but I have not learned any front-end tools, so it will live in a notebook for now 🙃.


```python
# Libraries
import datetime
import pytz
from dateutil import parser
from datetime import timezone
from dateutil.relativedelta import relativedelta
## String manipulation
import string
import re #regex
## Data Generation
import random
## Data Manipulation
import pandas as pd
import numpy as np
## Vector Math
from numpy.linalg import norm
## Math
import math
import matplotlib as mpl

class mortgage:
    def __init__ (self,cost,down,sdate,yr_length,intr,prop_tax,pts,cc_pct,hoa,util,ins,fha):
        # load from class
        self.cost = cost
        self.down = down
        self.yrs = yr_length
        self.length = yr_length * 12
        self.fha = fha
        self.sdate = pd.to_datetime(sdate)
        self.intr_curr = intr
        self.pts = pts
        self.cc_pct = cc_pct
        self.hoa = hoa
        self.util = util
        self.ins = ins
        self.proptax = prop_tax
        
        # general functions
        self.mortgage = self.cost - self.down
        self.ltv = self.down/self.mortgage
        self.months = list(range(self.length+1))
        self.intr_12mon = intr/12
        self.savings = self.cost * 0.02
        
        # property tax
        self.ann_proptax = self.cost * self.proptax
        self.mon_proptax = self.ann_proptax / 12
        
        # closing
        self.c_costs = self.mortgage * self.cc_pct
        
        # points
        self.pts_cost = self.mortgage * 0.01 * self.pts
        self.intr = self.intr_curr - (0.0025*self.pts)
        self.mon_intr = self.intr/12
        self.mon_intr_curr = self.intr_curr/12
        
        # pmi
        if self.ltv < 0.2:
            self.ann_pmi =  self.mortgage * 0.01
            self.mon_pmi = self.ann_pmi/12
        else:
            self.ann_pmi = 0
            self.mon_pmi = 0
            self.payoff_pmi = 0
        
        # fha
        if self.fha != 0:
            self.fha_pct = 0.0175
            self.ann_fha = self.mortgage * self.fha_pct
            self.mon_fha = self.ann_fha / 12
        else: 
            self.ann_fha = 0
            self.mon_fha = 0
    
        # end stats
        self.money_on_hand = self.down + self.c_costs + self.pts_cost
        self.mon_mortgage = self.mortgage*((self.mon_intr)*(1+self.mon_intr)**(self.length))/((1+self.mon_intr)**(self.length)-1)
        self.tot_mortgage = self.mon_mortgage * self.length
        self.tot_interest = self.tot_mortgage - self.mortgage
        self.tot_mon_cost = self.mon_mortgage + self.hoa + self.mon_proptax + self.mon_pmi + self.mon_fha + self.ins
        self.tot_upfront_cost = self.down + self.c_costs
        
        if self.ltv < 0.2:
            self.payoff_pmi = round(math.log((((self.cost*0.2)-self.down)*(self.mon_intr))/((self.mon_mortgage-(self.mon_intr*self.mortgage)))+1)/math.log(1+(self.mon_intr))/12,2)
        else: self.payoff_pmi = 0
        
        if self.pts > 0:
            self.self.mon_mortgage = self.mortgage*((self.mon_intr_curr)*(1+self.mon_intr_curr)**(self.length))/((1+self.mon_intr_curr)**(self.length)-1)
            self.mon_mortgage_p_diff = self.self.mon_mortgage - self.mon_mortgage
            self.payoff_pts = round(self.pts_cost/(self.mon_mortgage_p_diff*12),2)
        else: 
            self.mon_mortgage_nop = self.mon_mortgage
            self.mon_mortgage_p_diff = 0
            self.payoff_pts = 0
        
        self.tot_mortgage_cost = self.tot_mortgage + (self.ann_fha * 11) + (self.ann_pmi * self.payoff_pmi)
        self.tot_cost = self.tot_mortgage_cost + self.tot_upfront_cost
        
#         math.log((((self.cost*0.2)-self.down)*(self.mon_intr))/((self.mon_mortgage-(self.mon_intr*self.mortgage))+1))
#         /math.log(1+(self.mon_intr))/12
        
        # data frame
        self.df = pd.DataFrame({'month':self.months})
        
        months = list(range(self.length+1))
        months_real = []
        months_princ = []
        months_intr = []
        for n in months:
            date_real = pd.to_datetime(sdate) + relativedelta(months=n)
            months_real.append(date_real.date().strftime('%m-%d-%Y'))
        self.df['real_date'] = months_real
        
        for n in months:
            princ = (self.mon_mortgage-(self.mon_intr*self.mortgage))*((1+self.mon_intr)**(n)-1)/(self.mon_intr)
            intr = (self.mon_mortgage*n)-princ
            months_princ.append(princ)
            months_intr.append(round(intr,2))
        self.df['principal_paid'] = months_princ
        self.df['interest_paid'] = months_intr
        self.df['short_date'] = self.df['real_date'].apply(lambda x: pd.to_datetime(x).strftime('%B-%Y'))
        
    def mortgage_sentence(self):
        numb = [self.cost, self.down, self.mon_mortgage, self.tot_mortgage, self.tot_interest, self.mon_pmi, self.mon_fha,
                (self.ann_fha*11), self.hoa, self.mon_proptax, self.util, self.tot_mon_cost, self.pts_cost, self.c_costs,
                self.savings, self.tot_upfront_cost, self.ins]
        numb_list = []
        for z in numb:
            z = ('{:,.2f}'.format(round(z,2)))
            numb_list.append('$'+z)
        string = 'For a %s year mortgage on a %s with a %s down payment and a %.2f interest rate, your monthly mortgage payment will be %s. ' %(self.yrs, numb_list[0], numb_list[1], self.intr, numb_list[2])
        string2 = 'The total mortgage cost over %s years is %s, of which %s is interest. ' %(self.yrs, numb_list[3],numb_list[4])
        if self.ltv < 0.2 and self.fha == 0:
            string3 = 'Because your Loan-to-value (LTV) ratio is %.2f, which is lower than 0.2 (20%% downpayment), an additional Premium Mortgage Insurance (PMI) monthly payment of %s is needed. The additional PMI payment will end when the LTV  reaches 0.2. This will happen after %.2f years, assuming no re-financing and on-time payments. ' \
            %(float(self.ltv),numb_list[5], float(self.payoff_pmi))
        elif self.fha == 1:
            string3 = 'Because you have chosen an FHA loan, you will be paying an additional monthly Mortgage Insurance Premium (MIP) of %s for 11 years of the loan. This will add an additional %s to the total cost. ' \
            %(numb_list[6], numb_list[7])
        else: string3 = ''
        if self.hoa > 0:
            string4_1 = 'a monthly HOA payment of %s' %(numb_list[8])
        else: string4_1 = ''
        if self.mon_proptax > 0:
            string4_2 = 'the expected property tax of %s' %(numb_list[9])
        else: string4_2 = ''
        if self.util > 0:
            string4_3 = 'the expected monthly utility bill of %s' %( numb_list[10])
        else: string4_3 = ''
        if self.ins > 0:
            string4_4 = 'the expected monthly insurance of %s' %(numb_list[16])
        else: string4_4 = ''
        if (string4_1 == '' and string4_2 == '' and string4_3 == '' and string4_4 == ''):
            string4 = 'The total monthly payment is %s' %(numb_list[11])
        else:
            string4 = 'Additionally,'
            for x in [string4_1,string4_2,string4_3,string4_4]:
                if x != '':
                    string4 = string4 + ' ' + x + ', and'
            string4 = string4[:-4] + ' will make the monthly payment %s. ' %(numb_list[11])
        if self.pts > 0:
            string5 = 'In opting to buy %s Interest Point(s) for %s (-0.025 interest point per point) to bring down the current interest rate of %.2f to %.2f, it will take %.2f years of monthly mortgage payments to make up the upfront cost. ' %(self.pts, numb_list[12], self.intr_curr, self.intr, self.payoff_pts)
        else: string5 = ''
        string6 = 'You will pay %s upfront, %s of which are closing costs, and should have around %s of savings. ' \
        %(numb_list[15], numb_list[13], numb_list[14])
        return string + string2 + string3 + string4 + string5 + string6
    
    def mortgage_short(self):
        numb = [self.tot_mon_cost,
                self.tot_upfront_cost]
        numb_list = []
        for z in numb:
            z = ('{:,.2f}'.format(round(z,2)))
            numb_list.append('$'+z)
        string = 'Your monthly payment will be %s. Your total upfront costs will be around %s' %(numb_list[0], numb_list[1])
        return string
    
    def mortgage_list(self):
        list_df = pd.DataFrame(columns=['metric', 'value'])
        metric = ['Loan Cost', 'Downpayment', 'Monthly Mortgage', 'Total Mortgage Cost', 'Total Interest', 'Monthly PMI', \
                'Monthly FHA', 'Annual FHA', 'HOA', 'Monthly Property Tax', 'Utilities', 'Insurance', 'Total Monthly Cost', \
                'Points Upfront Cost', 'Closing Costs', 'Savings', 'Total Upfront Costs']
        numb = [self.cost, self.down, self.mon_mortgage, self.tot_mortgage, self.tot_interest, self.mon_pmi, \
                self.mon_fha, (self.ann_fha*11), self.hoa, self.mon_proptax, self.util, self.ins, self.tot_mon_cost, \
                self.pts_cost, self.c_costs, self.savings, self.tot_upfront_cost]
        numb_list = []
        for z in numb:
            z = ('{:,.2f}'.format(round(z,2)))
            numb_list.append('$'+z)
        list_df = pd.DataFrame({'metric':metric, 'value':numb_list})
        return list_df
        

    def mortgage_year(self,yr):
        dte = self.sdate + relativedelta(years=yr)
#         closest_mon = 
#         string = 
        return dte
```


```python
# (cost,down,sdate,length,intr,prop_tax,pts,cc_pct,hoa,util,fha)
import plotly.express as px
import plotly.graph_objects as go

test = mortgage(450000,50000,'2023-05-01',30,0.04,0.054,0,0.04,0,0,200,0)
test.df
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
      <th>month</th>
      <th>real_date</th>
      <th>principal_paid</th>
      <th>interest_paid</th>
      <th>short_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>05-01-2023</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>May-2023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>06-01-2023</td>
      <td>576.327849</td>
      <td>1333.33</td>
      <td>June-2023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>07-01-2023</td>
      <td>1154.576790</td>
      <td>2664.75</td>
      <td>July-2023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>08-01-2023</td>
      <td>1734.753228</td>
      <td>3994.23</td>
      <td>August-2023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>09-01-2023</td>
      <td>2316.863587</td>
      <td>5321.78</td>
      <td>September-2023</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>356</th>
      <td>356</td>
      <td>01-01-2053</td>
      <td>392424.588738</td>
      <td>287414.79</td>
      <td>January-2053</td>
    </tr>
    <tr>
      <th>357</th>
      <td>357</td>
      <td>02-01-2053</td>
      <td>394308.998549</td>
      <td>287440.04</td>
      <td>February-2053</td>
    </tr>
    <tr>
      <th>358</th>
      <td>358</td>
      <td>03-01-2053</td>
      <td>396199.689726</td>
      <td>287459.01</td>
      <td>March-2053</td>
    </tr>
    <tr>
      <th>359</th>
      <td>359</td>
      <td>04-01-2053</td>
      <td>398096.683207</td>
      <td>287471.68</td>
      <td>April-2053</td>
    </tr>
    <tr>
      <th>360</th>
      <td>360</td>
      <td>05-01-2053</td>
      <td>400000.000000</td>
      <td>287478.03</td>
      <td>May-2053</td>
    </tr>
  </tbody>
</table>
<p>361 rows × 5 columns</p>
</div>




```python
test = mortgage(450000,50000,'2023-05-01',30,0.04,0.054,0,0.04,0,0,200,0)

fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Principal and Interst Over Time")))
fig.add_trace(go.Scatter(x=test.df['short_date'], y=test.df['principal_paid'],
                    mode='lines',
                    name='Principal',
                    stackgroup='one'))
fig.add_trace(go.Scatter(x=test.df['short_date'], y=test.df['interest_paid'],
                    mode='lines',
                    name='Interest',
                    stackgroup='one'))
fig = fig.update_xaxes(showgrid=False, nticks=8)
fig = fig.update_layout(yaxis_tickprefix = '$', yaxis_tickformat = ',')
fig.show()
```
There was a plotly plot here!

```python
test = mortgage(450000,50000,'2023-05-01',30,0.04,0.054,0,0.04,0,0,200,0)
test.mortgage_list()
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
      <th>metric</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Loan Cost</td>
      <td>$450,000.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Downpayment</td>
      <td>$50,000.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Monthly Mortgage</td>
      <td>$1,909.66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Total Mortgage Cost</td>
      <td>$687,478.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Total Interest</td>
      <td>$287,478.03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Monthly PMI</td>
      <td>$333.33</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Monthly FHA</td>
      <td>$0.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Annual FHA</td>
      <td>$0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HOA</td>
      <td>$0.00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Monthly Property Tax</td>
      <td>$2,025.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Utilities</td>
      <td>$0.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Insurance</td>
      <td>$200.00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Total Monthly Cost</td>
      <td>$4,467.99</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Points Upfront Cost</td>
      <td>$0.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Closing Costs</td>
      <td>$16,000.00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Savings</td>
      <td>$9,000.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Total Upfront Costs</td>
      <td>$66,000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# (cost,down,sdate,length,intr,prop_tax,pts,cc_pct,hoa,util,fha)
mortgage(450000,50000,'2023-05-01',30,0.04,0.054,0,0.04,0,0,200,0).mortgage_sentence()
```




```
For a 30 year mortgage on a $450,000.00 with a $50,000.00 down payment and a 0.04 
interest rate, your monthly mortgage payment will be $1,909.66. The total mortgage 
cost over 30 years is $687,478.03, of which $287,478.03 is interest. Because your 
Loan-to-value (LTV) ratio is 0.12, which is lower than 0.2 (20% downpayment), an 
additional Premium Mortgage Insurance (PMI) monthly payment of $333.33 is needed. 
The additional PMI payment will end when the LTV  reaches 0.2. This will happen 
after 5.21 years, assuming no re-financing and on-time payments. Additionally, 
the expected property tax of $2,025.00, and the expected monthly insurance of 
$200.00, will make the monthly payment $4,467.99. You will pay $66,000.00 upfront, 
$16,000.00 of which are closing costs, and should have around $9,000.00 of savings. 
```




```python
mortgage(450000,50000,'2023-05-01',30,0.04,0.054,0,0.04,0,0,200,0).mortgage_short()
```




```
Your monthly payment will be $4,467.99. 
Your total upfront costs will be around $66,000.00
```


