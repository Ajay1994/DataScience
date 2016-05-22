# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 00:39:08 2016

@author: Pokemon
"""

import pandas as pd
import requests
import re

df = pd.read_csv('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\ZILL-Z77006_PRR.csv')
print(df.head())

df.to_csv('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\newcsv3.csv')

df.to_html('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\example.html')

fiddy_states = pd.read_html(requests.get('https://simple.wikipedia.org/wiki/List_of_U.S._states').text)
print(fiddy_states)

for abbv in fiddy_states[0][0][1:]:
    print(abbv)
    
    
df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                     index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                     index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                     index = [2001, 2002, 2003, 2004])
                     
concat = pd.concat([df1,df2])
print(concat)


df4 = df1.append(df2)
print(df4)

# A series is basically a single-columned dataframe. A series does have an index, 
#but, if you convert it to a list, it will be just those values. Whenever we say 
#something like df['column'], the return is a series. Series is kind of a vector
s = pd.Series([80,2,50], index=['HPI','Int_rate','US_GDP_Thousands'])
df4 = df1.append(s, ignore_index=True)
print(df4)


#merging ignores the index - kind of the joins in databases.
print(pd.merge(df1,df2, on='HPI'))

#You can share on multiple columns, here's an example of that:

print(pd.merge(df1,df2, on=['HPI','Int_rate']))

df4 = pd.merge(df1,df3, on='HPI')
df4.set_index('HPI', inplace=True)
print(df4)

############################################3
df1 = pd.DataFrame({
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55],
                    'Year':[2001, 2002, 2003, 2004]
                    })

df3 = pd.DataFrame({
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53],
                    'Year':[2001, 2003, 2004, 2005]})
                    
#we now have similar year columns, but different dates. df3
#has 2005 but not 2002, and df1 is the reverse of that. Now, what happens when we merge?
#it will print only those which have both in the merge column
merged = pd.merge(df1,df3, on='Year')
print(merged)

#Left - equal to left outer join SQL - use keys from left frame only
#Right - right outer join from SQL- use keys from right frame only.
#Outer - full outer join - use union of keys
#Inner - use only intersection of keys.
merged = pd.merge(df1,df3, on='Year', how='left')
merged.set_index('Year', inplace=True)
print(merged)
                  
