# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 21:41:39 2016

@author: Pokemon : Ajay Jaiswal
"""
import Quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]
    
states = state_list()

pickle_in = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\fiddy_states.pickle','rb')
HPI_data = pickle.load(pickle_in)
print(HPI_data)

#Change COlnames of the data
HPI_data.columns = states[1:10]

bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}

df = pd.DataFrame(bridge_height)

df.plot()
plt.show()

#We realize this is an outlier because it differs so greatly from the other values, 
#as well as the fact that it suddenly jumps and drops way more than any of the others. 
#Soundes like we're just applying standard deviation here.
#Two gigantic pics one showing sudden rise in value and another showing suddenn decrease in value.


df['STD'] = pd.rolling_std(df['meters'], 2)
print(df)

# Seeing this plot we say that if SD is greater than x than get rid of that x. How to find the value of x

df_std = df.describe()
print(df_std)

#we get get straight to the meters' std, which is 2067 and some change. 
#That's a pretty high figure, but it's still much lower than the STD for 
#that major fluctuation (4385)- which lies within a window of 2.Notion that within a window of two somewhere
# we have pretty high deviation
#Now, we can run through and remove all data that has 
#a standard deviation higher than that.

df_std = df.describe()['meters']['std']
print(df_std)

#Logically selecting only those of the rolling SD which lies within 1 Standard deviation of meter data.
df2 = df[ (df['STD'] < df_std) ]
df2['meters'].plot()
plt.show()





