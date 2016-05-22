# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 02:09:53 2016
@ Resampling : This is where we have some data that is sampled at a certain rate. 
For us, we have the Housing Price Index sampled at a one-month rate, but we could 
sample the HPI every week, every day, every minute, or more, but we could also resample 
at every year, every 10 years, and so on.
@author: Pokemon


Resample rule:
xL for milliseconds
xMin for minutes
xD for Days

Alias	Description
B	business day frequency
C	custom business day frequency (experimental)
D	calendar day frequency
W	weekly frequency
M	month end frequency
BM	business month end frequency
CBM	custom business month end frequency
MS	month start frequency
BMS	business month start frequency
CBMS	custom business month start frequency
Q	quarter end frequency
BQ	business quarter endfrequency
QS	quarter start frequency
BQS	business quarter start frequency
A	year end frequency
BA	business year end frequency
AS	year start frequency
BAS	business year start frequency
BH	business hour frequency
H	hourly frequency
T	minutely frequency
S	secondly frequency
L	milliseonds
U	microseconds
N	nanoseconds

How:
mean, sum, ohlc
"""

#Resamplaing is for changing the granuality of data like 
#instock prices you have the milisecond data and you want 
#to keep record of 5 years. Get milisec data of day, add them together 
#and avg dm and that wiil be at day level. 

#Note that when we resample, we mostly do upsampling. Downsampling - we dont add any new information.

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


#Sampling the data Anually

TX1yr = HPI_data['AR'].resample('A')
print(TX1yr.head())

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data["AR"].plot(ax = ax1, label = "Monthly Data")
TX1yr.plot(ax = ax1, label = "Yearly Data")
plt.legend()
plt.show()


#Open High low and close Sampling
AR1yr = HPI_data['AR'].resample('A', how='ohlc')
print(AR1yr.head())

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data["AR"].plot(ax = ax1, label = "Monthly Data")
AR1yr.plot(ax = ax1, label = "Yearly Data")
plt.legend()
plt.show()

