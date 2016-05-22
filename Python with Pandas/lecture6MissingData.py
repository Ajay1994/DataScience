# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 02:59:39 2016
We have a few options when considering the existence of missing data.

1. Ignore it - Just leave it there

2. Delete it - Remove all cases. Remove from data entirely. This means forfeiting the entire row of data.

3. Fill forward or backwards - This means taking the prior or following value and just filling it in.

4. Replace it with something static - For example, replacing all NaN data with -9999.

Each of these options has their own merits for a variety of reasons. 
Ignoring it requires no more work on our end. You may choose to ignore missing data for legal 
reasons, or maybe to retain the utmost integrity of the data. Missing data might also be very important data. 
For example, maybe part of your analysis is investigating signal drops from a server. 
In this case, maybe the missing data is super important to keep in the set.

@author: Pokemon
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

HPI_data['AZ1yr'] = HPI_data['AZ'].resample('A')
print(HPI_data[['AZ','AZ1yr']])
HPI_data[['AZ','AZ1yr']].plot() #AZ1yr wont be visible because it is fillled up with lots of NaN

#Copy the dataframe
HPI_data_copy = HPI_data.copy()

###############################################
#Choice 1 : drop the missing data NAN
###############################################
print(HPI_data_copy[['AZ','AZ1yr']].head(20))
HPI_data_copy.dropna(inplace=True) #After this the sampling will be one per year instead of the month
print(HPI_data_copy[['AZ','AZ1yr']].head(20))
HPI_data[['AZ','AZ1yr']].plot()
# Drop only if all the entries in a row are NAN
HPI_data.dropna(how='all',inplace=True)
#We can also the thresold


################################################
#Choice 2 : Fill NAN forward or backward
################################################
print(HPI_data_copy[['AZ','AZ1yr']].head(20))
HPI_data_copy.fillna(method='ffill',inplace=True)
print(HPI_data_copy[['AZ','AZ1yr']].head(20))
HPI_data_copy[['AZ','AZ1yr']].plot()


#this will be smwhat baised and it will fit the real line very good because it is
# like going to take data from future
HPI_data_copy.fillna(method='bfill',inplace=True)


################################################
#Choice 3 : Fill NAN with some value
################################################
print(HPI_data_copy[['AZ','AZ1yr']].head(20))
HPI_data_copy.fillna(value=-99999,inplace=True)
print(HPI_data_copy[['AZ','AZ1yr']].head(20))
HPI_data_copy[['AZ','AZ1yr']].plot()

#check for the no of the NAN, we only fill a limited no with some values
print(HPI_data_copy.isnull().values.sum())
HPI_data_copy.fillna(value=-99999,limit=10,inplace=True)
HPI_data_copy[['AZ','AZ1yr']].plot()
