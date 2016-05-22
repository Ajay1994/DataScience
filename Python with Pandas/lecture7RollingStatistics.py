# -*********- coding: utf-8 -****-
"""
Created on Thu Mar 10 03:45:48 2016
Rolling Statistics : Rolling mean, count, average and many more

One of the more popular rolling statistics is the moving average.
This takes a moving window of time, and calculates the average or
the mean of that time period as the current value. In our case,
we have monthly data. So a 10 moving average would be the current
value, plus the previous 9 months of data, averaged, and there 
we would have a 10 moving average of our monthly data.

@author: Pokemon
"""

#Rolling apply : you can write any function to do with the window
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

#find rolling mean in a window of 12 months - initially it will be null
HPI_data['AZ12MA'] = pd.rolling_mean(HPI_data['AZ'], 12)
HPI_data[['AZ','AZ12MA']].plot()

#finding Rolling SD - good because it help in finding problem points and outliers
#and also help us detect voltality in market.
#not same scale as housing prices but it is alwaz going to be small values
#so we graph on diff graph
HPI_data['AZ12STD'] = pd.rolling_std(HPI_data['AZ'], 12)
HPI_data[['AZ','AZ12MA','AZ12STD']].plot()



fig = plt.figure()
ax1 = plt.subplot2grid((2,1), (0,0)) #Grid is 2 tall and 1 wide : 1 graph on top and 1 on bottom
ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1) #share ax1 axix
HPI_data['AZ'].plot(ax=ax1)
HPI_data['AZ12MA'].plot(ax=ax1)
HPI_data['AZ12STD'].plot(ax=ax2)

plt.show()

"""
Another interesting visualization would be to compare the Texas HPI to the overall HPI. 
Then do a rolling correlation between the two of them. The assumption would be that when 
correlation was falling, there would soon be a reversion.

Every time correlation drops, you should in theory sell property in the are that is rising, 
and then you should buy property in the area that is falling. The idea is that, these two 
areas are so highly correlated that we can be very confident that the correlation will 
eventually return back to about 0.98. As such, when correlation is -0.5, we can be very 
confident in our decision to make this move, as the outcome can be one of the following:

HPI forever diverges like this and never returns (unlikely), the falling area rises up 
to meet the rising one, in which case we win, the rising area falls to meet the other 
falling one, in which case we made a great sale, or both move to re-converge, 
in which case we definitely won out.
"""

fig = plt.figure()
ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

AZ_AK_12corr = pd.rolling_corr(HPI_data['AZ'], HPI_data['AK'], 12)

HPI_data['AZ'].plot(ax=ax1, label="AZ HPI")
HPI_data['AK'].plot(ax=ax1, label="AK HPI")
ax1.legend(loc=4)

AZ_AK_12corr.plot(ax=ax2)

plt.show()