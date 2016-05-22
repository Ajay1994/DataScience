# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 13:10:35 2016

@author: Pokemon
"""
import Quandl
import pandas as pd
import pickle
# Not necessary, I just do this so I do not show my API key.
#api_key = open('quandlapikey.txt','r').read()
fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')

main_df = pd.DataFrame()

for abbv in fiddy_states[0][0][1:]:
    #print(abbv)
    query = "FMAC/HPI_"+str(abbv)
    df = Quandl.get(query)
    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df)
        
#When it comes to something like, machine learning, for example. You generally
#train a classifier, and then you can start immediately, and quickly, classifying 
#with that classifier. The problem is, a classifer can't be saved to a .txt or .csv file. 
#It's an object. Luckily, in programming, there are various terms for the process of saving 
#binary data to a file that can be accessed later. In Python, this is called pickling.
#You may know it as serialization, or maybe even something else. Python has a module 
#called Pickle, which will convert your object to a byte stream, or the reverse with unpickling.
        
# Pickle is way to save objects of any type including the machine learning classifier to a file. Serialization
pickle_out = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\fiddy_states.pickle','wb')
pickle.dump(main_df, pickle_out)
pickle_out.close()    


# to load the pickle data back to dataframe.
pickle_in = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\fiddy_states.pickle','rb')
HPI_data = pickle.load(pickle_in)
print(HPI_data)

#read API Key
api_key = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\quandleAPIKey.txt','r').read()


# function format of writing everything
def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]
    


def grab_initial_state_data():
    states = state_list()
    limStates = states[1:10]
    main_df = pd.DataFrame()
    for abbv in limStates:
        query = "FMAC/HPI_"+str(abbv)
        #print(query)
        df = Quandl.get(query, authtoken=api_key)
        print(query)
        if main_df.empty:
            main_df = df
            print("Hello")
        else:
            main_df = pd.concat([main_df, df], axis=1, join='inner')
            print("World")
    return main_df
    
main_df = grab_initial_state_data()

pickle_out = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\fiddy_states.pickle','wb')
pickle.dump(main_df, pickle_out)
pickle_out.close() 

# give the shape of the dataframe i.e, Dimensions
HPI_data.shape
#gives the no of the rows in the dataframe
len(HPI_data)

Count_Row=HPI_data.shape[0] #gives number of row count
Count_Col=HPI_data.shape[1] #gives number of col count

# added a new column TX2 to the original dataframe
HPI_data['TX2'] = HPI_data['Value'] * 2


#####################################################
# Plotting Some plots
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

main_df.plot()
plt.legend().remove()
plt.show()


##############################
df = df[1:5]
##############################
def HPI_Benchmark():
    df = Quandl.get("FMAC/HPI_USA")
    #We are starting the percentage increase from 0 instead of initial seed.
    #percentage increase in current value of the dataframe from the previous old value of df["Value"][0]
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    return df
    
stats = HPI_Benchmark()



###############################
#Plotting with the benchmark to show the comparison
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

main_df.plot(ax=ax1)
stats.plot(color='k',ax=ax1, linewidth=10)

plt.legend().remove()
plt.show()


###################################################
#If you want percent change in data frame 
query = "FMAC/HPI_"+str(abbv)
df = Quandl.get(query, authtoken=api_key)
print(query)
print(df.head())


####################################################
#How much every data is correlated with others (N*N) matrix
HPI_State_Correlation = main_df.corr()
#For every column of the co-relation matrix it gives the details like count of data der, mean of col
#what value in 25%- 1st quartile etc.
#min implies that one state A is 0.88 corelated with  B, ie. we have very correlated distribution
print(HPI_State_Correlation.describe())
print(HPI_State_Correlation)
HPI_State_Correlation.plot()
plt.show()
