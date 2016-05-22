# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 23:21:53 2016
@author: Pokemon
"""
import Quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
style.use('fivethirtyeight')

#read API Key
api_key = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\quandleAPIKey.txt','r').read()

def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]

states = state_list()

##################################################################
# Get HPI Data
df = Quandl.get("FMAC/HPI", authtoken="xCPtxkiFSsoYyE17A2uw")

main_df = pd.DataFrame()

for abbv in states:
    p = pd.DataFrame((df[abbv]-df[abbv][0]) / df[abbv][0] * 100.0)
    if main_df.empty:
        main_df = p
    else:
        main_df = main_df.join(p)

main_df.plot()
plt.legend().remove()
plt.show()

################################################################
pickle_out = open('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\sp500.pickle','wb')
pickle.dump(sp500, pickle_out)
pickle_out.close()   
################################################################

def HPI_Benchmark():
    df = Quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    return df

################################################################

def mortgage_30y():
    df = Quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    #Simply doing by M will lead NaN since we are having only one day data for that particular month and this wont add any information. 
    #We need more data point for the sampling, i.e more information about the folloing date of month. We can hack to subsample on day basis and we wl get repeated value
    #in columns then we can easily sample on the month basis.
    df=df.resample('1D')
    df=df.resample('M')
    print(df.head())
    return df


################################################################

HPI_data = pd.read_pickle('E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\fiddy_statesnew.pickle')

m30 = mortgage_30y()
HPI_Bench = HPI_Benchmark()
HPI_Bench.columns = ['United States']
m30.columns=['M30']
HPI = HPI_Bench.join(m30)
print(HPI.head())


#That's fairly expected. -0.74 is pretty strongly negative. Obviously not as beautifully 
#aligned as the various states were usually to eachother. This suggest that HPI is negative correlated with the 
#mortage rate.
print(HPI.corr())
################################################################
#-0.74 is pretty strongly negative. Obviously not as beautifully aligned as 
#the various states were usually to eachother, but this is still obviously a useful metric.
print(HPI.corr())

#look into the 30 year conventional mortgage rate. Now, this data should be very negatively 
#correlated with the House Price Index (HPI). Before even bothering with this code, I would 
#automatically assume and expect that the correlation wont be as negatively strong as the 
#higher-than-90% that we were getting with state HPI correlation, certainly less than -0.9, 
#but also it should be greater than -0.5. The interest rate is of course important, but correlation 
#to the overall HPI was so very strong because these were very similar statistics. The interest rate
# is of course related, but not as directly as other HPI values.

state_HPI_M30 = HPI_data.join(m30)
print(state_HPI_M30.corr())
#-ve corelation mean if mortage is less than HPI will be more but this negative co-relation is not that strong
print(state_HPI_M30.corr()['M30'])
print(state_HPI_M30.corr()['M30'].describe())


###################################################################
#Lecture 14

#There are two major economic indicators that come to mind out the gate: S&P 500 index (stock market) and GDP
# (Gross Domestic Product). I suspect the S&P 500 to be more correlated than the GDP, but the GDP is usually 
#a better overall economic indicator, so I may be wrong. Another macro indicator that I suspect might have 
#value here is the unemployment rate. If you're unemployed, you're probably not getting that mortgage. 

def sp500_data():
    df = Quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
    df["Adj Close"] = (df["Adj Close"]-df["Adj Close"][0]) / df["Adj Close"][0] * 100.0
    df=df.resample('M')
    df.rename(columns={'Adj Close':'sp500'}, inplace=True)
    df = df['sp500']
    return df

def gdp_data():
    df = Quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M')
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = Quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D')
    df=df.resample('M')
    return df
    
HPI_data = pd.read_pickle('fiddy_states3.pickle')
m30 = mortgage_30y()
sp500 = sp500_data()
#More money states bring in. More money for people to buy houses and more increase in housing prices.
gdp = gdp_data()
HPI_Bench = HPI_Benchmark()
unemployment = us_unemployment()
m30.columns=['M30']

HPI = HPI_data.join([HPI_Bench,m30,sp500,gdp,unemployment])
HPI.dropna(inplace=True)
print(HPI.corr())


############################################################################
HPI.to_pickle('HPI.pickle') # here we have all the data and we can add more in case we need . We can load this pickele add more data nd dump it again
############################################################################

############################################################################
                    #Machine Learing Classification#
############################################################################
#We compare current month HPI with the next two or three month HPI and if went down 
#we call it 0 and if it went up we call it 1. Then we can give it to a machine learing classifier
#and say these are my features like GDP and unemployment and what will be my HPI next month. We want a good accuracy.
#We need to generate that label.

#Read the pickle from the saved 
housing_data = pd.read_pickle('HPI.pickle')

#when we investing we dont care about the percentage change from
#beginning. We interested in percent change now and next 2-3 month frame

# we're going to lead in the dataset, and then convert all columns to percent change.
#This will help us to normalize all of the data.
housing_data = housing_data.pct_change()

# remove the infinitives and NaNs from the dataset.
import numpy as np
housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_data.dropna(inplace=True)

#we create a new column, which contains the future HPI. We can do this with a new method: .shift(). 
#This method will shift the column in question. Shifting by -1 means we're shifting down,
# so the value for the next point is moved back. This is our quick way of having the current value, and 
#the next period's value on the same row for easy comparison.

housing_data['US_HPI_future'] = housing_data['United States'].shift(-1)

print(housing_data[['US_HPI_future', 'United States']]).head()
housing_data.dropna(inplace=True)


def create_lebels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1 #Market is good and we can invest
    else:
        return 0


#let's show a custom way to apply a moving-window function. 
#We're going to just do a simple moving average example:
#notice that we just pass the "values" parameter. We do not need to code any 
#sort of "window" or "time-frame" handling, Pandas will handle that for us.
def moving_average(values):
    return mean(values) 

       
#Note we have pretty high bias becoz HPI mostly go up in graph.
#create labels and map it to the list
housing_data['label'] = list(map(create_lebels, housing_data['United States'], housing_data['US_HPI_future'] ))

#Rolling example usage
print(pd.rolling_apply(housing_data['M30'], 10, moving_average))

#-------------------------------------------------------------
# Machine Learning Application classifier
#--------------------------------------------------------------     

from sklearn import svm, preprocessing, cross_validation

#preprocessing - Preprocessing is used to adjust our dataset. Typically, machine learning will be a bit more accurate if your
# features are between -1 and 1 values. This does not mean this will always be true, always a good
# idea to check with and without the scaling that we'll do to be safe.

#Cross validation - Preprocessing is used to adjust our dataset. Typically, machine learning will be a bit more accurate if your features are between -1 and 1 values. 
#This does not mean this will always be true, always a good idea to check with and without the scaling that we'll do to be safe.
    
#Now, we can create our features and our labels for training/testing:

#return data frame without these columns
#define the featureset as the numpy array (this just converts the
# dataframe's contents to a multi-dimensional array) of the housing_data
# dataframe's contents, with the "label" and the "US_HPI_future" columns removed.

#Generally, with features and labels, you have X, y. The uppercase X is used to denote a feature set. The y is the label.
X = np.array(housing_data.drop(['label','US_HPI_future'], 1))
#converts data to hopeful range of -1 to +1
X = preprocessing.scale(X)

y = np.array(housing_data['label'])

# We're ready to split up our data into training and testing sets.
# We can do this ourselves, but we'll use the cross_validation import from earlier.

# this does is it splits up your features (X) and labels (y) into random training and testing groups for you.
# As you can see, the return is the feature training set, feature testing set, labels training, and labels 
#testing. We then are unpacking these into X_train, X_test, y_train, y_test. cross_validation.train_test_split
# takes the parameters of your features, labels, and then you can also specify the testing size (test_size), 
#which we've designated to be 0.2, meaning 20%.

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# going to use support vector classifcation with a linear kernel in this example
#establish the classifier that we intend to use
clf = svm.SVC(kernel = 'linear')

#Next, we want to train our classifier:
clf.fit(X_train, y_train)

#we could actually go ahead and make predictions from here, but let's test the 
#classifier's accuracy on known data:

print(clf.score(X_test, y_test))

clf.predict(X_test)




