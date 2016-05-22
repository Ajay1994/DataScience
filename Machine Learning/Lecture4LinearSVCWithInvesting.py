# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 12:48:14 2016

@author: Pokemon
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from matplotlib import style
style.use("ggplot")

def Build_Data_Set(features = ["DE Ratio",
                               "Trailing P/E"]):
    data_df = pd.DataFrame.from_csv("E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\yahoofinance\\key_stats.csv")
    # From there, we chop this to only include the first 100 rows of data.
    data_df = data_df[:100]

    X = np.array(data_df[features].values)

    y = (data_df["Status"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())


    return X,y
    

def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C= 1.0)
    clf.fit(X,y)
    
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx,yy, "k-", label="non weighted")

    plt.scatter(X[:, 0],X[:, 1],c=y)
    plt.ylabel("Trailing P/E")
    plt.xlabel("DE Ratio")
    plt.legend()

    plt.show()
    
Analysis()

"""
First, Debt/Equity. We can see here pretty clearly that out-performing stocks, at least in our small, 
sliced example, appear to be stocks that have low Debt/Equity. That's somewhat useful (though not as useful 
looking at today's companies), also even further un-useful for reasons discussed later (using the first 100 
rows uses on the first 100 data files, meaning we probably are not even getting out of the companies that 
start with the letter A here).

The other feature being used here is Trailing P/E, which is trailing price to earnings. Trailing, 
meaning according to older values. So what is the price, today, in accordance with earnings from the past, 
usually 12 months.

It only stands to reason that companies outperforming today are companies with a high current price compared 
to their previous earnings, as outperforming the market suggests price has indeed been rising.

"""



#############################################################################
########### With many features and Normalization ############################
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
style.use("ggplot")

FEATURES =  ['DE Ratio',
             'Trailing P/E',
             'Price/Sales',
             'Price/Book',
             'Profit Margin',
             'Operating Margin',
             'Return on Assets',
             'Return on Equity',
             'Revenue Per Share',
             'Market Cap',
             'Enterprise Value',
             'Forward P/E',
             'PEG Ratio',
             'Enterprise Value/Revenue',
             'Enterprise Value/EBITDA',
             'Revenue',
             'Gross Profit',
             'EBITDA',
             'Net Income Avl to Common ',
             'Diluted EPS',
             'Earnings Growth',
             'Revenue Growth',
             'Total Cash',
             'Total Cash Per Share',
             'Total Debt',
             'Current Ratio',
             'Book Value Per Share',
             'Cash Flow',
             'Beta',
             'Held by Insiders',
             'Held by Institutions',
             'Shares Short (as of',
             'Short Ratio',
             'Short % of Float',
             'Shares Short (prior ']

#Now we have our typical imports, and then a feature list that we intend to use.

#Next, we use a similar function as before to build our data set, with the use of preprocessing.scale():

def Build_Data_Set():
    data_df = pd.DataFrame.from_csv("key_stats.csv")
    data_df = data_df.reindex(np.random.permutation(data_df.index))
    #data_df = data_df[:100]

    X = np.array(data_df[FEATURES].values)#.tolist())

    y = (data_df["Status"]
         .replace("underperform",0)
         .replace("outperform",1)
         .values.tolist())

    X = preprocessing.scale(X)

    return X,y

#Notice that we're also commenting out the line that slices our data set.

#Now we're going to work on an analysis function:

def Analysis():

    test_size = 1000
    X, y = Build_Data_Set()
    print(len(X))

    
    clf = svm.SVC(kernel="linear", C= 1.0)
    clf.fit(X[:-test_size],y[:-test_size])

    correct_count = 0

    for x in range(1, test_size+1):
        if clf.predict(X[-x].reshape(1,-1))[0] == y[-x]:
            correct_count += 1
            print(clf.predict(X[-x].reshape(1,-1))[0], " And ",y[-x], " True")
        else:
            print(clf.predict(X[-x].reshape(1,-1))[0], " And ",y[-x], " False")
            
    #clf.predict()
    print("Accuracy:", (float(correct_count)/test_size) * 100.00)

Analysis()

"""
Our predictions are right more often than not, but, since this is an investing example, 
we're actually more curious about the "spectrum" in the end.
The goal is that our "loser" stocks that we pick are not not extremely bad. Consider for example:

Machine Learning Algorithm is 85% accurate, and chooses 100 stocks to invest in:
15 stocks perform an average -88% compared to market.
85 stocks perform an average +2% compared to market.
This equates to an average of -11.5% in the end compared to market.
(-15*0.88 + 85*0.02)

Next, instead, consider:

Machine Learning Algorthm is 52% accurate, and chooses 100 stocks to invest in:
48 stocks perform an average of -2% compared to market.
52 stocks perform an average of +2% compared to market.
This equates to an average +8% in the end compared to market.
"""


#############################################################################

#Randominzing Demo

def Randomizing():
    df = pd.DataFrame({"D1":range(5), "D2":range(5)})
    print(df)
    df2 = df.reindex(np.random.permutation(df.index))
    print(df2)


Randomizing()


#############################################################################

#Getting More and More Data

import pandas as pd
import os
from Quandl import Quandl
import time

auth_tok = open("quandlekey.txt", "r").read()
data = Quandl.get("WIKI/KO", trim_start = "2000-12-12", trim_end = "2014-12-30", authtoken=auth_tok)

print(data)


#####################################################################

df = pd.DataFrame()
path = "E:\\IIT KHARAGPUR\\Semester II\\Machine Learning\\DataSciencePython\\Dataset\\yahoofinance\\_KeyStats"

def Stock_Prices():
    statspath = path
    stock_list = [x[0] for x in os.walk(statspath)]

    print(stock_list)

    for each_dir in stock_list[1:]:
        try:
            ticker = each_dir.split("\\")[8]
            print(ticker)
            name = "WIKI/"+ticker.upper()
            data = Quandl.get(name,
                              trim_start = "2000-12-12",
                              trim_end = "2014-12-30",
                              authtoken=auth_tok)
            data[ticker.upper()] = data["Adj. Close"]
            df = pd.concat([df, data[ticker.upper()]], axis = 1)

        except Exception as e:
            print(str(e))
            time.sleep(10)
            try:
                ticker = each_dir.split("\\")[1]
                print(ticker)
                name = "WIKI/"+ticker.upper()
                data = Quandl.get(name,
                                  trim_start = "2000-12-12",
                                  trim_end = "2014-12-30",
                                  authtoken=auth_tok)
                data[ticker.upper()] = data["Adj. Close"]
                df = pd.concat([df, data[ticker.upper()]], axis = 1)
                df.to_csv("stock_prices.csv")
            except Exception as e:
                print(str(e))

    #df.to_csv("stock_prices.csv")
                
Stock_Prices()