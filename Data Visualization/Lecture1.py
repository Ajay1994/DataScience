# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:37:24 2016

@author: Pokemon
"""

import matplotlib.pyplot as plt

x = [1,2,3]
y = [5,7,4]

#plt.plot([1,2,3],[5,7,4])

plt.plot(x,y)

plt.xlabel("Plot Number")
plt.ylabel("Important Var")

plt.title("Graph\n Check it out")

plt.show()

"""
Next, we invoke the .plot method of pyplot to plot some coordinates. This .plot takes many parameters, 
but the first two here are 'x' and 'y' coordinates, which we've placed lists into. This means, we have 
3 coordinates according to these lists: 1,5 2,7 and 3,4.

The plt.plot will "draw" this plot in the background, but we need to bring it to the screen when we're 
ready, after graphing everything we intend to.
"""

#############################################################
######################### Legends ###########################
#############################################################

import matplotlib.pyplot as plt

x = [1,2,3]
y = [5,7,4]

x2 = [1,2,3]
y2 = [10,14,12]

#we add another parameter "label." This allows us to assign a name to the line, 
#which we can later show in the legend. 

plt.plot(x, y, label='First Line')
plt.plot(x2, y2, label='Second Line')


plt.xlabel('Plot Number')
plt.ylabel('Important var')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


#############################################################
############## Bar Charts and Histograms ####################
#############################################################

import matplotlib.pyplot as plt

plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Example one")

plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Example two", color='g')

plt.legend()

plt.xlabel('bar number')
plt.ylabel('bar height')

plt.title('Epic Graph\nAnother Line! Whoa')

plt.show()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#Histograms are mainly for showing the distribution of the data 
population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

#############################################################
################ Scatter Plots with Data ####################
#############################################################

#The idea of scatter plots is usually to compare two variables, or three if you are
#plotting in 3 dimensions, looking for correlation or groups.

#The plt.scatter allows us to not only plot on x and y, but it also lets us decide
#on the color, size, and type of marker we use.

# We can compare Age vs Cancer 
x = [1,2,3,4,5,6,7,8]
y = [5,2,4,2,1,4,5,2]

plt.scatter(x,y, label='skitscat', color='k', s=25, marker="o")
#S : denotes the size

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

#############################################################
################ Stackr Plots with Data #####################
#############################################################

#The idea of stack plots is to show "parts to the whole" over time. A stack plot is 
#basically like a pie-chart, only over time.

#Let's consider a situation where we have 24 hours in a day, and we'd like to see how
#we're spending out time. We'll divide our activities into: Sleeping, eating, working, and playing.

import matplotlib.pyplot as plt

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating =   [2,3,4,3,2]
working =  [7,8,7,2,2]
playing =  [8,5,7,8,13]

"""
 The problem is, we don't really know which color is which without looking back at the code. 
 The next problem is, with polygons, we cannot actually have "labels" for the data. So anywhere
 where there is more than just a line, with things like fills or stackplots like this, we cannot 
 label the specific parts inherently.
 
"""


plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])

plt.plot([],[],color='m', label='Sleeping', linewidth=5) #Plots label with the null data
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


#############################################################
################### Pie Plots with Data #####################
#############################################################

"""
Pie charts are a lot like the stack plots, only they are for a certain point in time. 
Typically, a Pie Chart is used to show parts to the whole, and often a % share.
"""

import matplotlib.pyplot as plt

slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']

plt.pie(slices,             #Distribution of the data
        labels=activities,  #what labels data is distributed
        colors=cols,        #Colors assigned in plot to activities
        startangle=90,      #At what angle the plot should start
        shadow= True,       #should we give shadow to plot
        explode=(0,0.1,0,0),#Explode the second activity to draw attention
        autopct='%1.1f%%')  #Add percentage to the plot

plt.title('Interesting Graph\nCheck it out')
plt.show()


###############################################################
################### Loading Data From File ####################
###############################################################

import matplotlib.pyplot as plt
import numpy as np

#Separate data into two variables separated by the comma

x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)
plt.plot(x,y, label='Loaded from file!')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()


####################################################################
################### Loading Data From Internet #####################
####################################################################

import matplotlib.pyplot as plt
import numpy as np
import urllib2
import matplotlib.dates as mdates

def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter
    
def graph_data(stock):

    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'

    source_code = urllib2.urlopen(stock_price_url).read().decode()

    stock_data = []
    split_source = source_code.split('\n')

    for line in split_source:
        split_line = line.split(',')
        if len(split_line) == 6:
            if 'values' not in line:
                stock_data.append(line)
                
    date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
                                                          delimiter=',',
                                                          unpack=True,
                                                          # %Y = full year. 2015
                                                          # %y = partial year 15
                                                          # %m = number month
                                                          # %d = number day
                                                          # %H = hours
                                                          # %M = minutes
                                                          # %S = seconds
                                                          # 12-06-2014
                                                          # %m-%d-%Y
                                                          converters={0: bytespdate2num('%Y%m%d')})
                                                          
    """
    What we are doing here, is unpacking these six elements to six variables, with numpy's loadtxt
    function. The first parameter here is stock_data, which is the data we're loading. Then, 
    we specify the delimiter, which is a comma in this case, then we specify that we indeed 
    want to unpack the variables here not just to one variable, but to this group of variables 
    we've defined. Finally, we use the optional "converters" parameter to specify what element we 
    want to convert (0), and then how we want to do that. We pass a function called bytespdate2num, 
    which doesn't quite exist yet, but we'll write that next.
    """       
    plt.plot_date(date, closep,'-', label='Price')
    # - means it is a solid line
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()    
    

graph_data('TSLA')


####################################################################
################### Loading Data and Graph Mod #####################
####################################################################
def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter
    

def graph_data(stock):

    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
    source_code = urllib2.urlopen(stock_price_url).read().decode()
    stock_data = []
    split_source = source_code.split('\n')
    for line in split_source:
        split_line = line.split(',')
        if len(split_line) == 6:
            if 'values' not in line and 'labels' not in line:
                stock_data.append(line)

    date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
                                                          delimiter=',',
                                                          unpack=True,
                                                          converters={0: bytespdate2num('%Y%m%d')})

    ax1.plot_date(date, closep,'-', label='Price')
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax1.grid(True)#, color='g', linestyle='-', linewidth=5)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.show()


graph_data('TSLA')


"""
date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
                                                          delimiter=',',
                                                          unpack=True)
    dateconv = np.vectorize(dt.datetime.fromtimestamp)
    date = dateconv(date)
"""

                                      

