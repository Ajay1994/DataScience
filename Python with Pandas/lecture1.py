import pandas as pd
import datetime
import pandas.io.data as web
import matplotlib.pyplot as plt
from matplotlib import style
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 8, 22)
#create start and end variables that are datetime objects, pulling data from Jan 1st 2010 to Aug 22nd 2015
df = web.DataReader("XOM", "yahoo", start, end)
print(df.head())
style.use('fivethirtyeight')
df['High'].plot()
plt.legend()
plt.show()