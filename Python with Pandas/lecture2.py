import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
web_stats = {'Day':[1,2,3,4,5,6],
             'Visitors':[43,34,65,56,29,76],
             'Bounce Rate':[65,67,78,65,45,52]}

df = pd.DataFrame(web_stats)
print(df.head())
print(df.tail())

#Set day as the index of the dataframe
df.set_index('Day', inplace=True)
print(df.Visitors)
print(df['visitors'])

print(df[['Visitors', 'Bounce Rate']])

print(df.Visitors.tolist())

print(np.array(df[['Visitors', 'Bounce Rate']]))

#to get a better stylish plots
style.use('fivethirtyeight')

df['Visitors'].plot()
plt.show()

df.plot()
plt.show()