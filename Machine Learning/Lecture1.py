# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 22:49:15 2016

@author: Pokemon
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# we're defining the digits variable, which is the loaded digit dataset.
digits = datasets.load_digits()

#you can reference the data's features and labels: 
#digits.data is the actual data (features).
print(digits.data)

#digits.target is the actual label we've assigned to the digits data.
print(digits.target)

len(digits.target) #1797

#Now that we've got the data ready, we're ready to do the machine learning.
# First, we specify the classifier:

clf = svm.SVC(gamma=0.001, C=100)

#With that done, now we're ready to train. It's best for clarity to go ahead and assign the value 
#into X (uppercase) and y.
#This loads in all but the last 10 data points, so we can use all of these for training. 
#Then, we can use the last 10 data points for testing. 
X,y = digits.data[:-10], digits.target[:-10]

#Next we train with:
clf.fit(X,y)

#This will predict what the 5th from the last element is.
print(clf.predict(digits.data[-5]))

#This just shows us an image of the number in question. 
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
