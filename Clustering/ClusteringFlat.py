# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:03:56 2016

@author: Pokemon

Flat Clustering : Flat clustering is where the scientist tells the machine how many categories to 
cluster the data into.

Hierarchical : Hierarchical clustering is where the machine is allowed to decide how many clusters
to create based on its own algorithms.

K-Means approaches the problem by finding similar means, repeatedly trying to find centroids that match 
with the least variance in groups

This repeatedly trying ends up leaving this algorithm with fairly poor performance, though performance
is an issue with all machine learning algorithms. This is why it is usually suggested that you use a 
highly stream-lined and efficient algorithm that is already tested heavily rather than creating your own.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()

####################################################
X = np.array([[1, 2],
              [5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11]])
              
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)


colors = ["g.","r.","c.","y."]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()
		