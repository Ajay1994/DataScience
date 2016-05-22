# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:22:19 2016

@author: Pokemon

Since we're using Scikit-learn here, we are using Ward's Method, which works by measuring
degrees of minimum variance to create clusters.

The specific algorithm that we're going to use here is Mean Shift.
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

"""
NumPy for the swift number crunching, then, from the clustering algorithms of scikit-learn, 
we import MeanShift.

We're going to be using the sample generator built into sklearn to create a dataset for us 
here, called make_blobs.


we're making our example data set. We've decided to make a dataset that originates from three 
center-points. One at 1,1, another at 5,5 and the other at 3,10. From here, we generate the sample,
unpacking to X and y. X is the dataset, and y is the label of each data point according to the sample 
generation.


What we have now is an example data set, with 500 randomized samples around the center points with 
a standard deviation of 1 for now.
"""

centers = [[1,1],[5,5],[3,10]]
X, _ = make_blobs(n_samples = 500, centers = centers, cluster_std = 1)

#The more samples you have, and the less standard deviation, the more accurate your predicted cluster 
#centers should be compared to the actual ones used to generate the randomized data.
 
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

plt.scatter(X[:,0],X[:,1])
plt.show()
	
 
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

#This is just a simple list of red, green, blue, cyan, black, yellow, and magenta multiplied by ten.
# We should be confident that we're only going to need three colors, but, with hierarchical clustering, 
#we are allowing the machine to choose, we'd like to have plenty of options. 
#This allows for 70 clusters, so that should be good enough.

colors = 10*['r.','g.','b.','c.','k.','y.','m.']

#Above, first we're iterating through all of the sample data points, plotting their coordinates,
#and coloring by their label # as an index value in our color list.

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)

plt.show()
