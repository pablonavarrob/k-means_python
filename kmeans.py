import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import glob
import imageio

# Define the number k of clusters
K = 3

# Create the test data set
x, _ = make_blobs(n_samples=500, n_features=2, cluster_std = 2.5)

# Generate random centers

# In this test I tried to add a new line to test the branching capabilities of Github so I can learn a bit more about it to work in Procure AI as an intern, yai!

# Define the function for the euclidean distance
def get_euclidean_distance(x, y):
    """ Returns the euclidean distance for a pair of arrays of the same
    dimensions. """
    d = np.sum((x-y)**2, axis=1)
    return np.sqrt(d)

def k_means(K, data, iterations=20):
    """ K-means with random initialization. """
    c = np.random.uniform(-5, 5, (K, 2))
    clusters = np.zeros(len(x), dtype=int)
    hist_centers = np.zeros((iterations, K, 2))
    hist_labels = np.zeros((iterations, len(x)))

    for iteration in range(iterations):
        for i in range(len(x)):
            # for each point, compute distance to centers
            dist_to_centers = get_euclidean_distance(x[i], c)
            # find the cluster with the minimum distance
            idx_cluster = np.where(dist_to_centers == np.min(dist_to_centers))[0][0]
            clusters[i] = idx_cluster

        # with that info, need to update the positions of the centers to the mean
        # of the defined cluster
        for j in range(K): # slow down the convergence
            c[j] = c[j]*0.85 + 0.15*np.mean(x[clusters == j], axis=0)

        # Plot
        fig, ax = plt.subplots(1, figsize=[10, 10])
        ax.scatter(data[:,0], data[:,1], s=4, c=clusters)
        ax.scatter(c[:,0], c[:,1], s=20, c='k')
        plt.savefig('{}.jpg'.format(iteration), dpi=100)

        hist_labels[iteration] = clusters
        hist_centers[iteration] = c

    return hist_centers, hist_labels
