"""Load data, sort to numerical information, and PCA the data
"""
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import StandardScaler


class PCA():
    def __init__(self, variance=None):
        self.scaler = None
        self.v = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.variance_frac = variance
    
    def fit(self, x):
        # Scale the data
        self.scaler = StandardScaler().fit(x)
        xs = self.scaler.transform(x)

        # Calculate svd
        s, vt = la.svd(xs, full_matrices=False)[1:]
        self.v = vt.T

        if self.variance_frac is not None:
            evr = s**2 / np.sum(s**2)
            cum_explained_var = np.cumsum(evr)
            inc = np.where(cum_explained_var < 0.95)
            ind = np.max(inc) + 2
            self.v = self.v[:, :ind]
        else:
            ind = s.shape[0]

        # Save the explained variance
        var = s ** 2
        self.explained_variance_ = var[:ind]
        self.explained_variance_ratio_ = var[:ind] / np.sum(var)

        return self
    
    def transform(self, x):
        # Scale the data from the previous values
        xs = self.scaler.transform(x)

        return xs @ self.v
    
    def fit_transform(self, x):
        self.fit(x)
        self.transform(x)

    def to_explain_variance_frac(self, f):
        cum_explained_var = np.cumsum(self.explained_variance_ratio_)
        inc = np.where(cum_explained_var < 0.95)
        return np.max(inc) + 1


class KMeans():
    def __init__(self):
        self.seeds_ = None
        self.k_ = None
        self.error_ = None
        self.conv_count_ = None
        self.best_seed_ = None
        self.num_starts_ = None
    
    def fit(self, data, k=3, num_starts=10):
        self.num_starts_ = num_starts
        self.k_ = k
        # Pick `k` initial seeds
        self.seeds_ = self.pick_random_seeds(data)

        # Make labels for initial guess
        labels = self.make_labels(data)

        # Calculate error from intial guess
        self.error_ = self.error(data, labels)

        for i in range(num_starts):
            seeds = self.pick_random_seeds(data)
            conv_count = 0
            old_seeds = np.zeros_like(seeds)
            while not(np.allclose(seeds, old_seeds)):
                conv_count += 1
                old_seeds = np.copy(seeds)
                labels = self.make_labels(data, seeds)
                seeds = self.new_seeds(data, labels)
            # See if the new seeds are better
            if self.error(data, labels) < self.error_:
                self.seeds_ = seeds
                self.error_ = self.error(data, labels)
                self.best_seed_ = i
                self.conv_count_ = conv_count
    
    def predict(self, data):
        l = self.make_labels(data)
        return l
    
    def distances(self, data, seeds=None):
        if seeds is None:
            seeds = self.seeds_
        ds = [[la.norm(s - d) for s in seeds] for d in data]
        return np.array(ds)
    
    def make_labels(self, data, seeds=None):
        if seeds is None:
            seeds = self.seeds_
        ds = self.distances(data, seeds)
        labels = np.argmin(ds, axis=1)
        return labels
    
    def new_seeds(self, data, labels):
        k = self.k_
        inds = [np.argwhere(labels == _) for _ in range(k)]
        new_seeds = [np.mean(data[i, :], axis=0) for i in inds]
        return np.array(new_seeds)
    
    def error(self, data, labels):
        seeds = self.seeds_
        e = 0
        for _ in range(labels.shape[0]):
            l = labels[_]
            e += la.norm(data[_] - seeds[l]) ** 2
        return e
    
    def pick_random_seeds(self, data):
        seed_ind = np.random.choice(
            np.arange(data.shape[0]),
            self.k_,
            replace=False
        )
        seeds = data[seed_ind, :]
        return seeds
