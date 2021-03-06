"""Load data, sort to numerical information, and PCA the data
"""
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


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
        self.scaler_ = StandardScaler()
        self.error_vs_conv_ = []
    
    def fit(self, data_unscaled, k=3, num_starts=10):
        data = self.scaler_.fit_transform(data_unscaled)
        self.num_starts_ = num_starts
        self.k_ = k
        # Pick `k` initial seeds
        # self.seeds_ = self.pick_random_seeds(data)
        self.seeds_ = self.initial_seeds(data)

        # Make labels for initial guess
        labels = self.make_labels(data)

        # Calculate error from intial guess
        e = self.error(data, labels)
        self.error_vs_conv_.append(e)
        self.error_ = e

        for i in range(num_starts):
            seeds = self.initial_seeds(data)
            conv_count = 0
            old_seeds = np.zeros_like(seeds)
            while not(np.allclose(seeds, old_seeds)):
                conv_count += 1
                old_seeds = np.copy(seeds)
                labels = self.make_labels(data, seeds)
                seeds = self.new_seeds(data, labels)
            # See if the new seeds are better
            e = self.error(data, labels, seeds)
            
            if e < self.error_:
                self.seeds_ = seeds
                self.error_ = self.error(data, labels)
                self.best_seed_ = i
                self.conv_count_ = conv_count

            self.error_vs_conv_.append(self.error_)
    
    def predict(self, data_unscaled):
        data = self.scaler_.transform(data_unscaled)
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
        inds = [np.ravel(np.argwhere(labels == _)) for _ in range(k)]
        new_seeds = [np.mean(data[i, :], axis=0) for i in inds]

        return np.array(new_seeds)
    
    def error(self, data, labels, seeds=None):
        if seeds is None:
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

    def intercluster_distance(self):
        us_seeds = self.scaler_.inverse_transform(self.seeds_)
        d = [[la.norm(s1 - s2) for s1 in us_seeds] for s2 in us_seeds]
        d = np.array(d)
        return d

    def min_intercluster_distance(self):
        if self.k_ == 1:
            raise ValueError("Can't calculate intercluster distance "
                             "with one cluster")
        d = self.intercluster_distance()
        d = d[np.nonzero(d)]
        return np.min(d)
        
    def intracluster_distance(self, data_unscaled):
        labels = self.predict(data_unscaled)
        intracluster_distances = []
        for k in range(self.k_):
            inds = np.argwhere(labels == k)[0][0]
            # print(inds)
            s = self.scaler_.inverse_transform(self.seeds_[k].reshape(1, -1))
            ds = la.norm(data_unscaled[inds] - s)
            intracluster_distances.append(np.max(ds))
        return np.array(intracluster_distances)
    
    def max_intracluster_distance(self, data_unscaled):
        d = self.intracluster_distance(data_unscaled)
        return np.max(d)
    
    def dunn_index(self, data_unscaled):
        inter = self.min_intercluster_distance()
        intra = self.max_intracluster_distance(data_unscaled)
        return inter / intra

    def initial_seeds(self, data):
        # Pick one initial seed from the data
        seed_ind = np.random.choice(
            np.arange(data.shape[0]),
            size=1
        )
        seeds = data[seed_ind]

        seed_count = 1

        while seed_count < self.k_:
            # Calculate the distance from this seed to all rows
            ds = self.distances(data, seeds) ** 2
            # Calculate the distance from each point to the closest seed
            min_ds = np.min(ds, axis=1)
            min_ds /= np.sum(min_ds)
            # Pick the row with the greatest distance
            # ns = data[np.argmax(min_ds)].reshape(1, -1)
            new_seed_ind = np.random.choice(
                np.arange(data.shape[0]),
                p=min_ds
            )
            ns = data[new_seed_ind, :].reshape(1, -1)
            seeds = np.vstack([seeds, ns])
            # Increment seed count
            seed_count += 1
        
        return seeds
    
    def plot_km(self, data_unscaled, cols=[0, 1]):
        labels = self.predict(data_unscaled)
        x = data_unscaled[:, cols[0]]
        y = data_unscaled[:, cols[1]]
        sns.scatterplot(x=x, y=y, hue=labels)
        plt.plot(
            self.scaler_.inverse_transform(self.seeds_)[:, cols[0]],
            self.scaler_.inverse_transform(self.seeds_)[:, cols[1]],
            "x"
        )
