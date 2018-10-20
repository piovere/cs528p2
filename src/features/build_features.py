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
        raise NotImplementedError
    
    def fit(self, x):
        raise NotImplementedError
