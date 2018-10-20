"""Load data, sort to numerical information, and PCA the data
"""
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import StandardScaler


class PCA():
    def __init__(self):
        self.scaler = None
        self.v = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, x):
        # Scale the data
        self.scaler = StandardScaler().fit(x)
        xs = self.scaler.transform(x)

        # Calculate svd
        s, vt = la.svd(xs, full_matrices=False)[1:]
        self.v = vt.T

        # Save the explained variance
        var = s ** 2
        self.explained_variance_ = var
        self.explained_variance_ratio_ = var / np.sum(var)

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
