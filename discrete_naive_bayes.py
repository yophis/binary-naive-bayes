import numpy as np


class DiscreteNaiveBayes:
    """Naive Bayes binary classifier that fits data in discretized fashion."""
    def __init__(self, prior1=None, prior0=None, bin_counts=10):
        self.trained = False
        self.prior1 = prior1
        self.prior0 = prior0
        self.bin_counts = bin_counts

    def discretize(self, x, y):
        """Discretize the given feature, with bin edges set up manually."""
        x, y = x[np.isfinite(x)], y[np.isfinite(x)]  # drop NaN and inf
        min_, max_ = np.nanmin(x), np.nanmax(x)
        bin_width = (max_-min_) / self.bin_counts
        
        bin_edges = [-np.inf] + [i*bin_width + min_ for i in range(1, self.bin_counts)] + [np.inf]
        bin_edges = np.array(bin_edges)

        bins1 = np.bincount(np.digitize(x[y == 1], bin_edges), minlength=self.bin_counts+1)
        bins0 = np.bincount(np.digitize(x[y == 0], bin_edges), minlength=self.bin_counts+1)

        return bins1[1:], bins0[1:], bin_edges
        
    @staticmethod
    def discretize2(self, x, y):
        """Discretize the given feature, with bin edges automatically determined by np.histogram."""
        x, y = x[np.isfinite(x)], y[np.isfinite(x)]  # drop NaN and inf
        _, bin_edges = np.histogram(x, bins=self.bin_counts)
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf

        bins1 = np.bincount(np.digitize(x[y == 1], bin_edges), minlength=self.bin_counts+1)
        bins0 = np.bincount(np.digitize(x[y == 0], bin_edges), minlength=self.bin_counts+1)

        return bins1[1:], bins0[1:], bin_edges

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if self.prior1 is None:
            self.prior1 = y.sum() / len(y)
        if self.prior0 is None:
            self.prior0 = 1 - self.prior1

        leaves, stays, edges = [], [], []
        for i in range(X.shape[1]):
            feat = X[:, i]
            bins1, bins0, bin_edges = self.discretize2(feat, y, bin_counts=self.bin_counts)
            leaves.append(bins1)
            stays.append(bins0)
            edges.append(bin_edges)
        leaves = np.array(leaves)
        stays = np.array(stays)

        # convert to log likelihood
        with np.errstate(divide='ignore'):
            log_leaves = np.log(leaves / leaves.sum(axis=1, keepdims=True))
            log_stays = np.log(stays / stays.sum(axis=1, keepdims=True))
        
        # super duper classy flooring
        log_leaves[~np.isfinite(log_leaves)] = 1e-2
        log_stays[~np.isfinite(log_stays)] = 1e-2

        # append 0 to each bins for handling NaN
        # as np.digitize will put NaN after the last bin
        self.log_leaves = np.concatenate([log_leaves, np.zeros((log_leaves.shape[0], 1))], axis=1)
        self.log_stays = np.concatenate([log_stays, np.zeros((log_stays.shape[0], 1))], axis=1)
        self.edges = np.array(edges)
        self.trained = True

    def predict(self, X, thresh=0.0):
        assert self.trained, 'The model has to be trained via fit method first.'

        X = np.array(X)
        result = np.array(np.log(self.prior1) - np.log(self.prior0)).repeat(X.shape[0])
        for i, edge in enumerate(self.edges):
            index_ = np.digitize(X[:, i], edge) - 1
            result += self.log_leaves[i, index_] - self.log_stays[i, index_]
        
        result = np.where(result > thresh, 1, 0)
        return result