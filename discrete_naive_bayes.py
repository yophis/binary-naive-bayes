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
        
    def discretize2(self, x, y):
        """Discretize the given feature, with bin edges automatically determined by np.histogram."""
        x, y = x[np.isfinite(x)], y[np.isfinite(x)]  # drop NaN and inf
        _, bin_edges = np.histogram(x, bins=self.bin_counts)
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf

        bin1 = np.bincount(np.digitize(x[y == 1], bin_edges), minlength=self.bin_counts+1)
        bin0 = np.bincount(np.digitize(x[y == 0], bin_edges), minlength=self.bin_counts+1)

        return bin1[1:], bin0[1:], bin_edges

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if self.prior1 is None:
            self.prior1 = y.sum() / len(y)
        if self.prior0 is None:
            self.prior0 = 1 - self.prior1

        bins1, bins0, edges = [], [], []
        for i in range(X.shape[1]):
            feat = X[:, i]
            bin1, bin0, bin_edges = self.discretize2(feat, y)
            bins1.append(bin1)
            bins0.append(bin0)
            edges.append(bin_edges)
        bins1 = np.array(bins1)
        bins0 = np.array(bins0)

        # convert to log likelihood
        with np.errstate(divide='ignore'):
            log_bins1 = np.log(bins1 / bins1.sum(axis=1, keepdims=True))
            log_bins0 = np.log(bins0 / bins0.sum(axis=1, keepdims=True))
        
        # super duper classy flooring
        log_bins1[~np.isfinite(log_bins1)] = 1e-2
        log_bins0[~np.isfinite(log_bins0)] = 1e-2

        # append 0 to each bins for handling NaN
        # as np.digitize will put NaN after the last bin
        self.log_bins1 = np.concatenate([log_bins1, np.zeros((log_bins1.shape[0], 1))], axis=1)
        self.log_bins0 = np.concatenate([log_bins0, np.zeros((log_bins0.shape[0], 1))], axis=1)
        self.edges = np.array(edges)
        self.trained = True

    def predict(self, X, thresh=0.0):
        assert self.trained, 'The model has to be trained via fit method first.'
        assert X.shape[1] == self.log_bins1.shape[0], 'The input shape mismatch the trained parameters.'

        X = np.array(X)
        result = np.array(np.log(self.prior1) - np.log(self.prior0)).repeat(X.shape[0])
        for i, edge in enumerate(self.edges):
            index_ = np.digitize(X[:, i], edge) - 1
            result += self.log_bins1[i, index_] - self.log_bins0[i, index_]
        
        result = np.where(result > thresh, 1, 0)
        return result