import numpy as np
import scipy


class GaussianNaiveBayes:
    """Naive Bayes binary classifier that estimates data with normal distribution"""
    def __init__(self, prior1=None, prior0=None):
        self.trained = False
        self.prior1 = prior1
        self.prior0 = prior0

    def fit(self, X, y):
        if self.prior1 is None:
            self.prior1 = y.sum() / len(y)
        if self.prior0 is None:
            self.prior0 = 1 - self.prior1

        self.means1 = X[y==1].mean(axis=0)
        self.sds1 = X[y==1].std(axis=0)
        self.means0 = X[y==0].mean(axis=0)
        self.sds0 = X[y==0].std(axis=0)
        self.trained = True

    def predict(self, X, thresh=0.0):
        assert self.trained, 'The model has to be trained via fit method first.'

        X = np.array(X)
        result = np.array(np.log(self.prior1) - np.log(self.prior0)).repeat(X.shape[0])
        for i in range(X.shape[1]):
            feat = X[:, i]
            m1, sd1 = self.means1[i], self.sds1[i]
            m0, sd0 = self.means0[i], self.sds0[i]

            # get log likelihoods
            with np.errstate(invalid='ignore'):
                l1 = np.log(scipy.stats.norm(m1, sd1).pdf(feat))
                l0 = np.log(scipy.stats.norm(m0, sd0).pdf(feat))

            # exclude pairs with NaN from comparison
            mask = np.isnan(l1) | np.isnan(l0)
            l1[mask] = 0
            l0[mask] = 0
            
            result += l1 - l0

        result = np.where(result > thresh, 1, 0)
        return result