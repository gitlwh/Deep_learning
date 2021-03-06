import numpy as np
import matplotlib.pyplot as plt
import math
import time
class Perceptron(object):
    """Perceptron classifer.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter :  int
        Passes over the training dataset.

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in ever epoch.

    """
    def __init__(self, eta=0.01,  n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        :param X: {array-like}, shape = [n_samples, n_features]
                  Training vectors, where n_samples
                  is the number of samples and
                  n_features is the number of features.
        :param y: array-like, shape = [n_samples]
                  Target values.
        :return:  self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                #print xi
                #print target
                nb = self.predict(xi)
                #print nb
                update = self.eta * (target - nb)
                #print update
                self.w_[1:] += update * xi
                self.w_[0] += update
                #print self.w_
                errors += int(update != 0.0)
                #print errors
            #print errors
            self.errors_.append(errors)
            print 'Error rate:'
            print float(errors)/len(X)
            w=self.w_
            a = -w[0] / w[2]
            b = -w[1] / w[2]
            xx = np.linspace(0, 10)
            yy = b * xx + a
            plt.plot(xx, yy, 'k-')
            plt.pause(0.3)
        return self

    def net_input(self, X):
        """Calculate net input"""
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return z

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
