import numpy as np
import math
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

    def fit1(self, X, y):
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
            
        return self

    def fit2(self, X, y):
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
            update=0
            value0=0
            value1=0
            for xi, target in zip(X, y):
                #print xi
                #print target
                nb = self.predict(xi)
                #print nb
                update = self.eta * (target - nb)
                value0+=update * xi/len(y)
                value1+=update/len(y)
                #print update
                #print update
                #print self.w_
                errors += int(target != nb)
                #print errors
            #print value0
            #print value1
            self.w_[1:] += value0/len(y)
            self.w_[0] += value1/len(y)
            #print self.w_
            #print errors
            self.errors_.append(errors)
            #print 'Error rate:'
            #print float(errors)/len(X)
            
        return self

    def fit3(self, X, y):
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
            a=0
            update=0
            for xi, target in zip(X, y):
                a+=1
                
                #print xi
                #print target
                nb = self.predict(xi)
                #print nb
                update += self.eta * (target - nb)
                #print update
                
                #print self.w_
                errors += int(target!=nb)
                if a==30:
                    self.w_[1:] += update * xi/a
                    self.w_[0] += update/a
                    update=0
                    a=0
                #print errors
            #print errors
            if a!=0:
                self.w_[1:] += update * xi/a
                self.w_[0] += update/a
            self.errors_.append(errors)
            print 'Error rate:'
            print float(errors)/len(X)
            
        return self

    def fit4(self, X, y):
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
        print X
        print y
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            update=0
            value0=0
            value1=0
            for xi, target in zip(X, y):
                #print xi
                #print target
                nb = self.net_input2(xi)
                nb2 = self.predict2(xi)
                #print nb
                #print nb2
                update = self.eta * (target - nb) * (1.0 - nb) * nb
                #print update
                value0+=update * xi
                value1+=update
                #print update
                #print self.w_
                errors += int(target!=nb2)
                #print errors
            #print self.net_input2(X)
            #print self.predict2(X)
            #print value0
            #print value1
            self.w_[1:] += value0
            self.w_[0] += value1
            #print self.w_
            #print errors
            self.errors_.append(errors)
            #print 'Error rate:'
            #print float(errors)/len(X)
            
        return self

    def net_input(self, X):
        """Calculate net input"""
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return z
    
    def net_input2(self, X):
        """Calculate net input"""
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        y = 1/(1+math.e**(-1*z))
        return y

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def predict2(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input2(X) >= 0.5, 1, 0)
