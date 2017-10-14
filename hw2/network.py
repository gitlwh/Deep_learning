"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation. 
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import math
from sklearn import preprocessing

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
            #print(np.shape(a))
        return a

    def feedforward1_5(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in list(zip(self.biases, self.weights))[:len(self.biases)-1]:
            a = sigmoid(np.dot(w, a)+b)
        b,w = self.biases[-1], self.weights[-1]
        a = softmax(np.dot(w, a)+b)
        return a

    def feedforward1_6(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in list(zip(self.biases, self.weights))[:len(self.biases)-1]:
            a = relu(np.dot(w, a)+b)
        b,w = self.biases[-1], self.weights[-1]
        a = softmax(np.dot(w, a)+b)
        return a



    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def SGD1_4(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            '''
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            '''
        return self.evaluate(test_data)

    def SGD1_5(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch1_5(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate1_5(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def SGD1_6(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch1_6(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate1_6(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def SGD1_7(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch1_7(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate1_6(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def SGD2_1(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        self.weights = [np.random.randn(y, x)*(1/math.sqrt(x)) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def SGD2_2(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        n = len(training_data)
        m = len(test_data)
        #print(np.shape(training_data))
        #print(np.shape(training_data[0][0]))
        #print(np.shape(np.array([i[0] for i in test_data[0][0]])))
        just_test_data = np.array([np.array([i[0] for i in test_data[j][0]]) for j in range(m)])
        #print(np.shape(np.array([i[0] for i in training_data[0][0]])))
        just_training_data = np.array([np.array([i[0] for i in training_data[j][0]]) for j in range(n)])
        #print(np.shape(just_test_data))
        
        if test_data: n_test = len(test_data)
        just_training_data = preprocessing.scale(just_training_data)
        just_test_data = preprocessing.scale(just_test_data)
        just_training_data = np.array([np.array([np.array([i]) for i in just_training_data[j]]) for j in range(n)])
        just_test_data = np.array([np.array([np.array([i]) for i in just_test_data[j]]) for j in range(m)])
        #print(just_training_data[0][0])
        #print(np.shape(just_training_data))
        #print(just_training_data[0])
        #print(np.shape(just_training_data[0]))
        training_data = np.array([np.array([x,y[1]]) for x,y in zip(just_training_data,training_data)])
        test_data = np.array([np.array([x,y[1]]) for x,y in zip(just_test_data,test_data)])
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
    
    def SGD2_3_1(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            self.nabla_bv = [np.zeros(b.shape) for b in self.biases]
            self.nabla_wv = [np.zeros(w.shape) for w in self.weights]
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch2_3_1(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))  

    def SGD2_3_2(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            self.nabla_bv = [np.zeros(b.shape) for b in self.biases]
            self.nabla_wv = [np.zeros(w.shape) for w in self.weights]
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch2_3_2(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))  

    def SGD2_4(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            self.nabla_bv = [np.zeros(b.shape) for b in self.biases]
            self.nabla_wv = [np.zeros(w.shape) for w in self.weights]
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch2_4(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))  

    def SGD2_5all(self, training_data, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        final = []
        final.append(self.SGD2_5_1(training_data, 10, 10, 3.0,test_data))

        print("first done")

        self.sizes=[784,30,10]
        self.num_layers = len(self.sizes)
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)*(1/math.sqrt(x)) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        
        final.append(self.SGD2_5_1(training_data, 10, 10, 3.0,test_data))

        print("second done")

        self.sizes=[784,30,10]
        self.num_layers = len(self.sizes)
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        final.append(self.SGD2_5_2(training_data, 10, 10, 3.0,test_data))
        print("third done")
        print(sum(int(major(x,a,m,y)) for (x, y),(a,b),(m,n) in zip(final[0],final[1],final[2])))
             

    def SGD2_5_1(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        return self.evaluate2_5_1(test_data)

    def SGD2_5_2(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch1_5(mini_batch, eta)
        return self.evaluate2_5_2(test_data)

    def SGD2_6(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            self.eta = eta
            self.sign = 1
            self.nabla_bv = [np.zeros(b.shape) for b in self.biases]
            self.nabla_wv = [np.zeros(w.shape) for w in self.weights]
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch2_6(mini_batch)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))  

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def update_mini_batch1_5(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop1_5(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def update_mini_batch1_6(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop1_6(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def update_mini_batch1_7(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop1_7(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def update_mini_batch2_3_1(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #v = mu * v - learning_rate * dx # integrate velocity
        mu=0.1
        self.nabla_bv = [mu*i-eta * nb/len(mini_batch) for i, nb in zip(self.nabla_bv, nabla_b)]
        self.nabla_wv = [mu*i-eta * nw/len(mini_batch) for i, nw in zip(self.nabla_wv, nabla_w)]
        self.weights = [w+nw for w, nw in zip(self.weights, self.nabla_wv)]
        self.biases = [b+nb for b, nb in zip(self.biases, self.nabla_bv)]

    def update_mini_batch2_3_2(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #v = mu * v - learning_rate * dx # integrate velocity
        mu=0.1
        prebv = np.array(self.nabla_bv)
        prewv = np.array(self.nabla_wv)
        self.nabla_bv = [mu*i-eta * nb/len(mini_batch) for i, nb in zip(self.nabla_bv, nabla_b)]
        self.nabla_wv = [mu*i-eta * nw/len(mini_batch) for i, nw in zip(self.nabla_wv, nabla_w)]
        self.weights = [w-mu*prev+(1+mu)*curv for prev,curv, w in zip(prewv,self.weights, self.nabla_wv)]
        self.biases = [b-mu*prev+(1+mu)*curv for prev,curv, b in zip(prebv,self.biases, self.nabla_bv)]


    def update_mini_batch2_4(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #backup_b = [np.zeros(b.shape) for b in self.biases]
        backup_w = [np.zeros(w.shape) for w in self.weights]
        rand = [list(set([np.random.randint(0,y) for i in range(int(y/2))])) for y in self.sizes]
        

        for i in range(len(self.biases)):
            bi=self.biases[i]
            we = self.weights[i]
            ra = rand[i]
            ra2 = rand[i+1]
            #bab = backup_b[i]
            baw = backup_w[i]
            for j in range(len(ra)):
                for a in range(len(we)):
                    baw[a][ra[j]]=we[a][ra[j]]
                    we[a][ra[j]]=np.zeros(baw[a][ra[j]].shape)
            #for j in range(len(ra2)):
                #bab[ra2[j]]=bi[ra2[j]]
                #bi[ra2[j]]=np.zeros(bab[ra2[j]].shape)

        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #print('w')
            #print(delta_nabla_w)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        for i in range(len(self.biases)):
                bi=self.biases[i]
                we = self.weights[i]
                ra = rand[i]
                ra2 = rand[i+1]
                #bab = backup_b[i]
                baw = backup_w[i]

                for j in range(len(ra)):
                    for a in range(len(we)):
                        we[a][ra[j]]=baw[a][ra[j]]
                    
                #for j in range(len(ra2)):
                    #bi[ra2[j]]=bab[ra2[j]]

    def update_mini_batch2_6(self, mini_batch):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        mysum = sum([sum([sum(j) for j in nabla_w[i]]) for i in range(len(nabla_w))])
        if mysum*self.sign<0:
            if(self.eta>0.5):
                self.eta=self.eta/2
            self.sign=self.sign*(-1)
        else:
            if(self.eta<10):
                self.eta=self.eta*2
        self.weights = [w-(self.eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #print(np.shape(x))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        #print("en")
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            #print('2')
            #print(nabla_w[-l])
        return (nabla_b, nabla_w)


    def backprop1_5(self, x, y):

        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in list(zip(self.biases, self.weights))[:len(self.biases)-1]:
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        b = self.biases[-1]
        w = self.weights[-1]
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backprop1_6(self, x, y):

        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in list(zip(self.biases, self.weights))[:len(self.biases)-1]:
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = relu(z)
            #print(activation)
            activations.append(activation)

        b = self.biases[-1]
        w = self.weights[-1]
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = reluDerivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backprop1_7(self, x, y):

        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in list(zip(self.biases, self.weights))[:len(self.biases)-1]:
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = relu(z)
            #print(activation)
            activations.append(activation)

        b = self.biases[-1]
        w = self.weights[-1]
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y)

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note about the variable l: Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. This numbering takes advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = leakyreluDerivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
               

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        #print(self.feedforward(test_data[0][0]))
        #print(test_data[0][1])
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate1_5(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward1_5(x)), y)
                        for (x, y) in test_data]
        #print(self.feedforward(test_data[0][0]))
        #print(test_data[0][1])
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate1_6(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward1_6(x)), y)
                        for (x, y) in test_data]
        #print(self.feedforward(test_data[0][0]))
        #print(test_data[0][1])
        return sum(int(x == y) for (x, y) in test_results)

    def evaluate2_5_1(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        #print(self.feedforward(test_data[0][0]))
        #print(test_data[0][1])
        return test_results

    def evaluate2_5_2(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward1_5(x)), y)
                        for (x, y) in test_data]
        #print(self.feedforward(test_data[0][0]))
        #print(test_data[0][1])
        return test_results


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def major(a,b,c,x):
    if a==b==c:
        return a==x
    if a==b or a==c:
        return a==x
    if b==c:
        return b==x
    return a==x

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def cross_entropy(t,y):
    """Derivative of the sigmoid function."""
    #print(-1*t*np.log(y))
    #print(np.shape(np.log(y)))
    #print(np.shape(t))
    return t*np.log(y)*(-1)

def relu(x):
    return np.maximum(x, 0, x)
def reluDerivative(x):
    return (x > 0) * 1 + (x <= 0) * 0
def leakyreluDerivative(x):
    return (x > 0) * 1 + (x <= 0) * .02
