import mnist_loader as ld
import network as nw
# Standard library
import pickle as cPickle
import gzip
# Third-party libraries
import numpy as np

training_data, validation_data, test_data = ld.load_data_wrapper()
#f = gzip.open('mnist.pkl.gz', 'rb')
#training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
mynetwork = nw.Network([784,30,10])
mynetwork.SGD(training_data, 30, 10, 50.0,test_data)