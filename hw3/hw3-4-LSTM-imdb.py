# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""

'''
Weiheng Li
wel615@lehigh.edu
CSE498 Deep learning homework 3
Implemented with the popular tflean library to implement lstm
The dataset is a preprocessed dataset which process every sentense into number according to the
frequency of every words.
'''



import tflearn
from tflearn.data_utils import pad_sequences
from tflearn.datasets import imdb
import numpy as np
import keras

batch_size = 32

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=500,
                                valid_portion=0.2)
trainX, trainY = train
testX, testY = test
#print(trainX)

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors

trainY = keras.utils.to_categorical(trainY,2)
testY = keras.utils.to_categorical(testY,2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=500, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.25)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
