import mnist_loader as ld
import network as nw
# Standard library
import pickle as cPickle
import gzip
# Third-party libraries
import numpy as np

#1.2,1.3
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD(training_data, 20, 10, 3.0,test_data)
'''
#1.4
'''
training_data, validation_data, test_data = ld.load_data_wrapper()

size = range(1, 20,5)
learningRate = np.arange(2.0,4.1,1)

best = 0;
for i in size:
	for j in learningRate:
		mynetwork = nw.Network([784,30,10])
		result = mynetwork.SGD1_4(training_data, 10, i, j,validation_data)
		if result > best:
			bestij = [i,j]
			best = result
print(bestij)
mynetwork.SGD(training_data, 20, bestij[0], bestij[1],test_data)
'''
#1.5
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD1_5(training_data, 30, 10, 3.0,test_data)
'''
#1.6
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD1_6(training_data, 30, 10, 3.0,test_data)
'''
#1.7
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD1_7(training_data, 30, 10, 3.0,test_data)
'''
#2.1
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_1(training_data, 20, 10, 3.0,test_data)
'''
#2.2
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_2(training_data, 20, 10, 3.0,test_data)
'''
#2.3(1)
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_3_1(training_data, 20, 10, 3.0,test_data)
'''
#2.3(2)
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_3_2(training_data, 20, 10, 3.0,test_data)
'''
#2.4
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_4(training_data, 20, 10, 3.0,test_data)
'''
#2.5
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_5all(training_data, test_data)
'''
#2.6
'''
training_data, validation_data, test_data = ld.load_data_wrapper()
mynetwork = nw.Network([784,30,10])
mynetwork.SGD2_6(training_data, 20, 10, 5.0,test_data)
'''

#3.2

def unpickle(file):
    with open('cifar-10-batches-py/'+file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def getdata(file,get="train"):
	mydata = unpickle(file)
	training_inputs = [np.reshape(x, (3072, 1)) for x in mydata['data']]
	if get=='test':
		training_results = mydata['labels']
	else:
		training_results = [vectorized_result(y) for y in mydata['labels']]
	training_data = list(zip(training_inputs, training_results))
	return training_data

	

def vectorized_result(j):
    """Return a 10-dimensional unit vector (one-hot encoding) with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def getalltraining():
	return getdata("data_batch_1")+getdata("data_batch_2")+getdata("data_batch_3")+getdata("data_batch_4")+getdata("data_batch_5")

def gettest():
	return getdata("test_batch","test")


mynetwork = nw.Network([3072,32,10])
mynetwork.SGD(getalltraining(), 20, 10, 3.0,gettest())


