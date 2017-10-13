import numpy as np
import matplotlib.pyplot as plt
from Perceptron4_3 import Perceptron

iris = np.loadtxt('iris2.txt') 

data = np.column_stack(iris)
label = data[4]
data = np.column_stack((data[0],data[1],data[2],data[3]))
num_folds = 10
subset_size = len(data)/num_folds
e=[]
w=[]

for i in range(num_folds):
    testing_this_round = data[i*subset_size:][:subset_size]
    testing_label = label[i*subset_size:][:subset_size]
    training_this_round = np.concatenate((data[:i*subset_size],data[(i+1)*subset_size:]))
    training_label = np.concatenate((label[:i*subset_size], label[(i+1)*subset_size:]))
    per=Perceptron(0.01,10)
    per.fit2(training_this_round,training_label)
    predict_this_round = per.predict(testing_this_round)
    num=0
    for j in range(len(predict_this_round)):
        if predict_this_round[j]!=testing_label[j]:
            num+=1
    print i
    #print 'error'
    #print num
    print 'error rate'
    print float(num)/len(predict_this_round)
    e.append(float(num)/len(predict_this_round))
    print 'weight'
    print per.w_
    w.append(per.w_)
print 'Average k-fold weight'
print sum(w)/len(w)
print 'Average k-fold error rate'
print sum(e)/len(e)

