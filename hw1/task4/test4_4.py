import numpy as np
import matplotlib.pyplot as plt
from Perceptron4_4 import Perceptron

iris = np.loadtxt('iris2.txt') 

data = np.column_stack(iris)
label = data[4]
for i in range(len(label)):
    if(label[i]==-1):
        label[i]=0
data = np.column_stack((data[0],data[1],data[2],data[3]))

per=Perceptron(0.01,20)
per.fit4(data,label)
print per.w_
print per.errors_
print per.net_input2(data)
print per.predict2(data)

iris = np.loadtxt('iris2.txt') 

data = np.column_stack(iris)
label = data[4]
data = np.column_stack((data[0],data[1],data[2],data[3]))


per3=Perceptron(0.01,20)
per3.fit2(data,label)
print per3.w_
print per3.errors_
print per3.predict(data)



