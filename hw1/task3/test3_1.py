import numpy as np
import matplotlib.pyplot as plt
from Perceptron3_1 import Perceptron


data = np.zeros( (100,2) )
label= np.zeros( (100,1) )

iris = np.loadtxt('iris.txt')
data = np.column_stack(iris)
label = data[2]
data = np.column_stack((data[0],data[1]))

per=Perceptron(0.01,10)
per.fit1(data,label)
predict=per.predict(data)
print predict
print per.errors_
per.fit2(data,label)
print per.predict(data)
print per.errors_
per.fit3(data,label)
print per.predict(data)
print per.errors_
