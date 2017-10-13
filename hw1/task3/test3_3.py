import numpy as np
import matplotlib.pyplot as plt
from Perceptron3_3 import Perceptron


iris = np.loadtxt('iris.txt')
data = np.column_stack(iris)
label = data[2]
data = np.column_stack((data[0],data[1]))

per=Perceptron(0.001,10)
per.fit1(data,label)
predict=per.predict(data)
print predict
print per.errors_

per2=Perceptron(0.00001,10)
per2.fit1(data,label)
predict2=per2.predict(data)
print predict2
print per2.errors_

per3=Perceptron(1,10)
per3.fit1(data,label)
predict3=per.predict(data)
print predict3
print per3.errors_
