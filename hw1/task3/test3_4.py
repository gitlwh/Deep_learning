import numpy as np
import matplotlib.pyplot as plt
from Perceptron3_4 import Perceptron


iris = np.loadtxt('iris.txt')
data = np.column_stack(iris)
label = data[2]
data = np.column_stack((data[0],data[1]))

per=Perceptron(0.01,20)
per.fit1(data,label)
predict=per.predict(data)
print predict
print per.errors_

per2=Perceptron(0.01,20)
per2.fit2(data,label)
predict2=per2.predict(data)
print predict2
print per2.errors_

per3=Perceptron(0.01,20)
per3.fit3(data,label)
predict3=per.predict(data)
print predict3
print per3.errors_
