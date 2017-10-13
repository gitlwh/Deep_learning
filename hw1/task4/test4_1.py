import numpy as np
import matplotlib.pyplot as plt
from Perceptron4_1 import Perceptron

iris = np.loadtxt('iris2.txt') 

data = np.column_stack(iris)
label = data[4]
data = np.column_stack((data[0],data[1],data[2],data[3]))

per=Perceptron(0.01,10)
per.fit2(data,label)
print per.predict(data)

