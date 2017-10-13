import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron


iris = np.loadtxt('iris.txt')
data = np.column_stack(iris)
label = data[2]
data = np.column_stack((data[0],data[1]))
#print data
plt.scatter([data[i][0] for i in range(len(data)) if label[i]==1],[data[i][1] for i in range(len(data)) if label[i]==1],color='red',marker='o')
plt.scatter([data[i][0] for i in range(len(data)) if label[i]==-1],[data[i][1] for i in range(len(data)) if label[i]==-1],color='blue',marker='*')
per=Perceptron(0.01,10)

per.fit(data,label)
plt.show()
print per.w_




