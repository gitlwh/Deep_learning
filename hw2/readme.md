Weiheng Li
wel615
1.1 logistic unit, sigmoid
nearly same
not good
A little better
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/preprocessing/data.py:164: UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
  warnings.warn("Numerical issues were encountered "
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/preprocessing/data.py:181: UserWarning: Numerical issues were encountered when scaling the data and might not be solved. The standard deviation of the data is probably very close to 0. 
  warnings.warn("Numerical issues were encountered "
/Users/shen/Desktop/CSE498Deep learning/hw2/t1/network.py:821: RuntimeWarning: overflow encountered in exp
  return 1.0/(1.0+np.exp(-z))
something wrong happens. Not good result.
not as good as not drop out 
I used baseline code, 2.1 question and 1.5 question to vote
output:
first done
second done
third done
9585
better than origin
shen-3:t1 shen$ python3 t1.py
Epoch 0: 8925 / 10000
Epoch 1: 8916 / 10000
Epoch 2: 9168 / 10000
Epoch 3: 9171 / 10000
Epoch 4: 9245 / 10000
Epoch 5: 9288 / 10000
Epoch 6: 9198 / 10000
Epoch 7: 9298 / 10000
Epoch 8: 9337 / 10000
Epoch 9: 9316 / 10000
Epoch 10: 9329 / 10000
Epoch 11: 9316 / 10000
Epoch 12: 9387 / 10000
Epoch 13: 9360 / 10000
Epoch 14: 9384 / 10000
Epoch 15: 9328 / 10000
Epoch 16: 9443 / 10000
Epoch 17: 9438 / 10000
Epoch 18: 9466 / 10000
Epoch 19: 9474 / 10000
slow and unstable
shen-3:t1 shen$ python3 t1.py
/Users/shen/Desktop/CSE498Deep learning/hw2/t1/network.py:854: RuntimeWarning: overflow encountered in exp
  return 1.0/(1.0+np.exp(-z))
Epoch 0: 1000 / 10000
Epoch 1: 1010 / 10000
Epoch 2: 1000 / 10000
Epoch 3: 1004 / 10000
Epoch 4: 1004 / 10000
Epoch 5: 1003 / 10000
Epoch 6: 1003 / 10000
Epoch 7: 1002 / 10000
Epoch 8: 1004 / 10000
Epoch 9: 1005 / 10000
Epoch 10: 1006 / 10000

Runtime warning happens, I don’t know if this influence. I think I should use CNN but it is too large. I have a interview on Monday and have no time to finish this. 
3.3
I reshape data into 3072 input, which is 3*32*32
