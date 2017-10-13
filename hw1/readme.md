Weiheng Li
wel615@lehigh.edu

This is a Readme file for hw1 of CSE498 deep learning.

1.1 Online
1.2 Stochastic gradient descent is a stochastic approximation of the gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minima or maxima by iteration. SGD uses the difference between target and predict result of each sample to update weight of inputs. In Stochastic Gradient Descent (SGD), the weight vector gets updated every time you read process a sample, whereas in Gradient Descent (GD) the update is only made after all samples are processed in the iteration. Thus, in an iteration in SGD, the weights number of times the weights are updated is equal to the number of examples, while in GD it only happens once. SGD is a whole lot faster. Large datasets often can't be held in RAM, which makes vectorization much less efficient. Rather, each sample or batch of samples must be loaded, worked with, the results stored, and so on. Minibatch SGD, on the other hand, is usually intentionally made small enough to be computationally tractable. Usually, this computational advantage is leveraged by performing many more iterations of SGD, making many more steps than conventional batch gradient descent. This usually results in a model that is very close to that which would be found via batch gradient descent, or better. Also, SGC can reach the best classifier in a faster rate, because the calculation cost less resource. But it is difficult to reach the ideal classifier because certain  sample may bring noise.
2.3 same to 2.2

3.1 see Perceptron3_1 and test3_1. fit1() is online, fit2() is fullbatch, fit3() is minibatch
3.2 see Perceptron3_1 and test3_1 
3.3 see Perceptron3_3 and test3_3. I used 1, 0.01. 0.00001 as learning rate. The prediction result are all the same. The weights are different according to learning rate.
3.4 see Perceptron3_4 and test3_4 The initial weight are randomized from -0.25 to 0.25, from -0.1 to 0.1 and all 0. The result is different from time to time. We can see that is the weight range is too large the result would be worse. It should be shuffled but keep in a limited range.

4.1 see Perceptron4_1 and test4_1
4.2 Cross-validation, sometimes called rotation estimation is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice. In a prediction problem, a model is usually given a dataset of known data on which training is run (training dataset), and a dataset of unknown data against which the model is tested (testing dataset). The goal of cross validation is to define a dataset to "test" the model in the training phase (i.e., the validation dataset), in order to limit problems like overfitting, give an insight on how the model will generalize to an independent dataset.
4.3 see Perceptron4_3 and test4_3. All would be printed.
4.4 see Perceptron4_4 and test4_4. I modify the parameters to 0.01 and 20. From the result, it seems linear is better.