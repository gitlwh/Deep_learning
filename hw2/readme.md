Weiheng Li
wel615
1.1 logistic unit, sigmoid1.2 Epoch 0: 8369 / 10000Epoch 1: 8877 / 10000Epoch 2: 9293 / 10000Epoch 3: 9288 / 10000Epoch 4: 9337 / 10000Epoch 5: 9403 / 10000Epoch 6: 9401 / 10000Epoch 7: 9441 / 10000Epoch 8: 9447 / 10000Epoch 9: 9461 / 10000Epoch 10: 9485 / 10000Epoch 11: 9445 / 10000Epoch 12: 9459 / 10000Epoch 13: 9458 / 10000Epoch 14: 9480 / 10000Epoch 15: 9498 / 10000Epoch 16: 9441 / 10000Epoch 17: 9456 / 10000Epoch 18: 9491 / 10000Epoch 19: 9482 / 10000Epoch 20: 9487 / 10000Epoch 21: 9480 / 10000Epoch 22: 9467 / 10000Epoch 23: 9491 / 10000Epoch 24: 9475 / 10000Epoch 25: 9455 / 10000Epoch 26: 9476 / 10000Epoch 27: 9504 / 10000Epoch 28: 9492 / 10000Epoch 29: 9480 / 100001.3When there are no hidden layers, the result is:Epoch 0: 5893 / 10000Epoch 1: 6359 / 10000Epoch 2: 6618 / 10000Epoch 3: 6626 / 10000Epoch 4: 6737 / 10000Epoch 5: 8044 / 10000Epoch 6: 8227 / 10000Epoch 7: 8195 / 10000Epoch 8: 8228 / 10000Epoch 9: 8229 / 10000Epoch 10: 8230 / 10000Epoch 11: 8195 / 10000Epoch 12: 8255 / 10000Epoch 13: 8258 / 10000Epoch 14: 8252 / 10000Epoch 15: 8270 / 10000Epoch 16: 8240 / 10000Epoch 17: 8250 / 10000Epoch 18: 8255 / 10000Epoch 19: 8279 / 10000Epoch 20: 8279 / 10000Epoch 21: 8294 / 10000Epoch 22: 8262 / 10000Epoch 23: 8247 / 10000Epoch 24: 8290 / 10000Epoch 25: 8266 / 10000Epoch 26: 8282 / 10000Epoch 27: 8253 / 10000Epoch 28: 8270 / 10000Epoch 29: 8256 / 10000The accuracy is worse than with one hidden layer case, but the speed is faster.When learning rate is too large, we set is as 50, the result is:Epoch 0: 986 / 10000Epoch 1: 1575 / 10000Epoch 2: 1573 / 10000Epoch 3: 1576 / 10000Epoch 4: 1580 / 10000Epoch 5: 1579 / 10000Epoch 6: 1573 / 10000Epoch 7: 1582 / 10000Epoch 8: 1581 / 10000Epoch 9: 1582 / 10000Epoch 10: 1595 / 10000Epoch 11: 1586 / 10000Epoch 12: 1585 / 10000Epoch 13: 1569 / 10000Epoch 14: 1567 / 10000Epoch 15: 1514 / 10000Epoch 16: 1299 / 10000Epoch 17: 1300 / 10000Epoch 18: 1305 / 10000Epoch 19: 1310 / 10000Epoch 20: 1314 / 10000Epoch 21: 1316 / 10000Epoch 22: 1321 / 10000Epoch 23: 1321 / 10000Epoch 24: 1323 / 10000Epoch 25: 1331 / 10000Epoch 26: 1336 / 10000Epoch 27: 1344 / 10000Epoch 28: 1352 / 10000Epoch 29: 1363 / 10000The accuracy is worse than with one hidden layer case, reach the final result in a fast paceWhen learning rate is too small, we set is as 0.1, the result is:Epoch 0: 4923 / 10000Epoch 1: 6648 / 10000Epoch 2: 7185 / 10000Epoch 3: 7481 / 10000Epoch 4: 7654 / 10000Epoch 5: 7764 / 10000Epoch 6: 7844 / 10000Epoch 7: 7922 / 10000Epoch 8: 7993 / 10000Epoch 9: 8042 / 10000Epoch 10: 8077 / 10000Epoch 11: 8090 / 10000Epoch 12: 8123 / 10000Epoch 13: 8153 / 10000Epoch 14: 8192 / 10000Epoch 15: 8213 / 10000Epoch 16: 8231 / 10000Epoch 17: 8239 / 10000Epoch 18: 8253 / 10000Epoch 19: 8259 / 10000Epoch 20: 8281 / 10000Epoch 21: 8294 / 10000Epoch 22: 8302 / 10000Epoch 23: 8302 / 10000Epoch 24: 8315 / 10000Epoch 25: 8325 / 10000Epoch 26: 8328 / 10000Epoch 27: 8329 / 10000Epoch 28: 8333 / 10000Epoch 29: 8341 / 10000Reach the final result in a slow pace1.4size = range(1, 20,5)learningRate = np.arange(2.0,4.1,1)the best batch size is 6 and learning rate is 2.0[6, 2.0]Epoch 0: 9399 / 10000Epoch 1: 9454 / 10000Epoch 2: 9458 / 10000Epoch 3: 9462 / 10000Epoch 4: 9445 / 10000Epoch 5: 9440 / 10000Epoch 6: 9449 / 10000Epoch 7: 9486 / 10000Epoch 8: 9492 / 10000Epoch 9: 9476 / 10000Epoch 10: 9482 / 10000Epoch 11: 9480 / 10000Epoch 12: 9481 / 10000Epoch 13: 9467 / 10000Epoch 14: 9485 / 10000Epoch 15: 9490 / 10000Epoch 16: 9473 / 10000Epoch 17: 9474 / 10000Epoch 18: 9481 / 10000Epoch 19: 9491 / 100001.5Epoch 0: 8929 / 10000Epoch 1: 9001 / 10000Epoch 2: 9109 / 10000Epoch 3: 9186 / 10000Epoch 4: 9320 / 10000Epoch 5: 9357 / 10000Epoch 6: 9280 / 10000Epoch 7: 9362 / 10000Epoch 8: 9276 / 10000Epoch 9: 9291 / 10000Epoch 10: 9383 / 10000Epoch 11: 9343 / 10000Epoch 12: 9387 / 10000Epoch 13: 9320 / 10000Epoch 14: 9438 / 10000Epoch 15: 9374 / 10000Epoch 16: 9407 / 10000Epoch 17: 9381 / 10000Epoch 18: 9455 / 10000Epoch 19: 9484 / 10000
nearly same1.6Epoch 0: 974 / 10000Epoch 1: 1135 / 10000Epoch 2: 1028 / 10000Epoch 3: 1135 / 10000Epoch 4: 980 / 10000Epoch 5: 1010 / 10000Epoch 6: 892 / 10000Epoch 7: 980 / 10000Epoch 8: 1032 / 10000Epoch 9: 1028 / 10000Epoch 10: 1135 / 10000Epoch 11: 1135 / 10000Epoch 12: 1009 / 10000Epoch 13: 1009 / 10000Epoch 14: 974 / 10000Epoch 15: 1032 / 10000Epoch 16: 1010 / 10000Epoch 17: 958 / 10000Epoch 18: 1135 / 10000Epoch 19: 974 / 10000
not good1.7leaky RELUA "dead" ReLU always outputs the same value (zero as it happens, but that is not important) for any input. Probably this is arrived at by learning a large negative bias term for its weights.In turn, that means that it takes no role in discriminating between inputs. For classification, you could visualise this as a decision plane outside of all possible input data.Once a ReLU ends up in this state, it is unlikely to recover, because the function gradient at 0 is also 0, so gradient descent learning will not alter the weights. "Leaky" ReLUs with a small positive gradient for negative inputs (y=0.01x when x < 0 say) are one attempt to address this issue and give a chance to recover.Leaky ReLUs allow a small, non-zero gradient when the unit is not active.shen-3:t1 shen$ python3 t1_2.py Epoch 0: 2571 / 10000Epoch 1: 1571 / 10000Epoch 2: 2882 / 10000Epoch 3: 2461 / 10000Epoch 4: 2054 / 10000Epoch 5: 2572 / 10000Epoch 6: 2733 / 10000Epoch 7: 3091 / 10000Epoch 8: 2061 / 10000Epoch 9: 2830 / 10000Epoch 10: 2182 / 10000Epoch 11: 2258 / 10000Epoch 12: 2893 / 10000Epoch 13: 2480 / 10000Epoch 14: 2521 / 10000Epoch 15: 2261 / 10000Epoch 16: 1973 / 10000Epoch 17: 1916 / 10000Epoch 18: 2262 / 10000Epoch 19: 2505 / 10000much better but not goode enough2.1shen-3:t1 shen$ python3 t1_2.py Epoch 0: 9309 / 10000Epoch 1: 9466 / 10000Epoch 2: 9495 / 10000Epoch 3: 9497 / 10000Epoch 4: 9525 / 10000Epoch 5: 9513 / 10000Epoch 6: 9499 / 10000Epoch 7: 9538 / 10000Epoch 8: 9558 / 10000Epoch 9: 9558 / 10000Epoch 10: 9568 / 10000Epoch 11: 9532 / 10000Epoch 12: 9532 / 10000Epoch 13: 9554 / 10000Epoch 14: 9573 / 10000Epoch 15: 9548 / 10000Epoch 16: 9581 / 10000Epoch 17: 9570 / 10000Epoch 18: 9565 / 10000Epoch 19: 9582 / 10000
A little better2.2
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/preprocessing/data.py:164: UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
  warnings.warn("Numerical issues were encountered "
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/sklearn/preprocessing/data.py:181: UserWarning: Numerical issues were encountered when scaling the data and might not be solved. The standard deviation of the data is probably very close to 0. 
  warnings.warn("Numerical issues were encountered "
/Users/shen/Desktop/CSE498Deep learning/hw2/t1/network.py:821: RuntimeWarning: overflow encountered in exp
  return 1.0/(1.0+np.exp(-z))Epoch 0: 7943 / 10000Epoch 1: 8054 / 10000Epoch 2: 8061 / 10000Epoch 3: 8067 / 10000Epoch 4: 8043 / 10000Epoch 5: 8032 / 10000Epoch 6: 8018 / 10000Epoch 7: 8017 / 10000Epoch 8: 8021 / 10000Epoch 9: 8019 / 10000Epoch 10: 8023 / 10000Epoch 11: 8014 / 10000Epoch 12: 8010 / 10000Epoch 13: 8010 / 10000Epoch 14: 8012 / 10000Epoch 15: 8012 / 10000Epoch 16: 8010 / 10000Epoch 17: 8009 / 10000Epoch 18: 8008 / 10000Epoch 19: 8004 / 10000
something wrong happens. Not good result.2.3Standard momentum methodEpoch 0: 9029 / 10000Epoch 1: 9270 / 10000Epoch 2: 9339 / 10000Epoch 3: 9360 / 10000Epoch 4: 9370 / 10000Epoch 5: 9399 / 10000Epoch 6: 9448 / 10000Epoch 7: 9436 / 10000Epoch 8: 9434 / 10000Epoch 9: 9456 / 10000Epoch 10: 9461 / 10000Epoch 11: 9451 / 10000Epoch 12: 9453 / 10000Epoch 13: 9459 / 10000Epoch 14: 9472 / 10000Epoch 15: 9472 / 10000Epoch 16: 9483 / 10000Epoch 17: 9495 / 10000Epoch 18: 9474 / 10000Epoch 19: 9494 / 10000Nesterov momentumnot good at all/Users/shen/Desktop/CSE498Deep learning/hw2/t1/network.py:391: RuntimeWarning: overflow encountered in exp  return 1.0/(1.0+np.exp(-z))Epoch 0: 859 / 10000/Users/shen/Desktop/CSE498Deep learning/hw2/t1/network.py:257: RuntimeWarning: overflow encountered in add  z = np.dot(w, activation)+bEpoch 1: 980 / 10000Epoch 2: 980 / 10000Epoch 3: 980 / 10000Epoch 4: 980 / 10000Epoch 5: 980 / 10000Epoch 6: 980 / 10000Epoch 7: 980 / 10000Epoch 8: 980 / 10000Epoch 9: 980 / 10000Epoch 10: 980 / 10000Epoch 11: 980 / 10000Epoch 12: 980 / 10000Epoch 13: 980 / 10000Epoch 14: 980 / 10000Epoch 15: 980 / 10000Epoch 16: 980 / 10000Epoch 17: 980 / 10000Epoch 18: 980 / 10000Epoch 19: 980 / 10000(1)and (3) is good, (2) occurs something wrong, result is bad2.4Epoch 0: 8402 / 10000Epoch 1: 8807 / 10000Epoch 2: 8969 / 10000Epoch 3: 9004 / 10000Epoch 4: 9063 / 10000Epoch 5: 9100 / 10000Epoch 6: 9147 / 10000Epoch 7: 9176 / 10000Epoch 8: 9201 / 10000Epoch 9: 9170 / 10000Epoch 10: 9182 / 10000Epoch 11: 9183 / 10000Epoch 12: 9222 / 10000Epoch 13: 9189 / 10000
not as good as not drop out 2.5
I used baseline code, 2.1 question and 1.5 question to vote
output:
first done
second done
third done
9585
better than origin2.6
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
slow and unstable3.2
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
