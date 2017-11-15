Weiheng Li for hw3
wel615

I implemented four of five tasks:

task1:
Implementing with keras, using Alexnet structure and using cifar10 as dataseet.
output:

x_train shape: (50000, 32, 32, 3)
50000 train samples
10000 test samples
Train on 50000 samples, validate on 10000 samples
Epoch 1/10
50000/50000 [==============================] - 446s 9ms/step - loss: 1.4829 - acc: 0.4935 - val_loss: 1.2030 - val_acc: 0.6033
Epoch 2/10
11744/50000 [======>.......................] - ETA: 4:59 - loss: 1.1710 - acc: 0.6373^C

We can see the acc is increasing but it is too slow, I have no time to rum for whole ten ecpoch.

task2:
Implementing with tensorflow, 
Using FCN-vgg16 as layer structure,
Using MS COCO image as dataset,
Using some code on the [Krzysztof Chalupka's github](https://github.com/kjchalup) to implement FCN-vgg16.
Using COCOAPI to process the image data
Before running it, you should put [vgg16.npy](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) in the root (500MB+), and keep the other files 
in the root. You also should put accordingly images and json files in ./images and ./annotations.

The images are too large and many, I have no time to run the result. I believe it works well on a faster machine. If you want to see the output you can run the test-FCN-COCO.py to see the output.


shen-3:hw3 shen$ python3 hw3-2-FCN-COCO.py 
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
loading annotations into memory...
Done (t=0.89s)
creating index...
index created!
/Users/shen/Desktop/CSE498Deep learning/hw3/vgg16.npy
npy file loaded
build model started
build model finished: 0s
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorflow/python/ops/gradients_impl.py:96: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
2017-11-14 23:37:00.652922: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/skimage/transform/_warps.py:84: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
  warn("The default mode, 'constant', will be changed to 'reflect' in "


Yes, it stucks here, and the size of ./logs is keep increasing, I believe my code is correct, there are thousands of large images in the ./images. It may take me days to process those stuff. I want to reduce the dataset but in that case I need to change the json file, which is also too large. If I open it, I can hardly do any other operations. If you want to see the output you can run the test-FCN-COCO.py to see the output. It should be same to this code.

task4:
Implemented with the popular [tflean](https://github.com/tflearn) library to implement lstm
The dataset is [IMDB](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification) dataset and has been preprocessed. Every sentense is processed into integer according to the frequency of every words.


shen-3:hw3 shen$ python3 hw3-4-LSTM-imdb.py 
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
Using TensorFlow backend.
2017-11-14 23:20:39.820216: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Downloading data from http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl
---------------------------------
Run id: A12IT5
Log directory: /tmp/tflearn_logs/
---------------------------------
Training samples: 20000
Validation samples: 5000
--
Training Step: 625  | total loss: 0.68358 | time: 141.176s
| Adam | epoch: 001 | loss: 0.68358 - acc: 0.5436 | val_loss: 0.68119 - val_acc: 0.5752 -- iter: 20000/20000
--
Training Step: 1250  | total loss: 0.67980 | time: 126.573s
| Adam | epoch: 002 | loss: 0.67980 - acc: 0.5430 | val_loss: 0.66422 - val_acc: 0.6012 -- iter: 20000/20000
--
Taining Step: 1874  | total loss: 0.63205 | time: 120.138s
| Adam | epoch: 003 | loss: 0.63205 - acc: 0.6316 -- iter: 19968/20000
^C

I have no time to finish running this, but you can see the acc is increasing.


task5:
Implementing FCN-VGG16 to classify [ISIC](https://isic-archive.com/#images) dataset, I download some images for benign and maligiant melanoma. I read the json file to label every image and classify with FCN.
The dataset is so much that I download only around two thousands of image and their accordingly json file.
My laptop is so slow that I have to comment some layer to test if my code is correct.
I replace the last several fully connected layer into conv layer
To run it, download some images and its according metadata (json) and divide them into two parts,
one is for training and one is for testing, put them in train and test files. Then, run this program.

shen-3:hw3 shen$ python3 hw3-5-FCN-ISIC.py 
Using TensorFlow backend.
/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/importlib/_bootstrap.py:205: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
2017-11-14 23:28:59.011650: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
x_train shape: (1144, 32, 32, 3)
1144 train samples
375 test samples
Train on 1144 samples, validate on 375 samples
Epoch 1/10
1144/1144 [==============================] - 32s 28ms/step - loss: 0.7364 - acc: 0.7255 - val_loss: 0.4852 - val_acc: 0.8000
Epoch 2/10
1144/1144 [==============================] - 31s 27ms/step - loss: 0.5426 - acc: 0.7552 - val_loss: 0.5326 - val_acc: 0.7973
Epoch 3/10
1144/1144 [==============================] - 30s 26ms/step - loss: 0.5563 - acc: 0.7343 - val_loss: 0.4693 - val_acc: 0.7973
Epoch 4/10
1144/1144 [==============================] - 30s 26ms/step - loss: 0.5236 - acc: 0.7692 - val_loss: 0.5234 - val_acc: 0.7627
Epoch 5/10
1144/1144 [==============================] - 30s 26ms/step - loss: 0.5198 - acc: 0.7640 - val_loss: 0.5001 - val_acc: 0.8000
Epoch 6/10
  32/1144 [..............................] - ETA: 25s - loss: 0.4960 - acc: 0.8125^C

 To save some time, I only run six epochs, we can see the acc is better and better, but not stable. Because there are only one thousand images to train, and I comment many layers to save time. If I tune the parameter better, it would perform much better.


