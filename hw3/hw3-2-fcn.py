from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import json
from scipy import ndimage
from PIL import Image
import scipy.misc
import os
import numpy as np
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
from keras.layers import *
'''
This proj is to use fcn to classify https://isic-archive.com/#images dataset
The dataset is so much that I download only around two thousands of image and their accordingly json file.
My laptop is so slow that I have to comment some layer to test if my code is correct.
This is an implentation of FCN_Vgg16 in the origin paper.
I replace the last several fully connected layer into conv layer


I download some images and its according metadata (json) and divide them into two parts,
one is for training and one is for testing. Then, run this program.
'''
def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


batch_size = 32
num_classes = 2
epochs = 10

def load_data(file):
	if file=='test':
		imgs = os.listdir("./test")
		num = len(imgs)
		validnum = 0;
		for i in range(num):
			if imgs[i][-2:]=='on':
				newdata = json.load(open("./test/"+imgs[i]))
				if newdata['meta']['clinical']["benign_malignant"]=="malignant" or newdata['meta']['clinical']["benign_malignant"]=="benign":
					validnum+=1

		data = np.empty((validnum,32,32,3),dtype="float32")
		label = np.empty((validnum,),dtype="uint8")
		index = 0
		for i in range(num):
			if imgs[i][-2:]=='on':
				newdata = json.load(open("./test/"+imgs[i]))
				if newdata['meta']['clinical']["benign_malignant"]=="malignant":
					label[index] = 1
					img = scipy.misc.imread("./test/"+imgs[i][:-3]+'pg')
					img=scipy.misc.imresize(img,[32,32,3])
					arr = np.asarray(img,dtype="float32")
					data[index,:,:,:] = arr
					index+=1
				elif newdata['meta']['clinical']["benign_malignant"]=="benign":
					label[index] = 0
					img = scipy.misc.imread("./test/"+imgs[i][:-3]+'pg')
					img=scipy.misc.imresize(img,[32,32,3])
					arr = np.asarray(img,dtype="float32")
					data[index,:,:,:] = arr
					index+=1
	if file=='train':
		imgs = os.listdir("./train")
		num = len(imgs)
		validnum = 0;
		for i in range(num):
			if imgs[i][-2:]=='on':
				newdata = json.load(open("./train/"+imgs[i]))
				if newdata['meta']['clinical']["benign_malignant"]=="malignant" or newdata['meta']['clinical']["benign_malignant"]=="benign":
					validnum+=1

		data = np.empty((validnum,32,32,3),dtype="float32")
		label = np.empty((validnum,),dtype="uint8")
		index = 0
		for i in range(num):
			if imgs[i][-2:]=='on':
				newdata = json.load(open("./train/"+imgs[i]))
				if newdata['meta']['clinical']["benign_malignant"]=="malignant":
					label[index] = 1
					img = scipy.misc.imread("./train/"+imgs[i][:-3]+'pg')
					img=scipy.misc.imresize(img,[32,32,3])
					arr = np.asarray(img,dtype="float32")
					data[index,:,:,:] = arr
					index+=1
				elif newdata['meta']['clinical']["benign_malignant"]=="benign":
					label[index] = 0
					img = scipy.misc.imread("./train/"+imgs[i][:-3]+'pg')
					img=scipy.misc.imresize(img,[32,32,3])
					arr = np.asarray(img,dtype="float32")
					data[index,:,:,:] = arr
					index+=1
	return data,label


# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train,y_train = load_data("train")
x_test,y_test = load_data("test")

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
weight_decay=0.
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
'''
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))


model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))


model.add(Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay)))
model.add(Dropout(0.5))
model.add(Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay)))
'''
model.add(Dropout(0.5))
model.add(Conv2D(num_classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay)))
model.add(BilinearUpSampling2D(size=(32, 32)))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])