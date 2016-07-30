# -*- coding: utf-8 -*-
######################################
"""
Created on 2016年 06月 02日 星期四 14:36:13 CST

@author: WGS
"""
######################################
#     import ConvNets 
######################################

#     ConvNets 
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU, LeakyReLU
import keras.layers.advanced_activations as adact
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
#from keras.utils.visualize_util import plot

#     collections 
from collections import Counter
import random, cPickle
#from cutslice3d import load_data
#from cutslice3d import ROW, COL, LABELTYPE, CHANNEL

#    nei cun diao yong 
import sys
import numpy as np
import os

import random
np.random.seed(1337)  # for reproducibility


######################################
#     参数设置 
######################################

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 10

# Training
batch_size = 5
nb_epoch = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

######################################
#     Load data 
######################################

# generate dummy data
X_test = np.random.random((100, 4, 28 , 28))
y_test = np.random.random(100)+20
X_train = np.random.random((100, 4, 28 ,28))
y_train = np.random.random(100)+20
print('Loading data...')

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print ( 'X_train',X_train[1])
print ( 'y_train',y_train[1:10])

#print ( "x_train",x_train)
#print ( "y_train",y_train)
#exit()

######################################
#     Set Model 
######################################
#print('Train...')
# train the model, iterating on the data in batches
# of 32 samples

# apply a 3x3 convolution with 64 output filters on a 256x256 image:
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=X_train.shape[-3:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(1))
model.add(Activation('relu'))
#model.add(Dropout(0.5))



"""
#卷积层
model.add(Convolution2D(64, 3, 3, border_mode='valid',input_shape=X_train.shape[-3:])) 
model.add(Activation('tanh'))
model.add(Dropout(0.25))

#卷积层
model.add(Convolution2D(128, 2, 2, border_mode='valid')) 
model.add(Activation('tanh'))
model.add(Dropout(0.25))

#卷积层
model.add(Convolution2D(128, 2, 2, border_mode='valid')) 
model.add(Activation('tanh'))
model.add(Dropout(0.25))

#卷积层
model.add(Convolution2D(128, 2, 2, border_mode='valid')) 
model.add(Activation('tanh'))
model.add(Dropout(0.25))

#扁平化
model.add(Flatten())

#全连接层
model.add(Dense(1024, init='normal'))
model.add(Activation('tanh'))
model.add(Dropout(0.25))

"""
#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
#model.add(LSTM(lstm_output_size))

#LSTM(output_dim, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)

#Softmax分类，输出是2类别
#predicted output
model.add(Dense(1))
#model.add(Dense(1, input_dim=5, activation='relu'))
model.add(Activation('relu'))

model_path='model.hdf5'
if os.path.exists(model_path):
    model.load_weights(model_path)

#############
#开始训练模型
##############
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
model.compile(optimizer='sgd',
              loss='mape',
              metrics=['accuracy'])


# activation --softmax,softsign,softplus,tanh,relu,sigmoid,hard_sigmoid,linear
#keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
#model.add(Dense(1, input_dim=5, activation='keras.layers.advanced_activations.LeakyReLU(alpha=0.3)'))
#model.add(Dense(1, input_dim=5, activation='relu'))
#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity,mape,msle,squared_hinge,hinge,


#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
his=model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,verbose=0)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print ('his',his.history)
# ================
# = 中间层output
# ================
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
layer_output = get_3rd_layer_output([X_test])[0]
print ('layer0_output',layer_output.shape)

get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = get_3rd_layer_output([X_test])[0]
print ('layer1_output',layer_output.shape)

get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([X_test])[0]
print ('layer2_output',layer_output.shape)

get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([X_test])[0]
print ('layer3_output',layer_output.shape)

get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[4].output])
layer_output = get_3rd_layer_output([X_test])[0]
print ('layer4_output',layer_output.shape)

#proba = model.predict_proba(X_test, batch_size=3)
#print('Test predict_proba:', proba)
#plot(model, to_file='model.png')

predict = model.predict(X_test, batch_size=batch_size, verbose=0)
print( 'predict',predict[1:20] )
#print( 'sum', np.sum(y_test-predict) )
print( 'mean',  np.mean(y_test-predict) )
exit()
######################################
#     Save ConvNets model
######################################

model.save_weights('MyConvNets.h5')
cPickle.dump(model, open('MyConvNets.pkl',"wb"))
json_string = model.to_json()
open(W_MODEL, 'w').write(json_string)
######################################
#     Load ConvNets model
######################################

model = cPickle.load(open('MyConvNets.pkl',"rb"))
exit()
