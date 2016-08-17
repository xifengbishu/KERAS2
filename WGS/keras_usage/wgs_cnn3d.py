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
from keras.layers.convolutional import Convolution3D, Convolution3D, MaxPooling3D
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

batch_size = 5
nb_epoch = 5

######################################
#     Load data 
######################################

# generate dummy data

#    X_test[0:999] = texture[2000:2999]
mu,sigma=0,0.1
X_test=np.random.normal(mu,sigma,(100,1, 2, 28 , 28))
X_train=np.random.normal(mu,sigma,(100,1, 2, 28 , 28))
y_test=np.random.normal(mu,sigma,(100))+5
y_train=np.random.normal(mu,sigma,(100))+5
print('Loading data...')

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print ( X_test[1])
print ( y_test)

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
#卷积层
model.add(Convolution3D(64, 2,3, 3, border_mode='valid',input_shape=X_train.shape[-4:])) 
model.add(Activation('relu'))
model.add(Dropout(0.25))

#卷积层
model.add(Convolution3D(128, 2, 2, 2, border_mode='valid')) 
model.add(Activation('relu'))
model.add(Dropout(0.25))

#卷积层
model.add(Convolution3D(128, 2, 2, 2, border_mode='valid')) 
model.add(Activation('relu'))
model.add(Dropout(0.25))

#卷积层
model.add(Convolution3D(128, 2, 2, 2, border_mode='valid')) 
model.add(Activation('relu'))
model.add(Dropout(0.25))

#扁平化
model.add(Flatten())

#"""
#全连接层
model.add(Dense(1024))
#model.add(Dense(1024, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#"""

#Softmax分类，输出是2类别
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mse', optimizer='rmsprop')

model.add(Dense(1, activation='relu'))

#############
#开始训练模型
##############
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#model.compile(loss='mse', optimizer='rmsprop')
#model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
#model.compile(optimizer='sgd',loss='mape')

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
             loss='mape')
#              metrics=['accuracy'])
model.compile(optimizer='rmsprop',
              loss='mse')


# activation --softmax,softsign,softplus,tanh,relu,sigmoid,hard_sigmoid,linear
#keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
#model.add(Dense(1, input_dim=5, activation='keras.layers.advanced_activations.LeakyReLU(alpha=0.3)'))
#model.add(Dense(1, input_dim=5, activation='relu'))
#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity,mape,msle,squared_hinge,hinge,


#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
hist=model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,verbose=1)

#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)

#proba = model.predict_proba(X_test, batch_size=3)
#print('Test predict_proba:', proba)
#plot(model, to_file='model.png')

predict = model.predict(X_test, batch_size=batch_size, verbose=0)
print( predict )
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
