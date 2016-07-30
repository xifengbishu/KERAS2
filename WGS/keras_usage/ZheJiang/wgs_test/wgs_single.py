#coding:utf-8
######################################
#     import ConvNets 
######################################

#     ConvNets 
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU, LeakyReLU
import keras.layers.advanced_activations as adact
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.callbacks import EarlyStopping

#     collections 
from collections import Counter
import random, cPickle
#from cutslice3d import load_data
#from cutslice3d import ROW, COL, LABELTYPE, CHANNEL

#    内存调用 
import sys
# for a single-input model with 2 classes (binary):
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.optimizers import SGD
#from keras.utils.visualize_util import plot

batch_size = 2
nb_epoch = 20

model = Sequential()
# activation --softmax,softsign,softplus,tanh,relu,sigmoid,hard_sigmoid,linear
#keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
#model.add(Dense(1, input_dim=5, activation='keras.layers.advanced_activations.LeakyReLU(alpha=0.3)'))
model.add(Dense(1, input_dim=5, activation='relu'))
#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity,mape,msle,squared_hinge,hinge,
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss='mape',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
exit()
X_test = np.random.random((1000, 784))
y_test = np.random.randint(2, size=(1000, 1))
X_train = np.random.random((1000, 784))
y_train = np.random.randint(2, size=(1000, 1))
#print('Loading data...')

#train=np.loadtxt('../DATA_train/training')
#test=np.loadtxt('../DATA_train/testing')
train=np.loadtxt('../DATA_train/training_svm-scale_BP')
test=np.loadtxt('../DATA_train/testing_svm-scale_BP')

X_train=train[:,1:6]
#X_train2=x_train.reshape(batch_size * 10,timesteps,data_dim)
#X_train[:,3]=X_train[:,3]*0.01
y_train=train[:,0]
#y_train2=y_train.reshape(batch_size * 10, nb_classes)
X_test=test[:,1:6]
#x_test2=x_test.reshape(batch_size * 10,timesteps,data_dim)
#X_test[:,3]=X_test[:,3]*0.01
y_test=test[:,0]
#y_test2=y_test.reshape(batch_size * 10, nb_classes)

#print('X_train shape:', X_train.shape)
#print('y_train shape:', y_train.shape)
#print ( X_test[1])
#print ( y_test[1])

#print ( "x_train",x_train)
#print ( "y_train",y_train)

#print('Train...')
# train the model, iterating on the data in batches
# of 32 samples
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
hist=model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,verbose=1)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#proba = model.predict_proba(X_test, batch_size=3)
#print('Test predict_proba:', proba)
#plot(model, to_file='model.png')

predict = model.predict(X_test, batch_size=batch_size, verbose=0)
print( predict )
#print( 'sum', np.sum(y_test-predict) )
print( 'mean',  np.mean(y_test-predict) )
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
