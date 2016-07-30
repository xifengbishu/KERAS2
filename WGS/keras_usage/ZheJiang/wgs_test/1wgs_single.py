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

#    nei cun diao yong 
import sys
# for a single-input model with 2 classes (binary):
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.optimizers import SGD
#from keras.utils.visualize_util import plot

batch_size = 2
nb_epoch = 5
# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

model = Sequential()

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        input_shape=(100,5)))
#                        input_dim=5, input_length=100))

model.add(MaxPooling1D(pool_length=pool_length))
model.add(Flatten())
#model.add(LSTM(5))
#model.add(Dense(1))
#model.compile(loss='mse', optimizer='rmsprop')
#model.add(Dense(1, activation='relu'))
# activation --softmax,softsign,softplus,tanh,relu,sigmoid,hard_sigmoid,linear
#keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
#model.add(Dense(1, input_dim=5, activation='keras.layers.advanced_activations.LeakyReLU(alpha=0.3)'))
#model.add(Dense(1, input_dim=5, activation='relu'))
#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity,mape,msle,squared_hinge,hinge,
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss='mape',
              metrics=['accuracy'])

# generate dummy data
import numpy as np

#train=np.loadtxt('../DATA_train/training_svm-scale_BP')
#test=np.loadtxt('../DATA_train/testing_svm-scale_BP')


mu,sigma=0,0.1
X_test=np.random.normal(mu,sigma,(100,5))
X_train=np.random.normal(mu,sigma,(100,5))
y_test=np.random.normal(mu,sigma,(100))+5
y_train=np.random.normal(mu,sigma,(100))+5


#print('Loading data...')


print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

print ( "X_train",X_train[1:10])
print ( "y_train",y_train[1:10])

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
