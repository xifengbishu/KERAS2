'''Trains a LSTM on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF+LogReg.

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb

max_features = 10000
maxlen = 5  # cut texts after this number of words (among top max_features most common words)
batch_size = 32


print('Loading data...')

train=np.loadtxt('./DATA_train/training')
test=np.loadtxt('./DATA_train/testing')

X_train=train[:,1:6]
X_train[:,3]=X_train[:,3]*0.01
y_train=train[:,0]
X_test=test[:,1:6]
X_test[:,3]=X_test[:,3]*0.01
y_test=test[:,0]

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
#exit()

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
exit()
# =========== np.fromfunction test =====
# =========== Test score: -3587.03657633
# =========== Test accuracy: 0.00416666666667

def funcx(i,j):
	return (i+5)*(j*2-7)
def funcy(i):
	return (i-3)*2-7
X_train = np.fromfunction(funcx,(240,10))
y_train = np.fromfunction(funcy,(240,))
X_test = np.fromfunction(funcx,(240,10))-7
y_test = np.fromfunction(funcy,(240,))
#X_train = np.empty((240,10),np.int32)
#y_train = np.empty((240),np.int32)
#X_test = np.empty((240,10),np.int32)
#y_test = np.empty((240),np.int32)
print ( X_train.dtype )
print ( y_train.dtype )
print ( X_train )
print ( y_train )
