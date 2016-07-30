# for a multi-input model with 10 classes:
from __future__ import print_function
import numpy as np
import os
import pydot
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.layers import Merge
from keras.utils.visualize_util import plot
batch_size = 32

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
from keras.utils.np_utils import to_categorical
data_1 = np.random.random((1000, 784))
data_2 = np.random.random((1000, 784))

# these are integers between 0 and 9
labels = np.random.randint(10, size=(1000, 1))
# we convert the labels to a binary matrix of size (1000, 10)
# for use with categorical_crossentropy
labels = to_categorical(labels, 10)

# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs
model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32)

score, acc = model.evaluate([data_1, data_2], labels,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
proba = model.predict_proba([data_1, data_2], batch_size=3)
predict = model.predict([data_1, data_2], batch_size=3, verbose=0)
print('Test predict_proba:', proba)
print('Test predict:', predict)
plot(model, to_file='model.png')
exit()
