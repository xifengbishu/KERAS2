'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
from keras.utils.visualize_util import plot
from keras.optimizers import SGD


# Embedding
max_features = 20000
maxlen = 5
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 4
nb_epoch = 20

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')

train=np.loadtxt('../DATA_train/training_svm-scale_BP')
test=np.loadtxt('../DATA_train/testing_svm-scale_BP')
X_train=train[:,1:6]
y_train=train[:,0]
X_test=test[:,1:6]
y_test=test[:,0]
print ( 'X_train',X_train )
print ( 'y_train',y_train )
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
model.add(Dense(1, activation='relu'))
#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity,mape,msle,squared_hinge,hinge,
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss='mape',
              metrics=['accuracy'])
#model.add(Activation('sigmoid'))

#model.compile(optimizer='sgd',
#              loss='mape',
#              metrics=['accuracy'])
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

print('Predicting')
predicted_output = model.predict(X_test, batch_size=batch_size)
print (predicted_output )
exit()
print('Ploting Results')
plt.subplot(2, 1, 1)
plt.plot(y_test)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()
