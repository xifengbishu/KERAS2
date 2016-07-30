#coding:utf-8
# for a single-input model with 2 classes (binary):
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

batch_size = 12
nb_epoch = 10
data_dim = 5
# since we are using stateful rnn tsteps can be set to 1
timesteps = 1
nb_classes = 1
# number of elements ahead that are used to make the prediction
lahead = 1

model = Sequential()
model.add(LSTM(50,
               batch_input_shape=(batch_size, timesteps, data_dim),
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               batch_input_shape=(batch_size, timesteps, data_dim),
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
#model.compile(loss='mse', optimizer=fast_compile)
#model.compile(loss='mse', optimizer=None)
model.compile(loss='mse', optimizer='rmsprop')
# activation --softmax,softsign,softplus,tanh,relu,sigmoid,hard_sigmoid,linear
#keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
#model.add(Dense(1, input_dim=5, activation='keras.layers.advanced_activations.LeakyReLU(alpha=0.3)'))
model.add(Dense(1, input_dim=data_dim, activation='relu'))
#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity,mape,msle,squared_hinge,hinge,
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss='mape',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
#X_test = np.random.random((1000, 784))
#y_test = np.random.randint(2, size=(1000, 1))
#X_train = np.random.random((1000, 784))
#y_train = np.random.randint(2, size=(1000, 1))
print('Loading data...')

#train=np.loadtxt('../DATA_train/training')
#test=np.loadtxt('../DATA_train/testing')
train=np.loadtxt('../DATA_train/training_svm-scale_BP')
test=np.loadtxt('../DATA_train/testing_svm-scale_BP')

#X_train[120,0,5]=train[:,1:6]

X_train = np.zeros((batch_size * 10, timesteps, data_dim))
y_train = np.zeros((batch_size * 10, timesteps))
X_test = np.zeros((batch_size * 10, timesteps, data_dim))
y_test = np.zeros((batch_size * 10, timesteps))
print ('len(X_train)',len(X_train))
for i in range(len(X_train)):
	X_train[i, 0, :] = train[i,1:6]
	y_train[i,0] = train[i,0]
for i in range(len(X_test)):
	X_test[i, 0, :] = test[i,1:6]
	y_test[i,0] = test[i,0]
#X_train2=X_train.reshape(batch_size * 10,timesteps,data_dim)
#X_train[:,3]=X_train[:,3]*0.01
#y_train=train[:,0]
#y_train2=y_train.reshape(batch_size * 10, nb_classes)
#X_test=test[:,1:6]
#x_test2=X_test.reshape(batch_size * 10,timesteps,data_dim)
#X_test[:,3]=X_test[:,3]*0.01
#y_test=test[:,0]
#y_test2=y_test.reshape(batch_size * 10, nb_classes)

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print ( X_test[1])
print ( y_test[1])

# ================
# = 中间层output
# ================
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
#print ('get_3rd_layer_output',get_3rd_layer_output)
layer_output = get_3rd_layer_output([X_test])[0]
print ('layer_output',layer_output)
#print ( "x_train",X_train)
#print ( "y_train",y_train)
#print('Train...')
# train the model, iterating on the data in batches
# of 32 samples
#model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,verbose=0)
his=model.fit(X_train, y_train,batch_size=batch_size,verbose=1,nb_epoch=nb_epoch,shuffle=False)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print ('his',his.history)
#proba = model.predict_proba(X_test, batch_size=3)
#print('Test predict_proba:', proba)
#plot(model, to_file='model.png')

predict = model.predict(X_test, batch_size=batch_size, verbose=0)
print( predict )
#print( 'sum', np.sum(y_test-predict) )
print( 'mean',  np.mean(y_test-predict) )
exit()
# --- PLOT ---
print('Ploting Results')
plt.subplot(2, 1, 1)
plt.plot(y_test-predict)
plt.title('Diff')
plt.subplot(2, 1, 2)
plt.plot(predict)
plt.title('Predicted')
plt.savefig('diff.jpg')
#plt.savefig("matplot_sample.jpg")
#plt.show()
# ------- output ---------
file_object = open('predict.txt', 'w')
for i in range(len(predict)):
	file_object.write(str(predict[i,0]))
	file_object.write('\n')
file_object.close( )
exit()
