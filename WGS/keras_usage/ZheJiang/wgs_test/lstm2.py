from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from numpy  import *

data_dim = 5
timesteps = 3
nb_classes = 3
batch_size = 4

# expected input batch shape: (batch_size, timesteps, data_dim)
# note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
# --softmax,softsign,softplus,tanh,relu,sigmoid,hard_sigmoid,linear
model.add(Dense(nb_classes, activation='tanh')) 

#model.compile(loss='categorical_crossentropy',poisson,cosine_proximity
model.compile(loss='cosine_proximity',
              optimizer='rmsprop',
              metrics=['accuracy'])

# generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, nb_classes))

# generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, nb_classes))


print('Loading data...')

train=np.loadtxt('../DATA_train/training_svm-scale_BP')
test=np.loadtxt('../DATA_train/testing_svm-scale_BP')

x_train=train[:,1:6]
x_train2=x_train.reshape(batch_size * 10,timesteps,data_dim)
#X_train[:,3]=X_train[:,3]*0.01
y_train=train[:,0]
y_train2=y_train.reshape(batch_size * 10, nb_classes)
x_test=test[:,1:6]
x_test2=x_test.reshape(batch_size * 10,timesteps,data_dim)
#X_test[:,3]=X_test[:,3]*0.01
y_test=test[:,0]
y_test2=y_test.reshape(batch_size * 10, nb_classes)

print('X_train shape:', x_train2.shape)
print('y_train shape:', y_train2.shape)

#print ( "x_train",x_train)
#print ( "y_train",y_train)

print('Train...')
model.fit(x_train2, y_train2,
          batch_size=batch_size, nb_epoch=20,
          validation_data=(x_test2, y_test2))
score, acc = model.evaluate(x_test2, y_test2,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
proba = model.predict_proba(x_test2, batch_size=batch_size)
print('Test predict_proba:', proba)
predict = model.predict(x_test2, batch_size=batch_size, verbose=0)
print('Test predict:', predict)
