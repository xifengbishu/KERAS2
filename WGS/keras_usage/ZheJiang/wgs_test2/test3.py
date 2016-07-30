from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD  
from keras.datasets import mnist  
import numpy

model = Sequential()  
model.add(Dense(784, 500, init='glorot_uniform')) # ....28*28=784  
model.add(Activation('tanh')) # .....tanh  
model.add(Dropout(0.5)) # ..50%.dropout

model.add(Dense(500, 500, init='glorot_uniform')) # ....500.  
model.add(Activation('tanh'))  
model.add(Dropout(0.5))

model.add(Dense(500, 10, init='glorot_uniform')) # .....10.........10  
model.add(Activation('softmax')) # .....softmax

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # ......lr....  
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical') # .......loss..

(X_train, y_train), (X_test, y_test) = mnist.load_data() # ..Keras...mnist...............

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) # ..mist........(num, 28, 28)..................784.  
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) # .............index.....one hot...  
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

# .............batch_size..batch_size.nb_epoch.......... shuffle..................
# verbose.............verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
# ...0.....1.........2...epoch.......
# show_accuracy.............
# validation_split................
model.fit(X_train, Y_train, batch_size=200, nb_epoch=100, shuffle=True, verbose=1, show_accuracy=True, validation_split=0.3)  
print 'test set'  
model.evaluate(X_test, Y_test, batch_size=200, show_accuracy=True, verbose=1)  
