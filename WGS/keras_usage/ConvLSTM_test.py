import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Convolution3D
from keras.layers.recurrent_convolutional import LSTMConv2D

all_data = np.loadtxt('../final_weekly_data.txt')
sample_size = all_data.shape[0]
all_data = all_data.reshape((sample_size, 1, 107, 72))

filter_size = 3

SE = all_data[:, :, 107/2:, 0:72/2]

ts_data = np.ones((155, 3, 1, 54, 36))

for i in range(ts_data.shape[0]):
    ts_data[i, :, :, :, :] = SE[i:i+3]
X = ts_data[:, :-1, :, :, :]
Y = ts_data[:, 1:, :, :, :]

X_train = X[:X.shape[0]*.6]
Y_train = Y[:Y.shape[0]*.6]

print X_train.shape, Y_train.shape
model = Sequential()
model.add(LSTMConv2D(2, 3, 3, input_shape=(2, 1, 54, 36),
          border_mode='same', dim_ordering='th', return_sequences=True))
model.add(LSTMConv2D(2, 3, 3, border_mode='same',
          dim_ordering='th', return_sequences=True))
model.add(LSTMConv2D(2, 3, 3, border_mode='same',
          dim_ordering='th', return_sequences=True))
model.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                        kernel_dim3=3, activation='sigmoid',
                        border_mode="same", dim_ordering="th"))
model.compile(loss='binary_crossentropy', optimizer='adadelta')
model.fit(X_train, Y_train, batch_size=32, verbose=1)
