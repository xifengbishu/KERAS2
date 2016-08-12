from keras.models import Sequential,Graph
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.recurrent_convolutional import LSTMConv2D

seq = Sequential()
seq.add(LSTMConv2D(nb_filter=15, nb_row=3, nb_col=3, input_shape=(10,40,40,1),
                   border_mode="same",return_sequences=True))
seq.add(LSTMConv2D(nb_filter=15,nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(LSTMConv2D(nb_filter=15, nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                      kernel_dim3=3, activation='sigmoid',
                   border_mode="same", dim_ordering="tf"))

seq.compile(loss="binary_crossentropy",optimizer="adadelta")
