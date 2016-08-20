#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''
#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from data2 import load_data
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
import random

from keras.models import Sequential,Graph
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.recurrent_convolutional import LSTMConv2D

######################################
#     参数设置 
######################################

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 10

# Training
batch_size = 2
nb_epoch = 5

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''


#加载数据
data, label = load_data()
#打乱数据
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
#print(data.shape[0], ' samples')
#print ('label',label[1:10])

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
#label = np_utils.to_categorical(label, 10)

print ('data shape',data.shape)
print ('label shape',label.shape)
#print ('data',data[1])
#print ('label',label[1])

###############
#开始建立CNN模型
###############

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
seq.fit(X_train, Y_train, batch_size=32, verbose=1)
# === predict ===
predict = model.predict(data[1:10], batch_size=batch_size, verbose=0)
print( 'predict' )
print( predict )
print ('label',label[1:10])
###########
# Save
###########
model.save_weights('wgs_ConvLSTM.h5',overwrite=True)
exit()
