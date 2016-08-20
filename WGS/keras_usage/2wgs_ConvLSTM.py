#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''
#导入各种用到的模块组件

from keras.models import Sequential,Graph
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.recurrent_convolutional import LSTMConv2D

from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, TimeDistributedDense, RepeatVector
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Merge
from six.moves import range
from data3 import load_data
import random
import numpy as np
import matplotlib.pyplot as plt



######################################
#     加载数据
######################################
#images, captions, next_words,images_pre, captions_pre, next_words_pre, scale, mean = load_data()

######################################
#     参数设置 
######################################
nb_samples = 360
nb_channels = 5
width = 100
height = 100

max_caption_len = 1
vocab_size = 12

batch_size = 2
nb_epoch = 10

# "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
# "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
# containing word index sequences representing partial captions.
# "next_words" is a numpy float array of shape (nb_samples, vocab_size)
# containing a categorical encoding (0s and 1s) of the next word in the corresponding
# partial caption.

images_input = np.random.random((nb_samples,nb_channels,width,height))
images_ouput = np.random.random((nb_samples,nb_channels,width,height))
#captions = np.random.random((nb_samples,max_caption_len))
#next_words = np.random.random ((nb_samples,vocab_size))
'''
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
'''


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
'''
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
