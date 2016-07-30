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
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from data2 import load_data
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.embeddings import Embedding

import sys
import numpy as np
import os

import random
np.random.seed(1337)  # for reproducibility

######################################
#     参数设置 
######################################
# GRN
max_caption_len = 6
vocab_size = 30

nb_samples = 1000
nb_width = 100
nb_height = 100

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 10

# Training
batch_size = 2
nb_epoch = 10

#  rnn
max_features = 10000
maxlen = 5  # cut texts after this number of words (among top max_features most common words)

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''


######################################
#           加载数据
######################################
'''
data, label = load_data()
#打乱数据
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

train=np.loadtxt('./DATA_train/training')
test=np.loadtxt('./DATA_train/testing')

X_train=train[:,1:6]
X_train[:,3]=X_train[:,3]*0.01
y_train=train[:,0]
X_test=test[:,1:6]
X_test[:,3]=X_test[:,3]*0.01
y_test=test[:,0]
'''
images = np.random.random((nb_samples,3,nb_width,nb_height)) # float array of shape (nb_samples, nb_channels=3, width, height)
partial_captions = np.random.random_integers(20, size=(nb_samples,max_caption_len) ) # numpy integer array of shape (nb_samples, max_caption_len)
next_words = np.random.random((nb_samples,vocab_size)) # numpy float array of shape (nb_samples, vocab_size)

test_images = np.random.random((nb_samples,3,nb_width,nb_height)) # float array of shape (nb_samples, nb_channels=3, width, height)
test_partial_captions = np.random.random_integers(20, size=(nb_samples,max_caption_len) ) # numpy integer array of shape (nb_samples, max_caption_len)
test_next_words = np.random.random((nb_samples,vocab_size)) # numpy float array of shape (nb_samples, vocab_size)

print('images shape:', images.shape)
print('partial_captions shape:', partial_captions.shape)
print('next_words shape:', next_words.shape)
print('images',images[1:3])
print('partial_captions',partial_captions[1:3])
print('next_words',next_words[1:3])
###############
#开始建立CNN模型
###############

#生成一个model
cnn_model = Sequential()
cnn_model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=images.shape[-3:])) 
cnn_model.add(Activation('relu'))
cnn_model.add(Convolution2D(8, 3, 3, border_mode='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Convolution2D(16, 3, 3, border_mode='valid')) 
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Activation('sigmoid'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, init='normal'))
'''
cnn_model.add(Activation('relu'))
cnn_model.add(Dense(1, init='normal'))
cnn_model.add(Activation('relu'))
'''
# ========= 
'''
# LSTM
LSTM_model = Sequential()
LSTM_model.add(Embedding(vocab_size, 256, input_length=maxlen, dropout=0.2))
LSTM_model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2,return_sequences=True))  # try using a GRU instead, for fun
LSTM_model.add(TimeDistributed(Dense(128)))
'''
LSTM_model = Sequential()
LSTM_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
LSTM_model.add(LSTM(output_dim=128, return_sequences=True))
LSTM_model.add(TimeDistributed(Dense(128)))
'''
LSTM_model.add(Dense(1))
LSTM_model.add(Activation('relu'))
'''
# let's repeat the image vector to turn it into a sequence.
cnn_model.add(RepeatVector(max_caption_len))
# ====================
# first, let's define an image model that
# will encode pictures into 128-dimensional vectors.
# it should be initialized with pre-trained weights.
image_model = Sequential()
image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Flatten())
image_model.add(Dense(128))

# let's load the weights from a save file.
#image_model.load_weights('weight_file.h5')

# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))

# let's repeat the image vector to turn it into a sequence.
image_model.add(RepeatVector(max_caption_len))

# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
model = Sequential()
model.add(Merge([cnn_model, LSTM_model], mode='concat',concat_axis=-1))
#model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('relu'))

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.compile(optimizer='rmsprop',
              loss='mse')

# "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
# "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
# containing word index sequences representing partial captions.
# "next_words" is a numpy float array of shape (nb_samples, vocab_size)
# containing a categorical encoding (0s and 1s) of the next word in the corresponding
# partial caption.
model.fit([images, partial_captions], next_words, batch_size=batch_size, nb_epoch=nb_epoch,verbose=0)

predict = model.predict([test_images, test_partial_captions], batch_size=batch_size, verbose=0)
print( 'predict' )
print( predict )
