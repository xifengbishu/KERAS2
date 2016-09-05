#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''
#�昼入各种用到的模块组件
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
from data66 import load_data
import random
import numpy as np
import matplotlib.pyplot as plt
import time


images, captions, next_words,images_pre, captions_pre, next_words_pre, scale, mean = load_data()

nb_samples = 360
nb_channels = 5
width = 100
height = 100

max_caption_len = 6
vocab_size = 12

batch_size = 6
nb_epoch = 20

# "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
# "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
# containing word index sequences representing partial captions.
# "next_words" is a numpy float array of shape (nb_samples, vocab_size)
# containing a categorical encoding (0s and 1s) of the next word in the corresponding
# partial caption.

#images = np.random.random((nb_samples,nb_channels,width,height))
#captions = np.random.random((nb_samples,max_caption_len))
#next_words = np.random.random ((nb_samples,vocab_size))
'''
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
'''
print ( 'captions',captions[1] )
print ( 'images',images[1] )
print ( 'next_words',next_words[1] )
print ( 'images.shape',images.shape)
print ( 'captions.shape',captions.shape)
print ( 'next_words.shape',next_words.shape)
print ( 'images_pre.shape',images_pre.shape)
print ( 'captions_pre.shape',captions_pre.shape)
print ( 'next_words_pre.shape',next_words_pre.shape)
print ('scale',scale)
print ('mean',mean)
#exit()
###############
#开始建立CNN模型
###############

cnn_model = Sequential()

cnn_model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=images.shape[-3:])) 
cnn_model.add(Activation('relu'))

cnn_model.add(Convolution2D(8, 3, 3, border_mode='valid'))
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

cnn_model.add(Convolution2D(16, 3, 3, border_mode='valid')) 
cnn_model.add(Activation('relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

cnn_model.add(Flatten())
cnn_model.add(Dense(128))



# first, let's define an image model that
# will encode pictures into 128-dimensional vectors.
# it should be initialized with pre-trained weights.
image_model = Sequential()
image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(nb_channels, width, height)))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(64, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))

image_model.add(Flatten())
image_model.add(Dense(128))


# next, let's define a RNN model that encodes sequences of words
# into sequences of 128-dimensional word vectors.
language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
#language_model.add(LSTM(output_dim=128, return_sequences=True))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))
#language_model.add(TimeDistributedDense(128))

# let's repeat the image vector to turn it into a sequence.
image_model.add(RepeatVector(max_caption_len))

# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
model = Sequential()
#model.add(Merge([cnn_model, language_model], mode='concat', concat_axis=-1))
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
# let's encode this vector sequence into a single vector
model.add(GRU(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('relu'))

# let's load the weights from a save file.
#model.load_weights('wgs_merge66_20160905_122053.h5')

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
rmspro = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=rmspro)

#model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_split=0.2)
model.fit([images,captions], next_words, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_split=0.2)


print('Predicting')
predicted_output = model.predict([images_pre,captions_pre],batch_size=batch_size)

# === 
next_words_pre += mean 
next_words_pre *= scale
predicted_output += mean 
predicted_output *= scale
# ===
f = open("predicted_output.txt",'w')
f.write(str(predicted_output))
f.close()
f = open("next_word.txt",'w')
f.write(str(next_words_pre))
f.close()


print ( 'next_words_pre',next_words_pre )
print('predicted_lable',predicted_output)

print('Plotting Results')
plt.subplot(2, 2, 1)
plt.plot(next_words_pre[1],'k')
plt.subplot(2, 2, 1)
plt.plot(predicted_output[1])
plt.title('Predicted1')

plt.subplot(2, 2, 2)
plt.plot(next_words_pre[6],'k')
plt.subplot(2, 2, 2)
plt.plot(predicted_output[6])
plt.title('Predicted6')

plt.subplot(2, 2, 3)
plt.plot(next_words_pre[2],'k')
plt.subplot(2, 2, 3)
plt.plot(predicted_output[2])
plt.title('Predicted2')

plt.subplot(2, 2, 4)
plt.plot(next_words_pre[4],'k')
plt.subplot(2, 2, 4)
plt.plot(predicted_output[4])
plt.title('Predicted4')
#plt.show()
plt.savefig('merge66.jpg')
#plt.savefig('merge2.jpg', dpi = 600)


###########
# Save
###########
now_timeStamp=int(time.time())
timeArray = time.localtime(now_timeStamp)
otherStyleTime = time.strftime("%Y%m%d_%H%M%S",timeArray)
print ( otherStyleTime )
model.save_weights('wgs_merge66_'+otherStyleTime+'.h5',overwrite=True)


''' 
print ( 'label',next_words[1:5] )
print('predicted_lable',predicted_output)

print('Plotting Results')
plt.subplot(2, 2, 1)
plt.plot(next_words[1])
plt.title('Expected1')
plt.subplot(2, 2, 2)
plt.plot(predicted_output[1])
plt.title('Predicted1')
plt.subplot(2, 2, 3)
plt.plot(next_words[2])
plt.title('Expected1')
plt.subplot(2, 2, 4)
plt.plot(predicted_output[2])
plt.title('Predicted1')
#plt.show()
plt.savefig('merge33.jpg', dpi = 600)
'''
