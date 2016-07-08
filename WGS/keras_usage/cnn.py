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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from six.moves import range
from data import load_data
import random


batch_size = 100
nb_epoch = 50


#加载数据
data, label = load_data()
#打乱数据
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
print(data.shape[0], ' samples')
print(data.shape, ' data.shape')
#print ( 'data',data[1] )
#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
#label = np_utils.to_categorical(label, 10)
#print ( 'label',label[1] )

###############
#开始建立CNN模型
###############

#生成一个model
model = Sequential()

#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
#border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
#激活函数用tanh
#你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
model.add(Convolution2D(4, 5, 5, border_mode='valid',input_shape=data.shape[-3:])) 
model.add(Activation('relu'))


#第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#第三个卷积层，16个卷积核，每个卷积核大小3*3
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(16, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#全连接层，先将前一层输出的二维特征图flatten为一维的。
#Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
#全连接有128个神经元节点,初始化方式为normal
model.add(Flatten())
model.add(Dense(1, init='normal'))
model.add(Activation('relu'))


#Softmax分类，输出是10类别
#model.add(Dense(1, init='normal'))
#model.add(Activation('tanh'))



#############
#开始训练模型
##############
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
rmspro = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=rmspro)


#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
#validation_split=0.2，将20%的数据作为验证集。
model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True,verbose=1,validation_split=0.2)
print('Predicting')
predicted_output = model.predict(data[1:5], batch_size=batch_size)
print ( 'label',label[1:5] )
print('predicted_lable',predicted_output)

"""
#使用data augmentation的方法
#一些参数和调用的方法，请看文档
datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images

# compute quantities required for featurewise normalization 
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(data)

for e in range(nb_epoch):
    print('-'*40)
    print('Epoch', e)
    print('-'*40)
    print("Training...")
    # batch train with realtime data augmentation
    progbar = generic_utils.Progbar(data.shape[0])
    for X_batch, Y_batch in datagen.flow(data, label):
        loss,accuracy = model.train(X_batch, Y_batch,accuracy=True)
        progbar.add(X_batch.shape[0], values=[("train loss", loss),("accuracy:", accuracy)] )

"""

