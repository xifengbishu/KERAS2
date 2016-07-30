#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
    http://www.jianshu.com/p/8c36a5e42d6c?utm_source=itdadao&utm_medium=referral#
'''

######################################
#     import ConvNets 
######################################

#     ConvNets 
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU, LeakyReLU
import keras.layers.advanced_activations as adact
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.callbacks import EarlyStopping

#     collections 
from collections import Counter
import random, cPickle
from cutslice3d import load_data
from cutslice3d import ROW, COL, LABELTYPE, CHANNEL

#    nei cun diao yong 
import sys

def create_model(data):

    model = Sequential()

    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=data.shape[-3:])) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(32, 3, 3, border_mode='valid')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, border_mode='valid')) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(LABELTYPE, init='normal'))
    model.add(Activation('softmax'))

    sgd = SGD(l2=0.0, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    return model

def load_data():    

    ######################################
    #    ..mat.......
    ######################################

    mat_training = h5py.File(DATAPATH_Training);
    mat_training.keys()
    Training_CT_x = mat_training[Training_CT_1];
    Training_CT_y = mat_training[Training_CT_2];
    Training_CT_z = mat_training[Training_CT_3];
    Training_PT_x = mat_training[Training_PT_1];
    Training_PT_y = mat_training[Training_PT_2];
    Training_PT_z = mat_training[Training_PT_3];
    TrLabel = mat_training[Training_label];
    TrLabel = np.transpose(TrLabel);
    Training_Dataset = len(TrLabel);

    mat_validation = h5py.File(DATAPATH_Validation);
    mat_validation.keys()
    Validation_CT_x = mat_validation[Validation_CT_1];
    Validation_CT_y = mat_validation[Validation_CT_2];
    Validation_CT_z = mat_validation[Validation_CT_3];
    Validation_PT_x = mat_validation[Validation_PT_1];
    Validation_PT_y = mat_validation[Validation_PT_2];
    Validation_PT_z = mat_validation[Validation_PT_3];
    VaLabel = mat_validation[Validation_label];
    VaLabel = np.transpose(VaLabel);
    Validation_Dataset = len(VaLabel);

    ######################################
    #    ...
    ######################################
    TrData = np.empty((Training_Dataset, CHANNEL, ROW, COL), dtype = "float32");
    VaData = np.empty((Validation_Dataset, CHANNEL, ROW, COL), dtype = "float32");

    ######################################
    #   cut  
    ######################################
    for i in range(Training_Dataset):
        TrData[i,0,:,:]=Training_CT_x[:,:,i];
        TrData[i,1,:,:]=Training_CT_y[:,:,i];
        TrData[i,2,:,:]=Training_CT_z[:,:,i];
        TrData[i,3,:,:]=Training_PT_x[:,:,i];    
        TrData[i,4,:,:]=Training_PT_y[:,:,i]; 
        TrData[i,5,:,:]=Training_PT_z[:,:,i];

    for i in range(Validation_Dataset):
        VaData[i,0,:,:]=Validation_CT_x[:,:,i];
        VaData[i,1,:,:]=Validation_CT_y[:,:,i];
        VaData[i,2,:,:]=Validation_CT_z[:,:,i];
        VaData[i,3,:,:]=Validation_PT_x[:,:,i];    
        VaData[i,4,:,:]=Validation_PT_y[:,:,i]; 
        VaData[i,5,:,:]=Validation_PT_z[:,:,i];

    print ('\\tThe dimension of each data and label, listed as folllowing:')
    print ('\\tTrData  : ', TrData.shape)
    print ('\\tTrLabel : ', TrLabel.shape)
    print ('\\tRange : ', np.amin(TrData[:,0,:,:]), '~', np.amax(TrData[:,0,:,:]))
    print ('\\t\\t', np.amin(TrData[:,1,:,:]), '~', np.amax(TrData[:,1,:,:]))
    print ('\\t\\t', np.amin(TrData[:,2,:,:]), '~', np.amax(TrData[:,2,:,:]))
    print ('\\t\\t', np.amin(TrData[:,3,:,:]), '~', np.amax(TrData[:,3,:,:]))
    print ('\\t\\t', np.amin(TrData[:,4,:,:]), '~', np.amax(TrData[:,4,:,:]))
    print ('\\t\\t', np.amin(TrData[:,5,:,:]), '~', np.amax(TrData[:,5,:,:]))
    print ('\\tVaData  : ', VaData.shape)
    print ('\\tVaLabel : ', VaLabel.shape)
    print ('\\tRange : ', np.amin(VaData[:,0,:,:]), '~', np.amax(VaData[:,0,:,:]))
    print ('\\t\\t', np.amin(VaData[:,1,:,:]), '~', np.amax(VaData[:,1,:,:]))
    print ('\\t\\t', np.amin(VaData[:,2,:,:]), '~', np.amax(VaData[:,2,:,:]))
    print ('\\t\\t', np.amin(VaData[:,3,:,:]), '~', np.amax(VaData[:,3,:,:]))
    print ('\\t\\t', np.amin(VaData[:,4,:,:]), '~', np.amax(VaData[:,4,:,:]))
    print ('\\t\\t', np.amin(VaData[:,5,:,:]), '~', np.amax(VaData[:,5,:,:]))

    return TrData, TrLabel, VaData, VaLabel
######################################
#     SVM
######################################
def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("\\n>> cnn-svm Accuracy")
    prt(testlabel, pred_testlabel)

######################################
#     Random Forests
######################################
def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=100,criterion='gini')
    rfClf.fit(traindata,trainlabel)    
    pred_testlabel = rfClf.predict(testdata)
    print("\\n>> cnn-rf Accuracy")
    prt(testlabel, pred_testlabel)


cnt = Counter(A)
for k,v in cnt.iteritems():
    print ('\\t', k, '-->', v)
# .....A............

print(">> Build Model ...")
model = create_model(TrData)

######################################
#     Training ConvNets Model
######################################
print(">> Training ConvNets Model ...")
print("\\tHere, batch_size =", BATCH_SIZE, ", epoch =", EPOCH, ", lr =", LR, ", momentum =", MOMENTUM)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
hist = model.fit(TrData, TrLabel,                 \
                batch_size=BATCH_SIZE,         \
                nb_epoch=EPOCH,             \
                shuffle=True,                 \
                verbose=1,                     \
                show_accuracy=True,         \
                validation_split=VALIDATION_SPLIT,         \
                callbacks=[early_stopping])

######################################
#     Test ConvNets model
######################################
print(">> Test the model ...")
pre_temp=model.predict_classes(VaData)

######################################
#     Save ConvNets model
######################################

model.save_weights('MyConvNets.h5')
cPickle.dump(model, open('MyConvNets.pkl',"wb"))
json_string = model.to_json()
open(W_MODEL, 'w').write(json_string)
######################################
#     Load ConvNets model
######################################

model = cPickle.load(open('MyConvNets.pkl',"rb"))
