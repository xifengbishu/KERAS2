#-*- coding: UTF-8 -*-
import os
import numpy as np
train=np.loadtxt('./DATA_train/training')
test=np.loadtxt('./DATA_train/testing')

X_train=train[:,1:6]
X_train[:,3]=X_train[:,3]*0.01
y_train=train[:,0]
X_test=test[:,1:6]
X_test[:,3]=X_test[:,3]*0.01
y_test=test[:,0]
print ( y_train )
print ( X_train )
#X_test=
#y_test=

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)



