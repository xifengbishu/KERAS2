#coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe

"""


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，图像大小28*28
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
	mnist_num = 4200
	data = np.empty((mnist_num,1,28,28),dtype="float32")
	#label = np.empty((mnist_num,),dtype="float32")
	label = np.empty((mnist_num,),dtype="uint8")
	#label = np.empty((42000,),dtype="uint8")
        
        
        X_test = np.random.random((mnist_num, 1, 28 , 28))
        y_test = np.random.random(mnist_num)+20
        X_train = np.random.random((mnist_num, 1, 28 ,28))
        y_train = np.random.random(mnist_num)+20

	imgs = os.listdir("./mnist")
	num = len(imgs)
	#print ('imgs num',num)
	#for i in range(num):
	for i in range(mnist_num):
	#for i in range(1):
		img = Image.open("./mnist/"+imgs[i])
		#print ('img',img)
		arr = np.asarray(img,dtype="float32")
		#print ('arr.shape',arr.shape)
		data[i,:,:,:] = arr
		label[i] = int(imgs[i].split('.')[0])
		#print ( 'imgs[i].split(.)',imgs[i].split('.'))
		#print ('data',data[i])
		#print ('label',label[i])
	return data,label




