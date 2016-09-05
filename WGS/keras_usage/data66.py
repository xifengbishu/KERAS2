#coding:utf-8
import numpy as np
#from Scientific.IO.NetCDF import NetCDFFile
from scipy.io import netcdf
#from netCDF4 import Dataset
import pylab as pl
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  

import os
from PIL import Image
import numpy as np

'''
for i in range(5,0,-1):
	print(i)
exit()
'''

def load_data():
	# --- hko_keras_79.01-11.12 ( 396 )	
	# --- era_keras_79.01-13.12 ( 420 )	
	# --- train 80.01-10.12 (13:372,360)
	# --- test  11.01-11.12 (373:396,12)

	nb_samples = 360
	nb_pre = 12
	nb_channels = 5
	width = 100
	height = 100

	max_caption_len = 6
	vocab_size = 12

	# --- label ---'
	txtfn='hko_keras_79-11'
	hko=np.loadtxt(txtfn,unpack='true')

	#label_all = np.empty((nb_samples,max_caption_len),dtype="float32")
	label_tra = np.empty((nb_samples,max_caption_len),dtype="float32")
	label_pre = np.empty((nb_pre,max_caption_len),dtype="float32")

	for i in range(nb_samples):
		for j in range(max_caption_len):
			label_tra[i,j] = hko[i+j+12-max_caption_len]
	for i in range(nb_pre):
		for j in range(max_caption_len):
       			label_pre[i,j] = hko[nb_samples+i+j+12-max_caption_len]
	'''
	print ( 'hko',hko[0:18] )
	print ( 'label_tra',label_tra[0:3])
	print ( 'hko_pre',hko[nb_samples+max_caption_len:nb_samples+max_caption_len+24])
	print ( 'label_pre',label_pre[0:3])
	print ( label_tra.shape )
	'''
	f = open("hko.txt",'w')
	f.write(str(hko))
	f.close()

	#exit()
	# --- next ---
	next_tra = np.empty((nb_samples,vocab_size),dtype="float32")
	next_pre = np.empty((nb_pre,vocab_size),dtype="float32")
	for i in range(nb_samples):
		for j in range(vocab_size):
			#k = i+j
			next_tra[i,j] = hko[i+j+12]
			#next_tra[i,j] = hko[i+j+max_caption_len-1]
			#next_tra[i,j] = hko[i+j+max_caption_len]
	for i in range(nb_pre):
		for j in range(vocab_size):
			#k = i+j
			next_pre[i,j] = hko[i+j+nb_samples+12]
			#next_pre[i,j] = hko[i+j+nb_samples+max_caption_len-1]
	'''
	print ( 'hko',hko[0:18] )
	print ( 'next_tra',next_tra[0:3])
	print ( 'hko_pre',hko[nb_samples+12:nb_samples+36])
	print ( 'next_pre',next_pre[0:3])
	print ( label_tra.shape )
	'''

	f = open("label_tra.txt",'w')
	f.write(str(label_tra))
	f.close()
	f = open("next_tra.txt",'w')
	f.write(str(next_tra))
	f.close()
	f = open("label_pre.txt",'w')
	f.write(str(label_pre))
	f.close()
	f = open("next_pre.txt",'w')
	f.write(str(next_pre))
	f.close()

	#exit()
	#print ( next[1:10])
	#print ( next.shape )
	# --- input cnn --- read matlab dat   
	# --- ERA 197901-201312 
	matfn='alldata.mat'  
	mat=sio.loadmat(matfn)  
	#print type(mat)
	#print mat
	sst1=mat['sst']
	t2m=mat['t2m']
	msl=mat['msl']
	u10=mat['u10']
	v10=mat['v10']

	lon=mat['lon']
	lat=mat['lat']
	time=mat['time']
        sst=np.nan_to_num(sst1)
	#print (sst1[1:100])
	#print (sst[1:100])
	print ( sst.shape )
	#exit()

	#print ( 'lon', lon)
	#print ( 'lat', lat)
	#data = np.empty((5,100,100,420),dtype="float32")
	data_tra = np.empty((nb_samples,nb_channels,width,height),dtype="float32")
	data_pre = np.empty((nb_pre,nb_channels,width,height),dtype="float32")
	for i in range(nb_samples):
		for j in range(width):
			for k in range(height):
				data_tra[i,0,j,k] = sst[k,99-j,i+12]
				data_tra[i,1,j,k] = t2m[k,99-j,i+12]
				data_tra[i,2,j,k] = msl[k,99-j,i+12]
				data_tra[i,3,j,k] = u10[k,99-j,i+12]
				data_tra[i,4,j,k] = v10[k,99-j,i+12]
	for i in range(nb_pre):
		for j in range(width):
			for k in range(height):
				data_pre[i,0,j,k] = sst[k,99-j,i+nb_samples+12]
				data_pre[i,1,j,k] = t2m[k,99-j,i+nb_samples+12]
				data_pre[i,2,j,k] = msl[k,99-j,i+nb_samples+12]
				data_pre[i,3,j,k] = u10[k,99-j,i+nb_samples+12]
				data_pre[i,4,j,k] = v10[k,99-j,i+nb_samples+12]
	#print 'origtion'
	#print (data[1:5,:,50,50])
	for i in range(nb_channels):
		sacle = np.max(data_tra[:,i,:,:])
		data_tra[:,i,:,:] /= sacle
		data_pre[:,i,:,:] /= sacle
		mean = np.std(data_tra[:,i,:,:])
		data_tra[:,i,:,:] -= mean
		data_pre[:,i,:,:] -= mean
		#print ('sacle',sacle,'mean',mean)
	
	#print 'nomalization'
	#print (data[1:5,:,50,50])
	#print (data[:,0,:,:])
	sacle = np.max(label_tra)
	label_tra /= sacle
	label_pre /= sacle
	mean = np.std(label_tra)
	label_tra -= mean
	label_pre -= mean

	sacle = np.max(next_tra)
	next_tra /= sacle
	next_pre /= sacle
	mean = np.std(next_tra)
	next_tra -= mean
	next_pre -= mean


	return data_tra,label_tra,next_tra,data_pre,label_pre,next_pre,sacle,mean


data_tra,label_tra,next_tra,data_pre,label_pre,next_pre,sacle,mean = load_data()
print ( 'data_tra.shape',data_tra.shape )
print ( 'data_pre',data_pre.shape )

print ( 'label_tra.shape',label_tra.shape )
print ( 'label_pre',label_pre.shape )

print ( 'next_tra',next_tra.shape )
print ( 'next_pre',next_pre.shape )

# ===============


'''
f = open("test.txt",'w')
f.write(str(label_tra))
f.close()
'''

'''
# === 0-1 ===
def Normalization(x):
	return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
x=[1,2,1,4,3,2,5,6,2,7]
b=Normalization(x)
images = np.random.random((100,10,10))
print ( b )
# === -1 - 1 ===
def Normalization2(x):
	return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
x=[1,2,1,4,3,2,5,6,2,7]
b=Normalization2(x)
print ( b )

	#归一化和零均值化
	
	scale = np.max(data)
	print ( 'sacle',scale )
	print ( 'data',data[100] )
	data /= scale
	print ( 'data',data[100] )
	mean = np.std(data)
	print ( 'mean',mean )
	data -= mean
	print ( 'data',data[100] )

'''

'''
print (sst[1:10])
print ( sst.shape )
x = np.linspace(0, 395, 396) 
print ( x.shape )

print ( x )

exit()
plt.figure(figsize=(8,4)) 
plt.plot(x,sst,label="$sin(x)$",color="red",linewidth=2)  
plt.xlabel("Time(s)")  
plt.ylabel("Volt")  
plt.title("PyPlot First Example")  

#plt.ylim(-1.2,1.2)  
#plt.show()  
#plt.plot(sstb)
#plt.title('Predicted1')
#plt.savefig('hko_keras_79-11.jpg')
#pl.savefig('hko_keras_79-11.jpg', dpi = 600)
#exit()
#matlab文件名  
matfn='/public/wind_flow/flow/alldata.mat'  
mat=sio.loadmat(matfn)  
#print type(mat)
#print mat
sst=mat['sst']
t2m=mat['t2m']
msl=mat['msl']
u10=mat['u10']
v10=mat['v10']

lon=mat['lon']
lat=mat['lat']
time=mat['time']
print ( sst.shape )
#print ( 'lon', lon)
#print ( 'lat', lat)
#data = np.empty((5,100,100,420),dtype="float32")
data = np.empty((420,5,100,100),dtype="float32")
num = 420
for i in range(num):
	for j in range(100):
		for k in range(100):
			data[i,0,j,k] = sst[k,99-j,i]
			data[i,1,j,k] = t2m[k,99-j,i]
			data[i,2,j,k] = msl[k,99-j,i]
			data[i,3,j,k] = u10[k,99-j,i]
			data[i,4,j,k] = v10[k,99-j,i]
#print data[:,0,:,:]
plt.figure(1)  
plt.contourf(data[1,0,:,:])  
plt.savefig('new_sst.jpg')
plt.figure(2)  
plt.quiver( data[1,3,:,:],data[1,4,:,:] )  
plt.savefig('new_uv.jpg')
#plt.show()  
exit()	
#data[1,:,:,:] = t2m
#data[2,:,:,:] = msl
#data[3,:,:,:] = u10
#data[4,:,:,:] = v10
print data[1]
print ( data.shape )

ww = data.reshape((5,100,100,420))

#print type(sst)
#print type(data)
plt.close('all') 
 
#xi=data['xi']  
#yi=data['yi']  
#ui=data['ui']  
#vi=data['vi']  

#plt.figure(1)  
#plt.quiver( data[3,:,:,5],data[4,:,:,5] )  
#plt.quiver( u10[::5,::5],v10[::5,::5],ui[::5,::5],vi[::5,::5])  
plt.figure(2)  
plt.contourf(ww[0,:,:,5])  
#plt.contourf(data[0,:,:,5])  
plt.show()  
  
#sio.savemat('saveddata.mat', {'xi': xi,'yi': yi,'ui': ui,'vi': vi})  
'''

'''

root_grp = Dataset('./DATA/nrt_global_allsat_msla_h_20160106_20160112.nc')

temp = root_grp.variables['sla']

for i in range(len(temp)):
    pl.clf()
    pl.contourf(temp[i])
    pl.show()
    raw_input('Press enter.')

exit()
a = np.loadtxt('2012030306.csv')
b = np.shape(a)
print ( b )
print ( a[0:6])


f = netcdf.netcdf_file('./DATA/nrt_global_allsat_msla_h_20160106_20160112.nc', 'r')
print f.history
time = f.variables['time']
print time.units
print time.shape
print time[:]
f.close()

#f = NetCDFFile('./DATA/nrt_global_allsat_msla_h_20160106_20160112.nc', 'r')
'''
