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


def load_data():
	nb_samples = 384
	nb_channels = 3
	width = 100
	height = 100

	max_caption_len = 1
	vocab_size = 12

	# --- lable ---'
	txtfn='hko_keras_79-11'
	hko=np.loadtxt(txtfn,unpack='true')

	label = np.empty((nb_samples,max_caption_len),dtype="float32")
	for i in range(nb_samples):
       		label[i] = hko[i] 
	#print ( label[1:10])
	#print ( label.shape )
	# --- next ---
	next = np.empty((nb_samples,vocab_size),dtype="float32")
	for i in range(nb_samples):
		for j in range(vocab_size):
			k = i+j
			next[i,j] = hko[i+j]

	#print ( next[1:10])
	#print ( next.shape )
	# --- input cnn --- read matlab dat   
	matfn='alldata.mat'  
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
	data = np.empty((nb_samples,nb_channels,width,height),dtype="float32")
	for i in range(nb_samples):
		for j in range(width):
			for k in range(height):
				data[i,0,j,k] = sst[k,99-j,i]
				data[i,1,j,k] = t2m[k,99-j,i]
				data[i,2,j,k] = msl[k,99-j,i]
				#data[i,3,j,k] = u10[k,99-j,i]
				#data[i,4,j,k] = v10[k,99-j,i]
	return data,label,next

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
