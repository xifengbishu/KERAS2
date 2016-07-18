#coding:utf-8
import numpy as np
#from Scientific.IO.NetCDF import NetCDFFile
from scipy.io import netcdf

#from netCDF4 import Dataset
import pylab as pl

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
  
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
print ( 'lon', lon)
print ( 'lat', lat)
exit()
data = np.empty((5,100,100,420),dtype="float32")
#ww = np.empty((420,5,100,100),dtype="float32")
data[0,:,:,:] = sst
data[1,:,:,:] = t2m
data[2,:,:,:] = msl
data[3,:,:,:] = u10
data[4,:,:,:] = v10
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
