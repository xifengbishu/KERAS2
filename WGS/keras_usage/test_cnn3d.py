c_Filter = [32,64]
c_Kernel = [7,7]
c_Nodes = [128,256]
c_Patch = [7,13]
c_Bias = 0.5
c_SamplesPerPatient = 200
c_Iterations = [1000,200]

model_p1 = Sequential()
model_p1.add(Convolution3D(nb_filter=c_Filter[0], len_conv_dim1=c_Kernel[0], len_conv_dim2=c_Kernel[0], len_conv_dim3=c_Kernel[0], init='normal', W_regularizer=l2(0.4), border_mode='valid', input_shape=(1,c_Patch[0],c_Patch[0],c_Patch[0])))
model_p1.add(Activation('relu'))
model_p1.add(Dropout(0.5))

model_p1.add(Flatten())
model_p1.add(Dense(c_Nodes[0],init='normal'))
model_p1.add(Activation('relu'))
model_p1.add(Dropout(0.5))

model_p1.add(Dense(1))
model_p1.add(Activation('sigmoid'))

model_p1.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06))

model_p1.fit_generator(construct_biased_dataset_generator(c_SamplesPerPatient,c_Bias,c_Patch[0],c_Images[0:2],c_Segs[0:2]),samples_per_epoch=4000,nb_epoch=c_Iterations[0],verbose=1,nb_worker=4)

EDIT: and to output the filters:

W_conv1 = model_p1.layers[0].get_weights()

for i in range(c_Filter[0]):
filter = np.zeros(shape=(c_Kernel[0],c_Kernel[0],c_Kernel[0]))
filter = (W_conv1[0])[i,0,:,:,:]
output_Image(filter, c_Path + str(i) + '.mhd')
