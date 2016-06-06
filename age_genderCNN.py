import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re, time

## To be used after face detection and cropping

def age_genderCNN(caffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, caffe_path + 'python')

	plt.rcParams['figure.figsize'] = (10, 10)        # large images
	plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	gender_model_prototxt = model_path + 'deploy_gender.prototxt'
	gender_model_trained = model_path + 'gender_net.caffemodel'
	age_model_prototxt = model_path + 'deploy_age.prototxt'
	age_model_trained = model_path + 'age_net.caffemodel'

	mean_path = model_path + 'age_gender_mean.npy'
	mu = np.load(mean_path).mean(1).mean(1)

	gender_net = caffe.Net(gender_model_prototxt,     # defines the structure of the model
	                gender_model_trained,  	# contains the trained weights
	                caffe.TEST)

	age_net = caffe.Net(age_model_prototxt,     # defines the structure of the model
	                age_model_trained,  	# contains the trained weights
	                caffe.TEST)

	gender_transformer = caffe.io.Transformer({'data': gender_net.blobs['data'].data.shape})
	gender_transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	gender_transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	gender_transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	gender_transformer.set_channel_swap('data', (2,1,0))
	
	age_transformer = caffe.io.Transformer({'data': age_net.blobs['data'].data.shape})
	age_transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	age_transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	age_transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	age_transformer.set_channel_swap('data', (2,1,0))

	# Assign batchsize
	batch_size = 10
	gender_data_blob_shape = gender_net.blobs['data'].data.shape
	gender_data_blob_shape = list(gender_data_blob_shape)
	gender_net.blobs['data'].reshape(batch_size, gender_data_blob_shape[1], gender_data_blob_shape[2], gender_data_blob_shape[3])

	age_data_blob_shape = age_net.blobs['data'].data.shape
	age_data_blob_shape = list(age_data_blob_shape)
	age_net.blobs['data'].reshape(batch_size, age_data_blob_shape[1], age_data_blob_shape[2], age_data_blob_shape[3])

	scores = None

	chunks_done = 0
	for chunk in [image_files[x:x+batch_size] for x in xrange(0, len(image_files), batch_size)]:
		print "Processing %.2f %%done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
		chunks_done = chunks_done + 1

		if len(chunk) < batch_size:
			gender_net.blobs['data'].reshape(len(chunk), gender_data_blob_shape[1], gender_data_blob_shape[2], gender_data_blob_shape[3])
			age_net.blobs['data'].reshape(len(chunk), age_data_blob_shape[1], age_data_blob_shape[2], age_data_blob_shape[3])

		gender_net.blobs['data'].data[...] = map(lambda y: gender_transformer.preprocess('data', caffe.io.load_image(y)), chunk)		
		gender_output = gender_net.forward()

		age_net.blobs['data'].data[...] = map(lambda y: age_transformer.preprocess('data', caffe.io.load_image(y)), chunk)		
		age_output = age_net.forward()

		#print gender_output , age_output

		if scores is None:
			scores = {}
			scores['gender_prob'] = gender_output['prob'].copy()
			scores['age_prob'] = age_output['prob'].copy()
			#fc8 = gender_net.blobs['fc8'].data[...].copy()
		else:
			scores['gender_prob'] = np.vstack((scores['gender_prob'],gender_output['prob']))
			scores['age_prob'] = np.vstack((scores['age_prob'],age_output['prob']))		

	gender_list = ['Male','Female']
	gender_labels = []
	age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
	age_labels = []

	for gender_proboutput, age_proboutput in zip(scores['gender_prob'], scores['age_prob']):

	 	gender = gender_list[gender_proboutput.argmax()]
	 	gender_labels.append(gender)
	 	age = age_list[age_proboutput.argmax()]
	 	age_labels.append(age)

	end = time.time()

	print "Time : %.3f \n"  %(end - start)

	return age_labels, gender_labels

