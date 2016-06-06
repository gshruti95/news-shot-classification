import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re, time

def googlenet(caffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, caffe_path + 'python')

	plt.rcParams['figure.figsize'] = (10, 10)        # large images
	plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	model_prototxt = model_path + 'deploy.prototxt'
	model_trained = model_path + 'bvlc_googlenet.caffemodel'

	mean_path = model_path + 'imagenet_mean.npy'
	mu = np.load(mean_path).mean(1).mean(1)

	net = caffe.Net(model_prototxt,     # defines the structure of the model
	                model_trained,  	# contains the trained weights
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))
	
	# Assign batchsize
	batch_size = 10
	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batch_size, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	scores = None

	chunks_done = 0
	for chunk in [image_files[x:x+batch_size] for x in xrange(0, len(image_files), batch_size)]:
		print "Processing %.2f %%done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
		chunks_done = chunks_done + 1

		if len(chunk) < batch_size:
			net.blobs['data'].reshape(len(chunk), data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

		net.blobs['data'].data[...] = map(lambda y: transformer.preprocess('data', caffe.io.load_image(y)), chunk)		
		output = net.forward()

		print output

		if scores is None:
			scores = {}
			scores['prob'] = output['prob'].copy()
		else:
			scores['prob'] = np.vstack((scores['prob'],output['prob']))
			
	places_labels = model_path + 'synset_words.txt'
	labels = np.loadtxt(places_labels, str, delimiter='\t')
	label_list = []

	for output_prob in scores['prob']:
		maxprob_label = labels[output_prob.argmax()].split(' ',1)[1]
		label_list.append(maxprob_label)

	#print label_list

	end = time.time()

	print "Time : %.3f \n"  %(end - start)

	return label_list
