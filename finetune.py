import os, sys, re, time
import numpy as np
import matplotlib.pyplot as plt
import caffe
import path_params

def mynet(pycaffe_path, model_path, image_files, mode, available_GPU_ID):

	start = time.time()

	sys.path.insert(0, pycaffe_path)

	plt.rcParams['figure.figsize'] = (10, 10)
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

	if mode == 'gpu':
		caffe.set_mode_gpu()
		caffe.set_device(available_GPU_ID[0])
	else:
		caffe.set_mode_cpu()

	my_labels = path_params.finetune_labels
	model_prototxt = path_params.finetune_prototxt
	model_trained = path_params.finetune_caffemodel

	mean_path = path_params.finetune_mean
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
		print "Processing %.2f%% done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
		chunks_done = chunks_done + 1

		if len(chunk) < batch_size:
			net.blobs['data'].reshape(len(chunk), data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

		net.blobs['data'].data[...] = map(lambda y: transformer.preprocess('data', caffe.io.load_image(y)), chunk)		
		output = net.forward()

		if scores is None:
			scores = {}
			scores['prob'] = output['prob'].copy()
		else:
			scores['prob'] = np.vstack((scores['prob'],output['prob']))

	labels = np.loadtxt(my_labels, str, delimiter='\t')
	[output_labels, labels_set] = get_labels(labels, scores)
	
	end = time.time()
	print "Time : %.3f \n"  %(end - start)

	return output_labels, labels_set

def get_labels(labels, scores):

	final_labels = []
	labels_set = []
	for idx, output_prob in enumerate(scores['prob']):

		toplabels_idx = output_prob.argsort()[::-1][:3]

		if output_prob[toplabels_idx[0]] > .2:
			maxprob_label = labels[output_prob.argmax()][1]
		else:
			maxprob_label = 'Problem/Unclassified'
		final_labels.append(maxprob_label)

		label_list = []
		for label_prob, label_idx in zip(output_prob[toplabels_idx], toplabels_idx):
			if label_prob > .2:
				label_list.append('(' + labels[label_idx][1] + ', ' + str(float('%.3f' %label_prob)) + ')')
		label_list = ', '.join(map(str, label_list))	
		labels_set.append(label_list)

	return final_labels, labels_set

