import os, sys, re, time
import numpy as np
import matplotlib.pyplot as plt
import caffe
import path_params

def get_labels(labels, scores, attribute_responses, scene_attributeNames):
	
	places_labels = []
	final_label_list = []
	scene_type_list = []
	scene_attributes_list = []

	for idx, output_prob in enumerate(scores['prob']):
		vote = 0
		toplabels_idx = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
	
		if output_prob[toplabels_idx[0]] > .1 :			 # threshold for bad labels	
			for top5_idx in toplabels_idx:
				if labels[top5_idx][-1] == '1':
					vote = vote + 1
			if vote > 2:
				scene_type = 'Indoor'
			else:
				scene_type = 'Outdoor'
		else:
			scene_type = 'None'

		label_list = []
		for label_prob, label_idx in zip(output_prob[toplabels_idx], toplabels_idx):
			if label_prob > .2 :
				label_list.append('(' + re.findall(r"[\w]+", labels[label_idx])[1] + ', ' + str(float('%.2f' %label_prob)) + ')')			

		label_list = ', '.join(map(str, label_list))
		places_labels.append(label_list)
		scene_type_list.append(scene_type)	
		print places_labels
		## Scene attributes
		attribute_response = attribute_responses[idx]
		attribute_index = attribute_response.argsort()[::-1][:5]
		scene_attributes = scene_attributeNames[attribute_index]
		scene_attributes = ", ".join(scene_attributes)
		scene_attributes_list.append(scene_attributes)

	return scene_type_list, places_labels, scene_attributes_list

def get_scene_attribute_responses(scene_attributeValues, fc7):
	# Returs the scene attributes for the fc7 features
	scene_attributeValues = np.transpose(scene_attributeValues)
	attribute_responses = np.dot(fc7, scene_attributeValues)

	return attribute_responses
	
def placesCNN(pycaffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, pycaffe_path)

	plt.rcParams['figure.figsize'] = (10, 10)
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	model_prototxt = path_params.placesCNN_prototxt
	model_trained = path_params.placesCNN_caffemodel

	mean_path = path_params.placesCNN_mean
	mu = np.load(mean_path).mean(1).mean(1)

	net = caffe.Net(model_prototxt,     # defines the structure of the model
	                model_trained,  	# contains the trained weights
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)
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
			fc7 = net.blobs['fc7'].data[...].copy()
			# fc8 = net.blobs['fc8'].data[...].copy()
			# fc6 = net.blobs['fc6'].data[...].copy()
			
		else:
			scores['prob'] = np.vstack((scores['prob'],output['prob']))
			fc7 = np.vstack((fc7, net.blobs['fc7'].data[...].copy()))
			# fc8 = np.vstack((fc8, net.blobs['fc8'].data[...].copy()))
			# fc6 = np.vstack((fc6, net.blobs['fc6'].data[...].copy()))
			
	places_labels = path_params.places_labels
	labels = np.loadtxt(places_labels, str, delimiter='\t')

	scene_attributeValues = np.loadtxt(path_params.scene_values, delimiter = ',')
	scene_attributeNames = np.loadtxt(path_params.scene_names, delimiter = '\n', dtype = str)
	attribute_responses = get_scene_attribute_responses(scene_attributeValues, fc7)

	scene_type_list, places_labels, scene_attributes_list = get_labels(labels, scores, attribute_responses, scene_attributeNames)
	
	end = time.time()
	print "Time : %.3f \n"  %(end - start)
	
	return fc7, scene_type_list, places_labels, scene_attributes_list