import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re, time

def placesCNN(caffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, caffe_path + 'python')

	plt.rcParams['figure.figsize'] = (10, 10)        # large images
	plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	model_prototxt = model_path + 'places205CNN_deploy_upgraded.prototxt'
	model_trained = model_path + 'places205CNN_iter_300000_upgraded.caffemodel'

	mean_path = model_path + 'places205CNN_mean.npy'
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
			fc8 = net.blobs['fc8'].data[...].copy()
			fc7 = net.blobs['fc7'].data[...].copy()
			fc6 = net.blobs['fc6'].data[...].copy()
			# pool5 = net.blobs['pool5'].data[...].copy()
			# conv5 = net.blobs['conv5'].data[...].copy()			
			# conv4 = net.blobs['conv4'].data[...].copy()
			# conv3 = net.blobs['conv3'].data[...].copy()
		else:
			scores['prob'] = np.vstack((scores['prob'],output['prob']))
			fc8 = np.vstack((fc8, net.blobs['fc8'].data[...].copy()))
			fc7 = np.vstack((fc7, net.blobs['fc7'].data[...].copy()))
			fc6 = np.vstack((fc6, net.blobs['fc6'].data[...].copy()))
			# pool5 = np.vstack((pool5, net.blobs['pool5'].data[...].copy()))
			# conv5 = np.vstack((conv5, net.blobs['conv5'].data[...].copy()))
			# conv4 = np.vstack((conv4, net.blobs['conv4'].data[...].copy()))
			# conv3 = np.vstack((conv3, net.blobs['conv3'].data[...].copy()))

	places_labels = model_path + 'IndoorOutdoor_places205.csv'
	labels = np.loadtxt(places_labels, str, delimiter='\t')

	scene_attributeValues = np.loadtxt(model_path + 'attributeValues.csv', delimiter = ',')
	scene_attributeNames = np.loadtxt(model_path + 'attributeNames.csv', delimiter = '\n', dtype = str)
	attribute_responses = get_scene_attribute_responses(scene_attributeValues, fc7)

	final_label_list, scene_type_list, final_labelset, scene_attributes_list = get_labels(labels, scores, attribute_responses, scene_attributeNames)
	
	for idx, item in enumerate(final_labelset):
		print "%d %s %s\n" %(idx+1, item, scene_type_list[idx]) 

	end = time.time()
	print "Time : %.3f \n"  %(end - start)
	
	return fc8, fc7, fc6, scene_type_list, scene_attributes_list #, final_label_list, scene_type_list, final_labelset, scene_attributes_list


def get_labels(labels, scores, attribute_responses, scene_attributeNames):
	
	final_labelset = []
	final_label_list = []
	scene_type_list = []
	scene_attributes_list = []

	for idx, output_prob in enumerate(scores['prob']):

		vote = 0
		#count = 0
		toplabels_idx = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

		maxprob_label = labels[output_prob.argmax()]
		maxprob_label = re.findall(r"[\w]+",maxprob_label)

		if output_prob[toplabels_idx[0]] > .1 :			 # threshold for bad labels
			
			for top5_idx in toplabels_idx:
				#if output_prob[top5_idx] > .08:
					#count = count + 1
				if labels[top5_idx][-1] == '1':
					vote = vote + 1

			if vote > 2:
				scene_type = 'indoor'
			else:
				scene_type = 'outdoor'

			# scene_type_no = maxprob_label[-1]
			# if scene_type_no == '1':
			# 	scene_type = 'indoor'
			# 	print 'scene type:', scene_type
			# else:
			# 	scene_type = 'outdoor'
			# 	print 'scene type:', scene_type_no

			if output_prob[toplabels_idx[0]] > .2 :
				final_label = maxprob_label[1]
				#print 'scores label:' , final_label
			else:
				final_label = "Unknown"
				#print 'scores label: Unknown'
		else:
			final_label = "Unknown"
			scene_type = 'unknown'
			#print "Did not return reasonably accurate label!"

		final_label_list.append(final_label)
		scene_type_list.append(scene_type)	

		label_set = []
		prob_set = []	

		no_label_flag = 0 

		for label_prob, label_idx in zip(output_prob[toplabels_idx],toplabels_idx):
			if label_prob > .2 :
				label_set.append(re.findall(r"[\w]+",labels[label_idx])[1])
				prob_set.append(label_prob)
				no_label_flag = 1

			label_list = zip(label_set,prob_set)
		
		if no_label_flag == 0:
			 label_set.append('None')
			 prob_set.append(0.000)
			 label_list = zip(label_set,prob_set)
				
		#print "probabilities and labels: %.3f %s" %(label_prob, re.findall(r"[\w]+",labels[label_idx])[1])
			
		label_list = "; ".join( "%s, %s" %tup for tup in label_list )
		final_labelset.append(label_list)

		## Scene attributes
		attribute_response = attribute_responses[idx]
		attribute_index = attribute_response.argsort()[::-1][:5]
		scene_attributes = scene_attributeNames[attribute_index]
		scene_attributes = ",".join(scene_attributes)
		scene_attributes_list.append(scene_attributes)

	return final_label_list, scene_type_list, final_labelset, scene_attributes_list

def get_scene_attribute_responses(scene_attributeValues, fc7):
	'''Returs the scene attributes for the fc7 features'''
	scene_attributeValues = np.transpose(scene_attributeValues)
	attribute_responses = np.dot(fc7, scene_attributeValues)

	return attribute_responses