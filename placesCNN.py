import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re

def placesCNNlabel_singleframe(caffe_path, model_path, image_file):

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
	net.blobs['data'].reshape(1,         # batch size
	                          3,         # 3-channel (BGR) images
	                          227, 227)

	
	input_image = caffe.io.load_image(image_file)
	transformed_image = transformer.preprocess('data', input_image)
	plt.imshow(input_image)

	net.blobs['data'].data[...] = transformed_image

	output = net.forward()

	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

	#print 'predicted class is:', output_prob.argmax()

	places_labels = model_path + 'IndoorOutdoor_places205.csv'

	labels = np.loadtxt(places_labels, str, delimiter='\t')

	toplabels_idx = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	maxprob_label = labels[output_prob.argmax()]
	maxprob_label = re.findall(r"[\w]+",maxprob_label)

	if output_prob[toplabels_idx[0]] > .2 :			 # threshold for bad labels
		final_label = maxprob_label[1]
		#print 'output label:' , maxprob_label[1]
		scene_type_no = maxprob_label[-1]
		if scene_type_no == '1':
			scene_type = 'indoor'
			#print 'scene type:', scene_type
		else:
			scene_type = 'outdoor'
			#print 'scene type:', scene_type
	else:
		scene_type = 'unknown'
		#print "Did not return reasonably accurate label!"

	label_set = []
	prob_set = []

	for label_prob, label_idx in zip(output_prob[toplabels_idx],toplabels_idx):
		if label_prob > .2 :
			label_set.append(re.findall(r"[\w]+",labels[label_idx])[1])
			prob_set.append(label_prob)
			final_set = zip(label_set,prob_set)
			#print "probabilities and labels: %.3f %s" %(label_prob, re.findall(r"[\w]+",labels[label_idx])[1])

	return final_label, scene_type, final_set
