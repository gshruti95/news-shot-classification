import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re

def placesCNNlabel(caffe_path, model_path, image_files):

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
	net.blobs['data'].reshape(10,         # batch size
	                          3,         # 3-channel (BGR) images
	                          227, 227)

	final_labelset = []
	final_label_list = []
	scene_type_list = []

	for image_file in image_files:

		input_image = caffe.io.load_image(image_file)
		transformed_image = transformer.preprocess('data', input_image)
		#plt.imshow(input_image)

		net.blobs['data'].data[...] = transformed_image

		output = net.forward()

		output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

		#print 'predicted class is:', output_prob.argmax()

		places_labels = model_path + 'IndoorOutdoor_places205.csv'

		labels = np.loadtxt(places_labels, str, delimiter='\t')

		toplabels_idx = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

		maxprob_label = labels[output_prob.argmax()]
		maxprob_label = re.findall(r"[\w]+",maxprob_label)


		if output_prob[toplabels_idx[0]] > .1 :			 # threshold for bad labels
			#print 'output label:' , maxprob_label[1]
			scene_type_no = maxprob_label[-1]
			if scene_type_no == '1':
				scene_type = 'indoor'
				#print 'scene type:', scene_type
			else:
				scene_type = 'outdoor'
				#print 'scene type:', scene_type

			if output_prob[toplabels_idx[0]] > .2 :
				final_label = maxprob_label[1]
			else:
				final_label = "Unknown"
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

	return final_label_list, scene_type_list, final_labelset
