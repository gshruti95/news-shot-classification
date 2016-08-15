import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re, time

def mynet(caffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, caffe_path + 'python')

	plt.rcParams['figure.figsize'] = (10, 10)        # large images
	plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	model_prototxt = model_path + 'deploy_3class.prototxt'
	model_trained = model_path + '3class_finetune_caffenet_iter_100000.caffemodel'

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

	my_labels = model_path + '3class.csv'
	labels = np.loadtxt(my_labels, str, delimiter='\t')

	[output_labels, labels_set] = get_labels(labels, scores)
	
	end = time.time()
	print "Time : %.3f \n"  %(end - start)

	return output_labels, labels_set

def get_labels(labels, scores):

	final_labels = []
	labels_set = []
	for idx, output_prob in enumerate(scores['prob']):

		toplabels_idx = output_prob.argsort()[::-1][:5]

		if output_prob[toplabels_idx[0]] > .2:
			maxprob_label = labels[output_prob.argmax()][1]
		else:
			maxprob_label = 'Problem/Unclassified'

		final_labels.append(maxprob_label)

		label_list = []
		for label_prob, label_idx in zip(output_prob[toplabels_idx], toplabels_idx):
			if label_prob > .2 :
				label_list.append((labels[label_idx][1], float('%.2f' %label_prob)))
		
		label_list = ', '.join(map(str, label_list))		
		labels_set.append(label_list)

	return final_labels, labels_set


def performance(orig_labels, output, labels_set):

	labels = []
	output_labels = []
	for label, output_label in zip(orig_labels, output):
		if label not in ['Commercial', 'Problem/Unclassified']:
			if label == 'Studio' or label == 'Reporter' or label == 'Hybrid':
				label = 'Newsperson(s)'
			elif label == 'Background_roll' or label == 'Talking_head' or label == 'Talking_head/Hybrid':
				label = 'Background_roll'
			labels.append(label)
			output_labels.append(output_label)

	p_np = 0
	p_s = 0
	p_r = 0
	p_h = 0
	p_bg = 0
	p_sp = 0
	p_w = 0
	p_g = 0
	p_prob = 0
	p_c = 0

	np = 0
	s = 0
	r = 0
	h = 0
	bg = 0
	sp = 0
	w = 0
	g = 0
	prob = 0
	c = 0

	crt_np = 0
	crt_s = 0
	crt_r = 0
	crt_h = 0
	crt_bg = 0
	crt_sp = 0
	crt_w = 0
	crt_g = 0
	crt_prob = 0
	crt_c = 0
	
	for i in range(len(output_labels)):
		
		print 'Max label: %s label: %s' %(output_labels[i], labels[i])
		print 'Result: ', labels_set[i]
		if output_labels[i] == 'Studio' and labels[i] not in ['Weather', 'Sports']:
			p_s += 1
			if labels[i] == output_labels[i]:
				crt_s += 1
		elif output_labels[i] == 'Reporter' and labels[i] not in ['Weather', 'Sports']:
			p_r += 1
			if labels[i] == output_labels[i]:
				crt_r += 1
		elif output_labels[i] == 'Hybrid' and labels[i] not in ['Weather', 'Sports']:
			p_h += 1
			if labels[i] == output_labels[i]:
				crt_h += 1
		elif output_labels[i] == 'Newsperson(s)' and labels[i] not in ['Weather', 'Sports']:
			p_np += 1
			if labels[i] == output_labels[i]:
				crt_np += 1
		elif output_labels[i] == 'Graphic' and labels[i] not in ['Weather', 'Sports']:
			p_g += 1 
			if labels[i] == output_labels[i]:
				crt_g += 1
		elif output_labels[i] == 'Weather' and labels[i] not in ['Weather', 'Sports']:
			p_w += 1
			if labels[i] == output_labels[i]:
				crt_w += 1
		elif output_labels[i] == "Sports" and labels[i] not in ['Weather', 'Sports']:
			p_sp += 1
			if labels[i] == output_labels[i]:
				crt_sp += 1
		elif (output_labels[i] == "Background_roll" or output_labels[i] == "Background roll") and labels[i] not in ['Weather', 'Sports']:
			p_bg += 1
			if labels[i] == output_labels[i]:
				crt_bg += 1
		elif output_labels[i] == 'Problem/Unclassified':
			p_prob += 1
			if labels[i] == output_labels[i]:
				crt_prob += 1

		if labels[i] == 'Studio':
			s += 1
		elif labels[i] == 'Reporter':
			r += 1
		elif labels[i] == 'Hybrid':
			h += 1
		elif labels[i] == 'Newsperson(s)':
			np += 1
		elif labels[i] == 'Graphic':
			g += 1	
		elif labels[i] == 'Weather':
			w += 1
		elif labels[i] == 'Sports':
			sp += 1
		elif labels[i] == 'Background_roll' or labels[i] == 'Background roll':
			bg += 1
		elif labels[i] == 'Commercial':
			c += 1
		elif labels[i] == 'Problem/Unclassified':
			prob += 1

	print "Predicted: NP: %d, G %d, W: %d, SP: %d, BG: %d" %(p_np, p_g, p_w, p_sp, p_bg)
	print "Correct: NP: %d, G: %d, W: %d, SP: %d, BG: %d"  %(crt_np, crt_g, crt_w, crt_sp, crt_bg)
	print "Total: NP: %d, G: %d, W: %d, SP: %d, BG: %d" %(np, g, w, sp, bg)

