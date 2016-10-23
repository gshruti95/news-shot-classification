import os, sys, re, time, ast
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cPickle
import path_params


def yolo(pycaffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, pycaffe_path)

	plt.rcParams['figure.figsize'] = (10, 10)
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	model_prototxt = path_params.yolo_prototxt
	model_trained = path_params.yolo_caffemodel

	net = caffe.Net(model_prototxt,     # defines the structure of the model
	                model_trained,  	# contains the trained weights
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	# transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	# transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	batch_size = 10
	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batch_size, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	scores = None
	images_widths = []
	images_heights = []
	chunks_done = 0
	for chunk in [image_files[x:x+batch_size] for x in xrange(0, len(image_files), batch_size)]:
		print "Processing %.2f%% done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
		chunks_done = chunks_done + 1

		if len(chunk) < batch_size:
			net.blobs['data'].reshape(len(chunk), data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

		net.blobs['data'].data[...] = map(lambda y: transformer.preprocess('data', caffe.io.load_image(y)), chunk)		
		chunk_widths = map(lambda y: caffe.io.load_image(y).shape[1], chunk)
		chunk_heights = map(lambda y: caffe.io.load_image(y).shape[0], chunk)
			
		images_widths += chunk_widths
		images_heights += chunk_heights
		# print images_widths
		output = net.forward()

		if scores is None:
			scores = {}
			scores['result'] = output['result'].copy()

		else:
			scores['result'] = np.vstack((scores['result'],output['result']))

	[person_count, obj_loc_set] = get_labels(scores, images_widths, images_heights)	

	end = time.time()
	print "Time : %.3f \n"  %(end - start)

	return person_count, obj_loc_set


def get_labels(scores, images_widths, images_heights):

	classes = np.loadtxt(path_params.yolo_labels, str, delimiter='\t')
	obj_loc_set = []
	persons_set = []
	for idx, output in enumerate(scores['result']):
		# print output
		w_img = images_widths[idx]
		h_img = images_heights[idx]
		# print w_img, h_img
		threshold = 0.2
		iou_threshold = 0.5
		num_class = 20
		num_box = 2
		grid_size = 7
		probs = np.zeros((7,7,2,20))
		class_probs = np.reshape(output[0:980],(7,7,20))
	#	print class_probs
		scales = np.reshape(output[980:1078],(7,7,2))
	#	print scales
		boxes = np.reshape(output[1078:],(7,7,2,4))
		offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

		boxes[:,:,:,0] += offset
		boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
		boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
		boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
		boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
			
		boxes[:,:,:,0] *= w_img
		boxes[:,:,:,1] *= h_img
		boxes[:,:,:,2] *= w_img
		boxes[:,:,:,3] *= h_img

		for i in range(2):
			for j in range(20):
				probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
		filter_mat_probs = np.array(probs>=threshold,dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs)
		boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]
			
		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0 : continue
			for j in range(i+1,len(boxes_filtered)):
				if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
					probs_filtered[j] = 0.0
			
		filter_iou = np.array(probs_filtered>0.0,dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]

		all_objs = []
		persons = 0
		obj_loc = []
		for i in range(len(boxes_filtered)):	
			x = int(boxes_filtered[i][0])
			y = int(boxes_filtered[i][1])	
			w = int(boxes_filtered[i][2])//2		
			h = int(boxes_filtered[i][3])//2
			loc = 'x,y,w,h= [' + str(x) + ', ' + str(y) + ', ' + str(w) + ', ' + str(h) + ']'
			if classes[classes_num_filtered[i]][1] == 'person':
				persons += 1
				obj_loc.append('(person[' + str(i+1) + '], ' + '{0:.2f}'.format(probs_filtered[i]) + ') -> ' + loc)
			else:
				# obj_loc.append('(' + classes[classes_num_filtered[i]][1] + '[' + str(i+1) + '], ' + '{0:.2f}'.format(probs_filtered[i]) + ') -> ' + loc)
				obj_loc.append('')

		new_obj_loc = ''
		for item in obj_loc:
			if item != '' and new_obj_loc != '':
				new_obj_loc += ', ' + item
			elif item!= '' and new_obj_loc == '':
				new_obj_loc = item

		# obj_loc = ', '.join(map(str, obj_loc))
		obj_loc_set.append(new_obj_loc)
		persons_set.append(str(persons))

		# print 'Persons count= ' + str(persons) + ' | ' + obj_loc
		# print obj_loc_set			

	return persons_set, obj_loc_set

def iou(box1,box2):

	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


# pycaffe_path = path_params.pycaffe_path
# yolo_path = path_params.yolo_path

# image_files = ['./full-clips/train/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595_keyframe0765.jpg'
# ,'./full-clips/train/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595_keyframe0745.jpg'
# ,'./full-clips/train/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595_keyframe0705.jpg']

# yolo(pycaffe_path, yolo_path, image_files)