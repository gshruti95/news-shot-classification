import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re, time
import cPickle

vehicle_dict = ['airliner', 'warplane', 'military plane', 'airship', 'dirigible', 'space shuttle', 'fireboat', 'gondola', 
'speedboat', 'lifeboat', 'canoe', 'container ship', 'containership', 'container vessel', 'liner', 'ocean liner', 'aircraft carrier', 'carrier', 'flattop',
  'attack aircraft carrier', 'submarine', 'pigboat', 'sub', 'U-boat', 'tank', 'army tank', 'armored combat vehicle', 'armoured combat vehicle', 
  'wreck', 'freight car', 'passenger car', 'coach', 'carriage', 'motor scooter', 'scooter', 'bicycle-built-for-two', 'tandem bicycle', 'tandem mountain bike', 
  'all-terrain bike', 'off-roader', 'electric locomotive', 'steam locomotive', 'ambulance', 'beach wagon', 'station wagon', 'wagon',
  'estate car', 'beach waggon', 'station waggon', 'waggon', 'cab', 'hack', 'taxi', 'taxicab', 'convertible', 'jeep', 'landrover', 'limousine', 'limo',
   'minivan', 'Model T', 'racer', 'race car', 'racing car', 'sports car', 'sport car', 'go-kart', 'golfcart', 'golf cart',
    'moped', 'snowplow', 'snowplough', 'fire engine', 'fire truck', 'garbage truck', 'dustcart', 'pickup', 'pickup truck',
     'tow truck', 'tow car', 'wrecker', 'trailer truck', 'tractor trailer', 'trucking rig', 'rig', 'articulated lorry', 'semi',
      'moving van', 'police van', 'police wagon', 'paddy wagon', 'patrol wagon', 'wagon', 'black Maria', 'recreational vehicle',
       'RV', 'R.V.', 'streetcar', 'tram', 'tramcar', 'trolley', 'trolley car', 'snowmobile', 'tractor', 'mobile home', 'manufactured home',
        'tricycle', 'trike', 'velocipede', 'unicycle', 'monocycle', 'horse cart', 'horse-cart', 'car wheel', 'traffic light', 'traffic signal', 'stoplight',
         'trolleybus', 'trolley coach', 'trackless trolley', 'bullet train', 'bullet','amphibious vehicle']

def googlenet(caffe_path, model_path, image_files):

	start = time.time()

	sys.path.insert(0, caffe_path + 'python')

	plt.rcParams['figure.figsize'] = (10, 10)        # large images
	plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
	plt.rcParams['image.cmap'] = 'gray'

	caffe.set_mode_cpu()

	model_prototxt = model_path + 'deploy.prototxt'
	model_trained = model_path + 'bvlc_googlenet.caffemodel'

	mean_file = model_path + 'imagenet_mean.npy'
	mu = np.load(mean_file).mean(1).mean(1)

	# net = caffe.Net(model_prototxt,     # defines the structure of the model
	#                 model_trained,  	# contains the trained weights
	#                 caffe.TEST)

	# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	# transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	# transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	# transformer.set_channel_swap('data', (2,1,0))
	
	net2 = caffe.Classifier(
            model_prototxt, model_trained,
            image_dims=(256, 256), raw_scale=255,
            mean=mu, channel_swap=(2, 1, 0))

	# Assign batchsize
	batch_size = 10
	# data_blob_shape = net.blobs['data'].data.shape
	# data_blob_shape = list(data_blob_shape)
	# net.blobs['data'].reshape(batch_size, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])


	count = 0

	for image in image_files:
		# print image_files[69]
		bet = cPickle.load(open('/home/shruti/gsoc/caffehome/caffe/data/ilsvrc12/imagenet.bet.pickle'))

		input_image = caffe.io.load_image(image)	
		another = net2.predict([input_image], oversample = True).flatten()
		count += 1

		bet['infogain'] -= np.array(bet['preferences']) * 0.1
		expected_infogain = np.dot(bet['probmat'], another[bet['idmapping']])
		expected_infogain *= bet['infogain']

		print len(expected_infogain)
		# sort the scores
		infogain_sort = expected_infogain.argsort()[::-1]
		bet_result = [(bet['words'][v], '%.5f' % expected_infogain[v]) for v in infogain_sort[:5]]

		print str(count) + ' bet result: %s', str(bet_result)

	# scores = None
	# chunks_done = 0
	# for chunk in [image_files[x:x+batch_size] for x in xrange(0, len(image_files), batch_size)]:
	# 	print "Processing %.2f%% done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
	# 	chunks_done = chunks_done + 1

	# 	# if len(chunk) < batch_size:
	# 		# net.blobs['data'].reshape(len(chunk), data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	# 	# net.blobs['data'].data[...] = map(lambda y: transformer.preprocess('data', caffe.io.load_image(y)), chunk)
	# 	input_images = map(lambda y: caffe.io.load_image(y), chunk)
	# 	print chunk
	# 	# input_image = caffe.io.load_image(image_files[0])	
	# 	# print input_image
	# 	# output = net.forward()
	# 	another = net2.predict(input_images, oversample = True).flatten()
	# 	print len(another), type(another)
		
	# 	# places_labels = model_path + 'synset_words.txt'
	# 	# labels = np.loadtxt(places_labels, str, delimiter='\t')
		
	# 	print len(another[bet['idmapping']])

	# 	vect = 1000
	# 	for another_img in [another[k:k+vect] for k in xrange(0,len(another),vect)]:

	# 		count += 1
	# 		bet['infogain'] -= np.array(bet['preferences']) * 0.1
	# 		expected_infogain = np.dot(bet['probmat'], another_img[bet['idmapping']])
	# 		expected_infogain *= bet['infogain']

	# 		print len(expected_infogain)
	# 		# sort the scores
	# 		infogain_sort = expected_infogain.argsort()[::-1]
	# 		bet_result = [(bet['words'][v], '%.5f' % expected_infogain[v]) for v in infogain_sort[:5]]

	# 		print str(count) + ' bet result: %s', str(bet_result)





		# if scores is None:
		# 	scores = {}
		# 	scores['prob'] = output['prob'].copy()
		# 	# allout = {}
		# 	# allout = another.copy()
		# else:
		# 	scores['prob'] = np.vstack((scores['prob'], output['prob']))
		# 	# allout = np.vstack((allout, another))
			
	# print "images len: ", len(image_files)
	# for idx, output_prob in enumerate(scores['prob']):
	# 	toplabels_idx = output_prob.argsort()[::-1][:5]
	# 	toplabels = [labels[toplabel_idx].split(' ',1)[1] for toplabel_idx in toplabels_idx]
	# 	maxprob_label = labels[output_prob.argmax()].split(' ',1)[1]
	# 	print "max label: ", maxprob_label, idx
	# 	print "top labels: ", toplabels
	# 	label_list.append(maxprob_label)
	# 	class_list.append('not vehicle')
	# 	for toplabel in toplabels:
	# 		if toplabel in vehicle_dict:
	# 			class_list[idx] = 'vehicle'
	# 			break

	# fordot = scores['prob']

	
	end = time.time()
	print "Googlenet Time : %.3f \n"  %(end - start)

	# new_lines = []
	# with open('vehicle_labels' + ".vis",'w') as file:
	# 	for output, output_label in zip(label_list, class_list): 
	# 		new_lines.append(output_label + ' ' + output + '\n')
	# 	file.writelines(new_lines)

	# return label_list
