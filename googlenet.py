import os, sys, re, time, ast
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cPickle
import path_params

def googlenet(pycaffe_path, model_path, image_files, mode, available_GPU_ID):

	start = time.time()

	sys.path.insert(0, pycaffe_path)

	plt.rcParams['figure.figsize'] = (10, 10)
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

	if mode == 'gpu':
		caffe.set_mode_gpu()
		caffe.set_device(available_GPU_ID[0])
	else
		caffe.set_mode_cpu()

	model_prototxt = path_params.googlenet_prototxt
	model_trained = path_params.googlenet_caffemodel

	mean_file = path_params.imagenet_mean
	mu = np.load(mean_file).mean(1).mean(1)

	net = caffe.Classifier(
            model_prototxt, model_trained,
            image_dims=(256, 256), raw_scale=255,
            mean=mu, channel_swap=(2, 1, 0))

	googlenet_category = []
	googlenet_labels = []
	with open(path_params.dictionary_file, 'r') as file:
		label_dict = ast.literal_eval(file.read())
	
	# Assign batchsize
	batch_size = 10
	chunks_done = 0
	for chunk in [image_files[x:x+batch_size] for x in xrange(0, len(image_files), batch_size)]:
		print "Processing %.2f%% done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
		chunks_done = chunks_done + 1

		input_images = map(lambda y: caffe.io.load_image(y), chunk)
		output = net.predict(input_images, oversample = False).flatten()
		
		vect = 1000
		for single_output in [output[k:k+vect] for k in xrange(0,len(output),vect)]:
			
			bet = cPickle.load(open(path_params.bet_pickle))
			bet['infogain'] -= np.array(bet['preferences']) * 0.1
			expected_infogain = np.dot(bet['probmat'], single_output[bet['idmapping']])
			expected_infogain *= bet['infogain']
			infogain_sort = expected_infogain.argsort()[::-1]
			
			counter = dict.fromkeys(['v', 'nf', 'w', 'p', 'b','sp',], 0)
			sums = dict.fromkeys(['v', 'nf', 'w', 'p', 'b','sp',], 0)
	
			fl_list = ['','','','','']
			label_list = []
			for v in infogain_sort[:5]:
				if expected_infogain[v] > .2:
					label_list.append('(' + bet['words'][v] + ', ' + str(float('%.2f' %expected_infogain[v])) + ')')
					if bet['words'][v] in label_dict['vehicle']:
						counter['v'] += 1
						sums['v'] += expected_infogain[v]
						if expected_infogain[v] > .6:
							fl_list[0] = 'Vehicle'
					if bet['words'][v] in label_dict['natural formation']:
						counter['nf'] += 1
						sums['nf'] += expected_infogain[v]
						if expected_infogain[v] > .6:
							fl_list[1] = 'Natural_formation'
					if bet['words'][v] in label_dict['weapon']:
						counter['w'] += 1
						sums['w'] += expected_infogain[v]
						if expected_infogain[v] > .6:
							fl_list[2] = 'Weapon'
					if bet['words'][v] in label_dict['person(s)']:
						counter['p'] += 1
						sums['p'] += expected_infogain[v]
						if expected_infogain[v] > .6:
							fl_list[3] = 'Person(s)/Clothing'
					if bet['words'][v] in label_dict['building/structure']:
						counter['b'] += 1
						sums['b'] += expected_infogain[v]
						if expected_infogain[v] > .6:
							fl_list[4] = 'Buidling/structure'
					if bet['words'][v] in label_dict['sports']:
						counter['sp'] += 1
						sums['sp'] += expected_infogain[v]

			label_list = ", ".join(map(str, label_list))
			googlenet_labels.append(label_list)

			if counter['v'] >= 3 or (counter['v'] == 2 and sums['v'] > .7):
				bet_result = 'Vehicle'
				fl_list[0] = ''
			elif counter['nf'] >= 3 or (counter['nf'] == 2 and sums['nf'] > .7):
				bet_result = 'Natural_formation'
				fl_list[1] = ''
			elif counter['w'] >= 3 or (counter['w'] == 2 and sums['w'] > .7):
				bet_result = 'Weapon'
				fl_list[2] = ''
			elif counter['p'] >= 3 or (counter['p'] == 2 and sums['p'] > .7):
				bet_result = 'Person(s)/Clothing'
				fl_list[3] = ''
			elif counter['b'] >= 3 or (counter['b'] == 2 and sums['b'] > .7):
				bet_result = 'Building/Structure'
				fl_list[4] = ''
			elif counter['sp'] >= 3 or (counter['sp'] == 2 and sums['sp'] > .7):
				bet_result = 'Sports'
			else:
				bet_result = 'Unclassified'

			other_result = ''
			for item in fl_list:
				if item != '' and other_result == '':
					other_result = ' ' + item
				elif item != '' and other_result != '':
					other_result += ', ' + item

			if other_result != '':		
				tmp = [bet_result, other_result]
				result = ','.join(tmp)
			else:
				result = bet_result
			googlenet_category.append(result)

	# print googlenet_labels

	end = time.time()
	print "Googlenet Time : %.3f \n"  %(end - start)

	return googlenet_category, googlenet_labels