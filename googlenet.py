import os, sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import re, time
import cPickle

label_dict = {}
# vehicle_dict = ['airliner', 'warplane', 'military plane', 'airship', 'dirigible', 'space shuttle', 'fireboat', 'gondola', 
# 'speedboat', 'lifeboat', 'canoe', 'container ship', 'containership', 'container vessel', 'liner', 'ocean liner', 'aircraft carrier', 'carrier', 'flattop',
#   'attack aircraft carrier', 'submarine', 'pigboat', 'sub', 'U-boat', 'tank', 'army tank', 'armored combat vehicle', 'armoured combat vehicle', 
#   'wreck', 'freight car', 'passenger car', 'coach', 'carriage', 'motor scooter', 'scooter', 'bicycle-built-for-two', 'tandem bicycle', 'tandem mountain bike', 
#   'all-terrain bike', 'off-roader', 'electric locomotive', 'steam locomotive', 'ambulance', 'beach wagon', 'station wagon', 'wagon',
#   'estate car', 'beach waggon', 'station waggon', 'waggon', 'cab', 'hack', 'taxi', 'taxicab', 'convertible', 'jeep', 'landrover', 'limousine', 'limo',
#    'minivan', 'Model T', 'racer', 'race car', 'racing car', 'sports car', 'sport car', 'go-kart', 'golfcart', 'golf cart',
#     'moped', 'snowplow', 'snowplough', 'fire engine', 'fire truck', 'garbage truck', 'dustcart', 'pickup', 'pickup truck',
#      'tow truck', 'tow car', 'wrecker', 'trailer truck', 'tractor trailer', 'trucking rig', 'rig', 'articulated lorry', 'semi',
#       'moving van', 'police van', 'police wagon', 'paddy wagon', 'patrol wagon', 'wagon', 'black Maria', 'recreational vehicle',
#        'RV', 'R.V.', 'streetcar', 'tram', 'tramcar', 'trolley', 'trolley car', 'snowmobile', 'tractor', 'mobile home', 'manufactured home',
#         'tricycle', 'trike', 'velocipede', 'unicycle', 'monocycle', 'horse cart', 'horse-cart', 'car wheel', 'traffic light', 'traffic signal', 'stoplight',
#          'trolleybus', 'trolley coach', 'trackless trolley', 'bullet train', 'bullet','amphibious vehicle']

label_dict['natural formation'] = ['geological formation', 'natural depression', 'natural elevation', 'mountain', 
								'ridge','reef','shore','spring','location','point','geographic point','natural object']

label_dict['vehicle'] = ['airliner', 'warplane', 'airship', 'space shuttle', 'fireboat', 
		'gondola', 'speedboat', 'lifeboat', 'canoe', 'yawl', 'catamaran', 'trimaran', 'container ship',
		 'liner', 'aircraft carrier', 'submarine', 'wreck', 'half track', 'tank', 'missile', 'bobsled', 
		 'dogsled', 'bicycle-built-for-two', 'mountain bike', 'freight car', 'passenger car', 
		  'motor scooter', 'forklift', 'electric locomotive', 'steam locomotive', 'amphibian', 'ambulance', 
		  'beach wagon', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'Model T', 'racer', 
		  'sports car', 'go-kart', 'golfcart', 'moped', 'snowplow', 'fire engine', 'garbage truck', 'pickup', 
		  'tow truck', 'trailer truck', 'moving van', 'police van', 'recreational vehicle', 'streetcar', 'snowmobile', 
		  'tractor', 'mobile home', 'horse cart', 'jinrikisha', 'oxcart', 'plane', 'wing', 
		  'crane', 'car wheel', 'traffic light', 'school bus', 'container', 'wheeled vehicle', 'bicycle',
		   'car', 'handcart', 'self-propelled vehicle', 'armored vehicle', 'locomotive', 'motor vehicle', 'car', 'motorcycle',
		    'minibike', 'truck', 'van', 'passenger van', 'tracked vehicle', 'trailer', 'wagon', 'cart', 'conveyance', 'vehicle',
		     'craft', 'aircraft', 'heavier-than-air craft', 'airplane', 'lighter-than-air craft', 'vessel', 'boat', 'motorboat',
		      'sea boat', 'small boat', 'sailing vessel', 'sailboat', 'ship', 'cargo ship', 'passenger ship', 'warship', 
		      'submersible', 'military vehicle', 'rocket', 'sled', 'litter', 'public transport', 'train', 'bus', 'trolleybus']

# Removed passenger train, bullet train, shopping cart, barrow, tricycle, unicycle, balloon, airfoil

label_dict['weapon'] = ['revolver','cannon','assault rifle','rifle','projectile','bulletproof vest','weapon','gun',
						'firearm','autoloader','automatic firearm','automatic rifle','machine gun','pistol']

label_dict['person(s)'] = ['covering', 'sunglass', 'lens', 'converging lens', 'military uniform', 'uniform', 'vestment', 'abaya', 'robe', 'gown', 'outerwear', 'crash helmet', 'helmet', 'headdress', 'necktie', 'suit', 'Windsor tie', 'neckwear', 'necktie', 'bow tie', 'academic gown', 'clothing', 'attire', 'disguise', 'hairpiece', 'protective garment', 'accessory', 'belt', 'outerwear', 'gown', 'garment', 'vest', 'swimsuit', 'trouser', 'scarf', 'neckwear', 'necktie', 'skirt', 'overgarment', 'cloak', 'coat', 'raincoat', 'sweater', 'pullover', 'robe', 'shirt', 'undergarment', 'handwear', 'glove', 'headdress', 'helmet', 'cap', 'hat', 'footwear', 'hosiery', 'tights', 'stocking', 'uniform', 'nightwear', 'apparel', 'workwear', "woman's clothing", 'dress']

label_dict['building/structure'] = ['structure', 'street sign', 'sign', 'arch', 'area', 'bridge', 'building', 'farm building', 'house', 'residence', 'religious residence', 'outbuilding', 'shed', 'place of worship', 'shrine', 'theater', 'building complex', 'factory', 'column', 'defensive structure', 'fortification', 'establishment', 'institution', 'penal institution', 'correctional institution', 'place of business', 'mercantile establishment', 'marketplace', 'shop', 'housing', 'dwelling', 'landing', 'memorial']

label_dict['sports'] = ['person','contestant','player','athlete','racket', 'sports implement', 'ball', 'game equipment', 'sports equipment', 'equipment', 'football helmet', 'gymnastic apparatus','ping-pong ball','ballplayer','volleyball']


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

	net = caffe.Classifier(
            model_prototxt, model_trained,
            image_dims=(256, 256), raw_scale=255,
            mean=mu, channel_swap=(2, 1, 0))

	# Assign batchsize
	batch_size = 10
	num = 0
	final_label = []
	other_label = []

	chunks_done = 0
	for chunk in [image_files[x:x+batch_size] for x in xrange(0, len(image_files), batch_size)]:
		print "Processing %.2f%% done ..." %((batch_size*chunks_done*100)/float(len(image_files)))
		chunks_done = chunks_done + 1

		input_images = map(lambda y: caffe.io.load_image(y), chunk)
		output = net.predict(input_images, oversample = False).flatten()
		
		vect = 1000
		for single_output in [output[k:k+vect] for k in xrange(0,len(output),vect)]:
			num += 1
			bet = cPickle.load(open('/home/shruti/gsoc/caffehome/caffe/data/ilsvrc12/imagenet.bet.pickle'))
			bet['infogain'] -= np.array(bet['preferences']) * 0.1
			expected_infogain = np.dot(bet['probmat'], single_output[bet['idmapping']])
			expected_infogain *= bet['infogain']
			# sort the scores
			infogain_sort = expected_infogain.argsort()[::-1]
			# bet_result = [(bet['words'][v], '%.5f' % expected_infogain[v]) for v in infogain_sort[:5]]
			# print str(num) + ' bet result: %s\n', str(bet_result) 
			# print '\n'

			count_v = 0
			count_na = 0
			count_w = 0
			count_c = 0
			count_p = 0
			count_sp = 0
			fl_list = [0,0,0,0,0]
			label_list = []
			for v in infogain_sort[:5]:
				label_list.append((bet['words'][v], '%.5f' % expected_infogain[v]))
				if expected_infogain[v] > .2:
					if bet['words'][v] in label_dict['vehicle']:
						count_v += 1
						if expected_infogain[v] > .6:
							fl_list[0] = 1
					elif bet['words'][v] in label_dict['natural formation']:
						count_na += 1
						if expected_infogain[v] > .6:
							fl_list[1] = 1
					elif bet['words'][v] in label_dict['weapon']:
						count_w += 1
						if expected_infogain[v] > .6:
							fl_list[2] = 1
					elif bet['words'][v] in label_dict['person(s)']:
						count_c += 1
						if expected_infogain[v] > .6:
							fl_list[3] = 1
					elif bet['words'][v] in label_dict['building/structure']:
						count_p += 1
						if expected_infogain[v] > .6:
							fl_list[4] = 1
					elif bet['words'][v] in label_dict['sports']:
						count_sp += 1


			if count_v >= 3:
				bet_result = ' Vehicle\n'
				final_label.append('Vehicle/Accident')
			elif count_na >= 3:
				bet_result = ' Natural formation\n'
				final_label.append('Natural formation')
			elif count_w >= 3:
				bet_result = ' Weapon\n'
				final_label.append('Weapon')
			elif count_c >= 3:
				bet_result = ' Person(s)\n'
				final_label.append('Person(s)')
			elif count_p >= 3:
				bet_result = ' Building/structure\n'
				final_label.append('Building/structure')
			elif count_sp >= 3:
				bet_result = ' Sports\n'
				final_label.append('Sports')
			else:
				bet_result = ' Not\n'
				final_label.append('Not')

			if fl_list[0] == 1:
				other_label.append('Vehicle/Accident')
				other_result = ', Vehicle'
			elif fl_list[1] == 1:
				other_label.append('Natural formation')
				other_result = ', Natural formation'
			elif fl_list[2] == 1:
				other_label.append('Weapon')
				other_result = ', Weapon'
			elif fl_list[3] == 1:
				other_label.append('Person(s)')
				other_result = ', Person(s)'
			elif fl_list[4] == 1:
				other_label.append('Building/structure')
				other_result = ', Building/structure'
			else:
				other_label.append('Not')
				other_result = ''

			print str(num) + bet_result + other_result
			print str(label_list) + '\n'

	end = time.time()
	print "Googlenet Time : %.3f \n"  %(end - start)

	return final_label