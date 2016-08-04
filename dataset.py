import os, sys
import fnmatch
import googlenet
import fileops
import cropframes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/test/'
def dataset(main_dir):

	dir_list = sorted(os.listdir(main_dir))

	features_data = []
	label_data = []
	fc6_data = []
	mean_feat = []
	googlenet_data = []
	

	for dir_name in dir_list:
		if os.path.isdir(main_dir + dir_name):

			# print main_dir + dir_name
			# print main_dir + dir_name + '/' + dir_name + '_shot_type_testuser.txt'

			if os.path.exists(main_dir + dir_name + '/' + dir_name + '_shot_type_testuser.txt'):

				
				with open(main_dir + dir_name + '/' + dir_name + '_shot_type_testuser.txt') as labels_file:
					labels = labels_file.readlines()				
				labels = [label.split('\t')[0] for label in labels]
				label_data += labels

				with open(main_dir + dir_name + '/' + 'cropped_places_fc7 .csv') as features_file:
					features = features_file.readlines()
				features = [feature.split('\n')[0] for feature in features]
				features_data += features

				# image_files = fileops.get_keyframeslist(main_dir + dir_name + '/')
				# image_files = cropframes.cropframes(main_dir + dir_name + '/', image_files)

				# print main_dir + dir_name + '/'
				# googlenet_labels = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
				# googlenet_data += googlenet_labels

				# for file in os.listdir(main_dir + dir_name):
					
				# 	if fnmatch.fnmatch(file,'*_shot_type*.txt'):
				# 		with open(main_dir + dir_name+'/'+file) as labels_file:
				# 			labels = labels_file.readlines()				
				# 		labels = [label.split('\t')[0] for label in labels]
				# 		label_data += labels

				# 	if fnmatch.fnmatch(file,'new_places_fc7 .csv'):
				# 		with open(main_dir + dir_name+'/'+file) as features_file:
				# 			features = features_file.readlines()
				# 		features = [feature.split('\n')[0] for feature in features]
				# 		features_data += features
					
					# with open(main_dir + dir_name+'/'+'new_places_fc6.csv') as fc6_file:
					# 	fc6 = fc6_file.readlines()			

					# for feature, fc6_item in zip(features, fc6):
					# 	feature = feature.split('\n')[0]
					# 	fc6_item = fc6_item.split('\n')[0]
					# 	feature = feature.split(',')
					# 	fc6_item = fc6_item.split(',')

					# 	new_vector = []
					# 	for item1, item2 in zip(feature,fc6_item):
					# 		# print type(item1)
					# 		# print item1
					# 		new_item = (float(item1) + float(item2))/2
					# 		# print type(new_item)
					# 		# print type(new_vector)
					# 		new_vector.append(new_item)
						
					# 	new_vector = str(new_vector).strip('[]')
					# 	mean_feat.append(new_vector)	


	## To exclude Commercial class etc.

	labels = []
	features = []
	glabels = []
	for label, feature in zip(label_data, features_data):
		# label = label.split('\t')[0]
		if label not in ['Commercial']:#,'Problem/Unclassified']:
			# if label == 'Vehicle/Accident':#, 'Background_roll','Background roll']:
			labels.append(label)
			# else:
				# labels.append('Not')
			features.append(feature)
			# glabels.append(glabel)	

	# p_v = 0
	# v = 0
	# crt_v = 0
	# for i in range(len(labels)):

	# 	if glabels[i] == 'Vehicle/Accident':
	# 		p_v += 1
	# 		if labels[i] == 'Vehicle/Accident':
	# 			crt_v += 1
	# 	if labels[i] == 'Vehicle/Accident':
	# 		v += 1

	# print "crt_v:%d v:%d p_v:%d" %(crt_v,v,p_v)

	# t_names = ['class Clothing', 'class Natural', 'class Not', 'class Place/building', 'class Vehicle', 'class Weapon']
	# print len(labels), len(glabels)
	# print(classification_report(labels, glabels, target_names = t_names))
	# print "Accuracy score: ", accuracy_score(labels, glabels)
	
	return features, labels
