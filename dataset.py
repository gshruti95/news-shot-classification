import os, sys
import fnmatch
# import classifier

# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/test/'
def dataset(main_dir):

	dir_list = sorted(os.listdir(main_dir))

	features_data = []
	label_data = []
	fc6_data = []
	mean_feat = []
	

	for dir_name in dir_list:
		if os.path.isdir(main_dir + dir_name):
			for file in os.listdir(main_dir + dir_name):
				
				if fnmatch.fnmatch(file,'new_places_fc7 .csv'):
					with open(main_dir + dir_name+'/'+file) as features_file:
						features = features_file.readlines()
					features = [feature.split('\n')[0] for feature in features]
					features_data += features
					
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

				if fnmatch.fnmatch(file,'*_shot_type*.txt'):
					with open(main_dir + dir_name+'/'+file) as labels_file:
						labels = labels_file.readlines()				
					labels = [label.split('\t')[0] for label in labels]
					label_data += labels


	## To exclude Commercial class etc.

	labels = []
	features = []
	for label, feature in zip(label_data, features_data):
		# label = label.split('\t')[0]
		if label not in ['Commercial', 'Problem/Unclassified', 'Background_roll','Background roll']:
			labels.append(label)
			features.append(feature)	


	return features, labels


