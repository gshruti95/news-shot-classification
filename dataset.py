import os, sys
import fnmatch
# import classifier

# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/test/'
def dataset(main_dir):

	dir_list = sorted(os.listdir(main_dir))

	features_data = []
	label_data = []

	for dir_name in dir_list:
		if os.path.isdir(main_dir + dir_name):
			for file in os.listdir(main_dir + dir_name):
				if fnmatch.fnmatch(file,'new_places_fc8.csv'):
					with open(main_dir + dir_name+'/'+file) as features_file:
						features = features_file.readlines()
					features = [feature.split('\n')[0] for feature in features]
					features_data += features

				if fnmatch.fnmatch(file,'*_shot_type*.txt'):
					with open(main_dir + dir_name+'/'+file) as labels_file:
						labels = labels_file.readlines()
					
					labels = [label.split('\t')[0] for label in labels]
					label_data += labels

	labels = []
	features = []
	for label, feature in zip(label_data, features_data):
		label = label.split('\t')[0]
		if label not in ['Commercial', 'Problem/Unclassified', 'Background_roll']:
			labels.append(label)
			features.append(feature)					
	
	# with open(main_dir + 'new_train_data.csv','w') as train_data_file:
	# 		train_data_file.writelines(features_data)
		
	# with open(main_dir + 'new_label_data.csv','w') as label_data_file:
	# 	label_data_file.writelines(label_data)

	# features_data = [data.split('\n')[0] for data in features_data]
	# label_data = [data.split('\n')[0] for data in label_data]
	# print "orig total " , len(label_data), len(features_data)
	# print "final total ", len(labels) , len(features)

	return features, labels


