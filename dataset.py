import os

def testset(clip_dir, fc7_file):

	test_features = []
	with open(clip_dir + fc7_file) as features_file:
		features = features_file.readlines()
	features = [feature.split('\n')[0] for feature in features]
	test_features += features

	return test_features

def ovo_trainset(train_labels, class_type):
	
	new_train_labels = []
	if class_type == 'np':
		for label in train_labels:
			if label == 'Studio' or label == 'Reporter' or label == 'Hybrid' or label == 'Newsperson(s)':
				label = 'Newsperson(s)'
			else:
				label = 'Not'
			new_train_labels.append(label)		

	return new_train_labels

def trainset(main_dir, annotations_file, fc7_file):

	dir_list = sorted(os.listdir(main_dir))

	features_data = []
	label_data = []
	ann_files = []
	all_keyframes = []

	for dir_name in dir_list:
		if os.path.isdir(main_dir + dir_name):
			if os.path.exists(main_dir + dir_name + '/' + dir_name + annotations_file):

				with open(main_dir + dir_name + '/' + dir_name + annotations_file) as labels_file:
					labels = labels_file.readlines()
				labels = [label.split('\t')[0] for label in labels]
				label_data += labels

				with open(main_dir + dir_name + '/' + fc7_file) as features_file:
					features = features_file.readlines()
				features = [feature.split('\n')[0] for feature in features]
				features_data += features

	## To exclude Commercial class etc.
	counter = dict.fromkeys(['bg', 'sp', 'w', 's', 'r', 'h', 'g'], 0)
	labels = []
	features = []
	
	for idx, label in enumerate(label_data):
		if label not in ['Commercial','Problem/Unclassified']:
			if label == 'Reporter':
				label = 'Newsperson(s)'
			elif label == 'Hybrid':
				label = 'Newsperson(s)'
			elif label == 'Studio':
				label = 'Newsperson(s)'
			elif label == 'Background_roll' or label == 'Talking_head' or label == 'Talking_head/Hybrid':	
				label = 'Background_roll'
			elif label == 'Graphic':
				label = 'Graphic'
			elif label == 'Weather' or label == 'Weather/Graphic' or label == 'Weather/Person':
				label = 'Weather'
			labels.append(label)
			features.append(features_data[idx])
	
	return features, labels