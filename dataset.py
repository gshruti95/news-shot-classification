import os, sys
import fnmatch
import googlenet
import fileops
import cropframes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def testset(clip_dir, fc7_file):

	test_features = []

	with open(clip_dir + fc7_file) as features_file:
		features = features_file.readlines()
	features = [feature.split('\n')[0] for feature in features]
	test_features += features

	return test_features

def ovo_trainset(train_labels, class_type):
	
	new_train_labels = []
	if class_type == 'newsperson':
		for label in train_labels:
			if label != 'Newsperson(s)':
				label = 'Not'
			new_train_labels.append(label)
	elif class_type == 'broll':
		for label in train_labels:
			if label != 'Background_roll':
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
				ann_files.append(main_dir + dir_name + '/' + dir_name + annotations_file)
				keyframes = [label.split('\t')[1] for label in labels]
				keyframes = [main_dir + keyframe.split('\n')[0] for keyframe in keyframes]
				all_keyframes += keyframes
				labels = [label.split('\t')[0] for label in labels]
				label_data += labels

				with open(main_dir + dir_name + '/' + fc7_file) as features_file:
					features = features_file.readlines()
				features = [feature.split('\n')[0] for feature in features]
				features_data += features

	## To exclude Commercial class etc.
	labels = []
	features = []
	bg_frames = []
	news = 0
	bg = 0
	g = 0
	w = 0
	sp = 0
	c = 0
	p = 0
	for label, keyframe in zip(label_data, all_keyframes):
		if label not in ['Commercial','Problem/Unclassified']:
			if label == 'Reporter' or label == 'Hybrid' or label == 'Studio':
				label = 'Newsperson(s)'
				news += 1
			elif label == 'Background_roll':
				bg_frames.append(keyframe)
				bg += 1
			elif label == 'Graphic':
				g += 1
			elif label == 'Weather':
				w += 1
			elif label == 'Sports':
				sp += 1
			labels.append(label)
			# features.append(feature)
		else:
			if label == 'Commercial':
				c += 1
			elif label == 'Problem/Unclassified':
				p += 1

	print len(labels)
	print news, bg, g, w, sp, c, p

	return labels, bg_frames, ann_files