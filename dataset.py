import os, sys
import fnmatch

# main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/test/'

dir_list = sorted(os.listdir(main_dir))

train_data = []
label_data = []

for dir_name in dir_list:
	for file in os.listdir(main_dir + dir_name):
		if fnmatch.fnmatch(file,'new_places_fc7 .csv'):
			with open(main_dir + dir_name+'/'+file) as features_file:
				features = features_file.readlines()
			features = [feature.split('\n')[0] + '\n' for feature in features]
			train_data += features

		if fnmatch.fnmatch(file,'*_shot_type*.txt'):
			with open(main_dir + dir_name+'/'+file) as labels_file:
				labels = labels_file.readlines()
			# print len(labels), len(features)
		
			# labels = [label.split('\t')[0] + '\n' for label in labels]
			
			# for label, feature in zip(labels, features):
			for label in labels:
				label = label.split('\t')[0]
				# if label not in ['Commercial','Problem/Unclassified']: 
				if label != 'Commercial':
					label = 'Not'

				label = label + '\n'
				label_data.append(label)
				# feature = feature.split('\n')[0] + '\n'
				# train_data.append(feature)
				# print "Iteration :" , len(label_data), len(train_data)

			# print "label: ", len(label_data), len(labels)	
				
			# print "features: " , len(train_data), len(features)

with open(main_dir + 'new_train_data.csv','w') as train_data_file:
		train_data_file.writelines(train_data)
	
with open(main_dir + 'new_label_data.csv','w') as label_data_file:
	label_data_file.writelines(label_data)

print "final " , len(label_data), len(train_data)	



