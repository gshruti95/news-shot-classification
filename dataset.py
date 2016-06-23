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
			# print "found"
			with open(main_dir + dir_name+'/'+file) as feat_file:
				feat = feat_file.readlines()
				feat = [feature.split('\n')[0] + '\n' for feature in feat]
	
			train_data += feat
			# print "feat: " , len(train_data), len(feat)
		
		if fnmatch.fnmatch(file,'*_shot_type*.txt'):
			# print "found in" , dir_name
			with open(main_dir + dir_name+'/'+file) as labels_file:
				labels = labels_file.readlines()

			labels = [label.split('\t')[0] + '\n' for label in labels]	
			label_data += labels
			# print "label: ", len(label_data), len(labels)

with open(main_dir + 'new_train_data.csv','w') as train_data_file:
		train_data_file.writelines(train_data)
	
with open(main_dir + 'new_label_data.csv','w') as label_data_file:
	label_data_file.writelines(label_data)

print "final " , len(label_data), len(train_data)	



