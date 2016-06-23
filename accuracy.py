import os, sys

counter = {'Indoor': 0, 'Outdoor': 0, 'None': 0,
				'Misleading': 0, 'Commercial': 0, 'Studio': 0, 'Hybrid': 0, 'Background_roll': 0,
				'Problem/Unclassified': 0, 'Reporter': 0, 'Weather': 0, 'Graphic': 0, 'Commercial': 0, 'Sports': 0}
total = {'Indoor': 0, 'Outdoor': 0, 'None': 0,
				'Misleading': 0, 'Commercial': 0, 'Studio': 0, 'Hybrid': 0, 'Background_roll': 0,
				'Problem/Unclassified': 0, 'Reporter': 0, 'Weather': 0, 'Graphic': 0, 'Commercial': 0, 'Sports': 0}				

# def Indoor(count)

def get_accuracy(filename, label_type, labels):

	labels = [label.capitalize() for label in labels]

	with open(filename + label_type + '_testuser.txt') as file:
		data = file.readlines()
		# with open(filename + '_shot_type_testuser.txt') as shot_file:
			# shot_data = shot_file.readlines()

	for idx, line in enumerate(data):
		data_type = line.split('\t')[0]
		x = 0
		cur_total = total[data_type]
		x = cur_total + 1
		total[data_type] = x

		if label_type == '_scene':
			if data_type == labels[idx]:
				tmp = 0
				cnt = counter[data_type]
				tmp = cnt + 1
				counter[data_type] = tmp

	# if label_type == '_shot_type':
	# 	print "entered " , studio_indices
	# 	for val in studio_indices:
	# 		print val
	# 		shot_type = data[val].split('\t')[0]
	# 		if shot_type == 'Studio' or shot_type == 'Hybrid':
	# 			tmp = 0
	# 			cnt = counter['Studio']
	# 			tmp = cnt + 1
	# 			counter['Studio'] = tmp
	# 	Studio_accuracy = float(counter['Studio']/float(total['Studio'] + total['Hybrid']))
	# 	print "Studio %.2f" %(Studio_accuracy*100)

	# else:
		# Indoor_accuracy = counter['Indoor']/float(total['Indoor'])
		# Outdoor_accuracy = counter['Outdoor']/float(total['Outdoor'])
	Scene_Accuracy = (counter['Outdoor'] + counter['Indoor'])*100/float(total['Outdoor'] + total['Indoor'])
		# print "Indoor %.2f" %(Indoor_accuracy*100)
		# print "Outdoor %.2f" %(Outdoor_accuracy*100)
	print "Scene accuracy %.2f \n" %(Scene_Accuracy)

			
	print "counter dict" , counter
	print 'total dict', total
	