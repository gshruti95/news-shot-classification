import dataset
import classifier

def pipeline(train_dir, test_dir):


	[train_data, train_labels] = dataset.dataset(train_dir)
	[test_data, test_labels] = dataset.dataset(test_dir)
	new_train_labels = []
	
	total = len(test_labels)
	s = 0
	r = 0
	h = 0
	bg = 0
	sp = 0
	w = 0
	c = 0
	g = 0
	prob = 0

	for label in test_labels:
		if label == 'Studio':
			s += 1
		elif label == 'Reporter':
			r += 1
		elif label == 'Hybrid':
			h += 1
		elif label == 'Graphic':
			g += 1
		elif label == 'Weather':
			w += 1
		elif label == "Sports":
			sp += 1
		elif label == "Background_roll":
			bg += 1
		elif label == 'Commercial':
			c += 1
		elif label == 'Problem/Unclassified':
			prob += 1

	correct = 0
	not_count = 0

	# for label, feature in zip(train_labels, train_data):
	# 	if label != 'Studio':
	# 		label = 'Not'
		
	# 	new_train_labels.append(label)
	# 	# new_train_data.append(feature)

	# for label in test_labels:
	# 	if label != 'Studio':
	# 		not_count += 1
	# 	# new_test_labels.append(label)

	# print "notcount: ", not_count

	[outp_len, test_data, test_labels] = classifier.classifier(train_data, train_labels, test_data, test_labels)
	
	# print "Total: " , s , r, h, g, w, sp , bg, c, prob
	# correct += outp_len
	# print "correct: ", outp_len
	# # print "total %d true studio %d : "  %(total, total-not_count)

	# new_train_labels = []
	# new_test_labels = []

	# for label in train_labels:
	# 	if label != 'Reporter':
	# 		label = 'Not'
	# 	new_train_labels.append(label)

	# total = len(test_labels)
	# not_count = 0
	# correct = 0
	# for label in test_labels:
	# 	if label != 'Reporter':
	# 		label = 'Not'
	# 		not_count += 1
	# 	new_test_labels.append(label)

	# [outp_len, test_data, test_labels] = classifier.classifier(train_data, new_train_labels, train_labels, test_data, new_test_labels, test_labels)
	# correct += outp_len
	# print "correct: ", correct
	# print "total %d true repo %d : " %(total, total-not_count)

	# new_train_labels = []
	# new_test_labels = []

	# for label in train_labels:
	# 	if label != 'Hybrid':
	# 		label = 'Not'
	# 	new_train_labels.append(label)

	# for label in test_labels:
	# 	if label != 'Hybrid':
	# 		label = 'Not'
	# 		not_count += 1
	# 	new_test_labels.append(label)

	# [outp_len, test_data, test_labels] = classifier.classifier(train_data, new_train_labels, train_labels, test_data, new_test_labels, test_labels)
	# # correct += outp_len
	# # print "correct: ", correct
	# print "total %d true hybrid %d "  %(total, total-not_count)

	# new_train_labels = []
	# new_test_labels = []

	# for label in train_labels:
	# 	if label != 'Weather':
	# 		label = 'Not'
	# 	new_train_labels.append(label)

	# for label in test_labels:
	# 	if label != 'Weather':
	# 		label = 'Not'
	# 	new_test_labels.append(label)

	# [outp_len, test_data, test_labels] = classifier.classifier(train_data, new_train_labels, train_labels, test_data, new_test_labels, test_labels)
	# correct += outp_len
	# print "correct: ", correct

	# new_train_labels = []
	# new_test_labels = []

	# for label in train_labels:
	# 	if label != 'Sports':
	# 		label = 'Not'
	# 	new_train_labels.append(label)

	# for label in test_labels:
	# 	if label != 'Sports':
	# 		label = 'Not'
	# 	new_test_labels.append(label)

	# [outp_len, test_data, test_labels] = classifier.classifier(train_data, new_train_labels, train_labels, test_data, new_test_labels, test_labels)
	# correct += outp_len
	# print "correct: ", correct

	# new_train_labels = []
	# new_test_labels = []

	# for label in train_labels:
	# 	if label != 'Graphic':
	# 		label = 'Not'
	# 	new_train_labels.append(label)

	# for label in test_labels:
	# 	if label != 'Graphic':
	# 		label = 'Not'
	# 	new_test_labels.append(label)

	# [outp_len, test_data, test_labels] = classifier.classifier(train_data, new_train_labels, train_labels, test_data, new_test_labels, test_labels)
	# correct += outp_len
	# print "correct: ", correct

	# new_train_labels = []
	# new_test_labels = []

	# for label in train_labels:
	# 	if label != 'Background_roll':
	# 		label = 'Not'
	# 	new_train_labels.append(label)

	# for label in test_labels:
	# 	if label != 'Background_roll':
	# 		label = 'Not'
	# 	new_test_labels.append(label)

	# [outp_len, test_data, test_labels] = classifier.classifier(train_data, new_train_labels, train_labels, test_data, new_test_labels, test_labels)
	# correct += outp_len
	# print "correct: ", correct

	# print correct*100/float(total)