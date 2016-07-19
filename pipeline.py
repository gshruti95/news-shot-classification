import dataset
import classifier

def pipeline(train_dir, test_dir):


	[train_data, train_labels] = dataset.dataset(train_dir)
	[test_data, test_labels] = dataset.dataset(test_dir)
	
	s = 0
	r = 0
	h = 0
	bg = 0
	sp = 0
	w = 0
	c = 0
	g = 0
	prob = 0

	## Combine studio repo and/or hybrid

	for idx, label in enumerate(test_labels):
		if label == 'Studio':
			s += 1
		elif label == 'Reporter':
			r += 1
			label = 'Studio'
			test_labels[idx] = label
		elif label == 'Hybrid':
			h += 1
			label = 'Studio'
			test_labels[idx] = label
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

	for idx, label in enumerate(train_labels):
		if label == 'Reporter':
			label = 'Studio'
			train_labels[idx] = label
		elif label == 'Hybrid':
			label = 'Studio'
			train_labels[idx] = label

	## To change train labels to single SVM ovo format

	new_train_labels = []
	new_test_labels = []
	not_count = 0

	for label, feature in zip(train_labels, train_data):
		if label != 'Background_roll':
			label = 'Not'		
		new_train_labels.append(label)

	for label in test_labels:
		if label != 'Background_roll':
			not_count += 1
			label = 'Not'
		new_test_labels.append(label)
	print "notcount: ", not_count

	##		
	# print "Train test split..."
	classifier.dataset_split(train_data, new_train_labels, test_data, new_test_labels)	
	print "Manual..."
	classifier.manual(train_data, new_train_labels, test_data, new_test_labels)

	# correct = 0
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