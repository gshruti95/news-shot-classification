import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import time


def classifier(train_data, train_labels, orig_train_labels, test_data, test_labels, orig_test_labels): 

	start = time.time()

	# train_data = pd.read_csv(train_dir + 'new_train_data.csv', header=-1)
	df_train_data = [data.split(',') for data in train_data]
	df_train_data = pd.DataFrame(df_train_data)
	df_train_data = df_train_data.astype(float)
	df_train_data = df_train_data.values

	# train_labels = pd.read_csv(train_dir + 'new_label_data.csv', header=-1)
	df_train_labels = pd.DataFrame(train_labels)
	df_train_labels = df_train_labels.values

	# test_data = pd.read_csv(test_dir + 'new_train_data.csv', header=-1)
	df_test_data = [data.split(',') for data in test_data]
	df_test_data = pd.DataFrame(df_test_data)
	df_test_data = df_test_data.astype(float)
	df_test_data = df_test_data.values
	# print (df_test_data)
	# test_labels = pd.read_csv(test_dir + 'new_label_data.csv', header=-1)
	df_test_labels = pd.DataFrame(test_labels)
	df_test_labels = df_test_labels.values

	# dt, dv = train_test_split(train_data,test_size=.33,random_state=42)

	# df_train_data = normalize(df_train_data)
	# df_test_data = normalize(df_test_data)
	# print (df_test_data)
	# print df_train_labels
	# print df_test_labels
	print 'Training data'

	# mysvm = svm.SVC(decision_function_shape='ovo')
	mysvm = svm.OneClassSVM()

	mysvm = mysvm.fit(df_train_data)	
	 
	print 'Predicting...'

	output = mysvm.predict(df_test_data)
	print len(output)
	end = time.time()
	print "Time taken: %.2f" %(end-start) 

	crt_outp = 0
	not_count = 0
	total_crt_outp = 0
	new_train_data = []
	new_test_data = []
	new_train_labels = []
	new_test_labels = []
	s = 0
	r = 0
	h = 0
	bg = 0
	sp = 0
	w = 0
	c = 0
	g = 0
	n = 0
	prob = 0
	crt_studio = 0
	crt_not = 0
	# print output


	for i in range(len(output)):
		if output[i] == 1 and df_test_labels[i][0] == 'Studio':
			crt_outp += 1
	# 	# if output[i] != 'Not':
	# 	# 	crt_outp = crt_outp + 1
	# 	# else:
	# 	# 	# new_train_data.append(train_data[i])
	# 	# 	new_test_data.append(test_data[i])
	# 	# 	# new_train_labels.append(orig_train_labels[i])
	# 	# 	new_test_labels.append(orig_test_labels[i]) 
	# 	# 	not_count += 1
	# 	if output[i] == 'Studio':
	# 		s += 1
	# 		if df_test_labels[i][0] == output[i]:
	# 			crt_studio += 1
	# 	elif output[i] == 'Reporter':
	# 		r += 1
	# 		# if df_test_labels[i][0] == output[i]:
	# 		# 	crt_studio += 1
	# 	elif output[i] == 'Hybrid':
	# 		h += 1
	# 		# if df_test_labels[i][0] == output[i]:
	# 		# 	crt_studio += 1
	# 	elif output[i] == 'Graphic':
	# 		g += 1 
	# 		# if df_test_labels[i][0] == output[i]:
	# 		# 	crt_studio += 1
	# 	elif output[i] == 'Weather':
	# 		w += 1
	# 		# if df_test_labels[i][0] == output[i]:
	# 		# 	crt_studio += 1
	# 	elif output[i] == "Sports":
	# 		sp += 1
	# 		# if df_test_labels[i][0] == output[i]:
	# 		# 	crt_studio += 1
	# 	elif output[i] == "Background_roll":
	# 		bg += 1
	# 	elif output[i] == 'Commercial':
	# 		c += 1
	# 	elif output[i] == 'Problem/Unclassified':
	# 		prob += 1
	# 	elif output[i] == 'Not':
	# 		n += 1
	# 		if df_test_labels[i][0] == output[i]:
	# 			crt_not += 1

	# 	if df_test_labels[i][0] == output[i]:
	# 		total_crt_outp = total_crt_outp + 1

	# print "totalcrtoutp: " , total_crt_outp
	# print "orig per: ", total_crt_outp*100/float(len(output))
	# # print "crtoutp %d not_count %d" %(crt_outp, not_count)
	# print s , r, h, g, w, sp , bg, c, prob, n
	# print "crt_studio %d crt_not %d" %(crt_studio, crt_not)

	return crt_outp , new_test_data, new_test_labels 

