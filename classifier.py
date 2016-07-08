import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def classifier(train_data, train_labels, test_data, test_labels): 

	start = time.time()

	df_train_data = [data.split(',') for data in train_data]
	df_train_data = pd.DataFrame(df_train_data)
	print type(df_train_data)
	df_train_data = df_train_data.astype(float)
	
	df_test_data = [data.split(',') for data in test_data]
	df_test_data = pd.DataFrame(df_test_data)
	df_test_data = df_test_data.astype(float)
	
	df_train_labels = pd.DataFrame(train_labels)
	df_test_labels = pd.DataFrame(test_labels)
	print df_train_labels

	df_trainset = [df_train_data, df_train_labels]
	comb_train = pd.concat(df_trainset, axis = 1)
	df_testset = [df_test_data, df_test_labels]
	comb_test = pd.concat(df_testset, axis = 1)
	df_comb = [comb_train, comb_test]
	final_train = pd.concat(df_comb)

	dt, dv = train_test_split(final_train,test_size=.33,random_state=42)

	df_train_data = df_train_data.values
	df_train_labels = df_train_labels.values
	df_test_data = df_test_data.values
	df_test_labels = df_test_labels.values

	print 'Training data'

	mysvm = svm.SVC(decision_function_shape='ovo')

	mysvm = mysvm.fit(df_train_data, df_train_labels)	
	 
	print 'Predicting...'

	output = mysvm.predict(df_test_data).astype(str)

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
	crt_s = 0
	crt_r = 0
	crt_h = 0
	crt_bg = 0
	crt_sp = 0
	crt_w = 0
	crt_c = 0
	crt_g = 0
	crt_not = 0
	# print output
	for i in range(len(output)):
		# if output[i] != 'Not':
		# 	crt_outp = crt_outp + 1
		# else:
		# 	# new_train_data.append(train_data[i])
		# 	new_test_data.append(test_data[i])
		# 	# new_train_labels.append(orig_train_labels[i])
		# 	new_test_labels.append(orig_test_labels[i]) 
		# 	not_count += 1
		if output[i] == 'Studio':
			s += 1
			if df_test_labels[i][0] == output[i]:
				crt_s += 1
		elif output[i] == 'Reporter':
			r += 1
			if df_test_labels[i][0] == output[i]:
				crt_r += 1
		elif output[i] == 'Hybrid':
			h += 1
			if df_test_labels[i][0] == output[i]:
				crt_h += 1
		elif output[i] == 'Graphic':
			g += 1 
			if df_test_labels[i][0] == output[i]:
				crt_g += 1
		elif output[i] == 'Weather':
			w += 1
			if df_test_labels[i][0] == output[i]:
				crt_w += 1
		elif output[i] == "Sports":
			sp += 1
			if df_test_labels[i][0] == output[i]:
				crt_sp += 1
		elif output[i] == "Background_roll":
			bg += 1
		elif output[i] == 'Commercial':
			c += 1
		elif output[i] == 'Problem/Unclassified':
			prob += 1
		elif output[i] == 'Not':
			n += 1
			if df_test_labels[i][0] == output[i]:
				crt_not += 1

		if df_test_labels[i][0] == output[i]:
			total_crt_outp = total_crt_outp + 1

	print "totalcrtoutp: " , total_crt_outp
	print "orig per: ", total_crt_outp*100/float(len(output))
	# print "crtoutp %d not_count %d" %(crt_outp, not_count)
	print "Predicted: " , s , r, h, g, w, sp , bg, c, prob, n
	print "Accuracy score: ", accuracy_score(df_test_labels, output)
	print "Recall score: ", recall_score(df_test_labels, output)
	print "F score: ", f1_score(df_test_labels, output)
	print "Precision score: ", precision_score(df_test_labels, output)
	print "Correct: " , crt_s, crt_r, crt_h, crt_g, crt_w, crt_sp, crt_not
	return crt_outp , new_test_data, new_test_labels 

