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
	# print type(df_train_data)
	df_train_data = df_train_data.astype(float)
	
	df_test_data = [data.split(',') for data in test_data]
	df_test_data = pd.DataFrame(df_test_data)
	df_test_data = df_test_data.astype(float)
	
	df_train_labels = pd.DataFrame(train_labels)
	df_test_labels = pd.DataFrame(test_labels)
	# print df_train_labels

	df_trainset = [df_train_data, df_train_labels]
	comb_train = pd.concat(df_trainset, axis = 1)
	df_testset = [df_test_data, df_test_labels]
	comb_test = pd.concat(df_testset, axis = 1)
	df_comb = [comb_train, comb_test]
	final_train = pd.concat(df_comb)
	# print final_train
	final_train = final_train.values

	dtrain, dvalidate = train_test_split(final_train,test_size=.33,random_state=1)

	# print "start ", len(dvalidate)

	df_train_data = df_train_data.values
	df_train_labels = df_train_labels.values
	df_test_data = df_test_data.values
	df_test_labels = df_test_labels.values

	print 'Training data'

	mysvm = svm.SVC(decision_function_shape='ovr')

	mysvm = mysvm.fit(dtrain[:,:-1], dtrain[:,-1])	
	 
	print 'Predicting...'

	output = mysvm.predict(dvalidate[:,:-1]).astype(str)

	end = time.time()
	print "Time taken: %.2f" %(end-start) 
	# print len(output)

	crt_outp = 0
	not_count = 0
	total_crt_outp = 0
	new_train_data = []
	new_test_data = []
	new_train_labels = []
	new_test_labels = []
	p_s = 0
	p_r = 0
	p_h = 0
	p_bg = 0
	p_sp = 0
	p_w = 0
	p_c = 0
	p_g = 0
	p_n = 0

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
			p_s += 1
			# if df_test_labels[i][0] == output[i]:
			# 	crt_s += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_s += 1
		elif output[i] == 'Reporter':
			p_r += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_r += 1
		elif output[i] == 'Hybrid':
			p_h += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_h += 1
		elif output[i] == 'Graphic':
			p_g += 1 
			if dvalidate[:,-1][i] == output[i]:
				crt_g += 1
		elif output[i] == 'Weather':
			p_w += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_w += 1
		elif output[i] == "Sports":
			p_sp += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_sp += 1
		elif output[i] == "Background_roll":
			p_bg += 1
		elif output[i] == 'Commercial':
			p_c += 1
		elif output[i] == 'Problem/Unclassified':
			prob += 1
		elif output[i] == 'Not':
			p_n += 1
			if df_test_labels[i][0] == output[i]:
				crt_not += 1

		if dvalidate[:,-1][i] == 'Studio':
			s += 1
			# print 's ', s
		elif dvalidate[:,-1][i] == 'Reporter':
			r += 1
			# print 'r ', r
		elif dvalidate[:,-1][i] == 'Hybrid':
			h += 1
			# print 'h ', h
		elif dvalidate[:,-1][i] == 'Graphic':
			g += 1	
			# print 'g ', g	
		elif dvalidate[:,-1][i] == 'Weather':
			w += 1
			# print 'w ', w
		elif dvalidate[:,-1][i] == 'Sports':
			sp += 1
			# print 'sp ', sp

		if dvalidate[:,-1][i] == output[i]:
			total_crt_outp = total_crt_outp + 1
			
		# print i, dvalidate[:,-1][i]	
		

	print "totalcrtoutp: " , total_crt_outp
	print "totaloutplen: " , len(output)
	print "orig per: ", total_crt_outp*100/float(len(output))

	print "Predicted: " , p_s , p_r, p_h, p_g, p_w, p_sp
	print "Accuracy score: ", accuracy_score(dvalidate[:,-1], output)
	print "Recall score: ", recall_score(dvalidate[:,-1], output)
	print "F score: ", f1_score(dvalidate[:,-1], output)
	print "Precision score: ", precision_score(dvalidate[:,-1], output)
	print "Correct: " , crt_s, crt_r, crt_h, crt_g, crt_w, crt_sp, crt_not
	print "Total: ", s, r, h, g, w, sp
	
	print "Label P: ", crt_s*100/float(p_s), crt_h*100/float(p_h), crt_g*100/float(p_g), crt_w*100/float(p_w), crt_sp*100/float(p_sp)
	print "Label R: ", crt_s*100/float(s), crt_h*100/float(h), crt_g*100/float(g), crt_w*100/float(w), crt_sp*100/float(sp)
	
	return crt_outp , new_test_data, new_test_labels 

