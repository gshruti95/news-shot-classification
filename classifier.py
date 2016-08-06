import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def dataset_split(train_data, train_labels, test_data, test_labels): 

	start = time.time()
	
	df_train_data = [data.split(',') for data in train_data]
	df_train_data = pd.DataFrame(df_train_data)
	df_train_data = df_train_data.astype(float)
	
	df_test_data = [data.split(',') for data in test_data]
	df_test_data = pd.DataFrame(df_test_data)
	df_test_data = df_test_data.astype(float)
	
	df_train_labels = pd.DataFrame(train_labels)
	df_test_labels = pd.DataFrame(test_labels)

	df_trainset = [df_train_data, df_train_labels]
	comb_train = pd.concat(df_trainset, axis = 1)
	df_testset = [df_test_data, df_test_labels]
	comb_test = pd.concat(df_testset, axis = 1)
	df_comb = [comb_train, comb_test]
	final_train = pd.concat(df_comb)
	final_train = final_train.values

	dtrain, dvalidate = train_test_split(final_train, test_size=.33, random_state=1)

	df_train_data = df_train_data.values
	df_train_labels = df_train_labels.values
	df_test_data = df_test_data.values
	df_test_labels = df_test_labels.values

	print 'Training data'

	mysvm = svm.SVC(decision_function_shape = 'ovo', kernel = 'linear')

	mysvm = mysvm.fit(dtrain[:,:-1], dtrain[:,-1])	

	return mysvm, dvalidate


def predict_split(mysvm, dvalidate):	
	 
	print 'Predicting...'

	output = mysvm.predict(dvalidate[:,:-1]).astype(str)

	# end = time.time()
	# print "Time taken: %.2f" %(end-start) 

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
	p_prob = 0
	p_v = 0
	p_notv = 0
	p_cr = 0
	p_notcr = 0

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
	v = 0
	notv = 0
	cr = 0
	notcr = 0

	crt_s = 0
	crt_r = 0
	crt_h = 0
	crt_bg = 0
	crt_sp = 0
	crt_w = 0
	crt_c = 0
	crt_g = 0
	crt_not = 0
	crt_prob = 0
	crt_v = 0
	crt_notv = 0
	crt_cr = 0
	crt_notcr = 0
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
		elif output[i] == "Background_roll" or output[i] == "Background roll":
			p_bg += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_bg += 1
		elif output[i] == 'Commercial':
			p_c += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_c += 1
		elif output[i] == 'Problem/Unclassified':
			prob += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_prob += 1
		elif output[i] == 'Not':
			p_n += 1
			if dvalidate[:,-1][i] == output[i]:
				crt_not += 1
		elif output[i] == 'Vehicle/Accident':
			p_v += 1
			if dvalidate[:,-1][i] == 'Vehicle/Accident':
				crt_v += 1
		elif output[i] == 'Not Vehicle/Accident':
			p_notv += 1
			if dvalidate[:,-1][i] == 'Not Vehicle/Accident':
				crt_notv += 1
		elif output[i] == 'Crowd':
			p_cr += 1
			if dvalidate[:,-1][i] == 'Crowd':
				crt_cr += 1
		elif output[i] == 'Not crowd':
			p_notcr += 1
			if dvalidate[:,-1][i] == 'Not crowd':
				crt_notcr += 1		

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
		elif dvalidate[:,-1][i] == 'Background_roll' or dvalidate[:,-1][i] == 'Background roll':
			bg += 1
		elif dvalidate[:,-1][i] == 'Commercial':
			c += 1
		elif dvalidate[:,-1][i] == 'Problem/Unclassified':
			prob += 1
			# print 'sp ', sp
		elif dvalidate[:,-1][i] == 'Vehicle/Accident':
			v += 1
		elif dvalidate[:,-1][i] == 'Not Vehicle/Accident':
			notv += 1
		elif dvalidate[:,-1][i] == 'Crowd':
			cr += 1
		elif dvalidate[:,-1][i] == 'Not crowd':
			notcr += 1
		elif dvalidate[:,-1][i] == 'Not':
			n += 1

		# if dvalidate[:,-1][i] == output[i]:
		# 	total_crt_outp = total_crt_outp + 1

		# print i, dvalidate[:,-1][i]	
		

	# print "totalcrtoutp: " , total_crt_outp
	# print "totaloutplen: " , len(output)
	# print "orig per: ", total_crt_outp*100/float(len(output))

	# print "Accuracy score: ", accuracy_score(dvalidate[:,-1], output)
	# print "Recall score: ", recall_score(dvalidate[:,-1], output, pos_label = None)
	# print "F score: ", f1_score(dvalidate[:,-1], output, pos_label = None)
	# print "Precision score: ", precision_score(dvalidate[:,-1], output, pos_label = None)

	print "Predicted: " , p_s , p_r, p_h, p_g, p_w, p_sp, p_bg, p_c, p_prob, p_v, p_notv, p_cr, p_notcr, p_n	
	print "Correct: " , crt_s, crt_r, crt_h, crt_g, crt_w, crt_sp, crt_bg, crt_c, crt_prob, crt_v, crt_notv, crt_cr, crt_notcr, crt_not
	print "Total: ", s, r, h, g, w, sp, bg, c, prob, v, notv, cr, notcr, n
	
	# print "Label P: ", crt_s*100/float(p_s), crt_r*100/float(p_r), crt_h*100/float(p_h), crt_g*100/float(p_g), crt_w*100/float(p_w), crt_sp*100/float(p_sp)#, crt_bg*100/float(p_bg), crt_c*100/float(p_c), crt_prob*100/float(p_prob)
	# print "Label R: ", crt_s*100/float(s), crt_r*100/float(r), crt_h*100/float(h), crt_g*100/float(g), crt_w*100/float(w), crt_sp*100/float(sp)#, crt_bg*100/float(bg), crt_c*100/float(c), crt_prob*100/float(prob)
	
	# return crt_outp , new_test_data, new_test_labels 

def manual(train_data, train_labels, test_data, test_labels):

	start = time.time()

	df_train_data = [data.split(',') for data in train_data]
	df_train_data = pd.DataFrame(df_train_data)
	df_train_data = df_train_data.astype(float)
	df_train_data = df_train_data.values

	df_train_labels = pd.DataFrame(train_labels)
	df_train_labels = df_train_labels.values

	df_test_data = [data.split(',') for data in test_data]
	df_test_data = pd.DataFrame(df_test_data)
	df_test_data = df_test_data.astype(float)
	df_test_data = df_test_data.values

	df_test_labels = pd.DataFrame(test_labels)
	df_test_labels = df_test_labels.values

	print 'Training data'

	mysvm = svm.SVC(decision_function_shape='ovo', kernel = 'linear')

	mysvm = mysvm.fit(df_train_data, df_train_labels)	

	return mysvm, df_test_labels

def predict(mysvm):

	start = time.time()
	print 'Predicting...'

	output = mysvm.predict(df_test_data).astype(str)

	end = time.time()
	print "Time taken: %.2f" %(end-start) 

	return output

def predict_manual(mysvm, df_test_labels):
	 
	print 'Predicting...'

	output = mysvm.predict(df_test_data).astype(str)

	# end = time.time()
	# print "Time taken: %.2f" %(end-start) 

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
	p_prob = 0
	p_v = 0
	p_notv = 0
	p_cr = 0
	p_notcr = 0

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
	v = 0
	notv = 0
	cr = 0
	notcr = 0

	crt_s = 0
	crt_r = 0
	crt_h = 0
	crt_bg = 0
	crt_sp = 0
	crt_w = 0
	crt_c = 0
	crt_g = 0
	crt_not = 0
	crt_prob = 0
	crt_v = 0
	crt_notv = 0
	crt_cr = 0
	crt_notcr = 0

	for i in range(len(output)):
		if output[i] == 'Studio':
			p_s += 1
			if df_test_labels[i][0] == output[i]:
				crt_s += 1
		elif output[i] == 'Reporter':
			p_r += 1
			if df_test_labels[i][0] == output[i]:
				crt_r += 1
		elif output[i] == 'Hybrid':
			p_h += 1
			if df_test_labels[i][0] == output[i]:
				crt_h += 1
		elif output[i] == 'Graphic':
			p_g += 1 
			if df_test_labels[i][0] == output[i]:
				crt_g += 1
		elif output[i] == 'Weather':
			p_w += 1
			if df_test_labels[i][0] == output[i]:
				crt_w += 1
		elif output[i] == "Sports":
			p_sp += 1
			if df_test_labels[i][0] == output[i]:
				crt_sp += 1
		elif output[i] == "Background_roll" or output[i] == "Background roll":
			p_bg += 1
			if df_test_labels[i][0] == output[i]:
				crt_bg += 1
		elif output[i] == 'Commercial':
			p_c += 1
			if df_test_labels[i][0] == output[i]:
				crt_c += 1
		elif output[i] == 'Problem/Unclassified':
			prob += 1
			if df_test_labels[i][0] == output[i]:
				crt_prob += 1
		elif output[i] == 'Not':
			p_n += 1
			if df_test_labels[i][0] == output[i]:
				crt_not += 1
		elif output[i] == 'Vehicle/Accident':
			p_v += 1
			if df_test_labels[i][0] == output[i]:
				crt_v += 1
		elif output[i] == 'Not Vehicle/Accident':
			p_notv += 1
			if df_test_labels[i][0] == output[i]:
				crt_notv += 1
		elif output[i] == 'Crowd':
			p_cr += 1
			if df_test_labels[i][0] == output[i]:
				crt_cr += 1
		elif output[i] == 'Not crowd':
			p_notcr += 1
			if df_test_labels[i][0] == output[i]:
				crt_notcr += 1

		if df_test_labels[i][0] == 'Studio':
			s += 1
			# print 's ', s
		elif df_test_labels[i][0] == 'Reporter':
			r += 1
			# print 'r ', r
		elif df_test_labels[i][0] == 'Hybrid':
			h += 1
			# print 'h ', h
		elif df_test_labels[i][0] == 'Graphic':
			g += 1	
			# print 'g ', g	
		elif df_test_labels[i][0] == 'Weather':
			w += 1
			# print 'w ', w
		elif df_test_labels[i][0] == 'Sports':
			sp += 1
		elif df_test_labels[i][0] == 'Background_roll' or df_test_labels[i][0] == 'Background roll':
			bg += 1
		elif df_test_labels[i][0] == 'Commercial':
			c += 1
		elif df_test_labels[i][0] == 'Problem/Unclassified':
			prob += 1		
		elif df_test_labels[i][0] == 'Vehicle/Accident':
			v += 1
		elif df_test_labels[i][0] == 'Not Vehicle/Accident':
			notv += 1
		elif df_test_labels[i][0] == 'Crowd':
			cr += 1
		elif df_test_labels[i][0] == 'Not crowd':
			notcr += 1
		elif df_test_labels[i][0] == 'Not':
			n += 1	

		# if df_test_labels[i][0] == output[i]:
		# 	total_crt_outp = total_crt_outp + 1

	# print "Accuracy score: ", accuracy_score(test_labels, output)
	# print "Recall score: ", recall_score(test_labels, output)
	# print "F score: ", f1_score(test_labels, output)
	# print "Precision score: ", precision_score(test_labels, output)

	print "Predicted: " , p_s , p_r, p_h, p_g, p_w, p_sp, p_bg, p_c, p_prob, p_v, p_notv, p_cr, p_notcr, p_n
	print "Correct: " , crt_s, crt_r, crt_h, crt_g, crt_w, crt_sp, crt_bg, crt_c, crt_prob, crt_v, crt_notv, crt_cr, crt_notcr, crt_not
	print "Total: ", s, r, h, g, w, sp, bg, c, prob, v, notv, cr, notcr, n
	
	# print "Label P: ", crt_s*100/float(p_s), crt_r*100/float(p_r), crt_h*100/float(p_h), crt_g*100/float(p_g), crt_w*100/float(p_w), crt_sp*100/float(p_sp)#, crt_bg*100/float(p_bg), crt_c*100/float(p_c), crt_prob*100/float(p_prob)
	# print "Label R: ", crt_s*100/float(s), crt_r*100/float(r), crt_h*100/float(h), crt_g*100/float(g), crt_w*100/float(w), crt_sp*100/float(sp)#, crt_bg*100/float(bg), crt_c*100/float(c), crt_prob*100/float(prob)
	
	# return crt_outp , new_test_data, new_test_labels 
