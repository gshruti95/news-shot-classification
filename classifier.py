import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm
import time
import dataset, cPickle

def classifier_dump(fpickle, train_dir, annotation_file, features_file, class_type):

	[train_data, train_labels] = dataset.trainset(train_dir, annotation_file, features_file)
	train_labels = dataset.ovo_trainset(train_labels, class_type)
	myclassifier = classifier_train(train_data, train_labels)
	with open(fpickle, 'w') as pickle_file:
		cPickle.dump(myclassifier, pickle_file)

def classifier_train(train_data, train_labels):

	start = time.time()

	df_train_data = [data.split(',') for data in train_data]
	df_train_data = pd.DataFrame(df_train_data)
	df_train_data = df_train_data.astype(float)
	df_train_data = df_train_data.values

	df_train_labels = pd.DataFrame(train_labels)
	df_train_labels = df_train_labels.values

	print 'Training data'

	myclassifier = svm.SVC(decision_function_shape='ovr', kernel = 'linear')
	myclassifier = myclassifier.fit(df_train_data, df_train_labels)

	end = time.time()
	print "Time taken to train: %.2f" %(end-start)	

	return myclassifier

def classifier_predict(myclassifier, test_data):

	start = time.time()

	df_test_data = [data.split(',') for data in test_data]
	df_test_data = pd.DataFrame(df_test_data)
	df_test_data = df_test_data.astype(float)
	df_test_data = df_test_data.values
	print 'Predicting...'

	output = myclassifier.predict(df_test_data).astype(str)

	end = time.time()
	print "Time taken to predict: %.2f" %(end-start) 

	return output