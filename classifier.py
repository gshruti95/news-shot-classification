import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm
from sklearn.cross_validation import train_test_split
import time

start = time.time()

main_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
test_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/test/'

df_data = pd.read_csv(main_dir + 'new_train_data.csv', header=-1)
train_data = df_data.values

df_labels = pd.read_csv(main_dir + 'new_label_data.csv', header=-1)
label_data = df_labels.values

df_test = pd.read_csv(test_dir + 'new_train_data.csv', header=-1)
test_data = df_test.values

test_labels = pd.read_csv(test_dir + 'new_label_data.csv', header=-1)
test_labels = test_labels.values

# dt, dv = train_test_split(train_data,test_size=.33,random_state=42)

print 'Training data'

mysvm = svm.SVC(decision_function_shape='ovr')

mysvm = mysvm.fit(train_data, label_data)	
 
print 'Predicting...'

output = mysvm.predict(test_data).astype(str)

end = time.time()
print "Time taken: %.2f" %(end-start) 

crt_outp = 0;
for i in range(len(output)):

	if test_labels[i][0] == output[i]:
		crt_outp = crt_outp +1;

print len(output)
print crt_outp

