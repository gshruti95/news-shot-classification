import pandas as pd
import numpy as np
import csv as csv
from sklearn import svm

df_data = pd.read_csv('attributeValues.csv', header=-1)
data = df_data.values

df_test = pd.read_csv('/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/fc7.csv', header=-1)

df_labels = pd.read_csv('attributeNames.csv',header=-1)
labels = df_labels.values

#print 'Training data'

mysvm = svm.SVC(decision_function_shape='ovr')

mysvm = mysvm.fit(data, labels)	
 
#print 'Predicting...'
output = mysvm.predict(df_test.values).astype(str)

print output