import os, sys, time
os.environ["GLOG_minloglevel"] = "2"
import fileops
import keyframes
import shotdetect
import facedetect
import placesCNN
import googlenet
import age_genderCNN
# import graphcluster, accuracy
import dataset, classifier
import pipeline
import cPickle
import cropframes
# import shot_labels

def main():
	root = '/home/shruti/gsoc/news-shot-classification/'

	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0100_US_KABC_Eyewitness_News_6PM_0-1793/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_FOX-News_The_OReilly_Factor_0-3595/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0100_US_KCBS_CBS_2_News_at_6_0-1735/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_0000_UK_KCET_BBC_World_News_America/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_0000_US_CNN_Anderson_Cooper_360/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_0600_US_KABC_KABC_7_News_at_11PM'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_0600_US_KCBS_CBS_2_News_at_11PM'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_0000_US_FOX-News_The_OReilly_Factor'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_1100_US_KNBC_Early_Today'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_1000_US_MSNBC_Morning_Joe'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-07-01_0000_US_HLN_Nancy_Grace'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-21_0635_US_KABC_Jimmy_Kimmel_Live'


	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2014-01-31_0230_US_KNBC_NBC_Nightly_News'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2014-05-10_2200_US_CNN_Situation_Room'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2014-05-10_0000_US_CNN_Anderson_Cooper_360'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2014-05-10_2300_US_KABC_Eyewitness_News_4PM'
	
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2014-01-21_1600_US_KCBS_The_Early_Show'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2014-01-21_2100_US_MSNBC_Martin_Bashir'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-03-16_2100_US_MSNBC_The_Ed_Show'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-03-25_1500_US_KCBS_This_Morning'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-06-14_1530_US_KCBS_Face_the_Nation'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-07-10_1700_US_FOX-News_Happening_Now'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-07-10_1900_US_CNN_Newsroom'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-01-03_1600_US_FOX-News_MediaBuzz'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-02-05_0230_US_KABC_World_News_Tonight_With_David_Muir'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-02-15_2300_US_FOX-News_Special_Report_with_Bret_Baier'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-02-15_2300_US_MSNBC_The_Place_for_Politics_2016'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-03-08_2000_US_KNBC_4_News_at_Noon'

	# Not found
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-03-16_1700_US_KABC_The_View'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-02-05_2300_US_MSNBC_MSNBC_Live'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2015-05-23_1230_US_KNBC_Today_Weekend'

	## Small clips

	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_148-628'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_1255-1735'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_1615-2095'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_1809-2289'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_2840-3320'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KCBS_CBS_Evening_News_0-472'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KCBS_CBS_Evening_News_0-478'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KCBS_CBS_Evening_News_263-743'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KCBS_CBS_Evening_News_560-1040'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KCBS_CBS_Evening_News_1192-1672'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_0-479'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_281-761'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_702-1182'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_792-1272'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_995-1475'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_1097-1577'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KNBC_NBC_Nightly_News_1460-1940'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0100_US_KCBS_CBS_2_News_at_6_258-738'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0100_US_KCBS_CBS_2_News_at_6_1030-1510'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-23_0030_US_KCBS_CBS_Evening_News_263-743'
	clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465'

	if clip_dir[-1] is not '/':
		clip_dir = clip_dir + '/'
	print clip_dir
	
	overall_start = time.time()

	output_filename = clip_dir.split('/')[-2]	
	clip_name = fileops.get_video_filename(clip_dir)

	timestamps = keyframes.keyframes(clip_dir, clip_name, output_filename)
	shot_boundaries, extra_timestamps = shotdetect.shotdetect(clip_dir, clip_name)
	py_images = fileops.get_pyframeslist(clip_dir, clip_name)
	keyframes_list = fileops.get_keyframeslist(clip_dir)

	image_files = fileops.rename_frames(timestamps, keyframes_list, extra_timestamps, py_images)
	# image_files = cropframes.cropframes(clip_dir, image_files)

	# studio_shots = graphcluster.get_graph_clusters(clip_dir, image_files)
	# print studio_shots
	# print len(studio_shots)
	# fileops.save_studio(clip_dir + output_filename, studio_shots)
	
	## Run a model and get labels for keyframe
 	
	# caffe_path = '/home/shruti/gsoc/caffehome/caffe/'
	# [fc8, fc7, fc6, scene_type_list, scene_attributes_list] = placesCNN.placesCNN(caffe_path, caffe_path + 'models/placesCNN/', image_files) 	

	# accuracy.get_accuracy(clip_dir + output_filename, '_scene', scene_type_list)
	# fileops.save_placesCNN_labels(clip_dir + output_filename, clip_dir + 'placesCNN_labels', output_label_list, scene_type_list, label_list, scene_attributes_list)

	# fileops.save_features(clip_dir + 'cropped_places_fc8', fc8)
	# print "Done fc8"
	# fileops.save_features(clip_dir + 'cropped_places_fc7 ', fc7)
	# print "done fc7"
	# fileops.save_features(clip_dir + 'cropped_places_fc6', fc6)
	# print "done fc6"

	# googlenet_dir = '/home/shruti/gsoc/news-shot-classification/clips/'
	# dataset.dataset(googlenet_dir)
	
	# googlenet_label_list = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
	# fileops.save_googlenet_labels(clip_dir + output_filename, clip_dir + 'googlenet_labels', label_list)
	# fileops.write_separate_labels(clip_dir + output_filename)
	
	train_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
	test_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/test/'	
	# mysvm = pipeline.pipeline(train_dir, test_dir)
	# classifier_label_list = classifier.predict(mysvm)


	# shot_labels.frame_labels(clip_dir + output_filename, timestamps, image_files, shot_boundaries, classifier_label_list, googlenet_label_list, scene_type_list, scene_attributes_list)

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()