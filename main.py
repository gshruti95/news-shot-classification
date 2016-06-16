import os, sys, time
os.environ["GLOG_minloglevel"] = "2"
import fileops
import keyframes
import shotdetect
import facedetect
import placesCNN
import googlenet
import age_genderCNN
import graphcluster

def main():
	root = '/home/shruti/gsoc/news-shot-classification'

	# clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/'
	clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_CNN_Anderson_Cooper_360_0-3595/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0100_US_KABC_Eyewitness_News_6PM_0-1793/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0000_US_FOX-News_The_OReilly_Factor_0-3595/'
	# clip_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/2016-06-07_0100_US_KCBS_CBS_2_News_at_6_0-1735/'
	if clip_dir[-1] is not '/':
		clip_dir = clip_dir + '/'
		print clip_dir
	
	overall_start = time.time()

	output_filename = clip_dir.split('/')[-2]	
	clip_name = fileops.get_video_filename(clip_dir)
	# shotdetect.shotdetect(clip_dir, clip_name)
	keyframes.get_keyframes(clip_dir, clip_name, output_filename)

	image_files = fileops.get_keyframeslist(clip_dir)
	#image_files = ['/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/keyframe039.jpg']

	graphcluster.get_graph_clusters(clip_dir, image_files)

	## Run a model and get labels for keyframe
 	
	caffe_path = '/home/shruti/gsoc/caffehome/caffe/' 
	# model_path = caffe_path + 'models/placesCNN/'
		
	[fc8, fc7, fc6, output_label_list, scene_type_list, label_list, scene_attributes_list] = placesCNN.placesCNN(caffe_path, caffe_path + 'models/placesCNN/', image_files)
	fileops.save_placesCNN_labels(clip_dir + output_filename, clip_dir + 'placesCNN_labels', output_label_list, scene_type_list, label_list, scene_attributes_list)
	fileops.save_features(clip_dir + 'fc8', fc8)
	fileops.save_features(clip_dir + 'fc7 ', fc7)
	# fileops.save_features(clip_dir + 'fc6', fc6)

	# label_list = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
	# fileops.save_googlenet_labels(clip_dir + output_filename, clip_dir + 'googlenet_labels', label_list)

	# fileops.write_separate_labels(clip_dir + output_filename)

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()