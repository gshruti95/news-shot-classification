import os, sys
import fileops
import keyframes
import shot_detect
import placesCNN

def main():

	clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/'
	clip_name = fileops.get_video_filename(clip_dir)
	
	#shot_detect.shot_detect(clip_dir, clip_name)
	#keyframes.get_keyframes(clip_dir, clip_name)
	
	## Run a model and get labels for keyframe
 
	caffe_path = '/home/shruti/gsoc/caffehome/caffe/' 
	model_path = caffe_path + 'models/placesCNN_upgraded/'
	image_files = fileops.get_keyframeslist(clip_dir)
	
	[fc8, fc7, fc6, output_label_list, scene_type_list, label_list] = placesCNN.placesCNN(caffe_path, model_path, image_files)

	fileops.save_features(clip_dir + 'fc8', fc7)
	fileops.save_features(clip_dir + 'fc7', fc7)
	fileops.save_features(clip_dir + 'fc6', fc6)

	fileops.save_placesCNN_labels(clip_dir + 'placesCNN_labels', output_label_list, scene_type_list, label_list)

if __name__ == '__main__':
	main()