import os, sys
import fileops
import keyframes
import shot_detect
import placesCNN
import googlenet
import age_genderCNN

def main():

	clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/gender/'
	if clip_dir[-1] is not '/':
		clip_dir = clip_dir + '/'
		print clip_dir
	
	clip_name = fileops.get_video_filename(clip_dir)
	shot_detect.shot_detect(clip_dir, clip_name)
	keyframes.get_keyframes(clip_dir, clip_name)
	
	## Run a model and get labels for keyframe
 
	caffe_path = '/home/shruti/gsoc/caffehome/caffe/' 
	model_path = caffe_path + 'models/age_gender/'
	image_files = fileops.get_keyframeslist(clip_dir)
	#image_files = ['/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/keyframe039.jpg']
	
	[fc7, fc6, output_label_list, scene_type_list, label_list] = placesCNN.placesCNN(caffe_path, model_path, image_files)
	label_list = googlenet.googlenet(caffe_path, model_path, image_files)
	age_labels, gender_labels = age_genderCNN.age_genderCNN(caffe_path, model_path, image_files)	

	fileops.save_age_gender_labels(clip_dir + 'age_gender_labels_test', age_labels, gender_labels)
	fileops.save_googlenet_labels(clip_dir + 'googlenet_labels', label_list)
	fileops.save_placesCNN_labels(clip_dir + 'placesCNN_labels', output_label_list, scene_type_list, label_list)
	fileops.save_features(clip_dir + 'fc8', fc8)
	#fileops.save_features(clip_dir + 'fc7 ', fc7)
	#fileops.save_features(clip_dir + 'fc6', fc6)


if __name__ == '__main__':
	main()