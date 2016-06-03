import os, sys
import fileops
import keyframes
import shot_detect
import placesCNN

def main():

	clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/'
	clip_name = fileops.get_video_filename(clip_dir)
	
	shot_detect.shot_detect(clip_dir, clip_name)
	keyframes.get_keyframes(clip_dir, clip_name)
	
	## Run a model and get labels for keyframe
 
	caffe_path = '/home/shruti/gsoc/caffehome/caffe/' 
	model_path = caffe_path + 'models/placesCNN_upgraded/'
	image_files = fileops.get_keyframeslist(clip_dir)
	
	output_label_list, scene_type_list, label_list = placesCNN.placesCNNlabel(caffe_path, model_path, image_files)
 
	with open(clip_dir + 'outputfile.txt', 'w') as file:
		for idx, output_label in enumerate(output_label_list): 
			file.write(scene_type_list[idx] + '|' + output_label + '|' + label_list[idx] + '\n')


if __name__ == '__main__':
	main()