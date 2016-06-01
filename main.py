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
	image_file = clip_dir + 'keyframe039.jpg'
	
	output_label, scene_type, label_list = placesCNN.placesCNNlabel_singleframe(caffe_path, model_path, image_file)

	label_list = "|".join( "%s, %s" %tup for tup in label_list )

	with open(clip_dir + 'outputfile.txt', 'w') as file:
			file.write(output_label + '|' + scene_type + '|' + label_list + '\n')


if __name__ == '__main__':
	main()