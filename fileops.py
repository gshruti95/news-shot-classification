import os,sys
import numpy as np
import csv


def save_features(filename, features):

	np.savetxt(filename + ".csv", features, fmt='%.6f', delimiter=' ')
	

def save_placesCNN_labels(filename, output_label_list, scene_type_list, label_list):

	with open(filename + '.csv', 'w') as file:
		for idx, output_label in enumerate(output_label_list): 
			file.write(scene_type_list[idx] + '|' + output_label + '|' + label_list[idx] + '\n')


def get_video_filename(clip_dir):

	source = os.listdir(clip_dir)

	mp4_flag = 0

	for file in source:
		if file.endswith(".mp4"):
			if mp4_flag == 0:
				clip_name = os.path.basename(file)
				mp4_flag = 1
			else:
				print "Multiple mp4 files! Quitting..."
				exit(0)

	return clip_name


def get_keyframeslist(clip_dir):

	keyframes_list = []
	source = sorted(os.listdir(clip_dir))

	for file in source:
		if file.endswith(".jpg"):
			image = clip_dir + os.path.basename(file)
			keyframes_list.append(image)

	return keyframes_list
