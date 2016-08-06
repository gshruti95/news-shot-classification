import os,sys
import numpy as np
import csv

def shot_labels(camtype, imagenet, scene, current_shot_timestamp):



def frame_labels(filename, timestamps, image_files, shot_boundaries, classifier_label_list, googlenet_label_list, scene_type_list, scene_attributes_list):
	
	for idx, boundary in enumerate(shot_boundaries):
		camtype = []
		imagenet = []
		scene = []
		if idx == 0:
			current_shot_timestamp = 0.00
		else:
			current_shot_timestamp = shot_boundaries[idx-1]

		while timestamps[count] <= boundary and count <= len(timestamps):
			line = timestamps[count] + '| ' + classifier_label_list[count] + '| ' + googlenet_label_list[count] + '| ' + scene_type_list[count] + '| ' + scene_attributes_list
			new_lines.append(line)
			count += 1
			camtype.append(classifier_label_list[count])
			imagenet.append(googlenet_label_list[count])
			scene.append(scene_type_list[count])

		shot_labels.shot_labels(camtype, imagenet, scene, current_shot_timestamp)

	with open(filename + '.vis', 'r') as file:
			new_lines = []
			count = 0