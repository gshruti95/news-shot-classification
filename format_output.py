import os,sys
import numpy as np
import csv

def shot_labels(camtype, imagenet, scene):

	camtype_dict = {}
	imagenet_dict = {}
	scene_dict = {}

	for idx, camtype_label in enumerate(camtype):
		if camtype_label in camtype_dict:
			camtype_dict[camtype_label] += 1
		else:
			camtype_dict[camtype_label] = 1

		if imagenet[idx] in imagenet_dict:
			imagenet_dict[imagenet[idx]] += 1
		else:
			imagenet_dict[imagenet[idx]] = 1

		if scene[idx] in scene_dict:
			scene_dict[scene[idx]] += 1
		else:
			scene_dict[scene[idx]] = 1

	cam_label = max(camtype_dict.iterkeys(), key = (lambda key: camtype_dict[key]))
	imagenet_label = max(imagenet_dict.iterkeys(), key = (lambda key: imagenet_dict[key]))
	scene_label = max(scene_dict.iterkeys(), key = (lambda key: scene_dict[key]))

	return cam_label, imagenet_label, scene_label

def output_labels(filename, timestamps, image_files, shot_boundaries, classifier_label_list, googlenet_label_list, scene_type_list, scene_attributes_list):
	
	with open(filename + '.vis', 'w+'): pass

	count = 0
	for idx, boundary in enumerate(shot_boundaries):
		camtype = []
		imagenet = []
		scene = []
		if idx == 0:
			current_shot_timestamp = 0.000
		else:
			current_shot_timestamp = shot_boundaries[idx-1]

		frame_labels = []
		while timestamps[count] <= boundary and count+1 <= len(timestamps):
			line = str(timestamps[count]) + '| ' + classifier_label_list[count] + '| ' + googlenet_label_list[count] + '| ' + scene_type_list[count] + '| ' + scene_attributes_list[count] + '\n'
			frame_labels.append(line)
			camtype.append(classifier_label_list[count])
			imagenet.append(googlenet_label_list[count])
			scene.append(scene_type_list[count])
			count += 1

		print "Boundary: ", idx, len(camtype), len(imagenet), len(scene)
		[cam, gnet, scenetype] = shot_labels(camtype, imagenet, scene)
		print idx, cam, gnet, scenetype

		boundary_label = "SHOT_BOUNDARY" + '| ' + str(current_shot_timestamp) + '| ' + cam + '| ' + gnet + '| ' + scenetype + '\n'
		
		with open(filename + '.vis', 'aw') as file:
			file.write(boundary_label)
			for frame_label in frame_labels:
				file.write(frame_label)
			