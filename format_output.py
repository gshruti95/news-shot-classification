import os,sys
import numpy as np
import csv
import time, datetime

def shot_labels(finetune_class, svm_class, imagenet, scene):

	finetune_dict = {}
	svm_dict = {}
	imagenet_dict = {}
	scene_dict = {}

	for idx, svm_label in enumerate(svm_class):
		if svm_label in svm_dict:
			svm_dict[svm_label] += 1
		else:
			svm_dict[svm_label] = 1

		if finetune_class[idx] in finetune_dict:
			finetune_dict[finetune_class[idx]] += 1
		else:
			finetune_dict[finetune_class[idx]] = 1

		if imagenet[idx] in imagenet_dict:
			imagenet_dict[imagenet[idx]] += 1
		else:
			imagenet_dict[imagenet[idx]] = 1

		if scene[idx] in scene_dict:
			scene_dict[scene[idx]] += 1
		else:
			scene_dict[scene[idx]] = 1

	finetune_label = max(finetune_dict.iterkeys(), key = (lambda key: finetune_dict[key]))
	svm_label = max(svm_dict.iterkeys(), key = (lambda key: svm_dict[key]))
	imagenet_label = max(imagenet_dict.iterkeys(), key = (lambda key: imagenet_dict[key]))
	scene_label = max(scene_dict.iterkeys(), key = (lambda key: scene_dict[key]))

	return finetune_label, svm_label, imagenet_label, scene_label

def output_labels(filename, name, timestamps, image_files, shot_boundaries, classifier_label_list, finetune_output, finetune_labels, googlenet_cat, googlenet_labels, scene_type_list, places_labels, scene_attributes_list, person_count, obj_loc_set):
	
	with open(filename + '.vis', 'w+'): pass

	date = name.split('_')[0]

	date = ''.join(map(str, date.split('-')))
	hour = name.split('_')[1]
	dh = date + hour
	video_time = time.mktime(datetime.datetime.strptime(dh, "%Y%m%d%H%M").timetuple())

	count = 0
	for idx, boundary in enumerate(shot_boundaries):
		finetune_class = []
		svm_class = []
		imagenet = []
		scene = []
		if idx == 0:
			current_shot_timestamp = 0.000
		else:
			current_shot_timestamp = shot_boundaries[idx-1]
		
		current_shot_timestamp = float('{0:.3f}'.format(current_shot_timestamp))
		new_cur_shot_time = video_time + current_shot_timestamp
		cur_shot_time_struct =  datetime.datetime.fromtimestamp(new_cur_shot_time)
		cur_shot_timestamp_string = cur_shot_time_struct.strftime("%Y%m%d%H%M%S.%f")[:-3]

		boundary = float('{0:.3f}'.format(boundary))
		new_boundary_time = video_time + boundary
		boundary_time_struct =  datetime.datetime.fromtimestamp(new_boundary_time)
		boundary_timestamp_string = boundary_time_struct.strftime("%Y%m%d%H%M%S.%f")[:-3]

		frame_labels = []
		while timestamps[count] <= boundary and count+1 <= len(timestamps):
			frame_time = float('{0:.3f}'.format(timestamps[count]))
			new_frame_time = video_time + frame_time
			frame_time_struct =  datetime.datetime.fromtimestamp(new_frame_time)
			frame_timestamp_string = frame_time_struct.strftime("%Y%m%d%H%M%S.%f")[:-3]

			finetune_line = frame_timestamp_string + '| ' + frame_timestamp_string + '| FINETUNED_SHOT_CLASS | ' + finetune_output[count] + ' | ' + finetune_labels[count] + '\n'			
			svm_line = frame_timestamp_string + '| ' + frame_timestamp_string + '| SVM_SHOT_CLASS | ' + classifier_label_list[count] + '\n'
			obj_line = frame_timestamp_string + '| ' + frame_timestamp_string + '| OBJ_CLASS | ' + googlenet_cat[count] + ' | ' + googlenet_labels[count] + '\n'
			scene_line = frame_timestamp_string + '| ' + frame_timestamp_string + '| SCENE_LOCATION | ' + scene_type_list[count] + ' | ' + places_labels[count] + '\n'
			attr_line = frame_timestamp_string + '| ' + frame_timestamp_string + '| SCENE_ATTRIBUTES | ' + scene_attributes_list[count] + '\n'
			yolo_line = frame_timestamp_string + '| ' + frame_timestamp_string + '| PERSONS COUNT = ' + person_count[count] + ' | ' + obj_loc_set[count] + '\n'
			frame_labels.append(finetune_line + svm_line + obj_line + scene_line + attr_line + yolo_line)
			finetune_class.append(finetune_output[count])
			svm_class.append(classifier_label_list[count])
			imagenet.append(googlenet_cat[count])
			scene.append(scene_type_list[count])
			count += 1

		[ft_shot_type, svm_shot_type, obj_type, scenetype] = shot_labels(finetune_class, svm_class, imagenet, scene)
		boundary_label = cur_shot_timestamp_string + '| ' + boundary_timestamp_string + '| SHOT_DETECTED >> | Finetuned_Shot_Class= ' + ft_shot_type + ' | SVM_Shot_Class= '  + svm_shot_type + ' | Obj_Class= ' + obj_type + ' | Scene_Type= ' + scenetype + '\n'
		
		with open(filename + '.vis', 'aw') as file:
			file.write(boundary_label)
			for frame_label in frame_labels:
				file.write(frame_label)
			