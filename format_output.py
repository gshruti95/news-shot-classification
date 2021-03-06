import os,sys
import numpy as np
import csv
import time, datetime

def shot_labels(finetune_class, svm_class, imagenet, scene, person):

	finetune_dict = {}
	svm_dict = {}
	imagenet_dict = {}
	scene_dict = {}
	person_dict = {'True':0, 'False':0}

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

		if int(person[idx]) > 0:
			person_dict['True'] += 1
		else:
			person_dict['False'] += 1


	finetune_label = max(finetune_dict.iterkeys(), key = (lambda key: finetune_dict[key]))
	svm_label = max(svm_dict.iterkeys(), key = (lambda key: svm_dict[key]))
	imagenet_label = max(imagenet_dict.iterkeys(), key = (lambda key: imagenet_dict[key]))
	scene_label = max(scene_dict.iterkeys(), key = (lambda key: scene_dict[key]))
	person_label = max(person_dict.iterkeys(), key = (lambda key: person_dict[key]))

	return finetune_label, svm_label, imagenet_label, scene_label, person_label

def output_labels(exec_time, filename, name, timestamps, image_files, shot_boundaries, classifier_label_list, finetune_output, finetune_labels, googlenet_cat, googlenet_labels, scene_type_list, places_labels, scene_attributes_list, person_count, obj_loc_set):
	 
	with open(filename + '.sht', 'w+') as out_file:
		credit_line = "SHT_01|" + exec_time + "|Source_Program=ShotClass-01.py|Source_Person=Shruti Gullapuram\n"
		out_file.write(credit_line)

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
		person = []
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

			finetune_line = frame_timestamp_string + '|' + frame_timestamp_string + '|SHT_01|FINETUNED_CLASS|' + finetune_output[count] + '|' + finetune_labels[count] + '\n'			
			svm_line = frame_timestamp_string + '|' + frame_timestamp_string + '|SHT_01|SVM_CLASS|' + classifier_label_list[count] + '\n'
			obj_line = frame_timestamp_string + '|' + frame_timestamp_string + '|SHT_01|OBJ_CLASS|' + googlenet_cat[count] + '|' + googlenet_labels[count] + '\n'
			scene_line = frame_timestamp_string + '|' + frame_timestamp_string + '|SHT_01|SCENE_LOCATION|' + scene_type_list[count] + '|' + places_labels[count] + '\n'
			attr_line = frame_timestamp_string + '|' + frame_timestamp_string + '|SHT_01|SCENE_ATTRIBUTES|' + scene_attributes_list[count] + '\n'
			yolo_line = frame_timestamp_string + '|' + frame_timestamp_string + '|SHT_01|YOLO/PERSONS|Count=' + person_count[count] + '|' + obj_loc_set[count] + '\n\n'
			
			frame_labels.append(finetune_line + svm_line + obj_line + scene_line + attr_line + yolo_line)
			finetune_class.append(finetune_output[count])
			svm_class.append(classifier_label_list[count])
			imagenet.append(googlenet_cat[count])
			scene.append(scene_type_list[count])
			person.append(person_count[count])
			count += 1

		[ft_shot_type, svm_shot_type, obj_type, scenetype, yolo_person] = shot_labels(finetune_class, svm_class, imagenet, scene, person)
		# if yolo_person == 'not':
		# 	boundary_label = cur_shot_timestamp_string + '|' + boundary_timestamp_string + '|SHT_01|SHOT_DETECTED >>|Finetuned_Shot_Class=' + ft_shot_type + '|SVM_Shot_Class='  + svm_shot_type + '|Obj_Class=' + obj_type + '|Scene_Type=' + scenetype + '\n'
		# else:
		# 	boundary_label = cur_shot_timestamp_string + '|' + boundary_timestamp_string + '|SHT_01|SHOT_DETECTED >>|' + yolo_person + '|Finetuned_Shot_Class=' + ft_shot_type + '|SVM_Shot_Class='  + svm_shot_type + '|Obj_Class=' + obj_type + '|Scene_Type=' + scenetype + '\n'
		boundary_label = cur_shot_timestamp_string + '|' + boundary_timestamp_string + '|SHT_01|SHOT_DETECTED|YOLO/PERSONS=' + yolo_person + '|FINETUNED_SHOT_CLASS=' + ft_shot_type + '|SVM_SHOT_CLASS='  + svm_shot_type + '|OBJ_SHOT_CLASS=' + obj_type + '|SCENE_TYPE=' + scenetype + '\n'

		with open(filename + '.sht', 'aw') as file:
			file.write(boundary_label)
			for frame_label in frame_labels:
				file.write(frame_label)