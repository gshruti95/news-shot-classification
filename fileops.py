import os,sys
import numpy as np
import csv

def save_studio(filename, label_list):
	with open(filename + ".vis",'r') as file:
		new_lines = []
		timestamps = []
		data = file.readlines()
		print len(data), len(label_list)

		for line in data:
			line = line.split('\n')[0]
			new_lines.append(line)
			timestamp = line.split('|')[0]
			timestamps.append(timestamp.split('\n')[0])

	with open(filename + ".vis",'w') as file:
		for output_label_index in label_list: 
			print output_label_index
			new_lines[output_label_index] += ('| Studio' + '\t' + '\n')
		file.writelines(new_lines)
	

def save_faces_count(filename, faces_count):
	with open(filename + ".vis",'r') as file:
		new_lines = []
		timestamps = []
		data = file.readlines()
		for line in data:
			line = line.split('\n')[0]
			new_lines.append(line)
			timestamp = line.split('|')[0]
			timestamps.append(timestamp.split('\n')[0])

	with open(filename + ".vis",'w') as file:
		for idx, count in enumerate(faces_count): 
			new_lines[idx] += ('| Face count:' + str(count) + '\t' + '\n')		
		file.writelines(new_lines)


def save_features(filename, features):
	
	np.savetxt(filename + ".csv", features, fmt='%.6f', delimiter=',')
	
def write_separate_labels(filename):

	with open(filename + ".vis",'r') as file:
		new_lines = []
		data = file.readlines()
		for line in data:
			timestamp = line.split('|')[0]
			temp = line.split('\t')
			new_lines.append(temp[0]+'\n')
			for write_line in temp:
				if write_line != temp[0] and write_line != temp[-1] and write_line != '\n':
					new_lines.append(timestamp + write_line + '\n')
				elif write_line == temp[-1] and write_line != '\n':
					new_lines.append(timestamp + write_line)

	with open(filename + ".vis",'w') as file:
		file.writelines(new_lines)

def save_placesCNN_labels(filename, csvname, output_label_list, scene_type_list, label_list, scene_attributes_list):

	with open(filename + ".vis",'r') as file:
		new_lines = []
		timestamps = []
		data = file.readlines()
		for line in data:
			line = line.split('\n')[0]
			new_lines.append(line)
			timestamp = line.split('|')[0]
			timestamps.append(timestamp.split('\n')[0])

	with open(filename + ".vis",'w') as file:
		for idx, output_label in enumerate(output_label_list):
			new_lines[idx] += ('| Scene_type= ' + (scene_type_list[idx] + '| ' + output_label + '| ' + label_list[idx] + '| Scene_attributes: ' +  scene_attributes_list[idx]) + '\t' +'\n')
		file.writelines(new_lines)

	with open(csvname + '.csv', 'w') as file:
		for idx, output_label in enumerate(output_label_list): 
			file.write(scene_type_list[idx] + '| ' + output_label + '| ' + label_list[idx] + '| ' + scene_attributes_list[idx] + '\n')

def save_googlenet_labels(filename, csvname, label_list):

	with open(filename + ".vis",'r') as file:
		new_lines = []
		timestamps = []
		data = file.readlines()
		for line in data:
			line = line.split('\n')[0]
			new_lines.append(line)
			timestamp = line.split('|')[0]
			timestamps.append(timestamp.split('\n')[0])

	with open(filename + ".vis",'w') as file:
		for idx, output_label in enumerate(label_list): 
			new_lines[idx] += ('| ' + "Objects: " + (output_label) + '\t' + '\n')
		file.writelines(new_lines)

	with open(csvname + '.csv', 'w') as file:
		for idx, output_label in enumerate(label_list): 
			file.write(output_label + '\n')

def save_age_gender_labels(filename, csvname, age_label_list, gender_label_list, single_faces_frameno, faces_count, multi_faces_frameno):

	with open(filename + ".vis",'r') as file:
		new_lines = []
		timestamps = []
		data = file.readlines()
		for line in data:
			line = line.split('\n')[0]
			new_lines.append(line)
			timestamp = line.split('|')[0]
			timestamps.append(timestamp.split('\n')[0])

	with open(filename + ".vis",'w') as file:
		for idx, count in zip(multi_faces_frameno, faces_count):
			if count > 0: 
				new_lines[idx[0]] += ('| Face count:' + str(count) )
			
		for idx, gender_label, age_label in zip(single_faces_frameno, gender_label_list,age_label_list):
			new_lines[idx] += ('| ' + (gender_label + '| ' + age_label) )

		new_lines = [new_line + '\t\n' for new_line in new_lines]
		file.writelines(new_lines)

	with open(csvname + '.csv', 'w') as file:
		for idx, gender_label in enumerate(gender_label_list): 
			file.write(gender_label + '| ' + age_label_list[idx] + '\n')

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
