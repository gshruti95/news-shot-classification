import os, sys, time, shutil
import fileops, cropframes
import keyframes


def main():

	if sys.argv[1] == 'labels':

		main_dir = './full-clips/train/'
		annotations_file = '_new_shot_type_testuser.txt'

		dir_list = sorted(os.listdir(main_dir))

		label_data = []
		for dir_name in dir_list:
			if os.path.isdir(main_dir + dir_name):
				if os.path.exists(main_dir + dir_name + '/' + dir_name + annotations_file):

					with open(main_dir + dir_name + '/' + dir_name + annotations_file) as labels_file:
						labels = labels_file.readlines()
					labels = [label.split('\t')[0] for label in labels]
					label_data += labels

		label_data = [label + '\n' for label in label_data]
		print len(label_data)

		with open('./train_labels_list.txt', 'w') as file:
			file.writelines(label_data)			

	elif sys.argv[1] == 'commercials':				## py org.py commercials /home/sxg755/dataset/train/all_frames/labels_list.txt ~/dataset/train/all_frames/sorted_keyframes_list.txt
													## ~/dataset/train/8class_train_keyframes/

		labels_f = sys.argv[2]						
		keyframes_f = sys.argv[3]
		main_dir = sys.argv[4]						## ~/dataset/train/8class_train_keyframes/


		# train_label_dir = '/home/sxg755/dataset/train/all_frames/labels_list.txt'
		# test_label_dir = '/home/sxg755/dataset/test/all_frames/labels_list.txt'
		
		# train_frames = '/home/sxg755/dataset/train/all_frames/cropped/'
		# test_frames = '/home/sxg755/dataset/test/all_frames/cropped/'

		# train_dir = '/home/sxg755/dataset/train/'
		# test_dir = '/home/sxg755/dataset/test/'
		
		label_dir = labels_f.rsplit('/',1)[0] + '/'
		frames_path = label_dir + 'cropped/'
		trainset_dir = label_dir.rsplit('/',2)[0] + '/'
		
		if not os.path.exists(main_dir):
			os.makedirs(main_dir)
		
		with open(labels_f, 'r') as lf:
			label_data = lf.readlines()
		label_data = [label.split('\n')[0] for label in label_data]
		with open(keyframes_f, 'r') as kf:
			all_frames = kf.readlines()
		all_frames = [keyframe.split('\n')[0] for keyframe in all_frames]
		keyframes_path = [frames_path + keyframe for keyframe in all_frames]

		final_list = []
		for idx, label in enumerate(label_data):
			if label not in ['Commercial','Problem/Unclassified', 'Black', 'Sports']:
				
				if label == 'Background_roll' or label == 'Background roll':	
					# label = 'Background_roll'
					final_list.append(main_dir + all_frames[idx] + ' 0\n')
				
				elif label == 'Graphic':
					final_list.append(main_dir + all_frames[idx] + ' 1\n')					
				
				elif label == 'Weather' or label == 'Weather/Graphic' or label == 'Weather/Person':
					label = 'Weather'				
					final_list.append(main_dir + all_frames[idx] + ' 1\n')
				
				# elif label == 'Sports':
				# 	final_list.append(main_dir + all_frames[idx] + ' 3\n')
				
				elif label == 'Studio':		
					# label = 'Newsperson(s)'
					final_list.append(main_dir + all_frames[idx] + ' 2\n')
				
				elif label == 'Reporter':
					final_list.append(main_dir + all_frames[idx] + ' 2\n')
				
				elif label == 'Hybrid' or label == 'Talking_head/Hybrid':
					label = 'Hybrid'
					final_list.append(main_dir + all_frames[idx] + ' 2\n')
				
				elif label == 'Talking_head':
					final_list.append(main_dir + all_frames[idx] + ' 0\n')

				shutil.copy(frames_path + all_frames[idx], main_dir)

		with open(main_dir + 'data.txt', 'w') as file:
			file.writelines(final_list)

		print len(final_list)
		

	else:

		clip_path = sys.argv[1] 								## ../../dir/video.mp4
		rel_clip_path = clip_path.rsplit('/',1)[0] + '/'		## ../../dir/
		clip_name = clip_path.rsplit('/',1)[1]					## video.mp4
		clip = clip_name.rsplit('.',1)[0]						## video
		output_filename = clip 									## video
		clip_dir = rel_clip_path								## ../../dir/

		temp = '/home/sxg755/dataset/train/new_all_frames/'		## ../../dir/all_frames/
		if not os.path.exists(temp):
			os.makedirs(temp)

		keyframe_times = keyframes.keyframes(temp, clip_path)
		keyframes_list = fileops.get_keyframeslist(temp, clip_path)

		image_files = cropframes.cropframes(temp, keyframes_list, clip_path)
		image_files = [image_file.rsplit('/',1)[1] + '\n' for image_file in image_files]

		with open(temp + 'keyframes_list.txt', 'aw') as file:
			file.writelines(image_files)


if __name__ == '__main__':
	main()