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

		with open('./labels_list.txt', 'w') as file:
			file.writelines(label_data)			

	elif sys.argv[1] == 'commercials':				## py org.py commercials ~/trainset/keyframes/labels_list.txt ~/trainset/keyframes/s_keyframes_list.txt

		labels_f = sys.argv[2]						## ../trainset/s_keyframes/label.txt
		keyframes_f = sys.argv[3]


		label_dir = labels_f.rsplit('/',1)[0] + '/'
		trainset_dir = label_dir.rsplit('/',2)[0] + '/'
		frames_path = label_dir + 'cropped/'
		temp = trainset_dir + 'train_keyframes/'
		test_dir = trainset_dir + 'test_keyframes/'

		if not os.path.exists(temp):
			os.makedirs(temp)
		if not os.path.exists(test_dir):
			os.makedirs(test_dir) 

		with open(labels_f, 'r') as lf:
			label_data = lf.readlines()
		label_data = [label.split('\n')[0] for label in label_data]
		with open(keyframes_f, 'r') as kf:
			keyframes = kf.readlines()
			keyframes = [keyframe.split('\n')[0] for keyframe in keyframes]
		keyframes_path = [frames_path + keyframe for keyframe in keyframes]

		newlines_np = []
		newlines_bg = []
		newlines_g = []
		newlines_w = []
		newlines_sp = []

		for idx, label in enumerate(label_data):
			if label not in ['Commercial','Problem/Unclassified']:
				if label == 'Reporter' or label == 'Hybrid' or label == 'Studio':		
					label = 'Newsperson(s)'
					newlines_np.append(keyframes[idx])
				elif label == 'Background_roll' or label == 'Talking_head' or label == 'Talking_head/Hybrid':	
					label = 'Background_roll'
					newlines_bg.append(keyframes[idx])
				elif label == 'Graphic':
					newlines_g.append(keyframes[idx])						
				elif label == 'Weather':					
					newlines_w.append(keyframes[idx])
				elif label == 'Sports':
					newlines_sp.append(keyframes[idx])

		# temp + keyframes[idx] + ' ' + label + '\n'
		total = len(label_data)
		tr_np = 2*len(newlines_np)/3
		tr_bg = 2*len(newlines_bg)/3
		tr_g = len(newlines_g) - 100
		tr_w = len(newlines_w) - 20
		tr_sp = len(newlines_sp) - 15
		
		np = 0
		bg = 0
		g = 0
		w = 0
		sp = 0

		train = []
		test = []
		for i in range(max(tr_np, tr_bg, tr_g, tr_w, tr_sp)):
			if np < tr_np:
				train.append(temp + newlines_np[np] + ' ' + '0\n')
				shutil.copy(frames_path + newlines_np[np], temp)
				np += 1
			elif tr_np <= np < len(newlines_np):
				test.append(test_dir + newlines_np[np] + ' ' + '0\n')
				shutil.copy(frames_path + newlines_np[np], test_dir)
				np += 1

			if bg < tr_bg:
				train.append(temp + newlines_bg[bg] + ' ' + '1\n')
				shutil.copy(frames_path + newlines_bg[bg], temp)
				bg += 1
			elif tr_bg <= bg < len(newlines_bg):
				test.append(test_dir + newlines_bg[bg] + ' ' + '1\n')
				shutil.copy(frames_path + newlines_bg[bg], test_dir)
				bg += 1	

			if g < tr_g:
				train.append(temp + newlines_g[g] + ' ' + '2\n')
				shutil.copy(frames_path + newlines_g[g], temp)
				g += 1
			elif tr_g <= g < len(newlines_g):
				test.append(test_dir + newlines_g[g] + ' ' + '2\n')
				shutil.copy(frames_path + newlines_g[g], test_dir)
				g += 1

			if w < tr_w:
				train.append(temp + newlines_w[w] + ' ' + '3\n')
				shutil.copy(frames_path + newlines_w[w], temp)
				w += 1
			elif tr_w <= w < len(newlines_w):
				test.append(test_dir + newlines_w[w] + ' ' + '3\n')
				shutil.copy(frames_path + newlines_w[w], test_dir)
				w += 1

			if sp < tr_sp:
				train.append(temp + newlines_sp[sp] + ' ' + '4\n')
				shutil.copy(frames_path + newlines_sp[sp], temp)
				sp += 1
			elif tr_sp <= sp < len(newlines_sp):
				test.append(test_dir + newlines_sp[sp] + ' ' + '4\n')
				shutil.copy(frames_path + newlines_sp[sp], test_dir)
				sp += 1

		with open(temp + 'train.txt', 'w') as file:
			file.writelines(train)
		with open(test_dir + 'test.txt', 'w') as file:
			file.writelines(test)



	else:

		clip_path = sys.argv[1] 								## ../../dir/video.mp4
		rel_clip_path = clip_path.rsplit('/',1)[0] + '/'		## ../../dir/
		clip_name = clip_path.rsplit('/',1)[1]					## video.mp4
		clip = clip_name.rsplit('.',1)[0]						## video
		output_filename = clip 									## video
		clip_dir = rel_clip_path								## ../../dir/

		temp = clip_dir + 'keyframes/'							## ../../dir/keyframes/
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