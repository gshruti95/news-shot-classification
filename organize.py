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

	elif sys.argv[1] == 'commercials':				## py org.py commercials /home/sxg755/trainset/keyframes/labels_list.txt ~/trainset/keyframes/sorted_keyframes_list.txt

		labels_f = sys.argv[2]						## ../trainset/s_keyframes/label.txt
		keyframes_f = sys.argv[3]


		label_dir = labels_f.rsplit('/',1)[0] + '/'
		trainset_dir = label_dir.rsplit('/',2)[0] + '/'
		frames_path = label_dir + 'cropped/'
		temp = trainset_dir + '8class_train_keyframes/'
		test_dir = trainset_dir + '8class_test_keyframes/'

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

		newlines_r = []
		newlines_s = []
		newlines_h = []
		newlines_th = []

		for idx, label in enumerate(label_data):
			if label not in ['Commercial','Problem/Unclassified']:
				if label == 'Reporter':
					newlines_r.append(keyframes[idx])
				elif label == 'Hybrid' or label == 'Talking_head/Hybrid':
					newlines_h.append(keyframes[idx])
				elif label == 'Studio':		
					# label = 'Newsperson(s)'
					newlines_s.append(keyframes[idx])
				elif label == 'Background_roll':	
					# label = 'Background_roll'
					newlines_bg.append(keyframes[idx])
				elif label == 'Talking_head':
					newlines_th.append(keyframes[idx])
				elif label == 'Graphic':
					newlines_g.append(keyframes[idx])						
				elif label == 'Weather':					
					newlines_w.append(keyframes[idx])
				elif label == 'Sports':
					newlines_sp.append(keyframes[idx])
		
		tr_np = 2*len(newlines_np)/3
		tr_bg = 2*len(newlines_bg)/3
		tr_g = 2*len(newlines_g)/3
		tr_w = 2*len(newlines_w)/3
		tr_sp = 2*len(newlines_sp)/3
		tr_s = 2*len(newlines_s)/3
		tr_r = 2*len(newlines_r)/3
		tr_h = 2*len(newlines_h)/3
		tr_th = 2*len(newlines_th)/3
		total = len(newlines_bg) + len(newlines_g) + len(newlines_w) + len(newlines_sp) \
				+ len(newlines_s) + len(newlines_th) + len(newlines_h) + len(newlines_r)
		
		np = 0
		
		bg = 0
		g = 0
		w = 0
		sp = 0
		s = 0
		r = 0
		h = 0
		th = 0

		train = []
		test = []
		for i in range(total):
			# if np < tr_np:
			# 	train.append(temp + newlines_np[np] + ' ' + '0\n')
			# 	shutil.copy(frames_path + newlines_np[np], temp)
			# 	np += 1
			# elif tr_np <= np < len(newlines_np):
			# 	test.append(test_dir + newlines_np[np] + ' ' + '0\n')
			# 	shutil.copy(frames_path + newlines_np[np], test_dir)
			# 	np += 1

			if bg < tr_bg:
				train.append(temp + newlines_bg[bg] + ' ' + '0\n')
				shutil.copy(frames_path + newlines_bg[bg], temp)
				bg += 1
			elif tr_bg <= bg < len(newlines_bg):
				test.append(test_dir + newlines_bg[bg] + ' ' + '0\n')
				shutil.copy(frames_path + newlines_bg[bg], test_dir)
				bg += 1	

			if g < tr_g:
				train.append(temp + newlines_g[g] + ' ' + '1\n')
				shutil.copy(frames_path + newlines_g[g], temp)
				g += 1
			elif tr_g <= g < len(newlines_g):
				test.append(test_dir + newlines_g[g] + ' ' + '1\n')
				shutil.copy(frames_path + newlines_g[g], test_dir)
				g += 1

			if w < tr_w:
				train.append(temp + newlines_w[w] + ' ' + '2\n')
				shutil.copy(frames_path + newlines_w[w], temp)
				w += 1
			elif tr_w <= w < len(newlines_w):
				test.append(test_dir + newlines_w[w] + ' ' + '2\n')
				shutil.copy(frames_path + newlines_w[w], test_dir)
				w += 1

			if sp < tr_sp:
				train.append(temp + newlines_sp[sp] + ' ' + '3\n')
				shutil.copy(frames_path + newlines_sp[sp], temp)
				sp += 1
			elif tr_sp <= sp < len(newlines_sp):
				test.append(test_dir + newlines_sp[sp] + ' ' + '3\n')
				shutil.copy(frames_path + newlines_sp[sp], test_dir)
				sp += 1

			if s < tr_s:
				train.append(temp + newlines_s[s] + ' ' + '4\n')
				shutil.copy(frames_path + newlines_s[s], temp)
				s += 1
			elif tr_s <= s < len(newlines_s):
				test.append(test_dir + newlines_s[s] + ' ' + '4\n')
				shutil.copy(frames_path + newlines_s[s], test_dir)
				s += 1

			if r < tr_r:
				train.append(temp + newlines_r[r] + ' ' + '5\n')
				shutil.copy(frames_path + newlines_r[r], temp)
				r += 1
			elif tr_r <= r < len(newlines_r):
				test.append(test_dir + newlines_r[r] + ' ' + '5\n')
				shutil.copy(frames_path + newlines_r[r], test_dir)
				r += 1

			if h < tr_h:
				train.append(temp + newlines_h[h] + ' ' + '6\n')
				shutil.copy(frames_path + newlines_h[h], temp)
				h += 1
			elif tr_h <= h < len(newlines_h):
				test.append(test_dir + newlines_h[h] + ' ' + '6\n')
				shutil.copy(frames_path + newlines_h[h], test_dir)
				h += 1

			if th < tr_th:
				train.append(temp + newlines_th[th] + ' ' + '7\n')
				shutil.copy(frames_path + newlines_th[th], temp)
				th += 1
			elif tr_th <= th < len(newlines_th):
				test.append(test_dir + newlines_th[th] + ' ' + '7\n')
				shutil.copy(frames_path + newlines_th[th], test_dir)
				th += 1


		print "Train test lengths ", len(train), len(test)
		print "Total ", total

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