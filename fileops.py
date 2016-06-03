import os,sys

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
	source = os.listdir(clip_dir)

	for file in source:
		if file.endswith(".jpg"):
			image = clip_dir + os.path.basename(file)
			keyframes_list.append(image)

	return keyframes_list
