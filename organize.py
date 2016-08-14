import os, sys, time, shutil
import fileops, cropframes
import keyframes


def main():

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