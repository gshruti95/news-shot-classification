import os, sys
import fileops
import keyframes
import shot_detect

def main():

	clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/'
	clip_name = fileops.get_video_filename(clip_dir)
	shot_detect.shot_detect(clip_dir,clip_name)
	keyframes.get_keyframes(clip_dir,clip_name)




if __name__ == '__main__':
	main()