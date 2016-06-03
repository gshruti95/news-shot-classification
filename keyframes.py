# ffmpeg -i 2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465.mp4 -vf "select='eq(pict_type,PICT_TYPE_I)'" -keyint_min 1 -g 5 -q:v 5 -vsync 2 -f image2 keyframe%03d.jpg -loglevel debug 2>&1 | grep "pict_type:I" | cut -d ' ' -f 4,6 >> keyframes-frame_ms.txt

import os, sys

def get_keyframes(clip_dir, clip_name):

	os.system("ffmpeg -i " + clip_dir + clip_name \
		+ " -vf \"select='eq(pict_type,PICT_TYPE_I)'\" -keyint_min 1 -g 5 -q:v 5 -vsync 2 -f image2 " \
		+ clip_dir + "keyframe%03d.jpg -loglevel debug 2>&1 | grep \"pict_type:I\" | cut -d ' ' -f 4,6 > " \
		+ clip_dir + "keyframes-num_ms.txt")

	print "Keyframes extracted!\n"
