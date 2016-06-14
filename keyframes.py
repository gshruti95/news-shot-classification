# ffmpeg -i 2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465.mp4 -vf "select='eq(pict_type,PICT_TYPE_I)'" -keyint_min 1 -g 5 -q:v 5 -vsync 2 -f image2 keyframe%03d.jpg -loglevel debug 2>&1 | grep "pict_type:I" | cut -d ' ' -f 4,6 >> keyframes-frame_ms.txt

import os, sys, time

def get_keyframes(clip_dir, clip_name, output_filename):

	start = time.time()
	print "Processing keyframes..."

	os.system("ffmpeg -i " + clip_dir + clip_name \
		+ " -vf \"select='eq(pict_type,PICT_TYPE_I)'\" -keyint_min 1 -g 5 -q:v 5 -vsync 2 -f image2 " \
		+ clip_dir + "keyframe%03d.jpg -loglevel debug 2>&1 | grep \"pict_type:I\" > " \
		+ clip_dir + output_filename + ".vis")

	with open(clip_dir + output_filename + ".vis",'r') as file:
		new_lines = []
		data = file.readlines()
		for line in data:
			line = line.split('t:')[1]
			line = line.split(' ')[0]
			line = "%0.3f\n" % round(float(line),2)
			new_lines.append(line)

	with open(clip_dir + output_filename + ".vis",'w') as file:
		file.writelines(new_lines)

	with open(clip_dir + "faces.vis",'w') as file:
		file.writelines(new_lines)	

	end = time.time()
	print "Keyframes extracted in %.2f!\n" %(end-start)


	
