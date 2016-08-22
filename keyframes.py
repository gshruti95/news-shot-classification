import os, sys, time

def keyframes(clip_dir, clip_path):

	start = time.time()
	print "Processing keyframes..."
	
	clip = clip_path.split('/')[-1]
	clip_name = clip.split('.')[0]

	os.system("ffmpeg -i " + clip_path \
		+ " -vf \"select='eq(pict_type,PICT_TYPE_I)'\" -keyint_min 1 -g 5 -q:v 5 -vsync 2 -f image2 " \
		+ clip_dir + clip_name + "_keyframe%04d.jpg -loglevel debug 2>&1 | grep \"pict_type:I\" > " \
		+ clip_dir + clip_name + "_tempkeyframes.times")

	with open(clip_dir + clip_name + "_tempkeyframes.times",'r') as file:
		data = file.readlines()
	timestamps = []
	stamps = []
	for line in data:
		line = line.split('t:')[1]
		line = line.split(' ')[0]
		line = "{0:.3f}".format(float(line))
		timestamps.append(float(line))
		stamps.append(line)

	with open(clip_dir + clip_name + "_keyframes.times",'w') as file:
		file.writelines("\n".join(stamps))

	end = time.time()
	print "Keyframes extracted in %.2f!\n" %(end-start)

	return timestamps

	
