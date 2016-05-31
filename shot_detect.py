import scenedetect

def shot_detect(clip_dir,clip_name):

	#clip_name = '/home/gsoc/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465.mp4'  # Path to video file.


	scene_list = []        # Scenes will be added to this list in detect_scenes().
	
	# Usually use one detector, but multiple can be used.
	detector_list = [scenedetect.detectors.ContentDetector(threshold = 20),
					 scenedetect.detectors.ThresholdDetector(threshold= 16)]

	video_fps, frames_read = scenedetect.detect_scenes_file(
	    clip_dir+clip_name, scene_list, detector_list)

	# scene_list now contains the frame numbers of scene boundaries.
	print scene_list

	# create new list with scene boundaries in milliseconds instead of frame #.
	scene_list_msec = [(1000.0 * x) / float(video_fps) for x in scene_list]

	# create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
	scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]

	with open(str(clip_dir)+'scene_list_tc.txt','w') as file:
		for item in scene_list_tc:
			print >> file, item

	with open(str(clip_dir)+'scene_list.txt','w') as file:
		for item in scene_list:
			print >> file, item

	print "Shots detected!"
