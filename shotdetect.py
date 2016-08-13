import scenedetect, time
import cv2, os, sys

def my_detect_scenes_file(path, scene_list, detector_list, stats_writer = None,
                  downscale_factor = 0, frame_skip = 0, quiet_mode = False,
                  perf_update_rate = -1, save_images = False,
                  timecode_list = None):
    """Performs scene detection on passed file using given scene detectors.
    Essentially wraps detect_scenes while handling all OpenCV interaction.
    For descriptions of arguments that are just passed through, see the
    detect_scenes(..) function documentation.
    Args:
        path:  A string containing the filename of the video to open.
        scene_list:  List to append frame numbers of any detected scene cuts.
        detector_list:  List of scene detection algorithms to run on the video.
        See detect_scenes(..) function documentation for details of other args.
    Returns:
        Tuple containing (video_fps, frames_read), where video_fps is a float
        of the video file's framerate, and frames_read is a positive, integer
        number of frames read from the video file.  Both values are set to -1
        if the file could not be opened.
    """

    cap = cv2.VideoCapture()
    frames_read = -1
    video_fps = -1
    if not timecode_list:
        timecode_list = [0, 0, 0]

    # Attempt to open the passed input (video) file.
    cap.open(path)
    # file_name = os.path.split(path)[1]
    file_name = path
    if not cap.isOpened():
        if not quiet_mode:
            print('[PySceneDetect] FATAL ERROR - could not open video %s.' % 
                path)
        return (video_fps, frames_read)
    elif not quiet_mode:
        print('[PySceneDetect] Parsing video %s...' % file_name)

    # Print video parameters (resolution, FPS, etc...)
    video_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    if not quiet_mode:
        print('[PySceneDetect] Video Resolution / Framerate: %d x %d / %2.3f FPS' % (
            video_width, video_height, video_fps ))
        if downscale_factor >= 2:
            print('[PySceneDetect] Subsampling Enabled (%dx, Resolution = %d x %d)' % (
                downscale_factor, video_width / downscale_factor, video_height / downscale_factor ))
        print('Verify that the above parameters are correct'
            ' (especially framerate, use --force-fps to correct if required).')

    # Convert timecode_list to absolute frames for detect_scenes() function.
    frames_list = []
    for tc in timecode_list:
        if isinstance(tc, int):
            frames_list.append(tc)
        elif isinstance(tc, float):
            frames_list.append(int(tc * video_fps))
        elif isinstance(tc, list) and len(tc) == 3:
            secs = float(tc[0] * 60 * 60) + float(tc[1] * 60) + float(tc[2])
            frames_list.append(int(secs * video_fps))
        else:
            frames_list.append(0)

    start_frame, end_frame, duration_frames = 0, 0, 0
    if len(frames_list) == 3:
        start_frame, end_frame, duration_frames = frames_list

    # Perform scene detection on cap object (modifies scene_list).
    frames_read = scenedetect.detect_scenes(cap, scene_list, detector_list, stats_writer,
                                downscale_factor, frame_skip, quiet_mode,
                                perf_update_rate, save_images, file_name,
                                start_frame, end_frame, duration_frames)

    # Cleanup and return number of frames we read.
    cap.release()
    return (video_fps, frames_read)


def shotdetect(clip_dir, clip_path):

	start = time.time()
	#clip_name = '/home/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465.mp4'  # Path to video file.


	scene_list = []        # Scenes will be added to this list in detect_scenes().
	
	# Usually use one detector, but multiple can be used.
	detector_list = [scenedetect.detectors.ContentDetector(threshold = 20)]

	video_fps, frames_read = my_detect_scenes_file(
	    clip_path, scene_list, detector_list, save_images = True)

	# scene_list now contains the frame numbers of scene boundaries.
	# print scene_list

	# create new list with scene boundaries in milliseconds instead of frame #.
	scene_list_sec = [round((x) / float(video_fps), 3) for x in scene_list]

	pyscene_timestamps = []
	for item in scene_list_sec:
		pyscene_timestamps.append(round(item - .01, 3))
		pyscene_timestamps.append(round(item + .01, 3))
	# create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
	# scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]

	# with open(clip_dir + 'scene_list_tc.txt','w') as file:
	# 	for item in scene_list_tc:
	# 		print >> file, item

	# with open(clip_dir + 'scene_list_sec.txt','w') as file:
	# 	for item in scene_list_sec:
	# 		print >> file, "{0:.3f}".format(float(item))

	end = time.time()

	print "Shots detected in %.3f!\n" %(end-start)

	return scene_list_sec, pyscene_timestamps