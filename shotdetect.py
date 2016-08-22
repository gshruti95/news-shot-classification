import scenedetect, time
import cv2, os, sys

def my_detect_scenes_file(path, scene_list, detector_list, stats_writer = None,
                  downscale_factor = 0, frame_skip = 0, quiet_mode = False,
                  perf_update_rate = -1, save_images = False,
                  timecode_list = None):

    cap = cv2.VideoCapture()
    frames_read = -1
    video_fps = -1
    if not timecode_list:
        timecode_list = [0, 0, 0]

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

    frames_read = scenedetect.detect_scenes(cap, scene_list, detector_list, stats_writer,
                                downscale_factor, frame_skip, quiet_mode,
                                perf_update_rate, save_images, file_name,
                                start_frame, end_frame, duration_frames)

    cap.release()
    return (video_fps, frames_read)


def shotdetect(clip_dir, clip_path):

	start = time.time()
	
	scene_list = []
	detector_list = [scenedetect.detectors.ContentDetector(threshold = 20)]
	video_fps, frames_read = my_detect_scenes_file(clip_path, scene_list, detector_list, save_images = True)
	scene_list_sec = [round((x) / float(video_fps), 3) for x in scene_list]

	pyscene_timestamps = []
	for item in scene_list_sec:
		pyscene_timestamps.append(round(item - .01, 3))
		pyscene_timestamps.append(round(item + .01, 3))
	# create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
	# scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]

	end = time.time()
	print "Shots detected in %.3f!\n" %(end-start)

	return scene_list_sec, pyscene_timestamps