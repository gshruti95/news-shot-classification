import os, sys, time, shutil
os.environ["GLOG_minloglevel"] = "2"
import fileops, cropframes
import keyframes, shotdetect
import placesCNN, googlenet, finetune, yolo
import dataset, classifier, cPickle
import format_output, sht_to_json
import path_params
import gpu_util as GPU

def process_news_video(video_name, mode, available_GPU_ID):
	overall_start = time.time()
	exec_time = time.strftime('%Y-%m-%d %H:%M')

	# Load model paths
	caffe_path = path_params.caffe_path
	pycaffe_path = path_params.pycaffe_path
	finetune_path = path_params.finetune_path
	placesCNN_path = path_params.placesCNN_path
	googlenet_path = path_params.googlenet_path
	yolo_path = path_params.yolo_path

	# Load required file paths
	fpickle = path_params.fpickle
	features_file = path_params.features_file

	# Start video processing
	clip_path = video_name								## ../../dir/video.mp4
	rel_clip_path = clip_path.rsplit('/',1)[0] + '/'	## ../../dir/
	clip_name = clip_path.rsplit('/',1)[1]				## video.mp4
	clip = clip_name.rsplit('.',1)[0]					## video
	output_filename = clip 								## video
	clip_dir = rel_clip_path + clip + '/'				## ../../dir/video/

	if not os.path.exists(clip_dir):
		os.makedirs(clip_dir)
	else:
		shutil.rmtree(clip_dir)
		os.makedirs(clip_dir)
	shutil.copy(clip_path, clip_dir)
	new_clip_path = clip_dir + clip_name				## ../../dir/video/video.mp4

	# Keyframe extraction and shot detection on a CPU node
	keyframe_times = keyframes.keyframes(clip_dir, new_clip_path)
	keyframes_list = fileops.get_keyframeslist(clip_dir, new_clip_path)
	[shot_boundaries, py_times] = shotdetect.shotdetect(clip_dir, new_clip_path)
	py_images = fileops.get_pyframeslist(clip_dir, clip_name)
	
	[all_images, all_timestamps] = fileops.rename_frames(clip_dir, keyframe_times, keyframes_list, py_times, py_images)
	if features_file == 'cropped_places_fc7.csv':
		image_files = cropframes.cropframes(clip_dir, all_images, new_clip_path)
		for image in all_images:
			os.remove(image)
	else:
		image_files = all_images
	os.remove(clip_dir + clip_name)
	print "Video preprocessing done...\n"
	
	# YOLO Person detection on a GPU node if available
	[person_count, obj_loc_set] = yolo.yolo(pycaffe_path, yolo_path, image_files, mode, available_GPU_ID)
	print "Retrieved yolo labels...\n"

	# placesCNN for scene type and scene attributes, GPU node used if available
	[fc7, scene_type_list, places_labels, scene_attributes_list] = placesCNN.placesCNN(pycaffe_path, placesCNN_path, image_files, mode, available_GPU_ID)
	fileops.save_features(clip_dir + features_file, fc7)
	print "Extracted fc7 features...\n"

	# SVM classification on a CPU node
	with open(fpickle, 'r') as pickle_file:
		myclassifier = cPickle.load(pickle_file)
	test_data = dataset.testset(clip_dir, features_file)
	classifier_label_list = classifier.classifier_predict(myclassifier, test_data)
	print "(SVM) Classified frames...\n"
	os.remove(clip_dir + features_file)

	# Finetuned net news shot category classification on a GPU node if available
	[finetune_output, finetune_labels] = finetune.mynet(pycaffe_path, finetune_path, image_files, mode, available_GPU_ID)
	print "(Finetuned net) Classified frames...\n"

	# Googlenet object category classification on a GPU node if available
	[googlenet_cat, googlenet_labels] = googlenet.googlenet(pycaffe_path, googlenet_path, image_files, mode, available_GPU_ID)
	print "Retrieved imagenet labels...\n"

	# Create output labels in the SHT format
	format_output.output_labels(exec_time, rel_clip_path + output_filename, output_filename, all_timestamps, image_files, shot_boundaries, classifier_label_list,
	finetune_output, finetune_labels, googlenet_cat, googlenet_labels, scene_type_list, places_labels, scene_attributes_list, person_count, obj_loc_set)
	shutil.rmtree(clip_dir)
	print "Processing complete!\n"
	
	# SHT to JSON lines conversion
	sht_to_json.sht_to_json(rel_clip_path + output_filename + '.sht')
	print "JSON lines ready!\n"

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':

	if len(sys.argv) != 2:
		print 'Usage: python '+sys.argv[0]+' <video-file-path>'
		exit(1)

	mode = 'cpu' #By default, we assume CPU mode
	available_GPU_ID = None
	try:
		available_GPU_ID = GPU.getFirstAvailable()
		mode = 'gpu'
		print 'GPU found!'
	except:
		mode = 'cpu'
		print 'GPU not found! Using CPU mode.'

	process_news_video(sys.argv[1], mode, available_GPU_ID)