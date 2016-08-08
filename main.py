import os, sys, time, shutil
os.environ["GLOG_minloglevel"] = "2"
import fileops, cropframes
import keyframes, shotdetect
import placesCNN, googlenet
import dataset, classifier, cPickle
import format_output
# import graphcluster, accuracy


def main():
	overall_start = time.time()

	clip_path = sys.argv[1] 								## ../../dir/video.mp4
	rel_clip_path = clip_path.rsplit('/',1)[0] + '/'		## ../../dir/
	clip_name = clip_path.rsplit('/',1)[1]					## video.mp4
	clip = clip_name.rsplit('.',1)[0]						## video
	output_filename = clip 									## video
	clip_dir = rel_clip_path + clip + '/'					## ../../dir/video/

	if not os.path.exists(clip_dir):
		os.makedirs(clip_dir)
	else:
		shutil.rmtree(clip_dir)
		os.makedirs(clip_dir)
	shutil.copy(clip_path, clip_dir)
	new_clip_path = clip_dir + clip_name

	keyframe_times = keyframes.keyframes(clip_dir, new_clip_path)
	keyframes_list = fileops.get_keyframeslist(clip_dir)
	shot_boundaries, py_times = shotdetect.shotdetect(clip_dir, new_clip_path)
	py_images = fileops.get_pyframeslist(clip_dir, clip_name)
	
	image_files, all_timestamps = fileops.rename_frames(clip_dir, keyframe_times, keyframes_list, py_times, py_images)
	# image_files = cropframes.cropframes(clip_dir, image_files)
	print "Video preprocessing done...\n"
	
	## Run a model and get labels for keyframe
 	
	caffe_path = '/home/shruti/gsoc/caffehome/caffe/'
	[fc7, scene_type_list, scene_attributes_list] = placesCNN.placesCNN(caffe_path, caffe_path + 'models/placesCNN/', image_files)
	fileops.save_features(clip_dir + 'cropped_places_fc7 ', fc7)
	print "Extracted fc7 features...\n"

	googlenet_label_list = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
	print "Retrieved imagenet labels...\n"
	
	# train_dir = '/home/shruti/gsoc/news-shot-classification/full-clips/train/'
	# [train_data, train_labels] = dataset.trainset(train_dir)
	# mysvm = classifier.classifier_train(train_data, train_labels)
	# with open('newsperson_classifier.pkl', 'w') as pickle_file:
	# 	cPickle.dump(mysvm, pickle_file)

	with open('newsperson_classifier.pkl', 'r') as pickle_file:
		mysvm = cPickle.load(pickle_file)
	test_data = dataset.testset(clip_dir)
	classifier_label_list = classifier.classifier_predict(mysvm, test_data)
	print "Classified frames...\n"

	format_output.output_labels(rel_clip_path + output_filename, all_timestamps, image_files, shot_boundaries, classifier_label_list, googlenet_label_list, scene_type_list, scene_attributes_list)
	
	shutil.rmtree(clip_dir)

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()