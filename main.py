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

	caffe_path = './mycaffe/'
	fpickle = './ovr_classifier.pkl'
	features_file = 'cropped_places_fc7'		

	## Test classifier accuracy
	if sys.argv[1] == 'testmode':

		test_dir = './full-clips/test/'	
		train_dir = './full-clips/train/'
		annotation_file = '_shot_type_testuser.txt'
		class_type = 'newsperson'
		# [train_data, train_labels] = dataset.trainset(train_dir, annotation_file, features_file)
		# # train_labels = dataset.ovo_trainset(train_labels, class_type)
		# mysvm = classifier.classifier_train(train_data, train_labels)
		# with open(fpickle, 'w') as pickle_file:
		# 	cPickle.dump(mysvm, pickle_file)

		with open(fpickle, 'r') as pickle_file:
			mysvm = cPickle.load(pickle_file)
		[test_data, test_labels] = dataset.trainset(test_dir, annotation_file, features_file)
		new_test_labels = dataset.ovo_trainset(test_labels, class_type)
		# new_test_labels = test_labels
		output = classifier.predict_testmode(mysvm, test_data, new_test_labels, test_labels)
		
		# for predicted, actual in zip(output, test_labels):
		# 	print predicted, actual
		print "Classified frames...\n"

		
	else:

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
		image_files = cropframes.cropframes(clip_dir, image_files)
		print "Video preprocessing done...\n"
		
		## Run a model and get labels for keyframe
	 	
		[fc7, scene_type_list, scene_attributes_list] = placesCNN.placesCNN(caffe_path, caffe_path + 'models/placesCNN/', image_files)
		fileops.save_features(clip_dir + features_file, fc7)
		print "Extracted fc7 features...\n"

		googlenet_label_list = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
		print "Retrieved imagenet labels...\n"

		with open(fpickle, 'r') as pickle_file:
			mysvm = cPickle.load(pickle_file)
		test_data = dataset.testset(clip_dir, features_file)
		classifier_label_list = classifier.classifier_predict(mysvm, test_data)
		print "Classified frames...\n"

		format_output.output_labels(rel_clip_path + output_filename, all_timestamps, image_files, shot_boundaries, classifier_label_list, googlenet_label_list, scene_type_list, scene_attributes_list)
		
		shutil.rmtree(clip_dir)

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()