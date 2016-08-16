import os, sys, time, shutil, errno
os.environ["GLOG_minloglevel"] = "2"
import fileops, cropframes
import keyframes, shotdetect
import placesCNN, googlenet
import dataset, classifier, cPickle
import format_output
import finetune


def main():
	overall_start = time.time()

	caffe_path = './mycaffe/'
	fpickle = './ovr_classifier.pkl'
	features_file = 'cropped_places_fc7 .csv'	
	finetune_path = caffe_path + 'models/finetune/ref_caffenet/' 
	

	## Test classifier accuracy
	if sys.argv[1] == 'testmode':

		fpickle = './new_ovr_classifier.pkl'
		class_type = 'newsperson'
		test_dir = './full-clips/test/'	
		train_dir = './full-clips/train/'
		annotation_file = '_new_shot_type_testuser.txt'	
		# classifier.classifier_dump(fpickle, train_dir, annotation_file, features_file)
		# with open(fpickle, 'r') as pickle_file:
		# 	myclassifier = cPickle.load(pickle_file)
		# [test_data, test_labels] = dataset.trainset(test_dir, annotation_file, features_file)
		# # new_test_labels = dataset.ovo_trainset(test_labels, class_type)
		# new_test_labels = test_labels
		# output_labels = classifier.predict_testmode(myclassifier, test_data, new_test_labels, test_labels)
		# print "Classified frames...\n"

		clip_dir = sys.argv[2]
		clip_name = fileops.get_video_filename(clip_dir)
		clip = clip_name.split('.')[0]
		clip_path = clip_dir + clip_name 
		keyframe_times = keyframes.keyframes(clip_dir, clip_path)
		keyframes_list = fileops.get_keyframeslist(clip_dir, clip_path)
		image_files = cropframes.cropframes(clip_dir, keyframes_list, clip_path)

		[fc7, scene_type_list, places_labels, scene_attributes_list] = placesCNN.placesCNN(caffe_path, caffe_path + 'models/placesCNN/', image_files)
		fileops.save_features(clip_dir + features_file, fc7)
		print "Extracted fc7 features...\n"

		# [output, labels_set] = finetune.mynet(caffe_path, finetune_path, image_files)
		# with open(clip_dir + clip + annotation_file, 'r') as file:
		# 	orig_labels = file.readlines()
		# orig_labels = [label.split('\t')[0] for label in orig_labels]
		# finetune.performance(orig_labels, output, labels_set, image_files)

	else:

		clip_path = sys.argv[1]
		print "Clip path ", clip_path								## ../../dir/video.mp4
		rel_clip_path = clip_path.rsplit('/',1)[0] + '/'
		print "rel_clip_path ", rel_clip_path		## ../../dir/
		clip_name = clip_path.rsplit('/',1)[1]	
		print "clip_name ", clip_name				## video.mp4
		clip = clip_name.rsplit('.',1)[0]
		print "clip ", clip						## video
		output_filename = clip 									## video
		clip_dir = rel_clip_path + clip + '/'
		print "clip_dir ", clip_dir					## ../../dir/video/

		# try:
  #   		os.mkdir(clip_dir)
		# except OSError as exc:
  #   		if exc.errno != errno.EEXIST:
  #       		raise exc
  #   		pass
		if not os.path.exists(clip_dir):
			print "making dir"
			os.makedirs(clip_dir)
		else:
			print "deleting"
			shutil.rmtree(clip_dir)
			time.sleep(10)
			print "making again"
			os.makedirs(clip_dir)
		time.sleep(1)
		print "copy video"
		shutil.copy(clip_path, clip_dir)
		new_clip_path = clip_dir + clip_name
		print new_clip_path					## ../../dir/video/video.mp4

		keyframe_times = keyframes.keyframes(clip_dir, new_clip_path)
		keyframes_list = fileops.get_keyframeslist(clip_dir, new_clip_path)
		[shot_boundaries, py_times] = shotdetect.shotdetect(clip_dir, new_clip_path)
		py_images = fileops.get_pyframeslist(clip_dir, clip_name)
		
		[all_images, all_timestamps] = fileops.rename_frames(clip_dir, keyframe_times, keyframes_list, py_times, py_images)
		if features_file == 'cropped_places_fc7 .csv':
			image_files = cropframes.cropframes(clip_dir, all_images, new_clip_path)
			print "removing extra images"
			for image in all_images:
				os.remove(image)
		else:
			image_files = all_images

		print "Removing clip copy"
		os.remove(clip_dir + clip_name)
		print "Video preprocessing done...\n"
		
		## Run a model and get labels for keyframe
		[fc7, scene_type_list, places_labels, scene_attributes_list] = placesCNN.placesCNN(caffe_path, caffe_path + 'models/placesCNN/', image_files)
		fileops.save_features(clip_dir + features_file, fc7)
		print "Extracted fc7 features...\n"

		with open(fpickle, 'r') as pickle_file:
			myclassifier = cPickle.load(pickle_file)
		test_data = dataset.testset(clip_dir, features_file)
		classifier_label_list = classifier.classifier_predict(myclassifier, test_data)
		print "Classified frames...\n"

		os.remove(clip_dir + features_file)

		[googlenet_cat, googlenet_labels] = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
		print "Retrieved imagenet labels...\n"

		format_output.output_labels(rel_clip_path + output_filename, all_timestamps, image_files, shot_boundaries, classifier_label_list, googlenet_cat, googlenet_labels, scene_type_list, places_labels, scene_attributes_list)
		
		shutil.rmtree(clip_dir)
		time.sleep(10)

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()