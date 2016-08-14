import os, sys, time, shutil
os.environ["GLOG_minloglevel"] = "2"
import fileops, cropframes
import keyframes, shotdetect
import placesCNN, googlenet, vgg_face
import dataset, classifier, cPickle
import format_output
# import graphcluster, accuracy


def main():
	overall_start = time.time()

	caffe_path = './mycaffe/'
	fpickle = './ovr_classifier.pkl'
	features_file = 'cropped_places_fc7 .csv'		

	## Test classifier accuracy
	if sys.argv[1] == 'testmode':

		# fc7 = vgg_face.vgg_face(caffe_path, caffe_path + 'models/vgg_face_caffe/', caffe_path + 'models/vgg_face_caffe/ak.png')
		# fileops.save_features(caffe_path + 'models/vgg_face_caffe/fc7.csv', fc7)

		# fpickle = './et100_classifier.pkl'
		test_dir = './full-clips/test/'	
		train_dir = './full-clips/train/'
		annotation_file = '_shot_type_testuser.txt'
		class_type = 'newsperson'
		# classifier.classifier_dump(fpickle, train_dir, annotation_file, features_file)


		# with open(fpickle, 'r') as pickle_file:
		# 	myclassifier = cPickle.load(pickle_file)
		# [test_data, test_labels] = dataset.trainset(test_dir, annotation_file, features_file)
		# # new_test_labels = dataset.ovo_trainset(test_labels, class_type)
		# new_test_labels = test_labels
		# output = classifier.predict_testmode(myclassifier, test_data, new_test_labels, test_labels)

		# print "Classified frames...\n"

		[train_data, train_labels, bg_frames, ann_files] = dataset.trainset(train_dir, annotation_file, features_file)
		print bg_frames
		print ann_files

		# with open(fpickle, 'r') as pickle_file:
		# 	myclassifier = cPickle.load(pickle_file)
		# test_data = dataset.testset(clip_dir, features_file)
		# classifier_label_list = classifier.classifier_predict(myclassifier, test_data)
		# print "Classified frames...\n"

		broll_dir = './broll/'
		if not os.path.exists(broll_dir):
			os.makedirs(broll_dir)

		cur_dir = 'nothing'
		ann_dir = ann_files[0].split('/')[-2]
		count = 0
		for idx, image in enumerate(bg_frames):
			image_dir = image.split('/')[-2]
			if image_dir != cur_dir:
				cur_dir = image_dir
				if not os.path.exists(broll_dir + cur_dir + '/'):
					os.makedirs(broll_dir + cur_dir + '/')
					print 'Made ' + broll_dir + cur_dir + '/'
					with open(broll_dir + cur_dir + '/' + 'cropped_places_fc7 .csv', 'w') as feat_file: pass

			shutil.copy(image, broll_dir + cur_dir + '/')
			with open(broll_dir + cur_dir + '/' + 'cropped_places_fc7 .csv', 'aw') as feat_file:
				feat_file.write(train_data[idx]+'\n')

			if ann_dir == cur_dir and count < len(ann_files):
				ann_dir = ann_files[count].split('/')[-2]
				file = ann_files[count]
				shutil.copy(file, broll_dir + cur_dir + '/')
				count += 1
				
		
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
		new_clip_path = clip_dir + clip_name					## ../../dir/video/video.mp4

		keyframe_times = keyframes.keyframes(clip_dir, new_clip_path)
		keyframes_list = fileops.get_keyframeslist(clip_dir)
		shot_boundaries, py_times = shotdetect.shotdetect(clip_dir, new_clip_path)
		py_images = fileops.get_pyframeslist(clip_dir, clip_name)
		
		orig_images, all_timestamps = fileops.rename_frames(clip_dir, keyframe_times, keyframes_list, py_times, py_images)
		if features_file == 'cropped_places_fc7 .csv':
			image_files = cropframes.cropframes(clip_dir, orig_images)
			for image in orig_images:
				os.remove(image)
		else:
			image_files = orig_images

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

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()