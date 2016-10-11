import os, sys, time, shutil
os.environ["GLOG_minloglevel"] = "2"
import fileops, cropframes
import keyframes, shotdetect
import placesCNN, googlenet, finetune, yolo
import dataset, classifier, cPickle
import format_output
import path_params

def main():
	overall_start = time.time()

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
	
	if sys.argv[1] == "snaps":

		# clip_dir = sys.argv[2]

		# source = sorted(os.listdir(clip_dir))
		# keyframes_list = []

		# for file in source:
		# 	if file.endswith(".jpg"):
		# 		image = clip_dir + os.path.basename(file)
		# 		print file
		# 		keyframes_list.append(image)

		# keyframes_list.sort(key = fileops.natural_sorting)

		# new_clip_path = clip_dir + 'temp.mp4'

		# if features_file == 'cropped_places_fc7.csv':
		# 	image_files = cropframes.cropframes(clip_dir, keyframes_list, new_clip_path)

		clip_path = sys.argv[2]								## ../../dir/video.mp4
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

		keyframe_times = keyframes.keyframes(clip_dir, new_clip_path)
		keyframes_list = fileops.get_keyframeslist(clip_dir, new_clip_path)
		if features_file == 'cropped_places_fc7.csv':
			image_files = cropframes.cropframes(clip_dir, keyframes_list, new_clip_path)
		print "Video preprocessing done...\n"
		
		[person_count, obj_loc_set] = yolo.yolo(pycaffe_path, yolo_path, image_files)
		print "Retrieved yolo labels...\n"

		shutil.rmtree(clip_dir)
		print "Processing complete!\n"
		# ## Run a model and get labels for keyframe
		# [fc7, scene_type_list, places_labels, scene_attributes_list] = placesCNN.placesCNN(pycaffe_path, placesCNN_path, image_files)
		# fileops.save_features(clip_dir + features_file, fc7)
		# print "Extracted fc7 features...\n"

		# with open(fpickle, 'r') as pickle_file:
		# 	myclassifier = cPickle.load(pickle_file)
		# test_data = dataset.testset(clip_dir, features_file)
		# classifier_label_list = classifier.classifier_predict(myclassifier, test_data)
		# print "(SVM) Classified frames...\n"

		# [finetune_output, finetune_labels] = finetune.mynet(pycaffe_path, finetune_path, image_files)
		# print "(Finetuned net) Classified frames...\n"

		# [googlenet_cat, googlenet_labels] = googlenet.googlenet(pycaffe_path, googlenet_path, image_files)
		# print "Retrieved imagenet labels...\n"

		with open('test/out.txt','w') as file:
			for idx, p_count in enumerate(person_count):
				file.write(str(idx+1))
				# file.write('| FINETUNED_SHOT_CLASS | ' + finetune_output[idx] + '| ' + finetune_labels[idx] + '\n')
				# file.write('| SVM_SHOT_CLASS | ' + classifier_label_list[idx] +'\n')
				# file.write('| OBJ_CLASS | ' + googlenet_cat[idx] + '| ' + googlenet_label + '\n')
				# file.write('| SCENE_LOCATION | ' + scene_type_list[idx] + '| ' + places_labels[idx] + '\n')
				# file.write('| SCENE_ATTRIBUTES | ' + scene_attributes_list[idx] + '\n')
				file.write('| PERSONS COUNT = ' + p_count + ' | ' + obj_loc_set[idx] + '\n')
				file.write('\n')

	
	else:

		clip_path = sys.argv[1]								## ../../dir/video.mp4
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
		
		[person_count, obj_loc_set] = yolo.yolo(pycaffe_path, yolo_path, image_files)
		print "Retrieved yolo labels...\n"

		## Run a model and get labels for keyframe
		[fc7, scene_type_list, places_labels, scene_attributes_list] = placesCNN.placesCNN(pycaffe_path, placesCNN_path, image_files)
		fileops.save_features(clip_dir + features_file, fc7)
		print "Extracted fc7 features...\n"

		with open(fpickle, 'r') as pickle_file:
			myclassifier = cPickle.load(pickle_file)
		test_data = dataset.testset(clip_dir, features_file)
		classifier_label_list = classifier.classifier_predict(myclassifier, test_data)
		print "(SVM) Classified frames...\n"
		os.remove(clip_dir + features_file)

		[finetune_output, finetune_labels] = finetune.mynet(pycaffe_path, finetune_path, image_files)
		print "(Finetuned net) Classified frames...\n"

		[googlenet_cat, googlenet_labels] = googlenet.googlenet(pycaffe_path, googlenet_path, image_files)
		print "Retrieved imagenet labels...\n"

		format_output.output_labels(rel_clip_path + output_filename, output_filename, all_timestamps, image_files, shot_boundaries, classifier_label_list,
		 finetune_output, finetune_labels, googlenet_cat, googlenet_labels, scene_type_list, places_labels, scene_attributes_list, person_count, obj_loc_set)
		# shutil.rmtree(clip_dir)
		print "Processing complete!\n"

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()