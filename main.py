import os, sys, time
os.environ["GLOG_minloglevel"] = "2"
import fileops
import keyframes
import shot_detect
import facedetect
import placesCNN
import googlenet
import age_genderCNN

def main():

	clip_dir = '/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/'
	root = '/home/shruti/gsoc/news-shot-classification'
	#clip_dir = '/home/shruti/gsoc/misc/frames/'
	if clip_dir[-1] is not '/':
		clip_dir = clip_dir + '/'
		print clip_dir
	
	overall_start = time.time()

	clip_name = fileops.get_video_filename(clip_dir)
	shot_detect.shot_detect(clip_dir, clip_name)

	output_filename = clip_name.split('.')[0]	
	keyframes.get_keyframes(clip_dir, clip_name, output_filename)
	image_files = fileops.get_keyframeslist(clip_dir)
	#image_files = ['/home/shruti/gsoc/news-shot-classification/clips/2016-05-22_2300_US_KABC_Eyewitness_News_4PM_0-465/keyframe039.jpg']

	## Detect faces

	#faces_list1, faces_count1 = facedetect.get_faces('./data/haarcascades/haarcascade_frontalface_default.xml', image_files)
	faces_count, faces_list, faces_frameno = facedetect.get_faces(clip_dir, image_files)
	#fileops.save_faces_count(clip_dir + output_filename, faces_count)

	## Run a model and get labels for keyframe
 	
	caffe_path = '/home/shruti/gsoc/caffehome/caffe/' 
	model_path = caffe_path + 'models/placesCNN/'
		
	[age_labels, gender_labels] = age_genderCNN.age_genderCNN(caffe_path, caffe_path + 'models/age_gender/', faces_list)
	print len(gender_labels), len(faces_frameno)	
	fileops.save_age_gender_labels(clip_dir + output_filename, clip_dir + 'age_gender_labels_test', age_labels, gender_labels, faces_frameno, faces_count)

	[fc8, fc7, fc6, output_label_list, scene_type_list, label_list, scene_attributes_list] = placesCNN.placesCNN(caffe_path, model_path, image_files)
	fileops.save_placesCNN_labels(clip_dir + output_filename, clip_dir + 'placesCNN_labels', output_label_list, scene_type_list, label_list, scene_attributes_list)
	fileops.save_features(clip_dir + 'fc8', fc8)
	fileops.save_features(clip_dir + 'fc7 ', fc7)
	fileops.save_features(clip_dir + 'fc6', fc6)

	label_list = googlenet.googlenet(caffe_path, caffe_path + 'models/bvlc_googlenet/', image_files)
	fileops.save_googlenet_labels(clip_dir + output_filename, clip_dir + 'googlenet_labels', label_list)

	fileops.write_separate_labels(clip_dir + output_filename)

	overall_end = time.time()	
	print "Total time taken: %.2f" %(overall_end-overall_start)

if __name__ == '__main__':
	main()