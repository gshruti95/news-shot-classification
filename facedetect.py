import os, sys
import dlib
from skimage import io
import time

def get_faces(clip_dir, image_files, frames):

	start = time.time()

	detector = dlib.get_frontal_face_detector()
	win = dlib.image_window()
	faces_count = []
	single_faces = []
	single_faces_frameno = []
	
	temp_dir = clip_dir + 'temp/'
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)

	for idx, image in enumerate(image_files):
		
		img = io.imread(image)
		# The 1 in the second argument indicates that we should upsample the image
		# 1 time.  This will make everything bigger and allow us to detect more
		# faces.
		dets = detector(img, 1)
		# print("Number of faces detected: {}".format(len(dets)))
		faces_count.append(len(dets))

		if 0 < len(dets) < 2:
			single_faces_frameno.append(frames[idx][0])
			img_cropped = img[dets[0].top():dets[0].bottom(), dets[0].left():dets[0].right()]
			io.imsave(temp_dir + str(frames[idx][0]) + '.jpg', img_cropped)
			single_faces.append(temp_dir + str(frames[idx][0]) + '.jpg')
	
		win.clear_overlay()
		win.set_image(img)
		win.add_overlay(dets)
	
		# dets, scores, idx = detector.run(img, 1)
		# for i, d in enumerate(dets):
			# print("Detection {}, score: {}, face_type:{}".format(
				# d, scores[i], idx[i]))
	end = time.time()	
	print "Face detection time: %.2f" %(end-start)

	return faces_count , single_faces , single_faces_frameno
		